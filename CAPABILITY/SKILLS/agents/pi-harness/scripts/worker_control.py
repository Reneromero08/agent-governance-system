#!/usr/bin/env python3
"""Persistent background worker control plane for the local Pi CLI."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from context_pack import pack_context
from pi_harness import READ_ONLY_TOOLS, build_task_packet, resolve_executable, sha256_text
from state_io import atomic_write, file_lock, read_json

SKILL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_DIR.parents[3]
DEFAULT_STATE_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "pi-harness"
CREATE_NO_WINDOW = 0x08000000
ALLOWED_STATE_ROOT = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
MAX_LOG_RETURN_BYTES = 1024 * 1024


def validate_identifier(value: str, label: str, max_length: int) -> str:
    if not value or len(value) > max_length or any(ch not in IDENTIFIER_CHARS for ch in value):
        raise ValueError(f"{label} must be 1-{max_length} letters, digits, hyphens, or underscores")
    return value


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(process_query, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_log_tail(path: Path, max_bytes: int = MAX_LOG_RETURN_BYTES) -> tuple[str, bool, int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size > max_bytes:
            stream.seek(-max_bytes, os.SEEK_END)
        data = stream.read(max_bytes)
    return data.decode("utf-8", errors="replace"), size > max_bytes, size


def split_values(values: Iterable[str] | str | None) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    return [str(value).strip() for value in values if str(value).strip()]


def resolve_scopes(workspace: Path, values: Iterable[str] | str | None) -> List[str]:
    resolved: List[str] = []
    for value in split_values(values):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        candidate = candidate.resolve()
        if not candidate.is_relative_to(workspace):
            raise ValueError(f"scope escapes workspace: {value}")
        resolved.append(str(candidate))
    return resolved


def resolve_shell_programs(values: Iterable[str] | str | None) -> Dict[str, str]:
    programs: Dict[str, str] = {}
    for value in split_values(values):
        alias, separator, command = value.partition("=")
        if not separator:
            command = alias
            alias = Path(command).stem
        validate_identifier(alias, "shell program alias", 64)
        executable = resolve_executable(command)
        if os.name == "nt" and Path(executable).suffix.lower() not in {".exe", ".com"}:
            raise ValueError("governed shell programs on Windows must be native .exe or .com files")
        if os.name != "nt" and not os.access(executable, os.X_OK):
            raise ValueError(f"governed shell program is not executable: {executable}")
        if alias in programs:
            raise ValueError(f"duplicate shell program alias: {alias}")
        programs[alias] = executable
        if len(programs) > 32:
            raise ValueError("governed shell supports at most 32 programs")
    return programs


class WorkerController:
    def __init__(self, state_dir: Path | str = DEFAULT_STATE_DIR, pi_command: Optional[str] = None):
        self.state_dir = Path(state_dir).resolve()
        self.workers_dir = self.state_dir / "workers"
        self.tasks_dir = self.state_dir / "tasks"
        self.sessions_dir = self.state_dir / "sessions"
        self.logs_dir = self.state_dir / "logs"
        self.pi_command = pi_command or os.environ.get("PI_COMMAND", "pi")

    def _worker_path(self, worker_id: str) -> Path:
        validate_identifier(worker_id, "worker_id", 64)
        return self.workers_dir / f"{worker_id}.json"

    def _worker_lock(self, worker_id: str) -> Path:
        validate_identifier(worker_id, "worker_id", 64)
        return self.workers_dir / f"{worker_id}.lock"

    def _task_path(self, task_id: str) -> Path:
        validate_identifier(task_id, "task_id", 80)
        return self.tasks_dir / f"{task_id}.json"

    def _task_lock(self, task_id: str) -> Path:
        validate_identifier(task_id, "task_id", 80)
        return self.tasks_dir / f"{task_id}.lock"

    def _read(self, path: Path) -> Dict[str, Any]:
        return read_json(path)

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        path = self._worker_path(worker_id)
        return self._read(path) if path.exists() else None

    def list_workers(self) -> List[Dict[str, Any]]:
        if not self.workers_dir.exists():
            return []
        return [self._read(path) for path in sorted(self.workers_dir.glob("*.json"))]

    def create_worker(
        self,
        worker_id: str,
        workspace: str,
        session_id: str = "",
        name: str = "",
        provider: str = "",
        model: str = "",
        read_roots: Iterable[str] | str | None = None,
        write_roots: Iterable[str] | str | None = None,
        allow_write: bool = False,
        allow_shell: bool = False,
        allow_extensions: bool = False,
        shell_programs: Iterable[str] | str | None = None,
    ) -> Dict[str, Any]:
        validate_identifier(worker_id, "worker_id", 64)
        if len(name) > 128 or len(provider) > 128 or len(model) > 256:
            raise ValueError("worker name/provider/model exceeds length limit")
        ws = Path(workspace).resolve()
        if not ws.is_dir():
            raise ValueError(f"workspace does not exist: {ws}")
        reads = resolve_scopes(ws, read_roots) or [str(ws)]
        writes = resolve_scopes(ws, write_roots)
        programs = resolve_shell_programs(shell_programs)
        if (allow_write or allow_shell) and not writes:
            raise ValueError("write or shell access requires at least one --write-root")
        if allow_shell and not programs:
            raise ValueError("--allow-shell requires at least one --shell-program")
        if programs and not allow_shell:
            raise ValueError("--shell-program requires --allow-shell")
        stable_id = session_id or str(uuid.uuid5(uuid.NAMESPACE_URL, f"pi-harness:{worker_id}:{ws}"))
        try:
            uuid.UUID(stable_id)
        except ValueError as exc:
            raise ValueError("session_id must be a UUID") from exc
        tools = list(READ_ONLY_TOOLS + (("edit", "write") if allow_write else ()) + (("bash",) if allow_shell else ()))
        worker = {
            "worker_id": worker_id,
            "name": name or worker_id,
            "workspace": str(ws),
            "session_id": stable_id,
            "session_dir": str(self.sessions_dir),
            "provider": provider,
            "model": model,
            "read_roots": reads,
            "write_roots": writes,
            "tools": tools,
            "allow_write": allow_write,
            "allow_shell": allow_shell,
            "allow_extensions": allow_extensions,
            "shell_programs": programs,
            "next_task": 1,
            "active_task_id": "",
        }
        with file_lock(self._worker_lock(worker_id)):
            if self.get_worker(worker_id):
                raise ValueError(f"worker already exists: {worker_id}")
            atomic_write(self._worker_path(worker_id), worker)
        return worker

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self._task_path(task_id)
        return self._read(path) if path.exists() else None

    def _refresh_worker(self, task: Dict[str, Any]) -> None:
        if task.get("status") in {"queued", "running"}:
            return
        worker_id = str(task["worker_id"])
        with file_lock(self._worker_lock(worker_id)):
            worker = self.get_worker(worker_id)
            if worker and worker.get("active_task_id") == task.get("task_id"):
                worker["active_task_id"] = ""
                atomic_write(self._worker_path(worker_id), worker)

    def submit_task(
        self,
        worker_id: str,
        task: str,
        constraints: str = "",
        timeout: int = 1800,
        context_files: Iterable[str] | None = None,
        context_texts: Iterable[str] | None = None,
        context_token_budget: int = 0,
        context_tokenizer: str = "cl100k_base",
    ) -> Dict[str, Any]:
        with file_lock(self._worker_lock(worker_id)):
            worker = self.get_worker(worker_id)
            if not worker:
                raise ValueError(f"worker not found: {worker_id}")
            active_id = str(worker.get("active_task_id") or "")
            if active_id:
                active = self.get_task(active_id)
                if active and active.get("status") in {"queued", "running"}:
                    active_pid = int(
                        active.get("runner_pid")
                        or active.get("pid")
                        or active.get("launcher_pid")
                        or 0
                    )
                    if active_pid and not process_is_alive(active_pid):
                        with file_lock(self._task_lock(active_id)):
                            current = self.get_task(active_id) or active
                            if current.get("status") in {"queued", "running"}:
                                current.update({
                                    "status": "failed",
                                    "exit_code": 70,
                                    "error": "background runner exited without finalizing the task",
                                })
                                atomic_write(self._task_path(active_id), current)
                        active = current
                    if active.get("status") in {"queued", "running"}:
                        raise ValueError(f"worker is busy with task {active_id}")
            sequence = int(worker.get("next_task", 1))
            if sequence < 1 or sequence > 999999:
                raise ValueError("worker task sequence is invalid or exhausted")
            task_id = f"{worker_id}-{sequence:06d}"
            stdout_path = self.logs_dir / f"{task_id}.jsonl"
            stderr_path = self.logs_dir / f"{task_id}.stderr.log"
            manual_context, context_manifest = pack_context(
                workspace=worker["workspace"],
                read_roots=worker["read_roots"],
                context_files=context_files or [],
                context_texts=context_texts or [],
                token_budget=context_token_budget,
                tokenizer=context_tokenizer,
            )
            prompt = build_task_packet(
                task=task,
                workspace=worker["workspace"],
                read_roots=worker["read_roots"],
                write_roots=worker["write_roots"],
                tools=worker["tools"],
                constraints=constraints,
                shell_programs=worker.get("shell_programs", {}),
                manual_context=manual_context,
            )
            record = {
                "task_id": task_id,
                "worker_id": worker_id,
                "session_id": worker["session_id"],
                "status": "queued",
                "pid": None,
                "launcher_pid": os.getpid(),
                "runner_pid": None,
                "agent_pid": None,
                "exit_code": None,
                "result": "",
                "error": "",
                "integrity": None,
                "prompt_sha256": sha256_text(prompt),
                "context_manifest": context_manifest,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "receipt_path": str(self.tasks_dir / f"{task_id}.receipt.json"),
            }
            task_path = self._task_path(task_id)
            atomic_write(task_path, record)
            spec_path = self.tasks_dir / f"{task_id}.spec.json"
            spec = {
                "task_path": str(task_path),
                "task_lock_path": str(self._task_lock(task_id)),
                "worker": worker,
                "prompt": prompt,
                "pi_command": self.pi_command,
                "timeout": max(1, min(int(timeout), 86400)),
                "max_log_bytes": 50 * 1024 * 1024,
            }
            atomic_write(spec_path, spec)
            worker["next_task"] = sequence + 1
            worker["active_task_id"] = task_id
            atomic_write(self._worker_path(worker_id), worker)
        flags = CREATE_NO_WINDOW if os.name == "nt" else 0
        runner = Path(__file__).resolve().parent / "task_runner.py"
        try:
            process = subprocess.Popen(
                [sys.executable, str(runner), str(spec_path)],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=os.name != "nt",
                creationflags=flags,
            )
        except OSError as exc:
            with file_lock(self._task_lock(task_id)):
                failed = self.get_task(task_id) or record
                failed.update({"status": "failed", "exit_code": 127, "error": f"runner launch failed: {exc}"})
                atomic_write(task_path, failed)
            self._refresh_worker(failed)
            return failed
        with file_lock(self._task_lock(task_id)):
            current = self.get_task(task_id) or record
            current["pid"] = process.pid
            atomic_write(task_path, current)
        return current

    def task_status(self, task_id: str) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"task not found: {task_id}")
        if task.get("status") in {"queued", "running"}:
            pid = int(task.get("runner_pid") or task.get("pid") or task.get("launcher_pid") or 0)
            if pid and not process_is_alive(pid):
                with file_lock(self._task_lock(task_id)):
                    current = self.get_task(task_id) or task
                    if current.get("status") in {"queued", "running"}:
                        current.update({
                            "status": "failed",
                            "exit_code": 70,
                            "error": "background runner exited without finalizing the task",
                        })
                        atomic_write(self._task_path(task_id), current)
                    task = current
        self._refresh_worker(task)
        return task

    def task_result(self, task_id: str) -> Dict[str, Any]:
        return self.task_status(task_id)

    def task_log(self, task_id: str) -> Dict[str, Any]:
        task = self.task_status(task_id)
        stdout = Path(task["stdout_path"]).resolve()
        stderr = Path(task["stderr_path"]).resolve()
        log_root = self.logs_dir.resolve()
        if not stdout.is_relative_to(log_root) or not stderr.is_relative_to(log_root):
            raise ValueError("task log path escapes harness log directory")
        stdout_text, stdout_truncated, stdout_bytes = read_log_tail(stdout) if stdout.exists() else ("", False, 0)
        stderr_text, stderr_truncated, stderr_bytes = read_log_tail(stderr) if stderr.exists() else ("", False, 0)
        return {
            "task_id": task_id,
            "status": task["status"],
            "stdout": stdout_text,
            "stdout_bytes": stdout_bytes,
            "stdout_truncated": stdout_truncated,
            "stderr": stderr_text,
            "stderr_bytes": stderr_bytes,
            "stderr_truncated": stderr_truncated,
        }

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        task = self.task_status(task_id)
        if task["status"] not in {"queued", "running"}:
            return task
        with file_lock(self._task_lock(task_id)):
            task = self.get_task(task_id) or task
            task["status"] = "cancelled"
            task["exit_code"] = 130
            atomic_write(self._task_path(task_id), task)
        runner_pid = int(task.get("runner_pid") or task.get("pid") or task.get("launcher_pid") or 0)
        agent_pid = int(task.get("agent_pid") or 0)
        if not agent_pid and runner_pid and process_is_alive(runner_pid):
            for _ in range(20):
                current = self.get_task(task_id) or task
                agent_pid = int(current.get("agent_pid") or 0)
                if agent_pid or not process_is_alive(runner_pid):
                    break
                time.sleep(0.05)
        pid = runner_pid or agent_pid
        if pid:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                        creationflags=CREATE_NO_WINDOW,
                    )
                else:
                    if agent_pid:
                        os.killpg(agent_pid, signal.SIGTERM)
                        for _ in range(20):
                            if not process_is_alive(agent_pid):
                                break
                            time.sleep(0.05)
                        if process_is_alive(agent_pid):
                            os.killpg(agent_pid, signal.SIGKILL)
                    if runner_pid:
                        os.kill(runner_pid, signal.SIGTERM)
            except (OSError, subprocess.SubprocessError):
                pass
        self._refresh_worker(task)
        return task


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run persistent local Pi workers.")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--pi-command", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("worker-create")
    create.add_argument("--worker-id", required=True)
    create.add_argument("--workspace", default=str(PROJECT_ROOT))
    create.add_argument("--session-id", default="")
    create.add_argument("--name", default="")
    create.add_argument("--provider", default="")
    create.add_argument("--model", default="")
    create.add_argument("--read-root", action="append", default=[])
    create.add_argument("--write-root", action="append", default=[])
    create.add_argument("--allow-write", action="store_true")
    create.add_argument("--allow-shell", action="store_true")
    create.add_argument("--allow-extensions", action="store_true")
    create.add_argument("--shell-program", action="append", default=[])
    sub.add_parser("worker-list")
    get_worker = sub.add_parser("worker-get")
    get_worker.add_argument("--worker-id", required=True)

    for command in ("task-status", "task-result", "task-log", "task-cancel"):
        item = sub.add_parser(command)
        item.add_argument("--task-id", required=True)
    for command, field in (("task-submit", "task"), ("prompt", "message")):
        item = sub.add_parser(command)
        item.add_argument("--worker-id", required=True)
        item.add_argument(f"--{field}", required=True)
        item.add_argument("--constraints", default="")
        item.add_argument("--timeout", type=int, default=1800)
        item.add_argument("--context-file", action="append", default=[])
        item.add_argument("--context-text", action="append", default=[])
        item.add_argument("--context-token-budget", type=int, default=0)
        item.add_argument("--context-tokenizer", default="cl100k_base")

    args = parser.parse_args(argv)
    state_dir = Path(args.state_dir).resolve()
    if not state_dir.is_relative_to(ALLOWED_STATE_ROOT.resolve()):
        emit({"error": f"state directory must be under {ALLOWED_STATE_ROOT}"})
        return 1
    ctl = WorkerController(state_dir, pi_command=args.pi_command)
    try:
        if args.command == "worker-create":
            value = ctl.create_worker(
                worker_id=args.worker_id,
                workspace=args.workspace,
                session_id=args.session_id,
                name=args.name,
                provider=args.provider,
                model=args.model,
                read_roots=args.read_root,
                write_roots=args.write_root,
                allow_write=args.allow_write,
                allow_shell=args.allow_shell,
                allow_extensions=args.allow_extensions,
                shell_programs=args.shell_program,
            )
        elif args.command == "worker-list":
            value = ctl.list_workers()
        elif args.command == "worker-get":
            value = ctl.get_worker(args.worker_id) or {"error": "not found"}
        elif args.command in {"task-submit", "prompt"}:
            message = args.task if args.command == "task-submit" else args.message
            value = ctl.submit_task(
                args.worker_id,
                message,
                constraints=args.constraints,
                timeout=args.timeout,
                context_files=args.context_file,
                context_texts=args.context_text,
                context_token_budget=args.context_token_budget,
                context_tokenizer=args.context_tokenizer,
            )
        elif args.command == "task-status":
            value = ctl.task_status(args.task_id)
        elif args.command == "task-result":
            value = ctl.task_result(args.task_id)
        elif args.command == "task-log":
            value = ctl.task_log(args.task_id)
        else:
            value = ctl.cancel_task(args.task_id)
        emit(value)
        return 0
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        emit({"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
