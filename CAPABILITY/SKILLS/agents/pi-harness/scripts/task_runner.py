#!/usr/bin/env python3
"""Run one Pi turn, monitor resource caps, and emit an integrity receipt."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pi_harness import build_pi_command, inspect_jsonl, sha256_text
from state_io import atomic_write, file_lock, read_json

CREATE_NO_WINDOW = 0x08000000
POLL_SECONDS = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminate_tree(process: subprocess.Popen[Any], force: bool = False) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                creationflags=CREATE_NO_WINDOW,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        process.kill()


def update_task(task_path: Path, lock_path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    with file_lock(lock_path):
        current = read_json(task_path)
        if current.get("status") == "cancelled":
            return current
        current.update(updates)
        atomic_write(task_path, current)
        return current


def register_agent_pid(task_path: Path, lock_path: Path, pid: int) -> dict[str, Any]:
    with file_lock(lock_path):
        current = read_json(task_path)
        current["agent_pid"] = pid
        atomic_write(task_path, current)
        return current


def finalize_task(
    task_path: Path,
    lock_path: Path,
    updates: dict[str, Any],
    receipt: dict[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    with file_lock(lock_path):
        current = read_json(task_path)
        if current.get("status") == "cancelled":
            return current
        current.update(updates)
        current["receipt_path"] = str(receipt_path)
        atomic_write(receipt_path, receipt)
        atomic_write(task_path, current)
        return current


def run_task(spec_path: Path) -> int:
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    task_path = Path(spec["task_path"]).resolve()
    lock_path = Path(spec["task_lock_path"]).resolve()
    tasks_dir = spec_path.parent
    if task_path.parent != tasks_dir or lock_path.parent != tasks_dir:
        raise ValueError("task or lock path escapes task directory")
    task = update_task(task_path, lock_path, {"status": "running", "runner_pid": os.getpid()})
    if task.get("status") == "cancelled":
        return 1

    stdout_path = Path(task["stdout_path"]).resolve()
    stderr_path = Path(task["stderr_path"]).resolve()
    receipt_path = Path(task["receipt_path"]).resolve()
    log_root = (tasks_dir.parent / "logs").resolve()
    if not stdout_path.is_relative_to(log_root) or not stderr_path.is_relative_to(log_root):
        raise ValueError("task log path escapes harness log directory")
    if receipt_path.parent != tasks_dir:
        raise ValueError("task receipt path escapes task directory")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    flags = CREATE_NO_WINDOW if os.name == "nt" else 0
    timeout = int(spec["timeout"])
    max_log_bytes = int(spec["max_log_bytes"])
    deadline = time.monotonic() + timeout
    failure = ""
    exit_code = 1
    process: subprocess.Popen[Any] | None = None

    try:
        command = build_pi_command(spec["worker"], spec["prompt"], spec["pi_command"])
        with file_lock(lock_path):
            if read_json(task_path).get("status") == "cancelled":
                return 1
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, \
                stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
            process = subprocess.Popen(
                command,
                cwd=spec["worker"]["workspace"],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                text=True,
                encoding="utf-8",
                errors="replace",
                close_fds=os.name != "nt",
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
            registered = register_agent_pid(task_path, lock_path, process.pid)
            if registered.get("status") == "cancelled":
                terminate_tree(process)
            while process.poll() is None:
                stdout.flush()
                stderr.flush()
                if stdout_path.stat().st_size + stderr_path.stat().st_size > max_log_bytes:
                    failure = f"Pi logs exceeded {max_log_bytes} bytes"
                    exit_code = 125
                    terminate_tree(process)
                    break
                if time.monotonic() >= deadline:
                    failure = f"Pi exceeded timeout of {timeout} seconds"
                    exit_code = 124
                    terminate_tree(process)
                    break
                time.sleep(POLL_SECONDS)
            try:
                observed_code = process.wait(timeout=10)
                if not failure:
                    exit_code = observed_code
            except subprocess.TimeoutExpired:
                terminate_tree(process, force=True)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                failure = failure or "Pi process did not terminate"
                exit_code = 126
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        failure = str(exc)
        if process is not None:
            terminate_tree(process)

    raw = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    integrity = inspect_jsonl(
        raw,
        workspace=str(spec["worker"]["workspace"]),
        write_roots=spec["worker"].get("write_roots", []),
    )
    if exit_code == 0 and not integrity["integrity_ok"]:
        failure = "Pi output failed integrity checks"
        exit_code = 65
    status = "complete" if exit_code == 0 else "failed"
    receipt = {
        "task_id": task["task_id"],
        "worker_id": task["worker_id"],
        "session_id": task["session_id"],
        "prompt_sha256": sha256_text(str(spec["prompt"])),
        "spec_sha256": sha256_file(spec_path),
        "stdout_sha256": sha256_file(stdout_path) if stdout_path.exists() else None,
        "stderr_sha256": sha256_file(stderr_path) if stderr_path.exists() else None,
        "result_sha256": integrity["result_sha256"],
        "exit_code": exit_code,
        "status": status,
        "integrity": integrity,
    }
    final = finalize_task(
        task_path,
        lock_path,
        {
            "status": status,
            "exit_code": exit_code,
            "result": integrity["result"],
            "error": failure,
            "integrity": integrity,
        },
        receipt,
        receipt_path,
    )
    return 0 if final.get("status") == "complete" else 1


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <task-spec.json>", file=sys.stderr)
        return 2
    return run_task(Path(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
