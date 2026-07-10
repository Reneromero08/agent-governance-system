#!/usr/bin/env python3
"""One-shot SSH/SCP transport for a future exactly-authorized Gate A smoke.

Nothing in this module runs at import time.  The host adapter imports and
constructs :class:`SshScpTransport` only after exact authority validation.  The
command runner is injectable; qualification tests use a fake transport and do
not construct this class or open a network connection.

The non-driving qualification baseline is zero command-runner calls before an
exact committed authority passes and zero transfer calls when namespace state
is present, unobservable, or malformed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import build_gate_a_execution_bundle as bundle


class TransportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportError(message)


@dataclass(frozen=True)
class HostExecutionRequest:
    target: str
    authority_path: Path
    authority_sha256: str
    reviewed_adapter_head: str
    independent_review_id: int
    execution_bundle_sha256: str
    schedule_sha256: str
    namespace_sha256: str
    remote_execution_root: str
    remote_output_root: str
    local_evidence_root: Path


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )


def _json_stdout(completed: subprocess.CompletedProcess[str], context: str) -> dict[str, Any]:
    require(completed.returncode == 0, f"{context} failed: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TransportError(f"{context} returned malformed JSON") from exc
    require(isinstance(value, dict), f"{context} must return an object")
    return value


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    root = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            require(member.isfile(), f"non-file evidence member rejected: {member.name}")
            target = (destination / member.name).resolve()
            require(os.path.commonpath((str(root), str(target))) == str(root), f"evidence member escapes root: {member.name}")
            require(not (destination / member.name).exists(), f"duplicate evidence member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            require(source is not None, f"evidence member unreadable: {member.name}")
            with target.open("xb") as output:
                output.write(source.read())
                output.flush()
                os.fsync(output.fileno())


def _inventory(root: Path) -> tuple[dict[str, Any], str]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), f"invalid evidence path: {path}")
        data = path.read_bytes()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    require(files, "copy-back evidence is empty")
    inventory = {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_EVIDENCE_INVENTORY_V1",
        "files": files,
    }
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return inventory, hashlib.sha256(canonical).hexdigest()


class SshScpTransport:
    """Closed, no-retry transport instantiated only after authority validates."""

    def __init__(self, *, command_runner: CommandRunner = _run_command):
        self._run = command_runner
        self._used = False

    def _command(
        self,
        argv: list[str],
        *,
        input_text: str | None = None,
        timeout: int = 120,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = self._run(argv, input_text=input_text, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TransportError(f"command timed out ({argv[0]}) with no retry") from exc
        if check:
            require(completed.returncode == 0, f"command failed ({argv[0]}): {completed.stderr.strip()}")
        return completed

    def execute(self, request: HostExecutionRequest) -> dict[str, Any]:
        require(not self._used, "transport may execute only once")
        self._used = True
        require(request.target == "root@192.168.137.100", "target mismatch")
        require(not request.local_evidence_root.exists(), "local evidence root must be absent")
        prefix = f"/root/.catcas_gate_a_{request.authority_sha256[:16]}"
        remote_stage = prefix + ".bundle.tar"
        remote_authority = prefix + ".authority.json"
        remote_archive = prefix + ".evidence.tar"
        remote_receipt = prefix + ".copy_back.json"

        with tempfile.TemporaryDirectory(prefix="gate_a_authorized_transport_") as tmp:
            temp = Path(tmp)
            deployment = temp / "bundle.tar"
            copied = temp / "evidence.tar"
            receipt_path = temp / "copy_back.json"
            bundle.write_deployment_archive(deployment, "HEAD")

            preflight_script = f'''import json, os\n\ndef state(path):\n    try:\n        os.lstat(path)\n    except FileNotFoundError:\n        return "absent"\n    except OSError as exc:\n        return "unobservable:" + type(exc).__name__\n    return "present"\n\nparent = "/root"\nprefix = {prefix!r}\ntry:\n    matches = sorted(os.path.join(parent, n) for n in os.listdir(parent) if os.path.join(parent, n).startswith(prefix))\nexcept OSError as exc:\n    print(json.dumps({{"inspection_complete": False, "error": type(exc).__name__}}))\n    raise SystemExit(0)\nprint(json.dumps({{\n    "inspection_complete": True,\n    "execution_root": state({request.remote_execution_root!r}),\n    "output_root": state({request.remote_output_root!r}),\n    "stage": state({remote_stage!r}),\n    "archive": state({remote_archive!r}),\n    "prefix_matches": matches,\n}}, sort_keys=True))\n'''
            preflight = _json_stdout(
                self._command(["ssh", request.target, "python3", "-"], input_text=preflight_script),
                "remote namespace preflight",
            )
            require(set(preflight) == {"inspection_complete", "execution_root", "output_root", "stage", "archive", "prefix_matches"}, "preflight key set mismatch")
            require(preflight["inspection_complete"] is True, "remote namespace unobservable")
            for key in ("execution_root", "output_root", "stage", "archive"):
                require(preflight[key] == "absent", f"remote {key} is not absent")
            require(preflight["prefix_matches"] == [], "authority-bound remote prefix collision")

            self._command(["scp", str(deployment), f"{request.target}:{remote_stage}"], timeout=120)
            self._command(["scp", str(request.authority_path), f"{request.target}:{remote_authority}"], timeout=120)

            execute_script = f'''import json, os, pathlib, signal, subprocess, sys, tarfile\nroot = pathlib.Path({request.remote_execution_root!r})\nstage = pathlib.Path({remote_stage!r})\noutput = pathlib.Path({request.remote_output_root!r})\narchive_path = pathlib.Path({remote_archive!r})\nroot.mkdir(mode=0o700, parents=False, exist_ok=False)\nwith tarfile.open(stage, "r") as archive:\n    base = root.resolve()\n    for member in archive.getmembers():\n        target = (root / member.name).resolve()\n        if not member.isfile() or os.path.commonpath((str(base), str(target))) != str(base):\n            raise SystemExit("unsafe deployment member")\n    archive.extractall(root)\ncmd = [sys.executable, "-B", str(root / "adapter/gate_a_target_runner.py"), "--execute-authorized",\n       "--authority-artifact", {remote_authority!r}, "--authority-sha256", {request.authority_sha256!r},\n       "--execution-bundle-sha256", {request.execution_bundle_sha256!r},\n       "--source-head", {request.reviewed_adapter_head!r}, "--independent-review-id", {str(request.independent_review_id)!r},\n       "--schedule-sha256", {request.schedule_sha256!r}, "--target", {request.target!r},\n       "--namespace-sha256", {request.namespace_sha256!r}, "--output-root", str(output)]\nprocess = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)\ntimed_out = False\ntry:\n    runner_stdout, runner_stderr = process.communicate(timeout=45)\nexcept subprocess.TimeoutExpired:\n    timed_out = True\n    os.killpg(process.pid, signal.SIGTERM)\n    try:\n        runner_stdout, runner_stderr = process.communicate(timeout=2)\n    except subprocess.TimeoutExpired:\n        os.killpg(process.pid, signal.SIGKILL)\n        runner_stdout, runner_stderr = process.communicate()\nif output.exists():\n    with tarfile.open(archive_path, "w") as archive:\n        for path in sorted(output.rglob("*")):\n            if path.is_file() and not path.is_symlink():\n                archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)\nprint(json.dumps({{"runner_return_code": process.returncode, "runner_stdout": runner_stdout, "runner_stderr": runner_stderr, "target_timeout": timed_out, "evidence_archive_created": archive_path.is_file()}}, sort_keys=True))\n'''
            execution = _json_stdout(
                self._command(["ssh", request.target, "python3", "-"], input_text=execute_script, timeout=180, check=True),
                "authorized target execution",
            )
            require(set(execution) == {"runner_return_code", "runner_stdout", "runner_stderr", "target_timeout", "evidence_archive_created"}, "execution receipt key set mismatch")
            require(execution["evidence_archive_created"] is True, "target evidence archive missing")
            self._command(["scp", f"{request.target}:{remote_archive}", str(copied)], timeout=120)
            _safe_extract(copied, request.local_evidence_root)
            inventory, inventory_sha256 = _inventory(request.local_evidence_root)
            inventory_path = request.local_evidence_root / "COPY_BACK_INVENTORY.json"
            inventory_path.write_text(json.dumps(inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            receipt = {
                "schema_id": "CAT_CAS_PHASE6B6_GATE_A_COPY_BACK_RECEIPT_V1",
                "remote_output_root": request.remote_output_root,
                "authority_sha256": request.authority_sha256,
                "execution_bundle_sha256": request.execution_bundle_sha256,
                "retained_evidence_custody_verified": True,
                "evidence_inventory_sha256": inventory_sha256,
                "copy_back_complete": True,
            }
            receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            self._command(["scp", str(receipt_path), f"{request.target}:{remote_receipt}"], timeout=120)

            cleanup_script = f'''import json, pathlib, shutil, subprocess, sys\nroot = pathlib.Path({request.remote_execution_root!r})\noutput = pathlib.Path({request.remote_output_root!r})\nreceipt = pathlib.Path({remote_receipt!r})\ncmd = [sys.executable, "-B", str(root / "adapter/gate_a_target_runner.py"), "--cleanup-after-verified-copy", "--output-root", str(output), "--authority-sha256", {request.authority_sha256!r}, "--execution-bundle-sha256", {request.execution_bundle_sha256!r}, "--copy-back-receipt", str(receipt)]\ncompleted = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)\nif completed.returncode == 0:\n    shutil.rmtree(root)\n    for path in ({remote_stage!r}, {remote_authority!r}, {remote_archive!r}, {remote_receipt!r}):\n        try:\n            pathlib.Path(path).unlink()\n        except FileNotFoundError:\n            pass\nprint(json.dumps({{"cleanup_return_code": completed.returncode, "execution_root_absent": not root.exists(), "output_root_absent": not output.exists()}}, sort_keys=True))\n'''
            cleanup = _json_stdout(
                self._command(["ssh", request.target, "python3", "-"], input_text=cleanup_script, timeout=120),
                "verified remote cleanup",
            )
            require(cleanup == {"cleanup_return_code": 0, "execution_root_absent": True, "output_root_absent": True}, "remote cleanup not verified")
            require(execution["target_timeout"] is False, "target-local execution timeout fired")
            require(execution["runner_return_code"] == 0, f"authorized runtime failed after evidence copy-back: {execution['runner_stderr'].strip()}")
            try:
                runtime_result = json.loads(execution["runner_stdout"])
            except json.JSONDecodeError as exc:
                raise TransportError("target runner result malformed") from exc
            return {
                "status": "GATE_A_AUTHORIZED_TRANSPORT_COMPLETE",
                "target_runner_result": runtime_result,
                "copy_back_inventory_sha256": inventory_sha256,
                "cleanup_verified": True,
                "automatic_retry": False,
                "transport_execution_count": 1,
            }
