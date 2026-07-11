#!/usr/bin/env python3
"""Closed process-scan custody shared by Gate A runtime and transport cleanup.

Qualification uses a synthetic clean process listing as its null baseline, then
mutates command, stream, parser, return-code, and forbidden-hit custody fields.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from typing import Any, Callable

PROCESS_COMMAND = ("ps", "-ww", "-eo", "pid=,comm=,args=")
PROCESS_PHASES = frozenset({"pre_runtime", "post_runtime", "post_cleanup"})
FORBIDDEN_MARKERS = (
    "combined_pdn_runner",
    "run_combined_campaign",
    "explicit_slot_runtime",
    "wrmsr",
    "rdmsr",
    "cpupower",
    "turbostat",
    "gate_a_worker",
)
RECEIPT_KEYS = frozenset({
    "schema_id",
    "phase",
    "exact_command",
    "command_sha256",
    "return_code",
    "raw_stdout",
    "raw_stderr",
    "raw_stdout_base64",
    "raw_stderr_base64",
    "stdout_sha256",
    "stderr_sha256",
    "parsed_forbidden_hits",
    "forbidden_filter_evaluated",
    "scan_complete",
    "timed_out",
    "failure",
})


class ProcessCustodyError(RuntimeError):
    pass


CommandRunner = Callable[..., subprocess.CompletedProcess]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProcessCustodyError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="surrogatepass")
    raise TypeError(f"process stream has unsupported type: {type(value).__name__}")


def _display(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _parse_hits(stdout: bytes) -> tuple[list[dict[str, Any]], str | None]:
    try:
        text = stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return [], "PROCESS_STDOUT_NOT_UTF8"
    if not text.strip():
        return [], "PROCESS_STDOUT_EMPTY"
    hits: list[dict[str, Any]] = []
    parsed = 0
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        match = re.match(r"^\s*(\d+)\s+(\S+)\s*(.*)$", raw_line)
        if match is None:
            return [], "PROCESS_STDOUT_MALFORMED"
        parsed += 1
        pid, comm, args = match.groups()
        haystack = f"{comm} {args}"
        for marker in FORBIDDEN_MARKERS:
            if marker in haystack:
                hits.append({
                    "marker": marker,
                    "pid": int(pid),
                    "comm": comm,
                    "args": args,
                    "line_sha256": hashlib.sha256(raw_line.encode("utf-8")).hexdigest(),
                })
    if parsed == 0:
        return [], "PROCESS_STDOUT_EMPTY"
    return hits, None


def _receipt(
    phase: str,
    *,
    return_code: int | None,
    stdout: bytes,
    stderr: bytes,
    timed_out: bool,
    failure: str | None,
) -> dict[str, Any]:
    hits, parse_failure = _parse_hits(stdout) if return_code == 0 and not timed_out and failure is None else ([], None)
    if failure is None and parse_failure is not None:
        failure = parse_failure
    complete = return_code == 0 and not timed_out and failure is None
    return {
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_PROCESS_RECEIPT_V1",
        "phase": phase,
        "exact_command": list(PROCESS_COMMAND),
        "command_sha256": _canonical_sha256(list(PROCESS_COMMAND)),
        "return_code": return_code,
        "raw_stdout": _display(stdout),
        "raw_stderr": _display(stderr),
        "raw_stdout_base64": base64.b64encode(stdout).decode("ascii"),
        "raw_stderr_base64": base64.b64encode(stderr).decode("ascii"),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "parsed_forbidden_hits": hits,
        "forbidden_filter_evaluated": complete,
        "scan_complete": complete,
        "timed_out": timed_out,
        "failure": failure,
    }


def scan_processes(
    phase: str,
    *,
    runner: CommandRunner = subprocess.run,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Return a receipt even when inspection fails so failure is retainable."""

    require(phase in PROCESS_PHASES, "unknown process-scan phase")
    try:
        completed = runner(
            list(PROCESS_COMMAND),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
        stdout = _bytes(completed.stdout)
        stderr = _bytes(completed.stderr)
        failure = None if completed.returncode == 0 else "PROCESS_COMMAND_NONZERO"
        return _receipt(
            phase,
            return_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            failure=failure,
        )
    except subprocess.TimeoutExpired as exc:
        return _receipt(
            phase,
            return_code=None,
            stdout=_bytes(exc.stdout),
            stderr=_bytes(exc.stderr),
            timed_out=True,
            failure="PROCESS_COMMAND_TIMEOUT",
        )
    except (OSError, TypeError, UnicodeError) as exc:
        return _receipt(
            phase,
            return_code=None,
            stdout=b"",
            stderr=str(exc).encode("utf-8", errors="replace"),
            timed_out=False,
            failure=f"PROCESS_COMMAND_UNOBSERVABLE:{type(exc).__name__}",
        )


def validate_process_receipt(receipt: dict[str, Any], *, expected_phase: str) -> None:
    require(expected_phase in PROCESS_PHASES, "unknown expected process-scan phase")
    require(isinstance(receipt, dict) and set(receipt) == RECEIPT_KEYS, "process receipt key set mismatch")
    require(receipt["schema_id"] == "CAT_CAS_PHASE6B6_GATE_A_PROCESS_RECEIPT_V1", "process receipt schema mismatch")
    require(receipt["phase"] == expected_phase, "process receipt phase mismatch")
    require(receipt["exact_command"] == list(PROCESS_COMMAND), "process command mismatch")
    require(receipt["command_sha256"] == _canonical_sha256(list(PROCESS_COMMAND)), "process command digest mismatch")
    try:
        stdout = base64.b64decode(receipt["raw_stdout_base64"], validate=True)
        stderr = base64.b64decode(receipt["raw_stderr_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ProcessCustodyError("process stream base64 malformed") from exc
    require(_display(stdout) == receipt["raw_stdout"], "process stdout display mismatch")
    require(_display(stderr) == receipt["raw_stderr"], "process stderr display mismatch")
    require(hashlib.sha256(stdout).hexdigest() == receipt["stdout_sha256"], "process stdout digest mismatch")
    require(hashlib.sha256(stderr).hexdigest() == receipt["stderr_sha256"], "process stderr digest mismatch")
    hits, parse_failure = _parse_hits(stdout)
    require(parse_failure is None, f"process listing malformed: {parse_failure}")
    require(hits == receipt["parsed_forbidden_hits"], "process forbidden-hit parsing mismatch")
    require(receipt["return_code"] == 0, "process scan returned nonzero")
    require(receipt["timed_out"] is False, "process scan timed out")
    require(receipt["failure"] is None, "process scan reported observation failure")
    require(receipt["forbidden_filter_evaluated"] is True, "forbidden filter was not evaluated")
    require(receipt["scan_complete"] is True, "process scan incomplete")
    require(receipt["parsed_forbidden_hits"] == [], "forbidden process present")


def render_remote_scan_script(phase: str) -> str:
    """Render the same closed scanner for a target-side post-cleanup SSH call."""

    require(phase in PROCESS_PHASES, "unknown remote process-scan phase")
    return f'''import base64, hashlib, json, re, subprocess\ncommand = {list(PROCESS_COMMAND)!r}\nmarkers = {list(FORBIDDEN_MARKERS)!r}\nphase = {phase!r}\ntry:\n    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)\n    out, err, rc, timed_out, failure = completed.stdout, completed.stderr, completed.returncode, False, None\n    if rc != 0: failure = "PROCESS_COMMAND_NONZERO"\nexcept subprocess.TimeoutExpired as exc:\n    out, err, rc, timed_out, failure = exc.stdout or b"", exc.stderr or b"", None, True, "PROCESS_COMMAND_TIMEOUT"\nexcept OSError as exc:\n    out, err, rc, timed_out, failure = b"", str(exc).encode("utf-8", "replace"), None, False, "PROCESS_COMMAND_UNOBSERVABLE:" + type(exc).__name__\nhits = []\nif rc == 0 and not timed_out and failure is None:\n    try:\n        text = out.decode("utf-8", "strict")\n        if not text.strip(): raise ValueError("empty")\n        for line in text.splitlines():\n            if not line.strip(): continue\n            match = re.match(r"^\\s*(\\d+)\\s+(\\S+)\\s*(.*)$", line)\n            if match is None: raise ValueError("malformed")\n            pid, comm, args = match.groups()\n            for marker in markers:\n                if marker in comm + " " + args:\n                    hits.append({{"marker": marker, "pid": int(pid), "comm": comm, "args": args, "line_sha256": hashlib.sha256(line.encode()).hexdigest()}})\n    except UnicodeDecodeError:\n        failure = "PROCESS_STDOUT_NOT_UTF8"\n    except ValueError as exc:\n        failure = "PROCESS_STDOUT_EMPTY" if str(exc) == "empty" else "PROCESS_STDOUT_MALFORMED"\ncomplete = rc == 0 and not timed_out and failure is None\ncanon = json.dumps(command, sort_keys=True, separators=(",", ":")).encode()\nreceipt = {{"schema_id":"CAT_CAS_PHASE6B6_GATE_A_PROCESS_RECEIPT_V1","phase":phase,"exact_command":command,"command_sha256":hashlib.sha256(canon).hexdigest(),"return_code":rc,"raw_stdout":out.decode("utf-8","replace"),"raw_stderr":err.decode("utf-8","replace"),"raw_stdout_base64":base64.b64encode(out).decode(),"raw_stderr_base64":base64.b64encode(err).decode(),"stdout_sha256":hashlib.sha256(out).hexdigest(),"stderr_sha256":hashlib.sha256(err).hexdigest(),"parsed_forbidden_hits":hits,"forbidden_filter_evaluated":complete,"scan_complete":complete,"timed_out":timed_out,"failure":failure}}\nprint(json.dumps(receipt, sort_keys=True))\n'''
