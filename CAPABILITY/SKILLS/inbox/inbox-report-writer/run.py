#!/usr/bin/env python3
"""Inbox report writer skill runner."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALLOWED_OUTPUT_ROOT = (PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs").resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat
import generate_inbox_ledger
import update_inbox_index
import hash_inbox_file


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def _atomic_write_bytes(path: Path, data: bytes, writer: Any = None) -> None:
    if not writer:
        raise RuntimeError("GuardedWriter required")
    # write_durable expects str, so decode if it's JSON bytes (which are ASCII/UTF-8 compatible)
    writer.write_durable(str(path), data.decode('utf-8'))


def _resolve_repo_path(path_str: str) -> Path:
    path = (PROJECT_ROOT / path_str).resolve()
    if not str(path).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path escapes repo root: {path_str}")
    return path


def _ensure_output_path(path: Path) -> None:
    if not str(path).startswith(str(ALLOWED_OUTPUT_ROOT)):
        raise ValueError(f"Output path must be under LAW/CONTRACTS/_runs: {path}")


def _normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    errors: List[str] = []
    status = "success"

    operation = str(payload.get("operation", "")).strip() or "generate_ledger"
    inbox_path = str(payload.get("inbox_path", "INBOX"))

    output: Dict[str, Any] = {
        "operation": operation,
        "status": "success",
        "ledger_path": "",
        "index_updated": False,
        "hash_valid": None,
        "errors": [],
    }

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS", "INBOX"])
    writer.open_commit_gate()

    try:
        if operation == "generate_ledger":
            ledger_path_str = payload.get("ledger_path")
            if not ledger_path_str:
                raise ValueError("ledger_path is required for generate_ledger")
            ledger_path = _resolve_repo_path(str(ledger_path_str))
            _ensure_output_path(ledger_path)
            # scanner: guarded
            writer.mkdir_durable(str(ledger_path.parent))
            inbox_dir = _resolve_repo_path(inbox_path)
            generate_inbox_ledger.generate_ledger(inbox_dir, ledger_path, quiet=True, writer=writer)
            output["ledger_path"] = _normalize_path(ledger_path.relative_to(PROJECT_ROOT))
        elif operation == "update_index":
            allow_write = bool(payload.get("allow_inbox_write", False))
            if not allow_write:
                raise ValueError("allow_inbox_write must be true for update_index")
            inbox_dir = _resolve_repo_path(inbox_path)
            updated = update_inbox_index.update_inbox_index(inbox_dir, quiet=True)
            output["index_updated"] = bool(updated)
        elif operation == "verify_hash":
            file_path_str = payload.get("file_path")
            if not file_path_str:
                raise ValueError("file_path is required for verify_hash")
            file_path = _resolve_repo_path(str(file_path_str))
            valid, stored_hash, computed_hash = hash_inbox_file.verify_hash(file_path)
            output["hash_valid"] = bool(valid)
            output["hash_stored"] = stored_hash
            output["hash_computed"] = computed_hash
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as exc:
        errors.append(str(exc))
        status = "error"

    output["status"] = status
    output["errors"] = errors

    _atomic_write_bytes(output_path, _canonical_json_bytes(output), writer)

    return 0 if status == "success" else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
