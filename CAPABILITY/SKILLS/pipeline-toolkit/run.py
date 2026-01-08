#!/usr/bin/env python3
"""
Pipeline Toolkit - Unified pipeline DAG operations skill.

Consolidates: pipeline-dag-scheduler, pipeline-dag-receipts, pipeline-dag-restore

Operations:
  - schedule: Deterministic DAG scheduling
  - receipts: Distributed execution receipts
  - restore: Receipt-gated DAG restoration

Note: This skill is a governance placeholder. Actual implementation is in
CATALYTIC-DPT/PIPELINES/ and TOOLS/catalytic.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

DURABLE_ROOTS = ["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]


def get_writer() -> GuardedWriter:
    """Get a configured GuardedWriter instance."""
    if not GuardedWriter:
        raise RuntimeError("GuardedWriter not available")
    writer = GuardedWriter(project_root=PROJECT_ROOT, durable_roots=DURABLE_ROOTS)
    writer.open_commit_gate()
    return writer


def write_output(output_path: Path, data: Dict[str, Any], writer: GuardedWriter) -> None:
    """Write JSON output using GuardedWriter."""
    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(data, indent=2, sort_keys=True) + "\n")


def op_schedule(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Deterministic DAG scheduling (governance placeholder)."""
    out = {
        "ok": False,
        "code": "NOT_IMPLEMENTED",
        "details": {
            "message": "Use repo implementation + tests; skill runner is a governance placeholder.",
            "operation": "schedule",
            "dag_spec_path": payload.get("dag_spec_path"),
            "runs_root": payload.get("runs_root", "CONTRACTS/_runs"),
        },
    }
    write_output(output_path, out, writer)
    return 0


def op_receipts(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Distributed execution receipts (governance placeholder)."""
    out = {
        "ok": False,
        "code": "NOT_IMPLEMENTED",
        "details": {
            "message": "Use repo implementation + tests; skill runner is a governance placeholder.",
            "operation": "receipts",
            "dag_spec_path": payload.get("dag_spec_path"),
            "runs_root": payload.get("runs_root", "CONTRACTS/_runs"),
        },
    }
    write_output(output_path, out, writer)
    return 0


def op_restore(payload: Dict[str, Any], output_path: Path, writer: GuardedWriter) -> int:
    """Receipt-gated DAG restoration (governance placeholder)."""
    out = {
        "ok": False,
        "code": "NOT_IMPLEMENTED",
        "details": {
            "message": "Use repo implementation + tests; skill runner is a governance placeholder.",
            "operation": "restore",
            "dag_spec_path": payload.get("dag_spec_path"),
            "runs_root": payload.get("runs_root", "CONTRACTS/_runs"),
        },
    }
    write_output(output_path, out, writer)
    return 0


OPERATIONS = {
    "schedule": op_schedule,
    "receipts": op_receipts,
    "restore": op_restore,
}


def main(input_path: Path, output_path: Path) -> int:
    """Main entry point."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    operation = payload.get("operation")
    if not operation:
        print("Error: 'operation' field is required")
        return 1

    if operation not in OPERATIONS:
        print(f"Error: Unknown operation '{operation}'. Valid: {', '.join(OPERATIONS.keys())}")
        return 1

    try:
        writer = get_writer()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    return OPERATIONS[operation](payload, output_path, writer)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
