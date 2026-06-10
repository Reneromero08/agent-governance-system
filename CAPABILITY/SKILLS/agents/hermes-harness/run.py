#!/usr/bin/env python3
"""Hermes Harness skill entry point (ADR-017 compliant).

Delegates to scripts/hermes_harness.py, which contains the full implementation.
This wrapper exists to satisfy the skill contract that every skill has a run.py
at its root.

Two entry paths:
    1. ADR-017 skill_run:        python run.py <input.json> <output.json>
    2. CLI delegation:           python run.py prompt|validate|run [args...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    _in_capability = True
except ImportError:
    def ensure_canon_compat(_path):
        return True
    GuardedWriter = None
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _in_capability = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_scripts = Path(__file__).resolve().parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))


def run_skill(input_path: Path, output_path: Path, writer: Optional[GuardedWriter] = None) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        raw = input_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            pass

    if not isinstance(payload, dict):
        print(f"Error: input JSON must be an object, got {type(payload).__name__}")
        return 1

    from hermes_harness import run_task, build_harness_prompt  # noqa: E402

    if "output" in payload and "output_contract" not in payload:
        payload["output_contract"] = payload["output"]

    if os.environ.get("CI") == "true":
        import sys as _sys
        print(f"[hermes-harness] CI=true detected, skipping API call", file=_sys.stderr)
        prompt = build_harness_prompt(
            task=str(payload.get("task", "")),
            workspace=payload.get("workspace", str(PROJECT_ROOT)),
            mode=payload.get("mode", "auto"),
            max_workers=int(payload.get("max_workers", 3)),
            constraints=payload.get("constraints", ""),
            output=payload.get("output_contract", ""),
            write_roots=payload.get("write_root"),
            read_roots=payload.get("read_root"),
            deny_roots=payload.get("deny_write_root"),
            search_policy=payload.get("search_policy", "artifact_only"),
            branch_policy=payload.get("branch_policy", "forbidden"),
        )
        result = {"ok": True, "skipped": True, "reason": "CI mode", "result": prompt, "task": payload, "mode": payload.get("mode", "auto")}
    else:
        try:
            result = run_task(
                task=str(payload.get("task", "")),
                workspace=payload.get("workspace", str(PROJECT_ROOT)),
                mode=payload.get("mode", "auto"),
                max_workers=int(payload.get("max_workers", 3)),
                constraints=payload.get("constraints", ""),
                output_contract=payload.get("output_contract", ""),
                conversation=str(payload.get("conversation", "")),
                conversation_new=bool(payload.get("conversation_new", False)),
                session_key=str(payload.get("session_key", "")),
                timeout=payload.get("timeout"),
            )
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "task": str(payload.get("task", "")), "mode": payload.get("mode", "auto")}

    output_data = json.dumps(result, indent=2)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        w = writer or (GuardedWriter(project_root=PROJECT_ROOT) if GuardedWriter else None)
        if w:
            try:
                rel_output = str(output_path.resolve().relative_to(PROJECT_ROOT))
                w.mkdir_auto(str(Path(rel_output).parent))
                w.write_auto(rel_output, output_data)
            except (ValueError, AttributeError):
                output_path.write_text(output_data, encoding="utf-8")
        else:
            output_path.write_text(output_data, encoding="utf-8")
    except Exception as exc:
        print(f"Error writing output: {exc}")
        return 1

    return 0


def main():
    if len(sys.argv) == 3:
        return run_skill(Path(sys.argv[1]), Path(sys.argv[2]))
    from hermes_harness import main as cli_main  # noqa: E402
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
