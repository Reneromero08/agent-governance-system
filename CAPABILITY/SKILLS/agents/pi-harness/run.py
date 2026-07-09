#!/usr/bin/env python3
"""Offline ADR-017 entry point for the Pi Harness skill."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parents[3]
SCRIPTS = SKILL_DIR / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pi_harness import build_task_packet  # noqa: E402


def atomic_output(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except (OSError, TypeError):
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def run_skill(input_path: Path, output_path: Path) -> int:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object")
        task = str(payload.get("task", "")).strip()
        if not task:
            raise ValueError("task is required")
        prompt = build_task_packet(
            task=task,
            workspace=str(payload.get("workspace", PROJECT_ROOT)),
            read_roots=payload.get("read_roots", []),
            write_roots=payload.get("write_roots", []),
            tools=payload.get("tools", ["read", "grep", "find", "ls"]),
            constraints=str(payload.get("constraints", "")),
            shell_programs=payload.get("shell_programs", {}),
        )
        result = {
            "ok": True,
            "skipped": True,
            "reason": "offline task-packet generation; live Pi execution is CLI-only",
            "result": prompt,
        }
        atomic_output(output_path, result)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"pi-harness: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        return 2
    return run_skill(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
