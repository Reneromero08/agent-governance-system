#!/usr/bin/env python3

"""
Template skill runner.

This is a skeleton script for a skill. Replace this file with implementation
code that performs the skill's action.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TOOLS.skill_runtime import ensure_canon_compat


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    # Template: echo the input as output (replace with actual logic)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("[skill] Template skill executed successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
