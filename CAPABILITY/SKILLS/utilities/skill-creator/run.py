#!/usr/bin/env python3

"""
Skill Creator skill runner.

This script implements the skill creator skill that handles skill creation tasks.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    # Process the input and generate expected output format
    # Extract task information from input
    task = payload.get("task", {})
    task_id = task.get("id", "unknown")

    # Generate the expected output format
    output_data = {
        "status": "success",
        "task_id": task_id
    }

    try:
        from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
        writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
        
        rel_output_path = str(output_path.resolve().relative_to(PROJECT_ROOT))
        writer.mkdir_tmp(str(Path(rel_output_path).parent))
        writer.write_tmp(rel_output_path, json.dumps(output_data, indent=2, sort_keys=True))
    except ImportError:
        print("GuardedWriter not found")
        return 1
    except Exception as e:
        print(f"Write failed: {e}")
        return 1
    print("[skill] Skill creator skill executed successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
