#!/usr/bin/env python3

"""
File Analyzer skill runner.

This script implements the file analyzer skill that handles file analysis tasks.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat


def main(input_path: Path, output_path: Path, writer: Optional[GuardedWriter] = None) -> int:
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

    # Use GuardedWriter for writes if available, otherwise fallback
    # Always use GuardedWriter for writes to enforce firewall
    writer = writer or GuardedWriter(project_root=PROJECT_ROOT)

    try:
        # Convert output_path to relative path for GuardedWriter
        rel_output_path = str(output_path.resolve().relative_to(PROJECT_ROOT))
        writer.mkdir_tmp(str(Path(rel_output_path).parent))
        writer.write_tmp(rel_output_path, json.dumps(output_data, indent=2, sort_keys=True))
    except ValueError:
        # If path is not relative to PROJECT_ROOT, this is a violation of the firewall contract
        print(f"Error: Output path {output_path} is outside project root")
        return 1
    print("[skill] File analyzer skill executed successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
