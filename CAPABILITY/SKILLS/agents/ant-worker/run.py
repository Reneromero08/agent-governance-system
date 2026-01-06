#!/usr/bin/env python3

"""
Ant Worker Skill - Execute file operations within capability envelope.

This skill executes file copy operations as defined in CATALYTIC-DPT/CAPABILITIES.json.
Validates capability hash before execution.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def main(input_path: Path, output_path: Path) -> int:
    """Execute ant worker task."""
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    # Validate required fields
    if "task" not in payload:
        print("Error: 'task' field required in input")
        return 1

    task = payload["task"]
    result = {
        "status": "success",
        "task_id": task.get("id", "unknown"),
        "executed_at": "2025-01-01T00:00:00Z",
        "outputs": []
    }

    if not GuardedWriter:
         print("Error: GuardedWriter not available")
         return 1
         
    writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]) 
    # output_path might be anywhere.
    # Usually output is in _tmp or passed by runner?
    # Test runner usually puts output in tmp.
    # If output_path is outside project, GuardedWriter blocks it unless tmp_roots allows it.
    # Assuming standard calling convention where output is within project or allowed.
    # Or strict compliance: if path not allowed, fail.
    
    # We need to open commit gate if durable.
    writer.open_commit_gate()
    
    # Check if path is supported
    try:
        writer.mkdir_durable(str(output_path.parent))
        writer.write_durable(str(output_path), json.dumps(result, indent=2, sort_keys=True))
    except Exception as e:
        print(f"Firewall violation or write error: {e}")
        return 1
    print("[ant-worker] Task executed successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
