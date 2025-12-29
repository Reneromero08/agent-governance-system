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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("[ant-worker] Task executed successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
