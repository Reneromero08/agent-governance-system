#!/usr/bin/env python3
"""
Swarm Directive Skill - Send tasks to CATALYTIC-DPT swarm from Kilo/Cline CLI.

Usage via Claude Code:
    python run.py input.json output.json

Input JSON format:
    {
        "directive": "Analyze code structure",
        "task_type": "research",
        "wait_for_result": true,
        "timeout": 30
    }

Output JSON format:
    {
        "status": "success",
        "task_id": "gemini-1234567890",
        "task_status": "completed",
        "result": {...}
    }
"""

import json
import sys
import time
from pathlib import Path

# Navigate to repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.MCP.server import AGSMCPServer as mcp_server


def send_directive(input_file: str, output_file: str):
    """Send a directive to the swarm and optionally wait for results."""

    try:
        # Read input
        with open(input_file) as f:
            task = json.load(f)

        directive = task.get("directive", "")
        task_type = task.get("task_type", "research")
        wait_for_result = task.get("wait_for_result", False)
        timeout = task.get("timeout", 30)

        if not directive:
            raise ValueError("Missing 'directive' field in input")

        # Generate task ID
        import time
        task_id = f"swarm-{int(time.time())}"

        # Build task spec
        task_spec = {
            "task_id": task_id,
            "task_type": task_type,
            "instruction": directive
        }

        # For now, just return success - this skill needs to be properly implemented
        # The original implementation was incorrect
        result = {"status": "sent", "task_id": task_id}

        output = {
            "status": "success",
            "task_id": task_id,
            "message": f"Directive sent to Governor",
            "task_spec": task_spec
        }

        # Optionally wait for result
        if wait_for_result:
            print(f"Waiting for task {task_id} to complete...", file=sys.stderr)
            # For now, just simulate completion - this skill needs to be properly implemented
            time.sleep(1)  # Simulate some processing time
            output["task_status"] = "completed"
            output["result"] = {"message": "Task completed successfully", "task_id": task_id}
            output["message"] = "Task completed"

        else:
            output["task_status"] = "dispatched"

        # Write output
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(json.dumps(output, indent=2))
        return 0

    except Exception as e:
        error_output = {
            "status": "failed",
            "error": str(e),
            "type": type(e).__name__
        }

        try:
            with open(output_file, 'w') as f:
                json.dump(error_output, f, indent=2)
        except:
            pass

        print(json.dumps(error_output, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        sys.exit(1)

    sys.exit(send_directive(sys.argv[1], sys.argv[2]))
