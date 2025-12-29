#!/usr/bin/env python3
"""
CATALYTIC-DPT/poll_tasks.py

Lightweight task poller for Ant Workers.
Polls MCP server for pending tasks and executes them via ant-worker skill.

Usage:
    python poll_tasks.py --agent Ant-1 --interval 5
"""

import argparse
import time
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Navigate up to CATALYTIC-DPT root
CATALYTIC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CATALYTIC_ROOT / "MCP"))
from server import mcp_server

SKILLS_DIR = CATALYTIC_ROOT / "SKILLS"


def execute_task(agent_id: str, task: dict) -> dict:
    """Execute a task via the ant-worker skill."""
    task_id = task.get("task_id", "unknown")
    task_spec = task.get("task_spec", {})

    print(f"[{agent_id}] Executing task: {task_id}")

    # Create temp input/output files
    run_dir = Path(__file__).parent.parent / "CONTRACTS" / "_runs" / f"{task_id}-{datetime.now().strftime('%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_file = run_dir / "input.json"
    output_file = run_dir / "output.json"

    # Write task spec
    with open(input_file, 'w') as f:
        json.dump(task_spec, f, indent=2)

    # Execute ant-worker
    ant_worker = SKILLS_DIR / "ant-worker" / "run.py"
    try:
        result = subprocess.run(
            [sys.executable, str(ant_worker), str(input_file), str(output_file)],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Read output
        if output_file.exists():
            with open(output_file) as f:
                output = json.load(f)
        else:
            output = {"status": "error", "message": "No output file created"}

        if result.returncode == 0:
            print(f"[{agent_id}] [OK] Task {task_id} completed")
            return {"status": "success", "result": output}
        else:
            print(f"[{agent_id}] [FAIL] Task {task_id} failed: {result.stderr}")
            return {"status": "failed", "result": output, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        print(f"[{agent_id}] [TIMEOUT] Task {task_id}")
        return {"status": "timeout", "message": "Task timed out after 120s"}
    except Exception as e:
        print(f"[{agent_id}] [ERROR] {e}")
        return {"status": "error", "message": str(e)}


def poll_loop(agent_id: str, interval: int):
    """Main polling loop."""
    print(f"\n{'='*50}")
    print(f"  {agent_id} - Task Poller Active")
    print(f"  Interval: {interval}s")
    print(f"{'='*50}\n")

    while True:
        try:
            # Check for pending tasks
            pending = mcp_server.get_pending_tasks(agent_id)

            if pending["pending_count"] > 0:
                for task in pending["tasks"]:
                    # Acknowledge task first
                    mcp_server.acknowledge_task(task["task_id"])

                    # Execute task
                    result = execute_task(agent_id, task)

                    # Report result
                    mcp_server.report_result(
                        task_id=task["task_id"],
                        from_agent=agent_id,
                        status=result.get("status", "error"),
                        result=result.get("result", {}),
                        errors=result.get("errors", [])
                    )
            else:
                print(f"[{agent_id}] No pending tasks. Sleeping {interval}s...")

        except KeyboardInterrupt:
            print(f"\n[{agent_id}] Shutting down...")
            break
        except Exception as e:
            print(f"[{agent_id}] Poll error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="CATALYTIC-DPT Task Poller")
    parser.add_argument("--agent", required=True, help="Agent ID (e.g., Ant-1)")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")
    args = parser.parse_args()

    poll_loop(args.agent, args.interval)


if __name__ == "__main__":
    main()
