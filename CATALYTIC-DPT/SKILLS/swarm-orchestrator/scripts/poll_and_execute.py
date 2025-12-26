#!/usr/bin/env python3
"""
CATALYTIC-DPT/poll_and_execute.py

Unified dispatcher - runs Governor OR Ant based on role.
Simpler than agent_loop.py, no external CLI dependencies.

Usage:
    python poll_and_execute.py --role Governor
    python poll_and_execute.py --role Ant-1
    python poll_and_execute.py --role Ant-2
"""

import argparse
import time
import json
import sys
from pathlib import Path
from datetime import datetime

# Navigate up to CATALYTIC-DPT root
CATALYTIC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CATALYTIC_ROOT / "LAB" / "MCP"))
from server import mcp_server

SKILLS_DIR = CATALYTIC_ROOT / "SKILLS"


def run_governor(interval: int):
    """Governor: receives directives and dispatches to Ant Workers."""
    print("\n" + "="*50)
    print("  GOVERNOR - Active")
    print("  Dispatches tasks to Ant Workers via MCP")
    print("="*50 + "\n")

    while True:
        try:
            try:
                directives = mcp_server.get_directives("Governor")
            except json.JSONDecodeError as e:
                print(f"[Governor] JSON parse error in directives: {e}")
                time.sleep(interval)
                continue

            if directives["pending_count"] > 0:
                for d in directives["directives"]:
                    dir_id = d["directive_id"]
                    print(f"[Governor] Processing directive: {dir_id}")

                    # Acknowledge
                    mcp_server.acknowledge_directive(dir_id)

                    # Parse directive and dispatch to workers
                    task_spec = d.get("context", {}).get("task_spec", {})
                    if not task_spec:
                        task_spec = {
                            "task_id": f"from-{dir_id}",
                            "task_type": "file_operation",
                            "operation": "read",
                            "files": []
                        }

                    # Dispatch to Ant-1
                    task_id = task_spec.get("task_id", f"task-{datetime.now().strftime('%H%M%S')}")
                    mcp_server.dispatch_task(
                        task_id=task_id,
                        task_spec=task_spec,
                        from_agent="Governor",
                        to_agent="Ant-1",
                        priority=5
                    )
                    print(f"[Governor] Dispatched {task_id} to Ant-1")
            else:
                print(f"[Governor] No directives. Polling in {interval}s...")

            # Check for results from workers
            results = mcp_server.get_results()
            if results["count"] > 0:
                print(f"[Governor] Received {results['count']} result(s) from workers")
                for r in results["results"]:
                    print(f"  - {r['task_id']}: {r['status']}")

        except KeyboardInterrupt:
            print("\n[Governor] Shutting down...")
            break
        except Exception as e:
            print(f"[Governor] Error: {e}")

        time.sleep(interval)


def run_ant(agent_id: str, interval: int):
    """Ant Worker: polls for tasks, executes them, reports back."""
    print("\n" + "-"*50)
    print(f"  {agent_id} - Active")
    print("  Executes tasks dispatched by Governor")
    print("-"*50 + "\n")

    import subprocess
    ant_worker = SKILLS_DIR / "ant-worker" / "run.py"

    while True:
        try:
            try:
                pending = mcp_server.get_pending_tasks(agent_id)
            except json.JSONDecodeError as e:
                print(f"[{agent_id}] JSON parse error in tasks: {e}")
                time.sleep(interval)
                continue

            if pending["pending_count"] > 0:
                for task in pending["tasks"]:
                    task_id = task["task_id"]
                    task_spec = task.get("task_spec", {})

                    print(f"[{agent_id}] Executing: {task_id}")
                    mcp_server.acknowledge_task(task_id)

                    # Create temp files
                    run_dir = Path(__file__).parent.parent / "CONTRACTS" / "_runs" / f"{task_id}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    input_file = run_dir / "input.json"
                    output_file = run_dir / "output.json"

                    with open(input_file, 'w') as f:
                        json.dump(task_spec, f, indent=2)

                    # Execute
                    try:
                        result = subprocess.run(
                            [sys.executable, str(ant_worker), str(input_file), str(output_file)],
                            capture_output=True,
                            text=True,
                            timeout=120
                        )

                        if output_file.exists():
                            with open(output_file) as f:
                                output = json.load(f)
                            status = output.get("status", "success")
                        else:
                            status = "failed"
                            output = {"error": result.stderr}

                        mcp_server.report_result(
                            task_id=task_id,
                            from_agent=agent_id,
                            status=status,
                            result=output,
                            errors=output.get("errors", [])
                        )
                        print(f"[{agent_id}] [OK] {task_id} -> {status}")

                    except subprocess.TimeoutExpired:
                        mcp_server.report_result(
                            task_id=task_id,
                            from_agent=agent_id,
                            status="timeout",
                            result={},
                            errors=["Task timed out after 120s"]
                        )
                        print(f"[{agent_id}] [TIMEOUT] {task_id}")

            else:
                print(f"[{agent_id}] No tasks. Polling in {interval}s...")

        except KeyboardInterrupt:
            print(f"\n[{agent_id}] Shutting down...")
            break
        except Exception as e:
            print(f"[{agent_id}] Error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="CATALYTIC-DPT Poll & Execute")
    parser.add_argument("--role", required=True, help="Governor, Ant-1, Ant-2")
    parser.add_argument("--interval", type=int, default=5, help="Poll interval (seconds)")
    args = parser.parse_args()

    if args.role == "Governor":
        run_governor(args.interval)
    elif args.role.startswith("Ant"):
        run_ant(args.role, args.interval)
    else:
        print(f"Unknown role: {args.role}")
        print("Valid roles: Governor, Ant-1, Ant-2")
        sys.exit(1)


if __name__ == "__main__":
    main()
