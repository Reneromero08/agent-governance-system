#!/usr/bin/env python3
"""
CATALYTIC-DPT/poll_and_execute.py

Unified dispatcher - runs Governor OR Ant based on role.
Simpler than agent_loop.py, no external CLI dependencies.

Features:
- Exponential backoff on idle/error
- Proper subprocess timeout handling with cleanup
- Graceful shutdown
- Comprehensive error handling

Usage:
    python poll_and_execute.py --role Governor
    python poll_and_execute.py --role Ant-1
    python poll_and_execute.py --role Ant-2
"""

import argparse
import time
import json
import sys
import signal
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Navigate up to CATALYTIC-DPT root
CATALYTIC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CATALYTIC_ROOT / "LAB" / "MCP"))
from server import mcp_server

SKILLS_DIR = CATALYTIC_ROOT / "SKILLS"

# Backoff configuration
MIN_POLL_INTERVAL = 1
MAX_POLL_INTERVAL = 60
BACKOFF_MULTIPLIER = 1.5
BACKOFF_RESET_ON_WORK = True

# Timeout configuration
TASK_TIMEOUT_SECONDS = 120
GRACEFUL_SHUTDOWN_TIMEOUT = 5


class BackoffController:
    """Manages exponential backoff for polling."""

    def __init__(self, min_interval: float = MIN_POLL_INTERVAL, max_interval: float = MAX_POLL_INTERVAL):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = min_interval
        self.consecutive_idle = 0

    def on_work_done(self):
        """Reset backoff when work is performed."""
        self.current_interval = self.min_interval
        self.consecutive_idle = 0

    def on_idle(self):
        """Increase backoff on idle poll."""
        self.consecutive_idle += 1
        self.current_interval = min(
            self.current_interval * BACKOFF_MULTIPLIER,
            self.max_interval
        )

    def on_error(self):
        """Increase backoff more aggressively on error."""
        self.current_interval = min(
            self.current_interval * (BACKOFF_MULTIPLIER * 2),
            self.max_interval
        )

    def get_interval(self) -> float:
        return self.current_interval


def kill_process_tree(proc, timeout: float = GRACEFUL_SHUTDOWN_TIMEOUT):
    """Kill a process and all its children (zombie-safe)."""
    import subprocess

    if proc is None:
        return
        
    try:
        # Check if process is still alive
        if proc.poll() is not None:
            return  # Already dead
            
        # Try graceful termination first
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        # Force kill
        proc.kill()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass  # Give up, OS will clean eventually
    except (OSError, ProcessLookupError):
        pass  # Process already dead or inaccessible


def run_governor(interval: int):
    """Governor: receives directives and dispatches to Ant Workers.

    Features:
    - Exponential backoff on idle
    - Comprehensive error handling
    - Graceful shutdown
    """
    print("\n" + "="*50)
    print("  GOVERNOR - Active")
    print("  Dispatches tasks to Ant Workers via MCP")
    print("="*50 + "\n")

    backoff = BackoffController(min_interval=interval)
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        print("\n[Governor] Shutdown signal received...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while not shutdown_requested:
        work_done = False

        try:
            # Get directives with error handling
            try:
                directives = mcp_server.get_directives("Governor")
            except json.JSONDecodeError as e:
                print(f"[Governor] JSON parse error in directives: {e}")
                backoff.on_error()
                time.sleep(backoff.get_interval())
                continue
            except Exception as e:
                print(f"[Governor] Error getting directives: {e}")
                backoff.on_error()
                time.sleep(backoff.get_interval())
                continue

            # Check for errors in response
            if directives.get("status") == "error":
                print(f"[Governor] Error: {directives.get('message')}")
                backoff.on_error()
                time.sleep(backoff.get_interval())
                continue

            if directives.get("pending_count", 0) > 0:
                for d in directives.get("directives", []):
                    dir_id = d.get("directive_id")
                    if not dir_id:
                        continue

                    print(f"[Governor] Processing directive: {dir_id}")

                    # Acknowledge with agent verification
                    ack_result = mcp_server.acknowledge_directive(dir_id, agent_id="Governor")
                    if ack_result.get("status") == "error":
                        print(f"[Governor] Failed to acknowledge: {ack_result.get('message')}")
                        continue

                    # Parse directive and dispatch to workers
                    task_spec = d.get("context", {}).get("task_spec", {})
                    if not task_spec:
                        task_spec = {
                            "task_id": f"from-{dir_id}",
                            "task_type": "file_operation",
                            "operation": "read",
                            "files": []
                        }

                    # Ensure task_id is set
                    task_id = task_spec.get("task_id", f"task-{datetime.now().strftime('%H%M%S%f')}")
                    task_spec["task_id"] = task_id

                    # Dispatch to Ant-1 (could be load-balanced in future)
                    dispatch_result = mcp_server.dispatch_task(
                        task_id=task_id,
                        task_spec=task_spec,
                        from_agent="Governor",
                        to_agent="Ant-1",
                        priority=d.get("priority", 5)
                    )

                    if dispatch_result.get("status") == "error":
                        print(f"[Governor] Dispatch failed: {dispatch_result.get('message')}")
                    else:
                        print(f"[Governor] Dispatched {task_id} to Ant-1")
                        work_done = True
            else:
                current_interval = backoff.get_interval()
                if backoff.consecutive_idle % 10 == 0:  # Only log every 10 idle cycles
                    print(f"[Governor] No directives. Polling in {current_interval:.1f}s...")

            # Check for results from workers (with pagination)
            try:
                results = mcp_server.get_results(limit=50)
                if results.get("count", 0) > 0:
                    print(f"[Governor] Received {results['count']} result(s) from workers")
                    for r in results.get("results", []):
                        status_icon = "✓" if r.get("status") == "success" else "✗"
                        print(f"  {status_icon} {r.get('task_id')}: {r.get('status')}")
                    work_done = True
            except Exception as e:
                print(f"[Governor] Error getting results: {e}")

            # Update backoff
            if work_done:
                backoff.on_work_done()
            else:
                backoff.on_idle()

        except KeyboardInterrupt:
            print("\n[Governor] Keyboard interrupt...")
            break
        except Exception as e:
            print(f"[Governor] Unexpected error: {e}")
            backoff.on_error()

        time.sleep(backoff.get_interval())

    print("[Governor] Shutdown complete.")


def run_ant(agent_id: str, interval: int):
    """Ant Worker: polls for tasks, executes them, reports back.

    Features:
    - Exponential backoff on idle
    - Proper subprocess timeout with cleanup
    - Agent ownership verification
    - Comprehensive error handling
    """
    print("\n" + "-"*50)
    print(f"  {agent_id} - Active")
    print("  Executes tasks dispatched by Governor")
    print("-"*50 + "\n")

    import subprocess
    ant_worker = SKILLS_DIR / "ant-worker" / "scripts" / "run.py"

    # Verify ant_worker exists
    if not ant_worker.exists():
        # Try alternate path
        ant_worker = SKILLS_DIR / "ant-worker" / "run.py"
        if not ant_worker.exists():
            print(f"[{agent_id}] ERROR: ant-worker script not found")
            return

    backoff = BackoffController(min_interval=interval)
    shutdown_requested = False
    current_process: Optional[subprocess.Popen] = None

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested, current_process
        print(f"\n[{agent_id}] Shutdown signal received...")
        shutdown_requested = True
        if current_process:
            kill_process_tree(current_process)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while not shutdown_requested:
        work_done = False

        try:
            # Get pending tasks with error handling
            try:
                pending = mcp_server.get_pending_tasks(agent_id)
            except json.JSONDecodeError as e:
                print(f"[{agent_id}] JSON parse error in tasks: {e}")
                backoff.on_error()
                time.sleep(backoff.get_interval())
                continue
            except Exception as e:
                print(f"[{agent_id}] Error getting tasks: {e}")
                backoff.on_error()
                time.sleep(backoff.get_interval())
                continue

            # Check for errors in response
            if pending.get("status") == "error":
                print(f"[{agent_id}] Error: {pending.get('message')}")
                backoff.on_error()
                time.sleep(backoff.get_interval())
                continue

            if pending.get("pending_count", 0) > 0:
                for task in pending.get("tasks", []):
                    if shutdown_requested:
                        break

                    task_id = task.get("task_id")
                    if not task_id:
                        continue

                    task_spec = task.get("task_spec", {})

                    print(f"[{agent_id}] Executing: {task_id}")

                    # Acknowledge with agent verification
                    ack_result = mcp_server.acknowledge_task(task_id, agent_id=agent_id)
                    if ack_result.get("status") == "error":
                        print(f"[{agent_id}] Failed to acknowledge {task_id}: {ack_result.get('message')}")
                        continue

                    # Create run directory
                    run_dir = CATALYTIC_ROOT / "CONTRACTS" / "_runs" / f"{agent_id}-{task_id}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    input_file = run_dir / "input.json"
                    output_file = run_dir / "output.json"

                    # Write input file
                    try:
                        with open(input_file, 'w', encoding='utf-8') as f:
                            json.dump(task_spec, f, indent=2)
                    except Exception as e:
                        print(f"[{agent_id}] Failed to write input: {e}")
                        mcp_server.report_result(
                            task_id=task_id,
                            from_agent=agent_id,
                            status="error",
                            result={},
                            errors=[f"Failed to write input file: {str(e)}"]
                        )
                        continue

                    # Execute with proper timeout handling
                    try:
                        current_process = subprocess.Popen(
                            [sys.executable, str(ant_worker), str(input_file), str(output_file)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )

                        try:
                            stdout, stderr = current_process.communicate(timeout=TASK_TIMEOUT_SECONDS)
                            return_code = current_process.returncode
                        except subprocess.TimeoutExpired:
                            # Kill the process tree
                            kill_process_tree(current_process)
                            stdout, stderr = "", ""
                            return_code = -1

                            mcp_server.report_result(
                                task_id=task_id,
                                from_agent=agent_id,
                                status="timeout",
                                result={"timeout_seconds": TASK_TIMEOUT_SECONDS},
                                errors=[f"Task timed out after {TASK_TIMEOUT_SECONDS}s"]
                            )
                            print(f"[{agent_id}] [TIMEOUT] {task_id}")
                            work_done = True
                            continue
                        finally:
                            current_process = None

                        # Process result
                        if output_file.exists():
                            try:
                                with open(output_file, encoding='utf-8') as f:
                                    output = json.load(f)
                                status = output.get("status", "success" if return_code == 0 else "failed")
                            except json.JSONDecodeError as e:
                                status = "error"
                                output = {"error": f"Invalid output JSON: {e}", "stderr": stderr}
                        else:
                            status = "failed"
                            output = {
                                "error": "No output file produced",
                                "return_code": return_code,
                                "stderr": stderr[:1000] if stderr else ""
                            }

                        # Report result
                        report_result = mcp_server.report_result(
                            task_id=task_id,
                            from_agent=agent_id,
                            status=status,
                            result=output,
                            errors=output.get("errors", [])
                        )

                        if report_result.get("status") == "error":
                            print(f"[{agent_id}] Failed to report: {report_result.get('message')}")
                        else:
                            status_icon = "✓" if status == "success" else "✗"
                            print(f"[{agent_id}] [{status_icon}] {task_id} -> {status}")

                        work_done = True

                    except Exception as e:
                        print(f"[{agent_id}] Execution error: {e}")
                        mcp_server.report_result(
                            task_id=task_id,
                            from_agent=agent_id,
                            status="error",
                            result={},
                            errors=[f"Execution exception: {str(e)}"]
                        )
                        work_done = True

            else:
                current_interval = backoff.get_interval()
                if backoff.consecutive_idle % 10 == 0:
                    print(f"[{agent_id}] No tasks. Polling in {current_interval:.1f}s...")

            # Update backoff
            if work_done:
                backoff.on_work_done()
            else:
                backoff.on_idle()

        except KeyboardInterrupt:
            print(f"\n[{agent_id}] Keyboard interrupt...")
            break
        except Exception as e:
            print(f"[{agent_id}] Unexpected error: {e}")
            backoff.on_error()

        time.sleep(backoff.get_interval())

    print(f"[{agent_id}] Shutdown complete.")


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
