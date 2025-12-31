#!/usr/bin/env python3
"""
Swarm Control CLI

Monitor and control swarm agents launched via swarm_launcher.py
Reads from .swarm/runs/<run_id>/ directories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def find_latest_run() -> Optional[Path]:
    """Find the most recent run directory."""
    # Use skill's runs directory relative to this script
    skill_dir = Path(__file__).parent.parent
    swarm_root = skill_dir / "runs"
    if not swarm_root.exists():
        return None

    runs = [d for d in swarm_root.iterdir() if d.is_dir() and (d / "registry.json").exists()]
    if not runs:
        return None

    # Sort by modification time of registry.json
    runs.sort(key=lambda d: (d / "registry.json").stat().st_mtime, reverse=True)
    return runs[0]


def get_run_dir(run_id: Optional[str] = None) -> Path:
    """Get run directory by ID or find latest."""
    skill_dir = Path(__file__).parent.parent
    if run_id:
        run_dir = skill_dir / "runs" / run_id
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        return run_dir
    else:
        run_dir = find_latest_run()
        if not run_dir:
            skill_dir = Path(__file__).parent.parent
            print(f"ERROR: No swarm runs found in {skill_dir / 'runs'}/", file=sys.stderr)
            print(f"Launch a swarm first with: python {Path(__file__).parent / 'swarm_launcher.py'}", file=sys.stderr)
            sys.exit(1)
        return run_dir


def load_registry(run_dir: Path) -> Dict:
    """Load registry.json"""
    registry_path = run_dir / "registry.json"
    if not registry_path.exists():
        print(f"ERROR: Registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid registry JSON: {e}", file=sys.stderr)
        sys.exit(1)


def is_process_alive(pid: int) -> bool:
    """Check if process with given PID is alive (Windows-compatible)."""
    if not pid:
        return False

    try:
        if os.name == "nt":
            # Windows: use tasklist
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True
            )
            return str(pid) in result.stdout
        else:
            # Unix: send signal 0
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False


def get_agent_status(agent: Dict) -> str:
    """Determine current agent status."""
    registry_status = agent.get("status", "unknown")

    if registry_status in ["exited", "terminated"]:
        return registry_status

    pid = agent.get("pid")
    if not pid:
        return "unknown"

    if is_process_alive(pid):
        return "running"
    else:
        return "exited"


def cmd_ls(args: argparse.Namespace) -> None:
    """List agents in a swarm run."""
    run_dir = get_run_dir(args.run)
    registry = load_registry(run_dir)

    print(f"Run ID: {registry.get('run_id', 'unknown')}")
    print(f"Run dir: {run_dir.absolute()}")
    print(f"Started: {registry.get('started_at', 'unknown')}")
    print()

    agents = registry.get("agents", [])
    if not agents:
        print("No agents found.")
        return

    # Print header
    header_cols = ["AGENT_ID", "PID", "STATUS", "LAST_LOG", "LOG_PATH"]
    col_widths = [15, 8, 12, 20, 40]
    header = " | ".join(col.ljust(w) for col, w in zip(header_cols, col_widths))
    print(header)
    print("-" * len(header))

    # Print each agent
    for agent in agents:
        agent_id = agent.get("agent_id", "unknown")
        pid = agent.get("pid", "")
        status = get_agent_status(agent)

        # Get last log time
        log_path = run_dir / f"{agent_id}.log"
        if log_path.exists():
            mtime = log_path.stat().st_mtime
            last_log = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            log_path_str = str(log_path) if args.verbose else f"{agent_id}.log"
        else:
            last_log = "N/A"
            log_path_str = "N/A"

        # Format row
        row_values = [agent_id, str(pid), status, last_log, log_path_str]
        row = " | ".join(val.ljust(w) for val, w in zip(row_values, col_widths))
        print(row)


def cmd_logs(args: argparse.Namespace) -> None:
    """Tail agent logs."""
    run_dir = get_run_dir(args.run)
    registry = load_registry(run_dir)

    if args.all:
        # Follow all agents
        agents = registry.get("agents", [])
        if not agents:
            print("No agents found.", file=sys.stderr)
            sys.exit(1)

        log_files = []
        for agent in agents:
            agent_id = agent["agent_id"]
            log_path = run_dir / f"{agent_id}.{'jsonl' if args.json else 'log'}"
            if log_path.exists():
                log_files.append((agent_id, log_path))

        if not log_files:
            print("No log files found.", file=sys.stderr)
            sys.exit(1)

        print(f"Following {len(log_files)} agents (Ctrl+C to stop)...", file=sys.stderr)
        print("", file=sys.stderr)

        follow_multiple_logs(log_files, use_json=args.json)

    else:
        # Single agent
        if not args.agent_id:
            print("ERROR: Must specify --agent-id or --all", file=sys.stderr)
            sys.exit(1)

        log_path = run_dir / f"{args.agent_id}.{'jsonl' if args.json else 'log'}"
        if not log_path.exists():
            print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
            sys.exit(1)

        tail_log(log_path, lines=args.lines, follow=args.follow, use_json=args.json)


def tail_log(log_path: Path, lines: int = 200, follow: bool = False, use_json: bool = False) -> None:
    """Tail a single log file."""
    try:
        # Read last N lines
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            for line in tail_lines:
                if use_json:
                    try:
                        event = json.loads(line)
                        print(f"[{event.get('ts', 'N/A')}] {event.get('line', '')}")
                    except json.JSONDecodeError:
                        print(line, end="")
                else:
                    print(line, end="")

        if follow:
            print(f"\n=== Following {log_path.name} (Ctrl+C to stop) ===", file=sys.stderr)
            last_size = log_path.stat().st_size

            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(last_size)
                    while True:
                        line = f.readline()
                        if line:
                            if use_json:
                                try:
                                    event = json.loads(line)
                                    print(f"[{event.get('ts', 'N/A')}] {event.get('line', '')}")
                                except json.JSONDecodeError:
                                    print(line, end="")
                            else:
                                print(line, end="")
                            sys.stdout.flush()
                        else:
                            time.sleep(0.1)
                            if not log_path.exists():
                                print(f"\n=== Log file removed ===", file=sys.stderr)
                                break
            except KeyboardInterrupt:
                print(f"\n=== Stopped following {log_path.name} ===", file=sys.stderr)

    except (OSError, UnicodeDecodeError) as e:
        print(f"ERROR: Cannot read log file: {e}", file=sys.stderr)
        sys.exit(1)


def follow_multiple_logs(log_files: List[tuple[str, Path]], use_json: bool = False) -> None:
    """Follow multiple log files concurrently, prefixing with agent_id."""
    file_positions: Dict[str, int] = {}

    # Initialize positions at end of each file
    for agent_id, log_path in log_files:
        try:
            file_positions[agent_id] = log_path.stat().st_size
        except OSError:
            file_positions[agent_id] = 0

    try:
        while True:
            for agent_id, log_path in log_files:
                if not log_path.exists():
                    continue

                try:
                    current_size = log_path.stat().st_size
                    prev_size = file_positions.get(agent_id, 0)

                    if current_size > prev_size:
                        # Read new content
                        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(prev_size)
                            new_content = f.read(current_size - prev_size)

                            for line in new_content.splitlines():
                                if use_json:
                                    try:
                                        event = json.loads(line)
                                        print(f"[{agent_id}] [{event.get('ts', 'N/A')}] {event.get('line', '')}")
                                    except json.JSONDecodeError:
                                        print(f"[{agent_id}] {line}")
                                else:
                                    print(f"[{agent_id}] {line}")

                        file_positions[agent_id] = current_size
                    elif current_size < prev_size:
                        # File truncated/replaced
                        file_positions[agent_id] = current_size

                except (OSError, PermissionError):
                    continue

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n=== Stopped following all logs ===", file=sys.stderr)


def cmd_status(args: argparse.Namespace) -> None:
    """Show overall swarm status."""
    run_dir = get_run_dir(args.run)
    registry = load_registry(run_dir)

    print(f"=== Swarm Status ===")
    print(f"Run ID: {registry.get('run_id', 'unknown')}")
    print(f"Started: {registry.get('started_at', 'unknown')}")
    print()

    agents = registry.get("agents", [])
    if not agents:
        print("No agents found.")
        return

    # Count by status
    status_counts = {}
    for agent in agents:
        status = get_agent_status(agent)
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Total agents: {len(agents)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print()

    # Show agents with non-zero exit codes
    failed_agents = [a for a in agents if a.get("exit_code", 0) != 0 and a.get("status") in ["exited", "terminated"]]
    if failed_agents:
        print("Failed agents:")
        for agent in failed_agents:
            print(f"  {agent['agent_id']}: exit_code={agent.get('exit_code', 'N/A')}")
    else:
        print("No failed agents.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Swarm control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  swarmctl ls                              # List agents in latest run
  swarmctl ls --run my-swarm-001           # List agents in specific run
  swarmctl logs governor --follow          # Follow governor logs
  swarmctl logs --all --follow             # Follow all agent logs
  swarmctl logs ant-1 --json               # View structured JSONL logs
  swarmctl status                          # Show overall swarm status
        """
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Run ID (defaults to latest)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ls command
    ls_parser = subparsers.add_parser("ls", help="List agents")
    ls_parser.add_argument("-v", "--verbose", action="store_true", help="Show full paths")

    # logs command
    logs_parser = subparsers.add_parser("logs", help="View agent logs")
    logs_parser.add_argument("agent_id", nargs="?", help="Agent ID to tail")
    logs_parser.add_argument("-n", "--lines", type=int, default=200, help="Number of lines (default: 200)")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow new output")
    logs_parser.add_argument("--all", action="store_true", help="Follow all agents")
    logs_parser.add_argument("--json", action="store_true", help="Use JSONL format")

    # status command
    subparsers.add_parser("status", help="Show overall swarm status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "ls":
        cmd_ls(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
