#!/usr/bin/env python3
"""
Swarm Launcher with Observability

Launches multiple agent processes with structured logging to .swarm/runs/<run_id>/
Enables terminal monitoring via swarmctl.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


def get_run_dir(run_id: Optional[str] = None) -> Path:
    """Get or create run directory."""
    # Use skill's runs directory relative to this script
    skill_dir = Path(__file__).parent.parent
    swarm_root = skill_dir / "runs"

    if run_id:
        run_dir = swarm_root / run_id
    else:
        # Generate run_id from timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = swarm_root / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_registry(run_dir: Path) -> Path:
    """Initialize empty registry.json"""
    registry_path = run_dir / "registry.json"
    registry = {
        "run_id": run_dir.name,
        "started_at": datetime.now().isoformat(),
        "agents": []
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry_path


def update_registry(registry_path: Path, agent_id: str, **fields: Any) -> None:
    """Update agent entry in registry."""
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {"run_id": registry_path.parent.name, "started_at": datetime.now().isoformat(), "agents": []}

    # Find or create agent entry
    agent_entry = None
    for agent in registry["agents"]:
        if agent["agent_id"] == agent_id:
            agent_entry = agent
            break

    if not agent_entry:
        agent_entry = {"agent_id": agent_id}
        registry["agents"].append(agent_entry)

    # Update fields
    agent_entry.update(fields)
    agent_entry["last_updated"] = datetime.now().isoformat()

    # Write atomically
    tmp_path = registry_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp_path.replace(registry_path)


class AgentLogger:
    """Captures agent output to .log and .jsonl files."""

    def __init__(self, run_dir: Path, agent_id: str, tee_to_console: bool = False):
        self.agent_id = agent_id
        self.tee_to_console = tee_to_console
        self.log_path = run_dir / f"{agent_id}.log"
        self.jsonl_path = run_dir / f"{agent_id}.jsonl"

        # Open files
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        self.jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

    def log_line(self, line: str, stream: str = "stdout") -> None:
        """Log a single line to both .log and .jsonl."""
        timestamp = datetime.now().isoformat()

        # Write to plain log
        self.log_file.write(line)
        if not line.endswith("\n"):
            self.log_file.write("\n")
        self.log_file.flush()

        # Write to JSONL
        event = {
            "ts": timestamp,
            "agent_id": self.agent_id,
            "stream": stream,
            "line": line.rstrip("\n")
        }
        self.jsonl_file.write(json.dumps(event) + "\n")
        self.jsonl_file.flush()

        # Optionally tee to console
        if self.tee_to_console:
            prefix = f"[{self.agent_id}] "
            print(f"{prefix}{line}", end="" if line.endswith("\n") else "\n")
            sys.stdout.flush()

    def close(self) -> None:
        """Close log files."""
        self.log_file.close()
        self.jsonl_file.close()


def stream_output(pipe: TextIO, logger: AgentLogger, stream_name: str) -> None:
    """Thread function to stream subprocess output to logger."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            logger.log_line(line, stream=stream_name)
    except Exception as e:
        logger.log_line(f"ERROR streaming {stream_name}: {e}\n", stream="stderr")
    finally:
        pipe.close()


def launch_agent(
    run_dir: Path,
    registry_path: Path,
    agent_id: str,
    command: List[str],
    tee_to_console: bool = False
) -> subprocess.Popen:
    """
    Launch a single agent process with logging.

    Returns the Popen object.
    """
    logger = AgentLogger(run_dir, agent_id, tee_to_console=tee_to_console)

    # Log startup
    logger.log_line(f"=== Starting agent {agent_id} ===\n")
    logger.log_line(f"Command: {' '.join(command)}\n")
    logger.log_line(f"Time: {datetime.now().isoformat()}\n")
    logger.log_line("=" * 60 + "\n")

    # Start process
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        # Update registry with PID
        update_registry(
            registry_path,
            agent_id,
            pid=proc.pid,
            command=command,
            status="running",
            started_at=datetime.now().isoformat()
        )

        # Start output streaming threads
        stdout_thread = threading.Thread(
            target=stream_output,
            args=(proc.stdout, logger, "stdout"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=stream_output,
            args=(proc.stderr, logger, "stderr"),
            daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Store threads and logger for cleanup
        proc._logger = logger  # type: ignore
        proc._threads = [stdout_thread, stderr_thread]  # type: ignore

        return proc

    except Exception as e:
        logger.log_line(f"FATAL: Failed to start process: {e}\n", stream="stderr")
        logger.close()
        raise


def wait_for_agents(agents: List[tuple[str, subprocess.Popen]], registry_path: Path) -> Dict[str, int]:
    """
    Wait for all agents to complete.

    Returns dict of {agent_id: exit_code}
    """
    results = {}

    for agent_id, proc in agents:
        exit_code = proc.wait()
        results[agent_id] = exit_code

        # Wait for output threads to finish
        for thread in proc._threads:  # type: ignore
            thread.join(timeout=1.0)

        # Close logger
        proc._logger.close()  # type: ignore

        # Update registry
        update_registry(
            registry_path,
            agent_id,
            status="exited",
            exit_code=exit_code,
            finished_at=datetime.now().isoformat()
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch swarm agents with observability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python TOOLS/swarm_launcher.py \\
    --agent governor "python agent_loop.py --role Governor" \\
    --agent ant-1 "python agent_loop.py --role Ant-1" \\
    --agent ant-2 "python agent_loop.py --role Ant-2" \\
    --run my-swarm-001 \\
    --tee

This will:
1. Create .swarm/runs/my-swarm-001/
2. Launch 3 agents with stdout/stderr capture
3. Write logs to .swarm/runs/my-swarm-001/<agent_id>.log
4. Write structured events to .swarm/runs/my-swarm-001/<agent_id>.jsonl
5. Maintain registry in .swarm/runs/my-swarm-001/registry.json
6. Optionally tee output to console

Monitor with:
  python TOOLS/swarmctl.py ls
  python TOOLS/swarmctl.py logs governor --follow
  python TOOLS/swarmctl.py logs --all --follow
        """
    )

    parser.add_argument(
        "--agent",
        action="append",
        nargs=2,
        metavar=("ID", "COMMAND"),
        required=True,
        help="Agent to launch: --agent <id> \"<command>\""
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Run ID (defaults to timestamp)"
    )
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Tee agent output to console (in addition to files)"
    )

    args = parser.parse_args()

    # Setup run directory
    run_dir = get_run_dir(args.run)
    registry_path = init_registry(run_dir)

    print(f"=== Swarm Launcher ===")
    print(f"Run ID: {run_dir.name}")
    print(f"Run dir: {run_dir.absolute()}")
    print(f"Agents: {len(args.agent)}")
    print(f"Tee to console: {args.tee}")
    print()

    # Launch all agents
    agents = []
    for agent_id, command_str in args.agent:
        print(f"Launching {agent_id}...")

        # Parse command (simple split - could be improved with shlex)
        command = command_str.split()

        proc = launch_agent(
            run_dir=run_dir,
            registry_path=registry_path,
            agent_id=agent_id,
            command=command,
            tee_to_console=args.tee
        )
        agents.append((agent_id, proc))

    print()
    print(f"All agents launched. Monitor with:")
    print(f"  python TOOLS/swarmctl.py ls --run {run_dir.name}")
    print(f"  python TOOLS/swarmctl.py logs --all --follow --run {run_dir.name}")
    print()

    # Wait for completion
    try:
        results = wait_for_agents(agents, registry_path)

        print()
        print("=== Swarm Complete ===")
        for agent_id, exit_code in results.items():
            status = "SUCCESS" if exit_code == 0 else f"FAILED ({exit_code})"
            print(f"  {agent_id}: {status}")

        # Exit with non-zero if any agent failed
        sys.exit(max(results.values()))

    except KeyboardInterrupt:
        print("\n\n=== Swarm Interrupted ===")
        print("Terminating agents...")

        for agent_id, proc in agents:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()

            update_registry(
                registry_path,
                agent_id,
                status="terminated",
                finished_at=datetime.now().isoformat()
            )

        sys.exit(130)


if __name__ == "__main__":
    main()
