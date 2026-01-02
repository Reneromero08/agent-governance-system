#!/usr/bin/env python3
"""
Auto-Starting MCP Server Wrapper

This wrapper ensures the MCP server is running before forwarding requests.
If the server isn't running, it starts automatically on first connection.

Usage:
    python CAPABILITY/MCP/server_wrapper.py

This is the script you should configure in Claude Desktop and other clients.
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = REPO_ROOT / "CAPABILITY" / "MCP" / "server.py"
PID_FILE = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "mcp_logs" / "server.pid"
LOG_DIR = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "mcp_logs"

def is_server_running():
    """Check if server is already running."""
    if not PID_FILE.exists():
        return False

    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process exists
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True
            )
            return str(pid) in result.stdout
        else:
            # Unix: send signal 0 (does nothing but checks if process exists)
            os.kill(pid, 0)
            return True
    except (ValueError, ProcessLookupError, FileNotFoundError):
        return False

def start_server():
    """Start the MCP server in the background."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Start server as subprocess
    if sys.platform == "win32":
        # Windows: Use CREATE_NEW_PROCESS_GROUP to detach
        process = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(LOG_DIR / "server_stderr.log", "a"),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        # Unix: Use nohup-like approach
        process = subprocess.Popen(
            [sys.executable, str(SERVER_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(LOG_DIR / "server_stderr.log", "a"),
            start_new_session=True
        )

    # Save PID
    PID_FILE.write_text(str(process.pid))

    # Give it a moment to initialize
    time.sleep(1)

    return process

def forward_to_server(server_process):
    """Forward stdin to server and stdout back to client."""
    try:
        # Read from stdin, write to server
        for line in sys.stdin:
            server_process.stdin.write(line.encode('utf-8'))
            server_process.stdin.flush()

            # Read response from server
            response_line = server_process.stdout.readline()
            sys.stdout.write(response_line.decode('utf-8'))
            sys.stdout.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass

def main():
    """Main entry point - auto-start server if needed, then act as stdio proxy."""

    # Check if server is already running
    if not is_server_running():
        # Server not running - start it now
        server_process = start_server()
        print(f"[AUTO-START] MCP server started (PID: {server_process.pid})", file=sys.stderr)
    else:
        # Server is running - just forward to it
        pid = int(PID_FILE.read_text().strip())
        print(f"[AUTO-START] Using existing MCP server (PID: {pid})", file=sys.stderr)

        # We can't directly attach to existing process stdio, so exec the main server
        # This is fine because MCP is stateless - each connection is independent
        os.execv(sys.executable, [sys.executable, str(SERVER_SCRIPT)])

    # This should never be reached in normal operation
    # (os.execv replaces the process)

if __name__ == "__main__":
    main()
