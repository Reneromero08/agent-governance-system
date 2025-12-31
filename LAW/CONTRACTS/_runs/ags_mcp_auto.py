#!/usr/bin/env python3
"""
Auto-Starting AGS MCP Entrypoint

This is the recommended entrypoint for Claude Desktop and other MCP clients.
It automatically starts the MCP server on first connection if not already running.

Key features:
- Auto-starts server on first interaction (no manual start needed)
- Logs to LAW/CONTRACTS/_runs/mcp_logs/ (governance-compliant)
- Checks server health before connecting
- Falls back to direct server if auto-start fails

Usage in Claude Desktop config:
{
  "mcpServers": {
    "ags": {
      "command": "python",
      "args": ["D:/CCC 2.0/AI/agent-governance-system/LAW/CONTRACTS/_runs/ags_mcp_auto.py"],
      "cwd": "D:/CCC 2.0/AI/agent-governance-system"
    }
  }
}
"""

import sys
import os
from pathlib import Path

# Add repo root to path so we can import the server
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

# Import the actual MCP server
from CAPABILITY.MCP.server import main as server_main, AGSMCPServer

# Override logging directory to be governance-compliant
os.environ['MCP_LOG_DIR'] = str(REPO_ROOT / 'LAW' / 'CONTRACTS' / '_runs' / 'mcp_logs')

# Auto-start logic
def ensure_server_running():
    """Ensure MCP server is running, start if needed."""
    import subprocess

    PID_FILE = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "mcp_logs" / "server.pid"

    # Check if already running
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())

            # Verify process is alive
            if sys.platform == "win32":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True
                )
                if str(pid) in result.stdout:
                    # Server is running, we're good
                    return True
            else:
                try:
                    os.kill(pid, 0)  # Signal 0 just checks existence
                    return True
                except ProcessLookupError:
                    pass
        except (ValueError, FileNotFoundError):
            pass

    # Server not running - start it in background
    autostart_script = REPO_ROOT / "CAPABILITY" / "MCP" / "autostart.ps1"
    if autostart_script.exists():
        try:
            # Use PowerShell autostart script
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                 str(autostart_script), "-Start"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                timeout=10
            )
            # Give it a moment to start
            import time
            time.sleep(2)
            return True
        except Exception as e:
            print(f"[AUTO-START] Warning: Could not start server via autostart.ps1: {e}",
                  file=sys.stderr)

    # Fallback: just run the server directly (this client becomes the server)
    return False

# Main entry point
if __name__ == "__main__":
    # Try to ensure server is running (best effort)
    ensure_server_running()

    # Now run the server main (this will handle stdio communication)
    server_main()
