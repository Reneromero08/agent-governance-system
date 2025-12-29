#!/usr/bin/env python3
"""
Stable entrypoint wrapper for the AGS MCP server.

This is an authored file (checked into git) so CI and fixtures can rely on it
existing in clean checkouts. It redirects MCP audit logs to an allowed output
root under CONTRACTS/_runs/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import CAPABILITY.MCP.server as mcp_server

mcp_server.LOGS_DIR = PROJECT_ROOT / "CONTRACTS" / "_runs" / "mcp_logs"

if __name__ == "__main__":
    mcp_server.main()

