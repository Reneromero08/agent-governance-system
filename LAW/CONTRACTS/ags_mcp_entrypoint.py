#!/usr/bin/env python3
"""Generated runtime entrypoint for AGS MCP server."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import CAPABILITY.MCP.server as mcp_server

# Redirect MCP audit logs to an allowed output root.
mcp_server.LOGS_DIR = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "mcp_logs"

if __name__ == '__main__':
    mcp_server.main()
