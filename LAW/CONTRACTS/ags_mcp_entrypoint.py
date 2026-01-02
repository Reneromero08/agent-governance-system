#!/usr/bin/env python3
"""
AGS MCP Runtime Entrypoint
This file is the canonical runtime entrypoint for the AGS MCP Server.
 It correctly sets up the path and launches the capability-defined server.
"""
import sys
from pathlib import Path

# Resolve repo root (2 levels up from LAW/CONTRACTS)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.MCP.server import main

if __name__ == "__main__":
    main()
