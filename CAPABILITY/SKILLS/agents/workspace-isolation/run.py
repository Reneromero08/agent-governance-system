#!/usr/bin/env python3
"""Entry point for workspace-isolation skill.

This skill enables parallel agent work using git worktrees.
See SKILL.md for full documentation.

Usage:
  python run.py create <task_id>
  python run.py status
  python run.py merge <task_id>
  python run.py cleanup <task_id>
  python run.py cleanup-stale [--apply]
"""

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "scripts"))

from workspace_isolation import main

if __name__ == "__main__":
    sys.exit(main())
