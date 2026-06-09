#!/usr/bin/env python3
"""Hermes Harness skill entry point (ADR-017 compliant).

Delegates to scripts/hermes_harness.py, which contains the full implementation.
This wrapper exists to satisfy the skill contract that every skill has a run.py
at its root.

Usage:
    python run.py validate --task "Audit repo" --mode audit
    python run.py prompt   --task "Audit repo"
    python run.py run      --task "Respond with: OK" --timeout 60
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the scripts directory is importable.
_scripts = Path(__file__).resolve().parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from hermes_harness import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
