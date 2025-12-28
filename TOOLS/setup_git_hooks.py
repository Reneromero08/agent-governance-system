#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = PROJECT_ROOT / ".githooks"


def _run(args: list[str]) -> int:
    res = subprocess.run(args, cwd=str(PROJECT_ROOT))
    return int(res.returncode)


def main() -> int:
    if not HOOKS_DIR.exists():
        print("ERROR: .githooks directory missing")
        return 1
    rc = _run(["git", "config", "core.hooksPath", ".githooks"])
    if rc != 0:
        print("ERROR: failed to set core.hooksPath")
        return rc
    print("OK: configured git hooksPath to .githooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

