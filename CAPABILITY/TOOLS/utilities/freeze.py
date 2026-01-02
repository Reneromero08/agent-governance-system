#!/usr/bin/env python3

"""
Freeze script for AGS.

This tool can generate a deterministic snapshot of the repository's dependencies
or environment.  For example, it could write out a `requirements.txt` with
pinned versions of Python packages or record a snapshot of the cortex.  Such
freezes are useful when migrating to a new version or reproducing results.

Currently this script is a placeholder.
"""

def main() -> int:
    print("[freeze] no functionality implemented yet")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())