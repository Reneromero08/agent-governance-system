#!/usr/bin/env python3

"""
Critic script for AGS.

This script analyzes the repository for potential governance violations.  It can
be run manually or in CI.  Example checks include:

- Ensuring canon files were updated alongside fixtures when appropriate.
- Preventing direct modification of files in `CANON/` outside of allowed paths.
- Checking for forbidden patterns (e.g. raw filesystem access).

Currently this is a placeholder implementation that always succeeds.
"""

import sys

def main() -> int:
    print("[critic] no checks implemented yet")
    return 0

if __name__ == "__main__":
    sys.exit(main())