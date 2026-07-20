#!/usr/bin/env python3
"""Target-side placeholder for the future relation-only transaction.

This module is intentionally inert in the frozen offline package. It refuses to
perform live work unless a future task creates a separate authority path.
"""

from __future__ import annotations

import sys


def main() -> int:
    print("relation-only target execution is not authorized by this offline package", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
