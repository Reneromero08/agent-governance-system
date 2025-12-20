#!/usr/bin/env python3

"""
Token linter for AGS.

This script scans the repository and warns about improper use of tokens.  It can
detect when a canonical term (defined in `CANON/GLOSSARY.md`) is mis-capitalised
or misused, or when a new token appears without being defined in the glossary.

Currently this is a placeholder implementation.
"""

def main() -> int:
    print("[lint_tokens] no checks implemented yet")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
