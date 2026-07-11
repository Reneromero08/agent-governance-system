#!/usr/bin/env python3
"""Validate Pi Harness fixture output."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(actual_path: str, expected_path: str) -> bool:
    actual = json.loads(Path(actual_path).read_text(encoding="utf-8"))
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    if actual.get("ok") != expected.get("ok"):
        print(f"FAIL: ok={actual.get('ok')!r}, expected={expected.get('ok')!r}", file=sys.stderr)
        return False
    text = json.dumps(actual, sort_keys=True)
    for value in expected.get("contains", []):
        if value not in text:
            print(f"FAIL: missing {value!r}", file=sys.stderr)
            return False
    for value in expected.get("forbidden", []):
        if value in text:
            print(f"FAIL: forbidden {value!r}", file=sys.stderr)
            return False
    return True


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <actual.json> <expected.json>", file=sys.stderr)
        return 2
    if validate(sys.argv[1], sys.argv[2]):
        print("Validation passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
