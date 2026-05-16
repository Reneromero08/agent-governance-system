#!/usr/bin/env python3

"""
Validator for coderabbit-comments skill.
Compares actual and expected JSON outputs.
"""

import json
import sys
from pathlib import Path


def main(actual_path: Path, expected_path: Path) -> int:
    try:
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading JSON files: {exc}")
        return 1

    if actual == expected:
        print("Validation passed")
        return 0

    print("Validation failed")
    # Show diff
    if actual.get("ok") != expected.get("ok"):
        print(f"  ok: expected {expected.get('ok')}, got {actual.get('ok')}")

    if actual.get("action") != expected.get("action"):
        print(f"  action: expected '{expected.get('action')}', got '{actual.get('action')}'")

    if actual.get("error") != expected.get("error"):
        print(f"  error: expected '{expected.get('error')}', got '{actual.get('error')}'")

    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
