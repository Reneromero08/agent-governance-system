#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main(actual_path: Path, expected_path: Path) -> int:
    try:
        actual = json.loads(actual_path.read_text())
        expected = json.loads(expected_path.read_text())
    except Exception as exc:
        print(f"Error reading JSON files: {exc}")
        return 1
    if actual == expected:
        print("Validation passed")
        return 0
    print("Validation failed")
    print("Actual:", actual)
    print("Expected:", expected)
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
