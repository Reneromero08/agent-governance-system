#!/usr/bin/env python3

"""
This script validates the outputs of the skill against expectations.
It is executed by the runner as part of fixture evaluation.

To implement validation logic, import the expected and actual data,
compare them and exit with code 0 on success or non-zero on failure.
"""

import json
import sys
from pathlib import Path

def main(input_path: Path, expected_path: Path) -> int:
    """Compare actual and expected JSON files."""
    actual_file = input_path
    expected_file = expected_path
    try:
        actual = json.loads(actual_file.read_text())
        expected = json.loads(expected_file.read_text())
    except Exception as exc:
        print(f"Error reading JSON files: {exc}")
        return 1
    if actual == expected:
        print("Validation passed")
        return 0
    else:
        print("Validation failed")
        print("Actual:", actual)
        print("Expected:", expected)
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
