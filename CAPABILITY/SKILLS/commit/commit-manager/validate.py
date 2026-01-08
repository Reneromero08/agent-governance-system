#!/usr/bin/env python3

"""
Validates the outputs of the commit-manager skill against expectations.
It is executed by the runner as part of fixture evaluation.
"""

import json
import sys
from pathlib import Path


def main(actual_path: Path, expected_path: Path) -> int:
    """Compare actual and expected JSON files."""
    try:
        actual = json.loads(actual_path.read_text())
        expected = json.loads(expected_path.read_text())
    except Exception as exc:
        print(f"Error reading JSON files: {exc}")
        return 1

    if actual == expected:
        print("Validation passed")
        return 0
    else:
        print("Validation failed")
        print("Actual:", json.dumps(actual, indent=2))
        print("Expected:", json.dumps(expected, indent=2))
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
