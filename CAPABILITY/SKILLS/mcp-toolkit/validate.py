#!/usr/bin/env python3
"""Validate mcp-toolkit output against expected JSON."""

import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python validate.py <actual.json> <expected.json>")
        return 1

    actual_path = Path(sys.argv[1])
    expected_path = Path(sys.argv[2])

    try:
        actual = _load_json(actual_path)
        expected = _load_json(expected_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Failed to load JSON: {exc}")
        return 1

    if actual == expected:
        print("Validation passed")
        return 0

    print("Validation failed: output does not match expected")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
