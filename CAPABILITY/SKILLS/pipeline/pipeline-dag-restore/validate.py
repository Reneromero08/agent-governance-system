#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main(actual_path: Path, expected_path: Path) -> int:
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    return 0 if actual == expected else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
