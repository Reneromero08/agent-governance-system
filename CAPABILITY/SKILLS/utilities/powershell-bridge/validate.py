#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: validate.py <actual.json> <expected.json>")
        return 1
    actual = load(Path(sys.argv[1]))
    expected = load(Path(sys.argv[2]))
    return 0 if actual == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
