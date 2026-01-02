#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("Usage: validate.py <actual.json> <expected.json>\n")
        return 2
    actual = _load(Path(argv[1]))
    expected = _load(Path(argv[2]))
    if actual == expected:
        return 0
    sys.stderr.write("Mismatch\n")
    sys.stderr.write(json.dumps({"actual": actual, "expected": expected}, sort_keys=True, separators=(",", ":")) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

