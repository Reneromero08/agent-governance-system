#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration_contract import build_plan, write_immutable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_immutable(args.output, build_plan())
    print(json.dumps({"path": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
