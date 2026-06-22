#!/usr/bin/env python3
"""Analyze only authorization-bound V2 engineering calibration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibration_contract import FALSE_AUTHORIZATIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    if evidence.get("hardware_executed") is not True:
        raise ValueError("real hardware calibration evidence required")
    if evidence.get("execution_class") != "AUTHORIZED_V2_SPECTRAL_CALIBRATION_NOT_ACQUISITION":
        raise ValueError("calibration evidence class mismatch")
    for key, expected in FALSE_AUTHORIZATIONS.items():
        if evidence.get(key) is not expected:
            raise ValueError(f"calibration evidence cannot set {key}")
    print(json.dumps({"classification": evidence["execution_class"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
