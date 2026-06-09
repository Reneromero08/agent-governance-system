#!/usr/bin/env python3
"""Hermes Harness output validator (ADR-017 compliant).

Accepts two JSON file paths:
    argv[1] = actual output   (produced by run.py)
    argv[2] = expected output (from fixtures/*/expected.json)

Returns exit code 0 if the actual output satisfies the expected contract,
exit code 1 otherwise.

Validation rules:
    - actual must be valid JSON with an "ok" key (for validate) or a non-empty
      string result (for run/prompt).
    - If expected contains "ok", actual["ok"] must match.
    - If expected contains "contains", every substring in the list must appear
      in the actual output text.
    - If expected contains "mode", actual["task"]["mode"] must match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(actual_path: str, expected_path: str) -> bool:
    actual_raw = Path(actual_path).read_text(encoding="utf-8")
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))

    # Try to parse actual as JSON; if it fails, treat it as raw text output.
    try:
        actual = json.loads(actual_raw)
    except (json.JSONDecodeError, ValueError):
        actual = actual_raw.strip()

    # Check "ok" field match.
    if "ok" in expected:
        if not isinstance(actual, dict):
            print(f"FAIL: expected JSON object with ok={expected['ok']}, got plain text", file=sys.stderr)
            return False
        if actual.get("ok") != expected["ok"]:
            print(f"FAIL: ok={actual.get('ok')}, expected={expected['ok']}", file=sys.stderr)
            return False

    # Check "mode" field match.
    if "mode" in expected:
        if not isinstance(actual, dict):
            print(f"FAIL: expected JSON object with mode, got plain text", file=sys.stderr)
            return False
        actual_mode = actual.get("task", {}).get("mode")
        if actual_mode != expected["mode"]:
            print(f"FAIL: mode={actual_mode}, expected={expected['mode']}", file=sys.stderr)
            return False

    # Check "contains" substrings.
    if "contains" in expected:
        text = actual_raw
        for substr in expected["contains"]:
            if substr not in text:
                print(f"FAIL: output missing required substring: {substr!r}", file=sys.stderr)
                return False

    # Check "non_empty" flag — actual output must not be blank.
    if expected.get("non_empty", False):
        if isinstance(actual, str) and not actual.strip():
            print("FAIL: output is empty but expected non-empty", file=sys.stderr)
            return False
        if isinstance(actual, dict) and not actual:
            print("FAIL: output is empty dict but expected non-empty", file=sys.stderr)
            return False

    return True


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <actual.json> <expected.json>", file=sys.stderr)
        return 2
    ok = validate(sys.argv[1], sys.argv[2])
    if ok:
        print("PASS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
