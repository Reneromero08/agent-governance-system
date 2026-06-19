#!/usr/bin/env python3
"""Execute one named suite from a previously generated push-plan JSON file."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PlannedSuiteError(RuntimeError):
    """Raised when a frozen plan is malformed or does not contain the suite."""


def load_plan(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise PlannedSuiteError(f"cannot read plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlannedSuiteError("plan root must be a JSON object")
    suites = payload.get("suites")
    if not isinstance(suites, list) or not suites:
        raise PlannedSuiteError("plan must contain a non-empty suites list")
    return payload


def suite_command(payload: dict, suite_name: str) -> list[str]:
    matches = [suite for suite in payload["suites"] if suite.get("name") == suite_name]
    if len(matches) != 1:
        raise PlannedSuiteError(
            f"expected exactly one suite named {suite_name!r}, found {len(matches)}"
        )
    command = matches[0].get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise PlannedSuiteError(f"suite {suite_name!r} has an invalid command")
    return command


def run_suite(payload: dict, suite_name: str) -> int:
    command = suite_command(payload, suite_name)
    started = time.perf_counter()
    print(
        f"[push-test-suite] START suite={suite_name} plan={payload.get('plan_hash', 'unknown')[:12]}",
        flush=True,
    )
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.perf_counter() - started
    print(
        f"[push-test-suite] END suite={suite_name} rc={result.returncode} elapsed={elapsed:.2f}s",
        flush=True,
    )
    return int(result.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    args = parser.parse_args(argv)
    try:
        payload = load_plan(args.plan)
        return run_suite(payload, args.suite)
    except PlannedSuiteError as exc:
        print(f"[push-test-suite] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
