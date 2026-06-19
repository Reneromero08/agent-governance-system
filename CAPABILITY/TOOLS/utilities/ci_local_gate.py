#!/usr/bin/env python3
"""Local commit and push verification gate.

Default mode runs the lightweight critic used for frequent commits.
``--full`` runs the mandatory risk-complete push gate and mints a HEAD-bound
verification receipt. ``--exhaustive`` runs every TESTBENCH test and implies
``--full``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
from CAPABILITY.TOOLS.utilities.push_test_plan import (
    PlanError,
    changed_paths,
    plan_payload,
    repo_python,
    run_plan,
)

TOKEN_FILE = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "ALLOW_PUSH.token"
writer = GuardedWriter(
    project_root=PROJECT_ROOT,
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=["LAW/CONTRACTS/_runs"],
)


def _git_stdout(args: Sequence[str], *, required: bool = False) -> str:
    result = subprocess.run(args, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if required:
            detail = (result.stderr or result.stdout or "unknown git error").strip()
            raise RuntimeError(f"git command failed: {' '.join(args)}: {detail}")
        return ""
    return (result.stdout or "").strip()


def _clean_status_path(path: str) -> str:
    return path.strip().strip('"').replace("\\", "/")


def _status_entry_paths(line: str) -> tuple[str, ...]:
    body = line[3:] if len(line) >= 4 else line
    if " -> " in body:
        source, destination = body.rsplit(" -> ", 1)
        return (_clean_status_path(source), _clean_status_path(destination))
    return (_clean_status_path(body),)


def _non_exempt_status_lines(text: str) -> list[str]:
    entries: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        paths = _status_entry_paths(line)
        if paths and all(path.startswith("THOUGHT/") for path in paths):
            continue
        entries.append(line)
    return entries


def _ensure_clean_tree(phase: str) -> bool:
    try:
        status = _git_stdout(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            required=True,
        )
    except RuntimeError as exc:
        sys.stderr.write(f"[ci-local-gate] FAIL: {exc}\n")
        return False

    entries = _non_exempt_status_lines(status)
    if not entries:
        return True

    sys.stderr.write(f"\n[ci-local-gate] FAIL: working tree is not clean {phase}.\n")
    sys.stderr.write("Lab-exempt THOUGHT/ paths are excluded. Remaining entries:\n")
    sys.stderr.write("\n".join(entries) + "\n")
    return False


def _run_stage(name: str, args: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    started = time.perf_counter()
    print(f"[ci-local-gate] START {name}", flush=True)
    result = subprocess.run(args, cwd=str(PROJECT_ROOT), env=merged_env, check=False)
    elapsed = time.perf_counter() - started
    print(f"[ci-local-gate] END {name} rc={result.returncode} elapsed={elapsed:.2f}s", flush=True)
    return int(result.returncode)


def _restore_generated_indexes() -> None:
    subprocess.run(
        [
            "git",
            "restore",
            "--source=HEAD",
            "--",
            "NAVIGATION/CORTEX/meta/FILE_INDEX.json",
            "NAVIGATION/CORTEX/meta/SECTION_INDEX.json",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _write_receipt(*, head: str, base_ref: str | None, payload: dict) -> None:
    receipt = {
        "type": "CI_OK",
        "head": head,
        "base_ref": base_ref,
        "mode": payload["mode"],
        "plan_hash": payload["plan_hash"],
        "risk_groups": [item["name"] for item in payload["risk_groups"]],
        "suites": [suite["name"] for suite in payload["suites"]],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    writer.open_commit_gate()
    writer.mkdir_durable(str(TOKEN_FILE.parent.relative_to(PROJECT_ROOT)), parents=True, exist_ok=True)
    writer.write_durable(
        str(TOKEN_FILE.relative_to(PROJECT_ROOT)),
        (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ci_local_gate.py")
    parser.add_argument("--full", action="store_true", help="Run mandatory risk-complete push verification")
    parser.add_argument("--exhaustive", action="store_true", help="Run every TESTBENCH test; implies --full")
    parser.add_argument("--base-ref", help="Compare BASE against HEAD when selecting conditional suites")
    parser.add_argument("--workers", type=int, default=4, help="pytest-xdist workers when available")
    parser.add_argument("--no-token", action="store_true", help="Run verification without writing a local receipt")
    args = parser.parse_args(argv)

    try:
        head = _git_stdout(["git", "rev-parse", "HEAD"], required=True)
    except RuntimeError as exc:
        sys.stderr.write(f"[ci-local-gate] FAIL: {exc}\n")
        return 1

    is_full = args.full or args.exhaustive
    if is_full and not _ensure_clean_tree("before checks"):
        return 1

    writer.open_commit_gate()
    tmp_root = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "pytest"
    writer.mkdir_tmp(str(tmp_root.relative_to(PROJECT_ROOT)), parents=True, exist_ok=True)
    test_env = {
        "TMPDIR": str(tmp_root),
        "TMP": str(tmp_root),
        "TEMP": str(tmp_root),
        "CUDA_VISIBLE_DEVICES": "",
    }

    total_start = time.perf_counter()
    rc = _run_stage("critic", [repo_python(), "CAPABILITY/TOOLS/governance/critic.py"])
    if rc != 0:
        return rc

    if not is_full:
        print(f"[ci-local-gate] FAST OK elapsed={time.perf_counter() - total_start:.2f}s (no push receipt)")
        print("[ci-local-gate] Run with --full before push.")
        return 0

    payload: dict | None = None
    base_ref: str | None = None
    rc = 1
    try:
        rc = _run_stage(
            "contracts",
            [repo_python(), "-u", "LAW/CONTRACTS/runner.py"],
            env={"CI": "true"},
        )
        if rc != 0:
            return rc

        try:
            paths, base_ref = changed_paths(args.base_ref)
            payload = plan_payload(
                paths,
                base_ref=base_ref,
                exhaustive=args.exhaustive,
                workers=max(args.workers, 0),
            )
        except PlanError as exc:
            sys.stderr.write(f"[ci-local-gate] FAIL: test planning failed: {exc}\n")
            return 2

        groups = ",".join(item["name"] for item in payload["risk_groups"]) or "none"
        print(
            f"[ci-local-gate] TEST PLAN mode={payload['mode']} base={base_ref or 'none'} "
            f"changed={len(paths)} groups={groups} "
            f"suites={','.join(s['name'] for s in payload['suites'])} "
            f"hash={payload['plan_hash'][:12]}",
            flush=True,
        )
        for item in payload["risk_groups"]:
            print(
                f"[ci-local-gate] RISK {item['name']} <- {','.join(item['matched_paths'])}",
                flush=True,
            )

        previous_env = os.environ.copy()
        os.environ.update(test_env)
        try:
            rc = run_plan(payload)
        finally:
            os.environ.clear()
            os.environ.update(previous_env)
        if rc != 0:
            return rc
    finally:
        # Preflight cleanliness guarantees these files had no user edits, so
        # restoring test-generated index churn is safe even on failed checks.
        _restore_generated_indexes()

    if not _ensure_clean_tree("after checks"):
        return 1

    if payload is None:
        sys.stderr.write("[ci-local-gate] FAIL: no test plan was produced\n")
        return 1

    if not args.no_token:
        _write_receipt(head=head, base_ref=base_ref, payload=payload)
        print(f"[ci-local-gate] RECEIPT {TOKEN_FILE.relative_to(PROJECT_ROOT)}")
        print("[ci-local-gate] Receipt remains valid for this HEAD, including network retry.")

    print(f"[ci-local-gate] FULL OK elapsed={time.perf_counter() - total_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
