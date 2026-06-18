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
from CAPABILITY.TOOLS.utilities.push_test_plan import changed_paths, plan_payload, run_plan

TOKEN_FILE = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "ALLOW_PUSH.token"
writer = GuardedWriter(
    project_root=PROJECT_ROOT,
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=["LAW/CONTRACTS/_runs"],
)


def _repo_python() -> str:
    venv = PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    if os.name == "nt":
        venv = venv.with_suffix(".exe")
    return str(venv) if venv.exists() else sys.executable


def _git_stdout(args: Sequence[str]) -> str:
    result = subprocess.run(args, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def _filter_thought_paths(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if line.strip() and not line.strip().replace("\\", "/").startswith("THOUGHT/")
    )


def _ensure_clean_tree() -> bool:
    staged = _filter_thought_paths(_git_stdout(["git", "diff", "--cached", "--name-only"]))
    unstaged = _filter_thought_paths(_git_stdout(["git", "diff", "--name-only"]))
    if not staged and not unstaged:
        return True
    sys.stderr.write("\n[ci-local-gate] FAIL: working tree is not clean after checks.\n")
    if staged:
        sys.stderr.write(f"\nStaged changes (lab-exempt paths excluded):\n{staged}\n")
    if unstaged:
        sys.stderr.write(f"\nUnstaged changes (lab-exempt paths excluded):\n{unstaged}\n")
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
            "checkout",
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
    parser.add_argument("--base-ref", help="Compare BASE...HEAD when selecting conditional suites")
    parser.add_argument("--workers", type=int, default=4, help="pytest-xdist workers when available")
    parser.add_argument("--no-token", action="store_true", help="Run verification without writing a local receipt")
    args = parser.parse_args(argv)

    writer.open_commit_gate()
    tmp_root = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "pytest"
    writer.mkdir_durable(str(tmp_root.relative_to(PROJECT_ROOT)), parents=True, exist_ok=True)
    test_env = {
        "TMPDIR": str(tmp_root),
        "TMP": str(tmp_root),
        "TEMP": str(tmp_root),
        "CUDA_VISIBLE_DEVICES": "",
    }

    head = _git_stdout(["git", "rev-parse", "HEAD"])
    if not head:
        sys.stderr.write("[ci-local-gate] FAIL: not a git repository or git is unavailable\n")
        return 1

    total_start = time.perf_counter()
    rc = _run_stage("critic", [_repo_python(), "CAPABILITY/TOOLS/governance/critic.py"])
    if rc != 0:
        return rc

    if not (args.full or args.exhaustive):
        print(f"[ci-local-gate] FAST OK elapsed={time.perf_counter() - total_start:.2f}s (no push receipt)")
        print("[ci-local-gate] Run with --full before push.")
        return 0

    rc = _run_stage(
        "contracts",
        [_repo_python(), "-u", "LAW/CONTRACTS/runner.py"],
        env={"CI": "true"},
    )
    if rc != 0:
        return rc

    paths, base_ref = changed_paths(args.base_ref)
    payload = plan_payload(
        paths,
        base_ref=base_ref,
        exhaustive=args.exhaustive,
        workers=max(args.workers, 0),
    )
    print(
        f"[ci-local-gate] TEST PLAN mode={payload['mode']} base={base_ref or 'none'} "
        f"changed={len(paths)} embeddings={payload['embedding_required']} "
        f"suites={','.join(s['name'] for s in payload['suites'])} "
        f"hash={payload['plan_hash'][:12]}",
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

    _restore_generated_indexes()
    if not _ensure_clean_tree():
        return 1

    if not args.no_token:
        _write_receipt(head=head, base_ref=base_ref, payload=payload)
        print(f"[ci-local-gate] RECEIPT {TOKEN_FILE.relative_to(PROJECT_ROOT)}")
        print("[ci-local-gate] Receipt remains valid for this HEAD, including network retry.")

    print(f"[ci-local-gate] FULL OK elapsed={time.perf_counter() - total_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
