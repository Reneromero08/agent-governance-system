#!/usr/bin/env python3
"""
Local CI gate helper.

Default goal (fast): run lightweight checks to support frequent commits.

Full goal (--full): run the same high-signal checks CI runs, then mint a one-time
pre-push token so `.githooks/pre-push` can skip re-running expensive checks.

Writes: LAW/CONTRACTS/_runs/ALLOW_PUSH.token
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
TOKEN_FILE = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "ALLOW_PUSH.token"

writer = GuardedWriter(
    project_root=PROJECT_ROOT,
    tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
    durable_roots=["LAW/CONTRACTS/_runs"]
)


def _run(args: Sequence[str], *, env: dict | None = None) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    res = subprocess.run(args, cwd=str(PROJECT_ROOT), env=merged_env)
    return int(res.returncode)


def _git_stdout(args: Sequence[str]) -> str:
    res = subprocess.run(args, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        return ""
    return (res.stdout or "").strip()


def _git_lines(args: Sequence[str]) -> List[str]:
    out = _git_stdout(args)
    if not out:
        return []
    return [line for line in out.splitlines() if line.strip()]


def _ensure_clean_tree() -> bool:
    staged = _git_stdout(["git", "diff", "--cached", "--name-only"])
    unstaged = _git_stdout(["git", "diff", "--name-only"])
    if staged or unstaged:
        sys.stderr.write("\n[ci-local-gate] FAIL: working tree is not clean after checks.\n")
        if staged:
            sys.stderr.write("\nStaged changes:\n")
            sys.stderr.write(staged + "\n")
        if unstaged:
            sys.stderr.write("\nUnstaged changes:\n")
            sys.stderr.write(unstaged + "\n")
        sys.stderr.write("\nTip: restore generated tracked files or commit them before pushing.\n")
        return False
    return True


def _classify_significant_paths(paths: Sequence[str]) -> Set[str]:
    significant: Set[str] = set()
    for path in paths:
        normalized = path.replace("\\", "/")
        if not normalized:
            continue
        if normalized.startswith("CAPABILITY/TOOLS/") and not normalized.endswith(".md"):
            significant.add(normalized)
        elif normalized.startswith("CAPABILITY/PRIMITIVES/") and not normalized.endswith(".md"):
            significant.add(normalized)
        elif normalized.startswith("CAPABILITY/PIPELINES/") and not normalized.endswith(".md"):
            significant.add(normalized)
        elif normalized.startswith("CAPABILITY/SKILLS/") and not normalized.endswith(".md"):
            significant.add(normalized)
        elif normalized.startswith("LAW/CONTRACTS/"):
            significant.add(normalized)
        elif normalized.startswith("NAVIGATION/CORTEX/"):
            significant.add(normalized)
        elif normalized.startswith(".github/workflows/"):
            significant.add(normalized)
        elif normalized.startswith(".githooks/"):
            significant.add(normalized)
    return significant


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="ci_local_gate.py")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full CI-aligned gate (critic + runner + pytest) and mint CI_OK token for pre-push.",
    )
    args = parser.parse_args(argv[1:])

    # Open commit gate before any durable writes
    writer.open_commit_gate()

    tmp_root = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / "pytest_tmp"
    writer.mkdir_durable(str(tmp_root.relative_to(PROJECT_ROOT)), parents=True, exist_ok=True)
    tmp_env = {
        "TMPDIR": str(tmp_root),
        "TMP": str(tmp_root),
        "TEMP": str(tmp_root),
    }

    head = _git_stdout(["git", "rev-parse", "HEAD"])
    if not head:
        sys.stderr.write("[ci-local-gate] FAIL: not a git repository (or git unavailable)\n")
        return 1

    print("[ci-local-gate] Running critic...")
    rc = _run([sys.executable, "CAPABILITY/TOOLS/governance/critic.py"])
    if rc != 0:
        return rc

    if not args.full:
        changed = sorted(
            set(_git_lines(["git", "diff", "--cached", "--name-only"]) + _git_lines(["git", "diff", "--name-only"]))
        )
        significant = sorted(_classify_significant_paths(changed))
        print("[ci-local-gate] FAST OK: critic passed (no token minted).")
        if significant:
            print("[ci-local-gate] Note: significant changes detected (runner/pytest recommended before push):")
            for path in significant[:10]:
                print(f"  - {path}")
            if len(significant) > 10:
                print(f"  ... and {len(significant) - 10} more")
            print("")
        print("[ci-local-gate] To run the full gate and mint a pre-push token:")
        print("  python CAPABILITY/TOOLS/utilities/ci_local_gate.py --full")
        return 0

    print("[ci-local-gate] Running contracts runner...")
    rc = _run([sys.executable, "-u", "LAW/CONTRACTS/runner.py"])
    if rc != 0:
        return rc

    print("[ci-local-gate] Running pytest (TESTBENCH) with parallel execution...")
    rc = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "CAPABILITY/TESTBENCH/",
            "-n", "auto",
            "-q",
            "--dist=loadfile",
        ],
        env=tmp_env,
    )
    if rc != 0:
        return rc

    # Tests intentionally regenerate/remove certain tracked artifacts; restore them so
    # the cleanliness check below reflects what will actually be pushed.
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
    )

    if not _ensure_clean_tree():
        return 1

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    token_line = f"CI_OK head={head} suite=critic,runner,pytest ts={ts}\n"
    writer.mkdir_durable(str(TOKEN_FILE.parent.relative_to(PROJECT_ROOT)), parents=True, exist_ok=True)
    writer.open_commit_gate()
    writer.write_durable(str(TOKEN_FILE.relative_to(PROJECT_ROOT)), token_line.encode("utf-8"))
    print(f"[ci-local-gate] FULL OK: wrote push token {TOKEN_FILE.relative_to(PROJECT_ROOT)}")
    print("[ci-local-gate] You can now git push (token will be consumed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
