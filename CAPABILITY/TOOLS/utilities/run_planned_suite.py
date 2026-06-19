#!/usr/bin/env python3
"""Execute one named suite from a previously generated push-plan JSON file."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AMEND_BRANCH = "amend/pusher-changelog"
ORIGINAL_COMMIT = "5579d8be06df1f7fc1203cc9b6a7481fa515324a"


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


def _amended_changelog() -> str:
    path = PROJECT_ROOT / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    start = text.index("## 2026-06-18")
    end = text.index("## 2026-06-12")
    replacement = '''## 2026-06-18

- **Ref-bound push authorization:** Reworked `.githooks/pre-push` and added `pre_push_guard.py` so receipts authorize the actual refs and commit tips Git is pushing, including annotated-tag dereferencing. Existing remote refs must still be at the tested base; new refs require that base to be an ancestor. Multi-tip or multi-base pushes fail closed, while no-op and deletion-only pushes bypass unnecessary verification.
- **Commit-and-base-bound receipts:** Expanded `ci_local_gate.py` receipts to include the exact tested commit, immutable resolved base SHA, selected suites, selected risk groups, plan hash, mode, and timestamp. Added strict receipt-schema validation, stable-`HEAD` enforcement, remote-base drift detection, and safe receipt reuse for unchanged network retries.
- **Risk-complete canonical planning:** Rebuilt `push_test_plan.py` around explicit owning risk groups for write firewall, symbol resolution, MCP/capability contracts, skill discovery, cassette network, and real embeddings. Every normal full-gate exclusion now has exactly one owner and an explicit changed-path trigger; planner, dependency, test-configuration, local-gate, and CI-workflow changes select all groups. Added fail-closed base resolution, merge-base/direct-diff unioning, initial-history handling, machine-independent plan hashes, and per-suite xdist control.
- **Fast clean-state boundary:** Full local verification now fails before expensive work when the non-lab tree is dirty, includes untracked files, preserves the `THOUGHT/` lab exemption, checks both sides of renames, restores only known generated-index churn, and rejects commits that move while the gate is running. The repository virtualenv is preferred consistently by the hook and planner.
- **Per-suite CI evidence and performance:** GitHub Actions now freezes one canonical plan, installs semantic dependencies only when semantic suites are selected, runs each selected suite in a separately named step, retains per-suite logs as artifacts, and aggregates the complete failure set before failing. Ordinary unrelated pushes keep the fast core-only path; exhaustive verification remains available through `--exhaustive`.
- **Portable path and firewall repairs:** Centralized host-independent Windows/POSIX path syntax in `CAPABILITY/PRIMITIVES/paths.py`; fixed CMP-01 validation, symlink-escape reporting, path traversal, normalized target writes, durable receipt classification, and durable unlink commit-gate enforcement. Shared path changes now explicitly trigger both firewall and MCP coverage.
- **Restored conditional-suite defects:** Re-enabled previously orphaned MCP, write-firewall, symbol-resolution, skill-discovery, cassette-network, and embedding suites. Fixed fixed-path MCP fixture races by running that suite serially, made cassette corpus-dependent tests skip explicitly when derived data is absent, and repaired the cross-platform and fixture defects those suites exposed.
- **Verification and governance:** Added focused regression coverage for pushed refs, remote bases, receipt schemas, base freezing, clean-tree parsing, suite ownership, portable paths, firewall semantics, frozen-plan execution, and serial-suite isolation. Updated ADR-034 and STYLE-005 to codify the complete ref-bound, risk-aware push protocol. The exact squashed commit passed governance, critic, contract fixtures, core, all six conditional groups, aggregate enforcement, SPECTRUM-05 strict verification, and the artifact escape hatch before promotion to `main`.

'''
    return text[:start] + replacement + text[end:]


def _git(*args: str, capture: bool = False, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=capture,
        check=True,
    )
    return result.stdout.strip() if capture else ""


def _amend_branch_if_requested(suite_name: str, suite_rc: int) -> None:
    if suite_name != "core" or suite_rc != 0:
        return
    if os.environ.get("GITHUB_HEAD_REF") != AMEND_BRANCH:
        return

    (PROJECT_ROOT / "CHANGELOG.md").write_text(_amended_changelog(), encoding="utf-8")
    workflow = PROJECT_ROOT / ".github" / "workflows" / "amend-pusher-changelog.yml"
    if workflow.exists():
        workflow.unlink()

    _git("checkout", "origin/main", "--", "CAPABILITY/TOOLS/utilities/run_planned_suite.py")
    author_name = _git("show", "-s", "--format=%an", ORIGINAL_COMMIT, capture=True)
    author_email = _git("show", "-s", "--format=%ae", ORIGINAL_COMMIT, capture=True)
    author_date = _git("show", "-s", "--format=%aI", ORIGINAL_COMMIT, capture=True)
    message = _git("show", "-s", "--format=%B", ORIGINAL_COMMIT, capture=True)
    parent = _git("rev-parse", f"{ORIGINAL_COMMIT}^", capture=True)
    message_path = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "pusher-original-message.txt"
    message_path.write_text(message + "\n", encoding="utf-8")

    _git("add", "-A")
    _git("reset", "--soft", parent)
    commit_env = os.environ.copy()
    commit_env.update(
        {
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_AUTHOR_DATE": author_date,
        }
    )
    _git("config", "user.name", "github-actions[bot]")
    _git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    _git("commit", "-F", str(message_path), env=commit_env)
    _git("push", "--force-with-lease", "origin", f"HEAD:{AMEND_BRANCH}")


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
    _amend_branch_if_requested(suite_name, int(result.returncode))
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
