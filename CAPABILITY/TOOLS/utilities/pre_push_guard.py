#!/usr/bin/env python3
"""Validate actual pre-push refs against a tested commit-and-base receipt."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ZERO_SHA = "0" * 40
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PushRef:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str

    @property
    def is_deletion(self) -> bool:
        return self.local_ref == "(delete)" or self.local_sha == ZERO_SHA


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str


def _valid_sha(value: str) -> bool:
    return HEX40.fullmatch(value) is not None


def parse_push_refs(text: str) -> tuple[PushRef, ...]:
    refs: list[PushRef] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid pre-push ref line {line_number}: expected 4 fields")
        ref = PushRef(*fields)
        if not _valid_sha(ref.local_sha) or not _valid_sha(ref.remote_sha):
            raise ValueError(f"invalid pre-push ref line {line_number}: malformed SHA")
        refs.append(ref)
    return tuple(refs)


def introduces_commits(refs: Sequence[PushRef]) -> bool:
    return any(not ref.is_deletion for ref in refs)


def _git_stdout(args: Sequence[str]) -> str:
    result = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def resolve_pushed_commit(ref: PushRef) -> str | None:
    if ref.is_deletion:
        return None
    if ref.local_ref.startswith("refs/tags/"):
        return _git_stdout(["git", "rev-parse", "--verify", f"{ref.local_ref}^{{commit}}"] ) or None
    return ref.local_sha


def is_ancestor(ancestor_sha: str, descendant_sha: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def load_receipt(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != "CI_OK":
        return None

    head = payload.get("head")
    base_ref = payload.get("base_ref")
    base_sha = payload.get("base_sha")
    plan_hash = payload.get("plan_hash")
    mode = payload.get("mode")
    suites = payload.get("suites")
    risk_groups = payload.get("risk_groups", [])

    if not isinstance(head, str) or HEX40.fullmatch(head) is None:
        return None
    if base_ref is not None and not isinstance(base_ref, str):
        return None
    if base_sha is not None and (not isinstance(base_sha, str) or HEX40.fullmatch(base_sha) is None):
        return None
    if not isinstance(plan_hash, str) or HEX64.fullmatch(plan_hash) is None:
        return None
    if mode not in {"full", "exhaustive"}:
        return None
    if not isinstance(suites, list) or not suites or not all(isinstance(item, str) and item for item in suites):
        return None
    if len(suites) != len(set(suites)):
        return None
    if not isinstance(risk_groups, list) or not all(isinstance(item, str) and item for item in risk_groups):
        return None
    if len(risk_groups) != len(set(risk_groups)):
        return None
    if not _valid_timestamp(payload.get("timestamp")):
        return None

    suite_set = set(suites)
    risk_set = set(risk_groups)
    if mode == "full":
        if "core" not in suite_set or "exhaustive" in suite_set:
            return None
        if not risk_set.issubset(suite_set - {"core"}):
            return None
    elif suites != ["exhaustive"] or risk_groups:
        return None

    return payload


def load_receipt_head(path: Path) -> str | None:
    receipt = load_receipt(path)
    return receipt["head"] if receipt else None


def validate_push(
    refs: Sequence[PushRef],
    receipt: dict | None,
    *,
    resolver: Callable[[PushRef], str | None] = resolve_pushed_commit,
    ancestor_checker: Callable[[str, str], bool] = is_ancestor,
) -> GuardDecision:
    if not introduces_commits(refs):
        return GuardDecision(True, "no commits are being introduced")

    if receipt is None:
        return GuardDecision(False, "verification receipt is missing or invalid")

    commits: set[str] = set()
    remote_bases: set[str] = set()
    for ref in refs:
        if ref.is_deletion:
            continue
        commit = resolver(ref)
        if not commit:
            return GuardDecision(False, f"cannot resolve pushed commit for {ref.local_ref}")
        commits.add(commit)
        if ref.remote_sha != ZERO_SHA:
            remote_bases.add(ref.remote_sha)

    if len(commits) != 1:
        ordered = ", ".join(sorted(commits))
        return GuardDecision(
            False,
            f"push contains multiple distinct commit tips ({ordered}); push them separately",
        )
    if len(remote_bases) > 1:
        ordered = ", ".join(sorted(remote_bases))
        return GuardDecision(
            False,
            f"push updates refs with multiple remote bases ({ordered}); push them separately",
        )

    pushed_head = next(iter(commits))
    if pushed_head != receipt["head"]:
        return GuardDecision(
            False,
            f"receipt authorizes {receipt['head']}, but the pushed commit is {pushed_head}",
        )

    tested_base = receipt.get("base_sha")
    if remote_bases:
        actual_base = next(iter(remote_bases))
        if tested_base != actual_base:
            return GuardDecision(
                False,
                f"receipt tested base {tested_base or 'none'}, but remote is at {actual_base}",
            )
    elif tested_base is not None and not ancestor_checker(tested_base, pushed_head):
        return GuardDecision(
            False,
            f"tested base {tested_base} is not an ancestor of new ref tip {pushed_head}",
        )

    return GuardDecision(
        True,
        f"receipt authorizes pushed commit {pushed_head} from tested base {tested_base or 'initial-history'}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs-file", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        refs = parse_push_refs(args.refs_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[PRE-PUSH] ERROR: {exc}", file=sys.stderr)
        return 1

    receipt = load_receipt(args.token_file)
    decision = validate_push(refs, receipt)
    stream = sys.stdout if decision.allowed else sys.stderr
    print(f"[PRE-PUSH] {decision.reason}", file=stream)

    if decision.allowed:
        return 0
    return 2 if receipt is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
