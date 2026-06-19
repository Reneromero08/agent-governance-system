#!/usr/bin/env python3
"""Validate a pre-push ref set against a HEAD-bound verification receipt.

Git's pre-push hook provides the refs that will actually be updated on stdin.
This module keeps the policy testable and ensures a receipt authorizes the
commit being introduced, not merely whichever branch happens to be checked out.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ZERO_SHA = "0" * 40


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


def parse_push_refs(text: str) -> tuple[PushRef, ...]:
    refs: list[PushRef] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid pre-push ref line {line_number}: expected 4 fields")
        refs.append(PushRef(*fields))
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


def load_receipt_head(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("type") != "CI_OK":
        return None
    head = payload.get("head")
    return head if isinstance(head, str) and len(head) == 40 else None


def validate_push(
    refs: Sequence[PushRef],
    receipt_head: str | None,
    *,
    resolver: Callable[[PushRef], str | None] = resolve_pushed_commit,
) -> GuardDecision:
    if not introduces_commits(refs):
        return GuardDecision(True, "no commits are being introduced")

    if not receipt_head:
        return GuardDecision(False, "verification receipt is missing or invalid")

    commits: set[str] = set()
    for ref in refs:
        if ref.is_deletion:
            continue
        commit = resolver(ref)
        if not commit:
            return GuardDecision(False, f"cannot resolve pushed commit for {ref.local_ref}")
        commits.add(commit)

    if len(commits) != 1:
        ordered = ", ".join(sorted(commits))
        return GuardDecision(
            False,
            f"push contains multiple distinct commit tips ({ordered}); push them separately",
        )

    pushed_head = next(iter(commits))
    if pushed_head != receipt_head:
        return GuardDecision(
            False,
            f"receipt authorizes {receipt_head}, but the pushed commit is {pushed_head}",
        )
    return GuardDecision(True, f"receipt authorizes pushed commit {pushed_head}")


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

    receipt_head = load_receipt_head(args.token_file)
    decision = validate_push(refs, receipt_head)
    stream = sys.stdout if decision.allowed else sys.stderr
    print(f"[PRE-PUSH] {decision.reason}", file=stream)

    if decision.allowed:
        return 0
    return 2 if receipt_head is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
