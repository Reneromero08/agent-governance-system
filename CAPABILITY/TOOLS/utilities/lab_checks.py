#!/usr/bin/env python3
"""Fast, side-effect-free checks for staged Lab payload."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_REPO_ROOT))

from CAPABILITY.TOOLS.utilities.lab_scope import (
    PathClass,
    ScopeViolation,
    classify_paths,
    parse_name_status_z,
    staged_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def git_blob(path: str, revision: str | None = None) -> bytes | None:
    object_name = f":{path}" if revision is None else f"{revision}:{path}"
    result = subprocess.run(
        ["git", "show", object_name], cwd=REPO_ROOT, capture_output=True, check=False
    )
    # Deleted paths have no stage-zero blob and need no content validation.
    return result.stdout if result.returncode == 0 else None


def check(paths: Sequence[str], *, revision: str | None = None) -> tuple[str, ...]:
    scope = classify_paths(paths)
    if scope.route != PathClass.LAB_PAYLOAD:
        raise ScopeViolation(
            f"focused Lab checks require LAB_PAYLOAD, got {scope.route.value if scope.route else 'EMPTY'}"
        )
    errors: list[str] = []
    for path in scope.paths[PathClass.LAB_PAYLOAD]:
        content = git_blob(path, revision)
        if content is None:
            continue
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        try:
            if path.endswith(".py"):
                ast.parse(text, filename=path)
            elif path.endswith(".json"):
                json.loads(text)
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head")
    args = parser.parse_args(argv)
    try:
        if bool(args.base) != bool(args.head):
            parser.error("--base and --head must be provided together")
        if args.base:
            result = subprocess.run(
                ["git", "diff", "--name-status", "-z", "--find-renames", f"{args.base}..{args.head}", "--"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise ScopeViolation("cannot read requested Lab validation range")
            paths = parse_name_status_z(result.stdout)
            revision = args.head
        else:
            paths = staged_paths()
            revision = None
        if not paths:
            print("[LAB] No staged payload to check.")
            return 0
        errors = check(paths, revision=revision)
    except ScopeViolation as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if errors:
        print("[LAB] Focused staged checks failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"[LAB] Focused staged checks passed ({len(set(paths))} paths).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
