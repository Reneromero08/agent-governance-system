#!/usr/bin/env python3
"""Canonical Lab boundary classifier, Git router, and runtime doctor.

The Git boundary is deliberately immutable in code.  ``.agent-root`` affects
agent instruction discovery only; it never grants commit or push authority.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = REPO_ROOT / "THOUGHT" / "LAB"
LAB_PREFIX = "THOUGHT/LAB/"
LAB_MARKER_TEXT = "ags-lab-root-v1\n"
ROOT_GOVERNANCE_SENTINEL = "Agent Operating Contract for the Agent Governance System (AGS)"
LAB_CONTRACT_SENTINEL = "AGS LAB ISOLATION CONTRACT V1"
ZERO_SHA = "0" * 40
SHA40 = re.compile(r"[0-9a-f]{40}")
REMOTE_NAME = re.compile(r"[A-Za-z0-9._-]+")

BOUNDARY_EXACT = frozenset(
    {
        "THOUGHT/LAB/.agent-root",
        "THOUGHT/LAB/AGENTS.md",
        "THOUGHT/LAB/LAB_CONTRACT.md",
        "THOUGHT/LAB/opencode.json",
    }
)
BOUNDARY_PREFIXES = ("THOUGHT/LAB/.codex/",)


class PathClass(str, Enum):
    LAB_PAYLOAD = "LAB_PAYLOAD"
    LAB_BOUNDARY_CONTROL = "LAB_BOUNDARY_CONTROL"
    MAIN = "MAIN"


class ScopeViolation(ValueError):
    """Raised when a path set crosses an isolation boundary."""


@dataclass(frozen=True)
class ScopeResult:
    route: PathClass | None
    paths: dict[PathClass, tuple[str, ...]]

    @property
    def empty(self) -> bool:
        return not any(self.paths.values())


@dataclass(frozen=True)
class PushRef:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str

    @property
    def is_deletion(self) -> bool:
        return self.local_ref == "(delete)" or self.local_sha == ZERO_SHA


def normalize_git_path(raw: str) -> str:
    """Return one safe repository-relative Git path."""
    if not raw or "\x00" in raw or "\\" in raw:
        raise ScopeViolation(f"invalid Git path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScopeViolation(f"path is not a canonical repository-relative path: {raw!r}")
    return path.as_posix()


def classify_path(raw: str) -> PathClass:
    path = normalize_git_path(raw)
    if path in BOUNDARY_EXACT or path == "THOUGHT/LAB/.codex" or any(
        path.startswith(prefix) for prefix in BOUNDARY_PREFIXES
    ):
        return PathClass.LAB_BOUNDARY_CONTROL
    if path.startswith(LAB_PREFIX):
        return PathClass.LAB_PAYLOAD
    return PathClass.MAIN


def classify_paths(paths: Iterable[str]) -> ScopeResult:
    grouped: dict[PathClass, list[str]] = {kind: [] for kind in PathClass}
    for raw in paths:
        path = normalize_git_path(raw)
        grouped[classify_path(path)].append(path)

    frozen = {kind: tuple(sorted(set(values))) for kind, values in grouped.items()}
    present = {kind for kind, values in frozen.items() if values}
    if not present:
        return ScopeResult(None, frozen)
    if PathClass.LAB_PAYLOAD in present and len(present) > 1:
        raise ScopeViolation(_violation_message(frozen))
    if present == {PathClass.LAB_BOUNDARY_CONTROL, PathClass.MAIN}:
        route = PathClass.LAB_BOUNDARY_CONTROL
    elif len(present) == 1:
        route = next(iter(present))
    else:
        raise ScopeViolation(_violation_message(frozen))
    return ScopeResult(route, frozen)


def _violation_message(paths: dict[PathClass, tuple[str, ...]]) -> str:
    lines = ["LAB SCOPE VIOLATION: experiment payload cannot cross governance boundaries"]
    for kind in PathClass:
        for path in paths[kind]:
            lines.append(f"  {kind.value}: {path}")
    return "\n".join(lines)


def parse_name_status_z(payload: bytes) -> tuple[str, ...]:
    """Parse ``git diff --name-status -z`` and retain both rename endpoints."""
    fields = payload.split(b"\x00")
    if fields and fields[-1] == b"":
        fields.pop()
    paths: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index].decode("ascii", "strict")
        index += 1
        count = 2 if status.startswith(("R", "C")) else 1
        if not status or index + count > len(fields):
            raise ScopeViolation("malformed NUL-delimited Git name-status stream")
        for _ in range(count):
            paths.append(fields[index].decode("utf-8", "surrogateescape"))
            index += 1
    return tuple(paths)


def parse_push_refs(text: str) -> tuple[PushRef, ...]:
    refs: list[PushRef] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 4:
            raise ScopeViolation(f"invalid pre-push line {line_number}: expected four fields")
        ref = PushRef(*fields)
        if not SHA40.fullmatch(ref.local_sha) or not SHA40.fullmatch(ref.remote_sha):
            raise ScopeViolation(f"invalid pre-push line {line_number}: malformed SHA")
        refs.append(ref)
    return tuple(refs)


def _git(args: Sequence[str], *, cwd: Path = REPO_ROOT, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=text, check=False
    )
    if result.returncode:
        stderr = result.stderr if text else result.stderr.decode("utf-8", "replace")
        raise ScopeViolation(f"git {' '.join(args)} failed: {stderr.strip()}")
    return result.stdout.strip() if text else result.stdout


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode in {0, 1}:
        return result.returncode == 0
    raise ScopeViolation(
        "git merge-base --is-ancestor failed: "
        + result.stderr.decode("utf-8", "replace").strip()
    )


def staged_paths() -> tuple[str, ...]:
    output = _git(["diff", "--cached", "--name-status", "-z", "--find-renames", "--"])
    assert isinstance(output, bytes)
    return parse_name_status_z(output)


def worktree_paths() -> tuple[str, ...]:
    tracked = _git(["diff", "HEAD", "--name-status", "-z", "--find-renames", "--"])
    untracked = _git(["ls-files", "--others", "--exclude-standard", "-z", "--"])
    assert isinstance(tracked, bytes) and isinstance(untracked, bytes)
    paths = list(parse_name_status_z(tracked))
    paths.extend(
        item.decode("utf-8", "surrogateescape")
        for item in untracked.split(b"\x00")
        if item
    )
    return tuple(paths)


def introduced_commits(refs: Sequence[PushRef], remote_name: str) -> tuple[str, ...]:
    if not REMOTE_NAME.fullmatch(remote_name):
        raise ScopeViolation(f"unsafe or unsupported remote name: {remote_name!r}")
    commits: set[str] = set()
    for ref in refs:
        if ref.is_deletion:
            continue
        if ref.remote_sha != ZERO_SHA and _git_is_ancestor(ref.remote_sha, ref.local_sha):
            args = ["rev-list", "--reverse", f"{ref.remote_sha}..{ref.local_sha}"]
        else:
            # Exclude every commit already reachable from this remote so a new
            # or rebased lab branch does not inherit and reclassify remote main
            # history. Fast-forward pushes retain the narrower exact range.
            args = ["rev-list", "--reverse", ref.local_sha, "--not", f"--remotes={remote_name}"]
        output = _git(args, text=True)
        assert isinstance(output, str)
        commits.update(line for line in output.splitlines() if line)
    return tuple(sorted(commits))


def commit_paths(commit: str) -> tuple[str, ...]:
    parents = _git(["rev-list", "--parents", "-n", "1", commit], text=True)
    assert isinstance(parents, str)
    if len(parents.split()) > 2:
        raise ScopeViolation(
            f"LAB SCOPE VIOLATION: merge commit {commit} is in introduced history; rebase to linear history"
        )
    output = _git(
        ["diff-tree", "--root", "--no-commit-id", "--name-status", "-z", "-r", "-M", commit]
    )
    assert isinstance(output, bytes)
    return parse_name_status_z(output)


def classify_push(refs: Sequence[PushRef], remote_name: str) -> ScopeResult:
    paths: list[str] = []
    for commit in introduced_commits(refs, remote_name):
        paths.extend(commit_paths(commit))
    return classify_paths(paths)


def classify_range(base: str, head: str, remote_name: str = "origin") -> ScopeResult:
    if not SHA40.fullmatch(head) or not (base == ZERO_SHA or SHA40.fullmatch(base)):
        raise ScopeViolation("range endpoints must be full 40-character Git SHAs")
    if base == ZERO_SHA:
        if not REMOTE_NAME.fullmatch(remote_name):
            raise ScopeViolation(f"unsafe or unsupported remote name: {remote_name!r}")
        args = ["rev-list", "--reverse", head, "--not", f"--remotes={remote_name}"]
    else:
        args = ["rev-list", "--reverse", f"{base}..{head}"]
    output = _git(args, text=True)
    assert isinstance(output, str)
    paths: list[str] = []
    for commit in output.splitlines():
        paths.extend(commit_paths(commit))
    return classify_paths(paths)


def canonical_runtime_path(path: Path, *, beneath: Path) -> Path:
    resolved = path.resolve(strict=False)
    root = beneath.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ScopeViolation(f"runtime path escapes {root}: {path} -> {resolved}")
    return resolved


def _all_prompt_text(payload: object) -> str:
    chunks: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(payload)
    return "\n".join(chunks)


def _load_toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ScopeViolation(f"cannot load required config {path}: {exc}") from exc


def _active_thread_prompt(expected_cwd: Path) -> object | None:
    """Read the current turn's persisted prompt without launching nested Codex."""
    thread_id = os.environ.get("CODEX_THREAD_ID")
    codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
    state_db = codex_home / "state_5.sqlite"
    if not thread_id or not state_db.is_file():
        return None
    try:
        with sqlite3.connect(f"file:{state_db}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT rollout_path FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
        if not row:
            return None
        events: list[object] = []
        instruction_event: object | None = None
        instruction_header = f"# AGENTS.md instructions for {expected_cwd.resolve()}"
        with Path(row[0]).open(encoding="utf-8") as stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except ValueError:
                    # A live rollout can expose a blank or not-yet-complete
                    # trailing record while the active task is writing it.
                    continue
                if (
                    event.get("type") == "event_msg"
                    and event.get("payload", {}).get("type") == "task_started"
                ):
                    events = []
                events.append(event)
                payload = event.get("payload", {})
                if (
                    event.get("type") == "response_item"
                    and payload.get("type") == "message"
                    and payload.get("role") == "user"
                ):
                    message_text = _all_prompt_text(payload.get("content", []))
                    if (
                        instruction_header in message_text
                        and LAB_CONTRACT_SENTINEL in message_text
                    ):
                        instruction_event = event
        turn_cwds = [
            event.get("payload", {}).get("cwd")
            for event in events
            if event.get("type") == "turn_context"
        ]
        if not turn_cwds or Path(str(turn_cwds[-1])).resolve() != expected_cwd.resolve():
            return None
        # World-state and user messages can quote historical governance. The
        # exact instruction record is the authoritative contamination check.
        return [instruction_event] if instruction_event is not None else events
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None


def doctor(cwd: Path, agent: str, *, prompt_json: Path | str | None = None) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        resolved_cwd = canonical_runtime_path(cwd, beneath=LAB_ROOT)
    except (OSError, ScopeViolation) as exc:
        return (str(exc),)

    marker = LAB_ROOT / ".agent-root"
    try:
        if marker.read_text(encoding="utf-8") != LAB_MARKER_TEXT:
            errors.append("Lab discovery marker is missing or has the wrong contract version")
    except OSError as exc:
        errors.append(f"Lab discovery marker is unavailable: {exc}")

    try:
        git_root = Path(str(_git(["rev-parse", "--show-toplevel"], cwd=resolved_cwd, text=True))).resolve()
        if git_root != REPO_ROOT.resolve():
            errors.append(f"wrong Git root: expected {REPO_ROOT.resolve()}, got {git_root}")
    except ScopeViolation as exc:
        errors.append(str(exc))

    cursor = resolved_cwd
    discovered: Path | None = None
    while True:
        if (cursor / ".agent-root").is_file() or (cursor / ".git").exists():
            discovered = cursor
            break
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    if discovered != LAB_ROOT.resolve():
        errors.append(f"effective project root is not the Lab root: {discovered}")

    if agent == "codex":
        project_config = _load_toml(LAB_ROOT / ".codex" / "config.toml")
        if "default_permissions" in project_config:
            errors.append("Lab project config must not replace the session's normal permissions")
        if "sandbox_mode" in project_config or "sandbox_workspace_write" in project_config:
            errors.append("Lab project config must not replace the session's normal sandbox settings")

        if prompt_json is None:
            prompt_payload = _active_thread_prompt(resolved_cwd)
            if prompt_payload is None:
                codex = shutil.which("codex")
                if not codex:
                    errors.append("Codex executable is unavailable for exact prompt inspection")
                    prompt_payload = []
                else:
                    probe = subprocess.run(
                        [codex, "debug", "prompt-input", "AGS_LAB_DOCTOR_PROBE"],
                        cwd=resolved_cwd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if probe.returncode:
                        errors.append(f"exact Codex prompt inspection failed: {probe.stderr.strip()}")
                        prompt_payload = []
                    else:
                        try:
                            prompt_payload = json.loads(probe.stdout)
                        except ValueError as exc:
                            errors.append(f"Codex prompt inspection returned invalid JSON: {exc}")
                            prompt_payload = []
        elif prompt_json == "-":
            try:
                prompt_payload = json.load(sys.stdin)
            except ValueError as exc:
                errors.append(f"cannot read prompt inspection input from stdin: {exc}")
                prompt_payload = []
        else:
            try:
                assert isinstance(prompt_json, Path)
                prompt_payload = json.loads(prompt_json.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"cannot read prompt inspection input: {exc}")
                prompt_payload = []
        prompt_text = _all_prompt_text(prompt_payload)
        if ROOT_GOVERNANCE_SENTINEL in prompt_text:
            errors.append("repository-root governance appears in the exact Codex prompt input")
        if LAB_CONTRACT_SENTINEL not in prompt_text:
            errors.append("Lab contract is absent from the exact Codex prompt input")
    elif agent == "opencode":
        try:
            inline = json.loads(os.environ.get("OPENCODE_CONFIG_CONTENT", ""))
        except ValueError:
            inline = None
        permission = inline.get("permission") if isinstance(inline, dict) else None
        edit_rules = permission.get("edit") if isinstance(permission, dict) else None
        read_rules = permission.get("read") if isinstance(permission, dict) else None
        if (
            os.environ.get("AGS_LAB_BOUNDARY") != "v1"
            or not isinstance(edit_rules, dict)
            or edit_rules.get("*") != "allow"
            or not isinstance(read_rules, dict)
            or read_rules.get("*") != "allow"
        ):
            errors.append("OpenCode inline Lab policy is absent or restricts normal file access")
        checked = json.loads((LAB_ROOT / "opencode.json").read_text(encoding="utf-8"))
        if "LAB_CONTRACT.md" not in checked.get("instructions", []):
            errors.append("OpenCode checked-in config does not include the Lab contract")

    return tuple(errors)


def _print_scope(result: ScopeResult, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "route": result.route.value if result.route else "EMPTY",
                    "paths": {kind.value: list(result.paths[kind]) for kind in PathClass},
                },
                sort_keys=True,
            )
        )
    else:
        print(result.route.value if result.route else "EMPTY")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify-paths")
    classify.add_argument("paths", nargs="+")
    classify.add_argument("--json", action="store_true")

    staged = subparsers.add_parser("check-staged")
    staged.add_argument("--json", action="store_true")

    push = subparsers.add_parser("check-push")
    push.add_argument("--remote-name", required=True)
    push.add_argument("--json", action="store_true")

    range_parser = subparsers.add_parser("classify-range")
    range_parser.add_argument("--base", required=True)
    range_parser.add_argument("--head", required=True)
    range_parser.add_argument("--remote-name", default="origin")
    range_parser.add_argument("--json", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--lab-session", action="store_true")

    diagnose = subparsers.add_parser("doctor")
    diagnose.add_argument("--cwd", type=Path, default=Path.cwd())
    diagnose.add_argument("--agent", choices=("codex", "opencode", "generic"), required=True)
    diagnose.add_argument("--prompt-json", type=lambda value: value if value == "-" else Path(value))

    args = parser.parse_args(argv)
    try:
        if args.command == "classify-paths":
            _print_scope(classify_paths(args.paths), as_json=args.json)
        elif args.command == "check-staged":
            _print_scope(classify_paths(staged_paths()), as_json=args.json)
        elif args.command == "check-push":
            refs = parse_push_refs(sys.stdin.read())
            _print_scope(classify_push(refs, args.remote_name), as_json=args.json)
        elif args.command == "classify-range":
            _print_scope(
                classify_range(args.base, args.head, args.remote_name), as_json=args.json
            )
        elif args.command == "audit":
            paths = staged_paths() if args.lab_session else worktree_paths()
            result = classify_paths(paths)
            _print_scope(result, as_json=args.json)
        elif args.command == "doctor":
            errors = doctor(args.cwd, args.agent, prompt_json=args.prompt_json)
            if errors:
                print("LAB DOCTOR FAILED", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                return 1
            print(f"LAB DOCTOR OK ({args.agent})")
        return 0
    except (OSError, ScopeViolation, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
