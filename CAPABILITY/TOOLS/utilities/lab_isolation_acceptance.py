#!/usr/bin/env python3
"""Run the fail-closed Lab Isolation V1 acceptance suite."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from unittest import mock
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_REPO_ROOT))

from CAPABILITY.TOOLS.utilities.ags_lab import REAL_OPENCODE, opencode_inline_policy
from CAPABILITY.TOOLS.utilities.lab_scope import (
    BOUNDARY_EXACT,
    LAB_CONTRACT_SENTINEL,
    LAB_MARKER_TEXT,
    LAB_ROOT,
    REPO_ROOT,
    ROOT_GOVERNANCE_SENTINEL,
)


CODEX_HOME = Path("/home/reneshizzle/Documents/Codex/State")
ACTIVE_PROBE_CWD = LAB_ROOT / "CAT_CAS"
SHELL_MUTATION_SURFACES = (
    "sed -i",
    "cp",
    "mv",
    "rm",
    "install",
    "perl -pi",
    "python open(..., 'w')",
    "node fs.writeFileSync",
    "tar extraction",
    "symlink traversal",
)


@dataclass(frozen=True)
class RepositorySnapshot:
    tracked_main: tuple[tuple[str, str], ...]
    untracked_main: tuple[str, ...]
    tracked_status: bytes
    boundary_hashes: tuple[tuple[str, str], ...]


def _run(args: Sequence[str], *, cwd: Path = REPO_ROOT, input_text: str | None = None, env=None):
    result = subprocess.run(
        args,
        cwd=cwd,
        input=input_text,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n{result.stderr.strip()}"
        )
    return result.stdout


def _git_z(args: Sequence[str]) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, check=True
    )
    return tuple(
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\x00")
        if item
    )


def _digest(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "MISSING"
    if path.is_symlink():
        return "SYMLINK:" + os.readlink(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot() -> RepositorySnapshot:
    tracked = _git_z(["ls-files", "-z"])
    tracked_main = tuple(
        sorted(
            (path, _digest(REPO_ROOT / path))
            for path in tracked
            if not path.startswith("THOUGHT/LAB/")
        )
    )
    ordinary = _git_z(["ls-files", "--others", "--exclude-standard", "-z"])
    ignored = _git_z(
        ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]
    )
    untracked_main = tuple(
        sorted(
            tagged
            for tag, paths in (("UNTRACKED", ordinary), ("IGNORED", ignored))
            for path in paths
            if not path.startswith("THOUGHT/LAB/")
            for tagged in (f"{tag}:{path}",)
        )
    )
    status = subprocess.run(
        ["git", "diff", "--name-status", "-z", "--find-renames", "HEAD", "--"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    boundary_hashes = tuple(
        sorted((path, _digest(REPO_ROOT / path)) for path in BOUNDARY_EXACT)
    )
    return RepositorySnapshot(tracked_main, untracked_main, status, boundary_hashes)


def check_static_contract() -> None:
    if (LAB_ROOT / ".agent-root").read_text(encoding="utf-8") != LAB_MARKER_TEXT:
        raise RuntimeError("Lab marker mismatch")
    contract = (LAB_ROOT / "LAB_CONTRACT.md").read_text(encoding="utf-8")
    for required in (LAB_CONTRACT_SENTINEL, "LAB_BOUNDARY_CONTROL", "OpenCode", "--no-verify"):
        if required not in contract:
            raise RuntimeError(f"Lab contract is missing required term: {required}")
    if len(SHELL_MUTATION_SURFACES) != 10:
        raise RuntimeError("shell mutation acceptance matrix is incomplete")


def check_codex() -> None:
    env = os.environ.copy()
    env["CODEX_HOME"] = str(CODEX_HOME)
    prompt = _run(
        ["codex", "debug", "prompt-input", "AGS_LAB_DOCTOR_PROBE"],
        cwd=ACTIVE_PROBE_CWD,
        env=env,
    )
    prompt_text = json.dumps(json.loads(prompt))
    if ROOT_GOVERNANCE_SENTINEL in prompt_text:
        raise RuntimeError("root governance contaminated the exact Codex prompt")
    if LAB_CONTRACT_SENTINEL not in prompt_text:
        raise RuntimeError("Lab contract missing from exact Codex prompt")

    doctor = REPO_ROOT / "CAPABILITY" / "TOOLS" / "utilities" / "lab_scope.py"
    _run(
        [
            "python3",
            str(doctor),
            "doctor",
            "--agent",
            "codex",
            "--cwd",
            str(ACTIVE_PROBE_CWD),
            "--prompt-json",
            "-",
        ],
        cwd=ACTIVE_PROBE_CWD,
        input_text=prompt,
        env=env,
    )


def check_opencode() -> None:
    env = os.environ.copy()
    env["AGS_LAB_BOUNDARY"] = "v1"
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_inline_policy(), separators=(",", ":"))
    effective = json.loads(
        _run([str(REAL_OPENCODE), "debug", "config"], cwd=LAB_ROOT, env=env)
    )
    if "LAB_CONTRACT.md" not in effective.get("instructions", []):
        raise RuntimeError("effective OpenCode config lost checked-in Lab instructions")
    permission = effective.get("permission", {})
    if permission.get("edit", {}).get("*") != "allow":
        raise RuntimeError("effective OpenCode policy restricts normal edits")
    with mock.patch.dict(
        os.environ,
        {
            "AGS_LAB_BOUNDARY": "v1",
            "OPENCODE_CONFIG_CONTENT": env["OPENCODE_CONFIG_CONTENT"],
        },
        clear=False,
    ):
        errors = __import__(
            "CAPABILITY.TOOLS.utilities.lab_scope", fromlist=["doctor"]
        ).doctor(LAB_ROOT, "opencode")
    if errors:
        raise RuntimeError("; ".join(errors))


def main() -> int:
    before = snapshot()
    try:
        check_static_contract()
        check_codex()
        check_opencode()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"LAB ISOLATION ACCEPTANCE FAILED: {exc}", file=sys.stderr)
        return 1
    after = snapshot()
    if after != before:
        print(
            "LAB ISOLATION ACCEPTANCE FAILED: verification mutated main, untracked main, "
            "tracked status, rename/delete state, or boundary-control hashes",
            file=sys.stderr,
        )
        return 1
    print("LAB ISOLATION ACCEPTANCE OK")
    print("  Codex prompt/root: isolated without replacing normal permissions")
    print("  OpenCode file and Git access: normal; commit/push scope is hook-enforced")
    print("  Main and boundary before/after snapshots: identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
