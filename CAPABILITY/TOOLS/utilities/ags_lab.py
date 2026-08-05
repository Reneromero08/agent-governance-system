#!/usr/bin/env python3
"""Launch supported agents inside the canonical AGS Lab boundary."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_REPO_ROOT))

from CAPABILITY.TOOLS.utilities.lab_scope import LAB_ROOT, REPO_ROOT, canonical_runtime_path, doctor


REAL_OPENCODE = Path("/home/reneshizzle/.opencode/bin/opencode")


def _managed_payloads(payload: object) -> set[Path]:
    found: set[Path] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            candidate = value.get("payload_path")
            if isinstance(candidate, str):
                found.add(Path(candidate).resolve())
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def resolve_scratch(explicit: Path | None) -> Path:
    manager = shutil.which("codex-scratch")
    if not manager:
        raise RuntimeError("codex-scratch is required for Lab runtime output")
    if explicit is None:
        result = subprocess.run(
            [manager, "current-dir"], capture_output=True, text=True, check=False
        )
        if result.returncode:
            raise RuntimeError(
                "no attributable managed Scratch payload; supply --scratch with an exact "
                "path returned by codex-scratch allocate"
            )
        candidate = Path(result.stdout.strip()).resolve()
    else:
        candidate = explicit.resolve(strict=True)

    status = subprocess.run(
        [manager, "status", "--json"], capture_output=True, text=True, check=False
    )
    if status.returncode:
        raise RuntimeError("cannot verify managed Scratch ownership")
    if candidate not in _managed_payloads(json.loads(status.stdout)):
        raise RuntimeError(f"Scratch path is not a controller-allocated payload: {candidate}")
    if str(candidate).startswith(("/dev/shm/", "/run/shm/")):
        raise RuntimeError("RAM-backed Scratch is forbidden")
    return candidate


def runtime_environment(scratch: Path) -> dict[str, str]:
    paths = {
        "TMPDIR": scratch / "tmp",
        "TEMP": scratch / "tmp",
        "TMP": scratch / "tmp",
        "XDG_CACHE_HOME": scratch / "cache",
        "PYTHONPYCACHEPREFIX": scratch / "pycache",
        "COVERAGE_FILE": scratch / "coverage" / ".coverage",
    }
    for path in set(paths.values()):
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in paths.items()})
    env["AGS_LAB_ROOT"] = str(LAB_ROOT)
    env["AGS_REPO_ROOT"] = str(REPO_ROOT)
    env["AGS_LAB_SCRATCH"] = str(scratch)
    env["AGS_LAB_BOUNDARY"] = "v1"
    return env


def opencode_inline_policy() -> dict:
    bash: dict[str, str] = {
        "*": "allow",
        # Keep only explicit attempts to bypass the repository's mechanical
        # scope enforcement out of the normal Lab route. Ordinary Bash and Git
        # commands remain available.
        "git *--no-verify*": "deny",
        "git -c core.hooksPath=*": "deny",
    }
    return {
        "permission": {
            "external_directory": {"*": "allow"},
            "read": {"*": "allow"},
            "edit": {"*": "allow"},
            "bash": bash,
        }
    }


def launch(agent: str, cwd: Path, scratch: Path | None, arguments: Sequence[str]) -> int:
    resolved_cwd = canonical_runtime_path(cwd, beneath=LAB_ROOT)
    payload = resolve_scratch(scratch)
    env = runtime_environment(payload)

    if agent == "codex":
        executable = shutil.which("codex")
        if not executable:
            raise RuntimeError("codex executable is unavailable")
        command = [
            executable,
            "-C",
            str(resolved_cwd),
            *arguments,
        ]
    else:
        if not REAL_OPENCODE.is_file():
            raise RuntimeError(f"OpenCode executable is unavailable: {REAL_OPENCODE}")
        # Overwrite inherited inline content. Checked-in instructions stay in
        # opencode.json; this layer contains only dynamic absolute permissions.
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_inline_policy(), separators=(",", ":"))
        command = [str(REAL_OPENCODE), str(resolved_cwd), *arguments]
    return subprocess.call(command, cwd=resolved_cwd, env=env)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("codex", "opencode"):
        launch_parser = subparsers.add_parser(name)
        launch_parser.add_argument("--cwd", type=Path, default=LAB_ROOT)
        launch_parser.add_argument("--scratch", type=Path)
        launch_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    diagnose = subparsers.add_parser("doctor")
    diagnose.add_argument("--agent", choices=("codex", "opencode", "generic"), required=True)
    diagnose.add_argument("--cwd", type=Path, default=Path.cwd())

    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            errors = doctor(args.cwd, args.agent)
            if errors:
                for error in errors:
                    print(f"LAB DOCTOR: {error}", file=sys.stderr)
                return 1
            print(f"LAB DOCTOR OK ({args.agent})")
            return 0
        remainder = args.arguments
        if remainder[:1] == ["--"]:
            remainder = remainder[1:]
        return launch(args.command, args.cwd, args.scratch, remainder)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ags-lab: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
