#!/usr/bin/env python3
"""Build and optionally execute the canonical pytest plan for a push.

The mandatory push gate is risk-complete rather than repository-exhaustive:
core deterministic tests always run, expensive integration groups run only when
relevant paths changed, and ``--exhaustive`` remains available for releases,
nightly verification, and explicit deep validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TESTBENCH = "CAPABILITY/TESTBENCH"

EMBEDDING_TESTS = (
    f"{TESTBENCH}/integration/test_canon_embedding.py",
    f"{TESTBENCH}/integration/test_adr_embedding.py",
    f"{TESTBENCH}/integration/test_model_registry.py",
)

CORE_IGNORES = (
    f"{TESTBENCH}/mcp-capability-tests",
    f"{TESTBENCH}/cassette_network",
    *EMBEDDING_TESTS,
    f"{TESTBENCH}/integration/test_skill_discovery.py",
    f"{TESTBENCH}/integration/test_stacked_symbol_resolution.py",
    f"{TESTBENCH}/integration/test_write_firewall_enforcement.py",
    f"{TESTBENCH}/pipeline/test_write_firewall.py",
)

EMBEDDING_PREFIXES = (
    "NAVIGATION/CORTEX/semantic/",
    "NAVIGATION/CORTEX/indexes/",
    "CAPABILITY/PRIMITIVES/canon_index",
    "CAPABILITY/PRIMITIVES/adr_index",
    "CAPABILITY/PRIMITIVES/model_registry",
)
EMBEDDING_EXACT = {
    "requirements.txt",
    "pytest.ini",
    "CAPABILITY/TOOLS/utilities/push_test_plan.py",
    *EMBEDDING_TESTS,
}


@dataclass(frozen=True)
class TestSuite:
    name: str
    paths: tuple[str, ...]
    extra_args: tuple[str, ...] = ()


def normalize_paths(paths: Iterable[str]) -> list[str]:
    return sorted({p.strip().replace("\\", "/") for p in paths if p.strip()})


def requires_embeddings(paths: Iterable[str]) -> bool:
    for path in normalize_paths(paths):
        if path in EMBEDDING_EXACT or any(path.startswith(prefix) for prefix in EMBEDDING_PREFIXES):
            return True
    return False


def repo_python() -> str:
    """Return the repository virtualenv interpreter when it exists."""
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _interpreter_has_xdist(python_executable: str) -> bool:
    if Path(python_executable).resolve() == Path(sys.executable).resolve():
        return importlib.util.find_spec("xdist") is not None
    result = subprocess.run(
        [python_executable, "-c", "import xdist"],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _git_lines(args: Sequence[str]) -> list[str]:
    result = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def resolve_base_ref(explicit: str | None = None) -> str | None:
    candidates = [explicit, os.environ.get("AGS_PUSH_BASE"), "@{upstream}", "origin/main", "HEAD^"]
    for candidate in candidates:
        if not candidate or set(candidate) == {"0"}:
            continue
        if _git_lines(["git", "rev-parse", "--verify", candidate]):
            return candidate
    return None


def changed_paths(base_ref: str | None = None, explicit_paths: Iterable[str] = ()) -> tuple[list[str], str | None]:
    explicit = normalize_paths(explicit_paths)
    if explicit:
        return explicit, base_ref

    resolved = resolve_base_ref(base_ref)
    committed: list[str] = []
    if resolved:
        committed = _git_lines(["git", "diff", "--name-only", f"{resolved}...HEAD"])
        if not committed:
            committed = _git_lines(["git", "diff", "--name-only", f"{resolved}..HEAD"])

    local = _git_lines(["git", "diff", "--name-only"])
    staged = _git_lines(["git", "diff", "--cached", "--name-only"])
    return normalize_paths([*committed, *local, *staged]), resolved


def build_plan(paths: Iterable[str], *, exhaustive: bool = False) -> list[TestSuite]:
    if exhaustive:
        return [TestSuite("exhaustive", (TESTBENCH,))]

    core_args = tuple(f"--ignore={path}" for path in CORE_IGNORES)
    suites = [TestSuite("core", (TESTBENCH,), core_args)]
    if requires_embeddings(paths):
        suites.append(TestSuite("embeddings", EMBEDDING_TESTS))
    return suites


def pytest_command(
    suite: TestSuite,
    *,
    workers: int = 0,
    python_executable: str | None = None,
) -> list[str]:
    interpreter = python_executable or repo_python()
    command = [
        interpreter,
        "-m",
        "pytest",
        *suite.paths,
        "-q",
        "--tb=short",
        "--durations=25",
        *suite.extra_args,
    ]
    if workers > 0 and _interpreter_has_xdist(interpreter):
        command.extend(["-n", str(workers), "--dist=loadfile"])
    return command


def plan_payload(paths: Iterable[str], *, base_ref: str | None, exhaustive: bool, workers: int) -> dict:
    normalized = normalize_paths(paths)
    suites = build_plan(normalized, exhaustive=exhaustive)
    payload = {
        "mode": "exhaustive" if exhaustive else "full",
        "base_ref": base_ref,
        "changed_paths": normalized,
        "embedding_required": requires_embeddings(normalized),
        "suites": [
            {**asdict(suite), "command": pytest_command(suite, workers=workers)}
            for suite in suites
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["plan_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def run_plan(payload: dict) -> int:
    started = time.perf_counter()
    for suite in payload["suites"]:
        print(f"[push-test-plan] Running {suite['name']} suite")
        suite_start = time.perf_counter()
        result = subprocess.run(suite["command"], cwd=str(PROJECT_ROOT), check=False)
        elapsed = time.perf_counter() - suite_start
        print(f"[push-test-plan] {suite['name']} finished rc={result.returncode} elapsed={elapsed:.2f}s")
        if result.returncode != 0:
            return int(result.returncode)
    print(f"[push-test-plan] All suites passed in {time.perf_counter() - started:.2f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", help="Compare BASE...HEAD when determining changed paths")
    parser.add_argument("--changed-file", action="append", default=[], help="Explicit changed path; repeatable")
    parser.add_argument("--exhaustive", action="store_true", help="Run the entire TESTBENCH without exclusions")
    parser.add_argument("--workers", type=int, default=0, help="pytest-xdist workers when xdist is installed")
    parser.add_argument("--run", action="store_true", help="Execute the computed plan")
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON")
    args = parser.parse_args(argv)

    paths, base = changed_paths(args.base_ref, args.changed_file)
    payload = plan_payload(paths, base_ref=base, exhaustive=args.exhaustive, workers=max(args.workers, 0))

    if args.json or not args.run:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"[push-test-plan] mode={payload['mode']} base={payload['base_ref'] or 'none'} "
            f"changed={len(payload['changed_paths'])} embeddings={payload['embedding_required']} "
            f"plan={payload['plan_hash'][:12]}"
        )

    return run_plan(payload) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
