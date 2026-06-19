#!/usr/bin/env python3
"""Build and optionally execute the canonical pytest plan for a push.

The mandatory push gate is risk-complete rather than repository-exhaustive:
deterministic core tests always run, each excluded infrastructure suite owns an
explicit changed-path risk group, and ``--exhaustive`` remains available for
releases, nightly verification, and deliberate deep validation.
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
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TESTBENCH = "CAPABILITY/TESTBENCH"

EMBEDDING_TESTS = (
    f"{TESTBENCH}/integration/test_canon_embedding.py",
    f"{TESTBENCH}/integration/test_adr_embedding.py",
    f"{TESTBENCH}/integration/test_model_registry.py",
)
SKILL_DISCOVERY_TESTS = (f"{TESTBENCH}/integration/test_skill_discovery.py",)
MCP_CAPABILITY_TESTS = (f"{TESTBENCH}/mcp-capability-tests",)
CASSETTE_NETWORK_TESTS = (f"{TESTBENCH}/cassette_network",)
WRITE_FIREWALL_TESTS = (
    f"{TESTBENCH}/integration/test_write_firewall_enforcement.py",
    f"{TESTBENCH}/pipeline/test_write_firewall.py",
)
SYMBOL_RESOLUTION_TESTS = (f"{TESTBENCH}/integration/test_stacked_symbol_resolution.py",)

GLOBAL_TRIGGER_EXACT = {
    ".github/workflows/contracts.yml",
    "pytest.ini",
    "requirements.txt",
    f"{TESTBENCH}/conftest.py",
    f"{TESTBENCH}/01_core/test_push_test_plan.py",
    "CAPABILITY/TOOLS/utilities/ci_local_gate.py",
    "CAPABILITY/TOOLS/utilities/push_test_plan.py",
}
GLOBAL_TRIGGER_GLOBS = ("requirements*.txt",)


class PlanError(RuntimeError):
    """Raised when the planner cannot safely determine the change set."""


@dataclass(frozen=True)
class TestSuite:
    name: str
    paths: tuple[str, ...]
    extra_args: tuple[str, ...] = ()
    xdist: bool = True


@dataclass(frozen=True)
class RiskGroup:
    name: str
    tests: tuple[str, ...]
    trigger_prefixes: tuple[str, ...] = ()
    trigger_exact: tuple[str, ...] = ()
    trigger_globs: tuple[str, ...] = ()
    xdist: bool = True


RISK_GROUPS = (
    RiskGroup(
        "write-firewall",
        WRITE_FIREWALL_TESTS,
        trigger_exact=(
            "CAPABILITY/PRIMITIVES/paths.py",
            "CAPABILITY/PRIMITIVES/repo_digest.py",
            "CAPABILITY/PRIMITIVES/write_firewall.py",
            "CAPABILITY/TOOLS/utilities/guarded_writer.py",
        ),
    ),
    RiskGroup(
        "symbol-resolution",
        SYMBOL_RESOLUTION_TESTS,
        trigger_prefixes=("CAPABILITY/TOOLS/scl/",),
        trigger_exact=(
            "CAPABILITY/PRIMITIVES/scl_validator.py",
            "CAPABILITY/TOOLS/codebook_lookup.py",
        ),
    ),
    RiskGroup(
        "mcp-capability",
        MCP_CAPABILITY_TESTS,
        trigger_prefixes=(
            "CAPABILITY/MCP/",
            "CAPABILITY/PIPELINES/",
            "CAPABILITY/SKILLS/agents/ant-worker/",
            "CAPABILITY/SKILLS/mcp-toolkit/",
            "CAPABILITY/TOOLS/catalytic/",
            "NAVIGATION/CORTEX/cassettes/",
            "NAVIGATION/CORTEX/network/",
        ),
        trigger_exact=(
            "CAPABILITY/PRIMITIVES/paths.py",
            "CAPABILITY/PRIMITIVES/registry_validators.py",
            "CAPABILITY/TOOLS/ags.py",
            "LAW/CONTRACTS/ags_mcp_entrypoint.py",
            "TOOLS/catalytic.py",
        ),
        xdist=False,
    ),
    RiskGroup(
        "skill-discovery",
        SKILL_DISCOVERY_TESTS,
        trigger_prefixes=("NAVIGATION/CORTEX/semantic/",),
        trigger_exact=("CAPABILITY/PRIMITIVES/skill_index.py",),
        trigger_globs=("CAPABILITY/SKILLS/**/SKILL.md",),
    ),
    RiskGroup(
        "cassette-network",
        CASSETTE_NETWORK_TESTS,
        trigger_prefixes=(
            "NAVIGATION/CORTEX/cassettes/",
            "NAVIGATION/CORTEX/network/",
            "NAVIGATION/CORTEX/semantic/",
        ),
        trigger_exact=(
            "CAPABILITY/MCP/cortex_geometric.py",
            "CAPABILITY/PRIMITIVES/elo_db.py",
        ),
    ),
    RiskGroup(
        "embeddings",
        EMBEDDING_TESTS,
        trigger_prefixes=(
            "NAVIGATION/CORTEX/indexes/",
            "NAVIGATION/CORTEX/semantic/",
        ),
        trigger_exact=(
            "CAPABILITY/PRIMITIVES/adr_index.py",
            "CAPABILITY/PRIMITIVES/canon_index.py",
            "CAPABILITY/PRIMITIVES/model_registry.py",
        ),
    ),
)


def _owned_test_paths(groups: Iterable[RiskGroup]) -> tuple[str, ...]:
    owners: dict[str, str] = {}
    ordered: list[str] = []
    for group in groups:
        if not group.tests:
            raise PlanError(f"risk group has no tests: {group.name}")
        for path in group.tests:
            previous = owners.get(path)
            if previous is not None:
                raise PlanError(
                    f"conditional test path {path} has multiple owners: {previous}, {group.name}"
                )
            owners[path] = group.name
            ordered.append(path)
    return tuple(ordered)


CORE_IGNORES = _owned_test_paths(RISK_GROUPS)


def normalize_paths(paths: Iterable[str]) -> list[str]:
    return sorted({p.strip().replace("\\", "/") for p in paths if p.strip()})


def _path_is_within(path: str, root: str) -> bool:
    root = root.rstrip("/")
    return path == root or path.startswith(root + "/")


def _is_global_trigger(path: str) -> bool:
    return path in GLOBAL_TRIGGER_EXACT or any(fnmatchcase(path, pattern) for pattern in GLOBAL_TRIGGER_GLOBS)


def matched_paths(group: RiskGroup, paths: Iterable[str]) -> tuple[str, ...]:
    matches: list[str] = []
    for path in normalize_paths(paths):
        if _is_global_trigger(path):
            matches.append(path)
            continue
        if path in group.trigger_exact or any(path.startswith(prefix) for prefix in group.trigger_prefixes):
            matches.append(path)
            continue
        if any(fnmatchcase(path, pattern) for pattern in group.trigger_globs):
            matches.append(path)
            continue
        if any(_path_is_within(path, test_path) for test_path in group.tests):
            matches.append(path)
    return tuple(matches)


def selected_risk_groups(paths: Iterable[str]) -> tuple[tuple[RiskGroup, tuple[str, ...]], ...]:
    normalized = normalize_paths(paths)
    selected: list[tuple[RiskGroup, tuple[str, ...]]] = []
    for group in RISK_GROUPS:
        reasons = matched_paths(group, normalized)
        if reasons:
            selected.append((group, reasons))
    return tuple(selected)


def requires_embeddings(paths: Iterable[str]) -> bool:
    return any(group.name == "embeddings" for group, _ in selected_risk_groups(paths))


def repo_python() -> str:
    """Return the repository virtualenv interpreter when it exists."""
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


@lru_cache(maxsize=None)
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


def _git_result(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _git_lines(args: Sequence[str], *, required: bool = False) -> list[str]:
    result = _git_result(args)
    if result.returncode != 0:
        if required:
            detail = (result.stderr or result.stdout or "unknown git error").strip()
            raise PlanError(f"git command failed: {' '.join(args)}: {detail}")
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_ref_exists(candidate: str) -> bool:
    return bool(_git_lines(["git", "rev-parse", "--verify", candidate]))


def resolve_base_ref(explicit: str | None = None) -> str | None:
    if explicit:
        if set(explicit) == {"0"}:
            return None
        if not _git_ref_exists(explicit):
            raise PlanError(f"explicit base ref does not resolve: {explicit}")
        return explicit

    environment_base = os.environ.get("AGS_PUSH_BASE")
    if environment_base:
        if set(environment_base) == {"0"}:
            return None
        if not _git_ref_exists(environment_base):
            raise PlanError(f"AGS_PUSH_BASE does not resolve: {environment_base}")
        return environment_base

    for candidate in ("@{upstream}", "origin/main", "HEAD^"):
        if _git_ref_exists(candidate):
            return candidate
    return None


def _git_diff_lines(range_spec: str) -> list[str] | None:
    result = _git_result(["git", "diff", "--name-only", range_spec])
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def changed_paths(base_ref: str | None = None, explicit_paths: Iterable[str] = ()) -> tuple[list[str], str | None]:
    explicit = normalize_paths(explicit_paths)
    if explicit:
        return explicit, base_ref

    resolved = resolve_base_ref(base_ref)
    committed: list[str]
    if resolved:
        merge_base_diff = _git_diff_lines(f"{resolved}...HEAD")
        direct_diff = _git_diff_lines(f"{resolved}..HEAD")
        if merge_base_diff is None and direct_diff is None:
            raise PlanError(f"cannot diff resolved base {resolved} against HEAD")
        committed = [*(merge_base_diff or []), *(direct_diff or [])]
    else:
        # Initial/untracked history: treat every tracked path as changed rather
        # than silently under-selecting conditional suites.
        committed = _git_lines(["git", "ls-tree", "-r", "--name-only", "HEAD"], required=True)

    local = _git_lines(["git", "diff", "--name-only"], required=True)
    staged = _git_lines(["git", "diff", "--cached", "--name-only"], required=True)
    return normalize_paths([*committed, *local, *staged]), resolved


def build_plan(paths: Iterable[str], *, exhaustive: bool = False) -> list[TestSuite]:
    if exhaustive:
        return [TestSuite("exhaustive", (TESTBENCH,))]

    core_args = tuple(f"--ignore={path}" for path in CORE_IGNORES)
    suites = [TestSuite("core", (TESTBENCH,), core_args)]
    suites.extend(
        TestSuite(group.name, group.tests, xdist=group.xdist)
        for group, _ in selected_risk_groups(paths)
    )
    return suites


def pytest_command(
    suite: TestSuite,
    *,
    workers: int = 0,
    python_executable: str | None = None,
    xdist_available: bool | None = None,
) -> list[str]:
    interpreter = python_executable or repo_python()
    if xdist_available is None:
        xdist_available = workers > 0 and _interpreter_has_xdist(interpreter)
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
    if workers > 0 and xdist_available and suite.xdist:
        command.extend(["-n", str(workers), "--dist=loadfile"])
    return command


def plan_payload(paths: Iterable[str], *, base_ref: str | None, exhaustive: bool, workers: int) -> dict:
    normalized = normalize_paths(paths)
    suites = build_plan(normalized, exhaustive=exhaustive)
    selected = selected_risk_groups(normalized) if not exhaustive else ()
    interpreter = repo_python()
    xdist_available = workers > 0 and _interpreter_has_xdist(interpreter)
    effective_workers = workers if xdist_available else 0
    risk_groups = [
        {"name": group.name, "matched_paths": list(reasons)}
        for group, reasons in selected
    ]
    semantic_suites = [asdict(suite) for suite in suites]
    hash_material = {
        "mode": "exhaustive" if exhaustive else "full",
        "base_ref": base_ref,
        "changed_paths": normalized,
        "risk_groups": risk_groups,
        "suites": semantic_suites,
        "workers": effective_workers,
    }
    encoded = json.dumps(hash_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        **hash_material,
        "embedding_required": any(item["name"] == "embeddings" for item in risk_groups),
        "python": interpreter,
        "xdist": xdist_available,
        "suites": [
            {
                **asdict(suite),
                "command": pytest_command(
                    suite,
                    workers=workers,
                    python_executable=interpreter,
                    xdist_available=xdist_available,
                ),
            }
            for suite in suites
        ],
        "plan_hash": hashlib.sha256(encoded).hexdigest(),
    }
    return payload


def run_plan(payload: dict) -> int:
    started = time.perf_counter()
    for suite in payload["suites"]:
        print(f"[push-test-plan] Running {suite['name']} suite", flush=True)
        suite_start = time.perf_counter()
        result = subprocess.run(suite["command"], cwd=str(PROJECT_ROOT), check=False)
        elapsed = time.perf_counter() - suite_start
        print(f"[push-test-plan] {suite['name']} finished rc={result.returncode} elapsed={elapsed:.2f}s", flush=True)
        if result.returncode != 0:
            return int(result.returncode)
    print(f"[push-test-plan] All suites passed in {time.perf_counter() - started:.2f}s", flush=True)
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

    try:
        paths, base = changed_paths(args.base_ref, args.changed_file)
        payload = plan_payload(paths, base_ref=base, exhaustive=args.exhaustive, workers=max(args.workers, 0))
    except PlanError as exc:
        print(f"[push-test-plan] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json or not args.run:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"[push-test-plan] mode={payload['mode']} base={payload['base_ref'] or 'none'} "
            f"changed={len(payload['changed_paths'])} groups="
            f"{','.join(item['name'] for item in payload['risk_groups']) or 'none'} "
            f"plan={payload['plan_hash'][:12]}"
        )

    return run_plan(payload) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
