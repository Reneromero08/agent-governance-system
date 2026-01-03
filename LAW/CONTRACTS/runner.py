#!/usr/bin/env python3

"""
Contract runner for the Agent Governance System.

This script discovers and runs fixtures under the `LAW/CONTRACTS/fixtures/` directory
and skill fixtures under `CAPABILITY/SKILLS/*/fixtures/`. A fixture consists of an input
(`input.json` or other files) and an expected output (`expected.json`).
The runner executes the relevant skill or validation script, then compares
actual output to expected output.

Any fixture that fails will cause the runner to exit with a non-zero exit code.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SKILLS_DIR = PROJECT_ROOT / "CAPABILITY" / "SKILLS"
RUNS_DIR = Path(__file__).parent / "_runs"
DEFAULT_VALIDATE = SKILLS_DIR / "_TEMPLATE" / "validate.py"

SYSTEM1_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
CORTEX_DB = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated" / "cortex.db"


def run_process(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def run_process_stream(args: List[str], *, cwd: Path) -> int:
    """
    Run a subprocess while streaming stdout/stderr to the console.

    The contract runner is often executed in environments where stdout can be
    buffered; streaming avoids "looks stuck" behavior for long-running steps.
    """
    res = subprocess.run(args, cwd=str(cwd))
    return res.returncode


def run_validation(validate_script: Path, actual_path: Path, expected_path: Path) -> subprocess.CompletedProcess:
    return run_process([
        sys.executable,
        str(validate_script),
        str(actual_path),
        str(expected_path),
    ])


def iter_contract_inputs() -> List[Path]:
    return sorted(FIXTURES_DIR.rglob("input.json"))


def iter_skill_inputs() -> List[Tuple[Path, Path]]:
    fixtures = []
    for category_dir in sorted(SKILLS_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        # Iterate through skills in each category
        for skill_dir in sorted(category_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue
            fixtures_root = skill_dir / "fixtures"
            if not fixtures_root.exists():
                continue
            for input_path in sorted(fixtures_root.rglob("input.json")):
                fixtures.append((skill_dir, input_path))
    return fixtures


def ensure_navigation_dbs() -> int:
    """
    Ensure required navigation DBs exist before running fixtures.

    CI builds these DBs earlier, but local runs of the contract runner should be
    self-sufficient and deterministic.
    """
    steps: List[Tuple[Path, List[str]]] = []

    cortex_build = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "cortex.build.py"
    system1_reset = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "reset_system1.py"

    if not CORTEX_DB.exists():
        steps.append((cortex_build, [sys.executable, str(cortex_build)]))

    # system1-verify fixtures assert the DB matches current repo content. Rebuild it
    # deterministically before running fixtures (mirrors CI behavior).
    steps.append((system1_reset, [sys.executable, str(system1_reset)]))

    for script_path, cmd in steps:
        if not script_path.exists():
            print(f"[contracts/runner] Missing required build script: {script_path}")
            return 1
        print(f"[contracts/runner] Building navigation DB via {script_path.relative_to(PROJECT_ROOT)}", flush=True)
        rc = run_process_stream(cmd, cwd=PROJECT_ROOT)
        if rc != 0:
            print(f"[contracts/runner] Build failed (rc={rc})", flush=True)
            return 1

    return 0


def run_contract_fixture(input_path: Path) -> int:
    fixture_dir = input_path.parent
    expected = fixture_dir / "expected.json"
    if not expected.exists():
        print(f"Skipping {fixture_dir}: no expected.json")
        return 0
    if not DEFAULT_VALIDATE.exists():
        print(f"Missing default validator at {DEFAULT_VALIDATE}")
        return 1
    print(f"Running contract fixture in {fixture_dir}.", flush=True)
    result = run_validation(DEFAULT_VALIDATE, input_path, expected)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    else:
        print(result.stdout)
    if result.returncode != 0:
        print(f"!!! FAILURE: {fixture_dir} !!!")
        return 1
    return 0


def run_skill_fixture(skill_dir: Path, input_path: Path) -> int:
    fixture_dir = input_path.parent
    expected = fixture_dir / "expected.json"
    if not expected.exists():
        print(f"Skipping {fixture_dir}: no expected.json")
        return 0

    run_script = skill_dir / "run.py"
    validate_script = skill_dir / "validate.py"
    if not run_script.exists():
        print(f"Missing run.py for skill {skill_dir.name} at {run_script}")
        return 1
    if not validate_script.exists():
        validate_script = DEFAULT_VALIDATE

    relative_fixture = fixture_dir.relative_to(skill_dir / "fixtures")
    output_dir = RUNS_DIR / "fixtures" / skill_dir.name / relative_fixture
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_path = output_dir / "actual.json"

    print(f"Running skill fixture in {fixture_dir}.", flush=True)
    rc = run_process_stream([
        sys.executable,
        "-u",
        str(run_script),
        str(input_path),
        str(actual_path),
    ], cwd=PROJECT_ROOT)
    if rc != 0:
        print(f"!!! FAILURE (EXEC): {fixture_dir} !!!")
        return 1

    result = run_validation(validate_script, actual_path, expected)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print(f"!!! FAILURE (VAL): {fixture_dir} !!!")
        return 1
    print(result.stdout)
    if result.returncode != 0:
        print(f"!!! FAILURE: {fixture_dir} !!!")
        return 1
    return 0


def run_fixtures() -> int:
    failures = 0
    failures += ensure_navigation_dbs()
    if failures:
        return failures
    for input_path in iter_contract_inputs():
        failures += run_contract_fixture(input_path)
    for skill_dir, input_path in iter_skill_inputs():
        failures += run_skill_fixture(skill_dir, input_path)
    return failures


if __name__ == "__main__":
    failures = run_fixtures()
    if failures:
        print(f"\n{failures} fixture(s) failed.")
        print("FAILED FIXTURES:")
        # We need to track names, but changing return type of run_fixtures is invasive.
        # Instead, let's print failure immediately with a distinct marker.
        sys.exit(1)
    print("All fixtures passed")
    sys.exit(0)
