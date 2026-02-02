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

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Literal, Optional

@dataclass
class FixtureResult:
    name: str
    status: Literal["passed", "failed", "skipped"]
    duration_ms: int
    error: Optional[str] = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SKILLS_DIR = PROJECT_ROOT / "CAPABILITY" / "SKILLS"
RUNS_DIR = Path(__file__).parent / "_runs"
DEFAULT_VALIDATE = SKILLS_DIR / "_TEMPLATE" / "validate.py"

# Cassette network handles semantic search
# See: NAVIGATION/CORTEX/cassettes/*.db
CASSETTES_DIR = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"


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

    Navigation is now handled by the cassette network (NAVIGATION/CORTEX/cassettes/).
    This function verifies at least some cassettes exist.

    In CI environments (CI=true), this check is skipped since cassettes are
    derived artifacts that aren't tracked in git.
    """
    import os
    if os.environ.get("CI") == "true":
        print("[contracts/runner] CI environment detected - skipping cassette check", flush=True)
        return 0

    if not CASSETTES_DIR.exists():
        print(f"[contracts/runner] Cassettes directory not found: {CASSETTES_DIR}")
        return 1

    cassette_dbs = list(CASSETTES_DIR.glob("*.db"))
    if not cassette_dbs:
        print(f"[contracts/runner] No cassette databases found in {CASSETTES_DIR}")
        return 1

    print(f"[contracts/runner] Found {len(cassette_dbs)} cassette databases", flush=True)
    return 0


def run_contract_fixture(input_path: Path, quiet: bool = False) -> FixtureResult:
    fixture_dir = input_path.parent
    fixture_name = str(fixture_dir.relative_to(FIXTURES_DIR))
    expected = fixture_dir / "expected.json"

    if not expected.exists():
        if not quiet:
            print(f"Skipping {fixture_dir}: no expected.json")
        return FixtureResult(name=fixture_name, status="skipped", duration_ms=0)

    if not DEFAULT_VALIDATE.exists():
        if not quiet:
            print(f"Missing default validator at {DEFAULT_VALIDATE}")
        return FixtureResult(
            name=fixture_name,
            status="failed",
            duration_ms=0,
            error="Missing default validator"
        )

    if not quiet:
        print(f"Running contract fixture in {fixture_dir}.", flush=True)

    start = time.perf_counter()
    result = run_validation(DEFAULT_VALIDATE, input_path, expected)
    elapsed = time.perf_counter() - start
    duration_ms = int(elapsed * 1000)

    if result.returncode != 0:
        if not quiet:
            print(result.stdout)
            print(result.stderr)
            print(f"!!! FAILURE: {fixture_dir} !!!")
        error_msg = result.stderr.strip() if result.stderr.strip() else "Validation failed"
        return FixtureResult(name=fixture_name, status="failed", duration_ms=duration_ms, error=error_msg)
    else:
        if not quiet:
            print(result.stdout)
            print(f"[contracts/runner] OK ({elapsed:.2f}s)", flush=True)
        return FixtureResult(name=fixture_name, status="passed", duration_ms=duration_ms)


def run_skill_fixture(skill_dir: Path, input_path: Path, quiet: bool = False) -> FixtureResult:
    fixture_dir = input_path.parent
    relative_fixture = fixture_dir.relative_to(skill_dir / "fixtures")
    fixture_name = f"{skill_dir.name}/{relative_fixture}"
    expected = fixture_dir / "expected.json"

    if not expected.exists():
        if not quiet:
            print(f"Skipping {fixture_dir}: no expected.json")
        return FixtureResult(name=fixture_name, status="skipped", duration_ms=0)

    run_script = skill_dir / "run.py"
    validate_script = skill_dir / "validate.py"
    if not run_script.exists():
        if not quiet:
            print(f"Missing run.py for skill {skill_dir.name} at {run_script}")
        return FixtureResult(
            name=fixture_name,
            status="failed",
            duration_ms=0,
            error="Missing run.py"
        )

    if not validate_script.exists():
        validate_script = DEFAULT_VALIDATE

    output_dir = RUNS_DIR / "_tmp" / "fixtures" / skill_dir.name / relative_fixture
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_path = output_dir / "actual.json"

    if not quiet:
        print(f"Running skill fixture in {fixture_dir}.", flush=True)

    fixture_start = time.perf_counter()
    rc = run_process_stream([
        sys.executable,
        "-u",
        str(run_script),
        str(input_path),
        str(actual_path),
    ], cwd=PROJECT_ROOT)

    if rc != 0:
        elapsed = time.perf_counter() - fixture_start
        duration_ms = int(elapsed * 1000)
        if not quiet:
            print(f"!!! FAILURE (EXEC): {fixture_dir} !!!")
        return FixtureResult(
            name=fixture_name,
            status="failed",
            duration_ms=duration_ms,
            error="Execution failed"
        )

    exec_elapsed = time.perf_counter() - fixture_start

    validate_start = time.perf_counter()
    result = run_validation(validate_script, actual_path, expected)
    validate_elapsed = time.perf_counter() - validate_start
    total_elapsed = time.perf_counter() - fixture_start
    duration_ms = int(total_elapsed * 1000)

    if result.returncode != 0:
        if not quiet:
            print(result.stdout)
            print(result.stderr)
            print(f"!!! FAILURE (VAL): {fixture_dir} !!!")
        error_msg = result.stderr.strip() if result.stderr.strip() else "Validation failed"
        return FixtureResult(name=fixture_name, status="failed", duration_ms=duration_ms, error=error_msg)

    if not quiet:
        print(result.stdout)
        print(
            f"[contracts/runner] OK (exec={exec_elapsed:.2f}s, validate={validate_elapsed:.2f}s, total={total_elapsed:.2f}s)",
            flush=True,
        )
    return FixtureResult(name=fixture_name, status="passed", duration_ms=duration_ms)


def matches_filter(fixture_name: str, pattern: Optional[str]) -> bool:
    """Check if a fixture name matches the filter pattern."""
    if pattern is None:
        return True
    return pattern.lower() in fixture_name.lower()


def run_fixtures(filter_pattern: Optional[str] = None, quiet: bool = False) -> List[FixtureResult]:
    """
    Run all fixtures and return results.

    Args:
        filter_pattern: Optional string to filter fixtures by name
        quiet: If True, suppress output (for JSON mode)

    Returns:
        List of FixtureResult objects
    """
    results: List[FixtureResult] = []

    # Ensure navigation DBs are built
    if ensure_navigation_dbs() != 0:
        # If DB build fails, return a failed result
        return [FixtureResult(
            name="navigation_db_setup",
            status="failed",
            duration_ms=0,
            error="Failed to build navigation databases"
        )]

    # Run contract fixtures
    for input_path in iter_contract_inputs():
        fixture_dir = input_path.parent
        fixture_name = str(fixture_dir.relative_to(FIXTURES_DIR))
        if matches_filter(fixture_name, filter_pattern):
            result = run_contract_fixture(input_path, quiet=quiet)
            results.append(result)

    # Run skill fixtures
    for skill_dir, input_path in iter_skill_inputs():
        fixture_dir = input_path.parent
        relative_fixture = fixture_dir.relative_to(skill_dir / "fixtures")
        fixture_name = f"{skill_dir.name}/{relative_fixture}"
        if matches_filter(fixture_name, filter_pattern):
            result = run_skill_fixture(skill_dir, input_path, quiet=quiet)
            results.append(result)

    return results


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Contract runner for the Agent Governance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                    # Run all fixtures
  python runner.py --json             # Output results as JSON
  python runner.py --filter skill1    # Run only fixtures matching 'skill1'
  python runner.py --json --filter db # JSON output for fixtures matching 'db'
        """
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON array"
    )
    parser.add_argument(
        "--filter",
        type=str,
        metavar="PATTERN",
        help="Run only fixtures whose name contains PATTERN (case-insensitive)"
    )

    args = parser.parse_args()

    # Run fixtures
    results = run_fixtures(filter_pattern=args.filter, quiet=args.json)

    # Output results
    if args.json:
        # JSON mode: print results as JSON array
        json_results = [asdict(r) for r in results]
        print(json.dumps(json_results, indent=2))
    else:
        # Standard mode: print summary
        failures = sum(1 for r in results if r.status == "failed")
        if failures:
            print(f"\n{failures} fixture(s) failed.")
            print("FAILED FIXTURES:")
            for result in results:
                if result.status == "failed":
                    print(f"  - {result.name}: {result.error}")
            sys.exit(1)
        print("All fixtures passed")

    # Exit with appropriate code
    failures = sum(1 for r in results if r.status == "failed")
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
