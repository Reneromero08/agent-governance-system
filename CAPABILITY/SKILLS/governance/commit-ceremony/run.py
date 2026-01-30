#!/usr/bin/env python3
"""
Skill: commit-ceremony

Run failsafe checks and return ceremony checklist with staged files.

Contract-style wrapper:
  python run.py <input.json> <output.json>

Deterministic fixture support:
- If `dry_run` is true with mock values, returns deterministic output
- Otherwise, runs actual checks
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    GuardedWriter = None


def _load_json(path: Path) -> Dict[str, Any]:
    """Load and validate JSON input."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("input must be a JSON object")
    return obj


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON output using GuardedWriter."""
    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        sys.exit(1)

    writer = GuardedWriter(
        PROJECT_ROOT,
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]
    )

    rel_path = path.resolve().relative_to(PROJECT_ROOT)
    writer.mkdir_auto(str(rel_path.parent))
    content = json.dumps(obj, indent=2, sort_keys=True)
    writer.write_auto(str(rel_path), content)


def run_critic() -> Dict[str, Any]:
    """Run the governance critic."""
    critic_path = PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "governance" / "critic.py"

    try:
        result = subprocess.run(
            [sys.executable, str(critic_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace"
        )
        output = result.stdout if result.stdout else result.stderr
        return {
            "passed": result.returncode == 0,
            "tool": "TOOLS/governance/critic.py",
            "output": output.strip()[-500:] if output else ""
        }
    except Exception as e:
        return {
            "passed": False,
            "tool": "TOOLS/governance/critic.py",
            "output": f"Error: {str(e)}"
        }


def run_contract_runner() -> Dict[str, Any]:
    """Run the contract runner."""
    runner_path = PROJECT_ROOT / "LAW" / "CONTRACTS" / "runner.py"

    try:
        result = subprocess.run(
            [sys.executable, str(runner_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace"
        )
        output = result.stdout if result.stdout else result.stderr
        return {
            "passed": result.returncode == 0,
            "tool": "LAW/CONTRACTS/runner.py",
            "output": output.strip()[-500:] if output else ""
        }
    except Exception as e:
        return {
            "passed": False,
            "tool": "LAW/CONTRACTS/runner.py",
            "output": f"Error: {str(e)}"
        }


def get_staged_files() -> List[str]:
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace"
        )
        return [f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        return []


def get_git_status() -> str:
    """Get short git status."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace"
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def run_ceremony(
    dry_run: bool = False,
    mock_critic_passed: Optional[bool] = None,
    mock_runner_passed: Optional[bool] = None,
    mock_staged_files: Optional[List[str]] = None,
    mock_git_status: Optional[str] = None
) -> Dict[str, Any]:
    """Run the commit ceremony checks."""

    if dry_run:
        # Use mock values for deterministic testing
        critic_result = {
            "passed": mock_critic_passed if mock_critic_passed is not None else True,
            "tool": "TOOLS/governance/critic.py",
            "output": "[MOCK] Critic check passed" if mock_critic_passed else "[MOCK] Critic check failed"
        }
        runner_result = {
            "passed": mock_runner_passed if mock_runner_passed is not None else True,
            "tool": "LAW/CONTRACTS/runner.py",
            "output": "[MOCK] Runner check passed" if mock_runner_passed else "[MOCK] Runner check failed"
        }
        staged_files = mock_staged_files if mock_staged_files is not None else []
        git_status = mock_git_status if mock_git_status is not None else ""
    else:
        # Run actual checks
        critic_result = run_critic()
        runner_result = run_contract_runner()
        staged_files = get_staged_files()
        git_status = get_git_status()

    files_staged = len(staged_files) > 0
    ready_for_commit = critic_result["passed"] and runner_result["passed"] and files_staged

    if ready_for_commit:
        prompt = f"Ready for the Chunked Commit Ceremony? Shall I commit these {len(staged_files)} files?"
    else:
        prompt = "Ceremony cannot proceed - failsafe checks must pass and files must be staged."

    return {
        "checklist": {
            "1_failsafe_critic": critic_result,
            "2_failsafe_runner": runner_result,
            "3_files_staged": files_staged,
            "4_ready_for_commit": ready_for_commit
        },
        "staged_files": staged_files,
        "staged_count": len(staged_files),
        "git_status": git_status,
        "ceremony_prompt": prompt
    }


def main(argv: list) -> int:
    """Main entry point."""
    if len(argv) != 3:
        sys.stderr.write("Usage: run.py <input.json> <output.json>\n")
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2])

    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        inp = _load_json(input_path)
    except Exception as e:
        print(f"Error reading input: {e}")
        return 1

    dry_run = inp.get("dry_run", False)
    mock_critic_passed = inp.get("_mock_critic_passed")
    mock_runner_passed = inp.get("_mock_runner_passed")
    mock_staged_files = inp.get("_mock_staged_files")
    mock_git_status = inp.get("_mock_git_status")

    result = run_ceremony(
        dry_run=dry_run,
        mock_critic_passed=mock_critic_passed,
        mock_runner_passed=mock_runner_passed,
        mock_staged_files=mock_staged_files,
        mock_git_status=mock_git_status
    )

    # Merge input with result for traceability
    output = {**inp, **result}

    try:
        _write_json(output_path, output)
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
