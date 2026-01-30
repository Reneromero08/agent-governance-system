#!/usr/bin/env python3
"""
Skill: critic-run

Run TOOLS/governance/critic.py to verify governance compliance.

Contract-style wrapper:
  python run.py <input.json> <output.json>
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


def run_critic(verbose: bool = False) -> Dict[str, Any]:
    """Run the governance critic and return structured result."""
    critic_path = PROJECT_ROOT / "CAPABILITY" / "TOOLS" / "governance" / "critic.py"

    if not critic_path.exists():
        return {
            "passed": False,
            "output": f"Critic script not found: {critic_path}",
            "exit_code": 1
        }

    cmd = [sys.executable, str(critic_path)]
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace"
        )

        passed = result.returncode == 0
        output = result.stdout + result.stderr

        return {
            "passed": passed,
            "output": output.strip(),
            "exit_code": result.returncode
        }
    except Exception as e:
        return {
            "passed": False,
            "output": f"Critic execution error: {str(e)}",
            "exit_code": 1
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

    verbose = bool(inp.get("verbose", False))
    result = run_critic(verbose)

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
