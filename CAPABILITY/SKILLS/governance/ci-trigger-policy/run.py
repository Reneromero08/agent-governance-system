#!/usr/bin/env python3
"""
Run script for CI trigger policy fixture.

This script executes the CI trigger policy validation and returns the result.
"""

import json
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

def check_workflow_triggers(workflow_path: Path) -> dict:
    """Check if workflow has forbidden push triggers to main."""
    try:
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        violations = []

        # Check 'on' triggers
        triggers = workflow.get('on', {})

        # If 'on' is a list (shorthand), it's likely safe
        if isinstance(triggers, list):
            if 'push' in triggers:
                violations.append("Workflow uses shorthand 'push' trigger without branch restriction")

        # If 'on' is a dict, check push.branches
        elif isinstance(triggers, dict):
            push_config = triggers.get('push')
            if push_config is not None:
                # push exists - check if it targets main
                if isinstance(push_config, dict):
                    branches = push_config.get('branches', [])
                    if 'main' in branches or 'master' in branches:
                        violations.append(f"Workflow triggers on push to main/master: {branches}")
                else:
                    # push: without config means all branches
                    violations.append("Workflow has unrestricted 'push' trigger")

        return {
            "ok": len(violations) == 0,
            "policy": "no_push_to_main",
            "violations": violations
        }

    except Exception as e:
        return {
            "ok": False,
            "policy": "no_push_to_main",
            "violations": [f"Error reading workflow: {e}"]
        }

def main(input_path: Path, output_path: Path) -> int:
    try:
        if not ensure_canon_compat(Path(__file__).resolve().parent):
            sys.exit(1)
            
        payload = json.loads(input_path.read_text())
        workflow_file = payload.get("workflow_file")

        if not workflow_file:
            raise ValueError("Missing workflow_file in input")

        # Resolve the workflow file path relative to project root
        project_root = Path(__file__).resolve().parents[4]
        workflow_path = project_root / workflow_file

        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow file does not exist: {workflow_path}")

        result = check_workflow_triggers(workflow_path)

        if not GuardedWriter:
            print("Error: GuardedWriter not available")
            return 1

        writer = GuardedWriter(PROJECT_ROOT, tmp_roots=["LAW/CONTRACTS/_runs/_tmp"], durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
        writer.open_commit_gate()

        # Convert absolute path to relative path from repo root
        rel_output_path = output_path.resolve().relative_to(PROJECT_ROOT)
        writer.write_auto(str(rel_output_path), json.dumps(result, indent=2))
        return 0

    except Exception as e:
        error_result = {
            "ok": False,
            "policy": "no_push_to_main",
            "violations": [f"Error in CI trigger policy check: {e}"]
        }
        if GuardedWriter:
             writer = GuardedWriter(PROJECT_ROOT, tmp_roots=["LAW/CONTRACTS/_runs/_tmp"], durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
             writer.open_commit_gate()
             # Convert absolute path to relative path from repo root
             rel_output_path = output_path.resolve().relative_to(PROJECT_ROOT)
             writer.write_auto(str(rel_output_path), json.dumps(error_result, indent=2))
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    sys.exit(main(input_path, output_path))
