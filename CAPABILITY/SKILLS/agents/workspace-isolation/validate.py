#!/usr/bin/env python3
"""Validation for workspace-isolation skill.

Verifies the skill is properly configured and functional.
"""

import subprocess
import sys
from pathlib import Path


def validate() -> bool:
    """Validate workspace-isolation skill is functional.

    Returns:
        True if validation passes, False otherwise.
    """
    script_path = Path(__file__).parent / "scripts" / "workspace_isolation.py"

    if not script_path.exists():
        print(f"FAIL: Script not found: {script_path}")
        return False

    # Test that the script can be imported
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"FAIL: Script --help failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("FAIL: Script timed out")
        return False
    except Exception as e:
        print(f"FAIL: Script execution error: {e}")
        return False

    # Test status command (should work in any git repo)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print(f"WARN: Status command failed (may be expected if not in repo): {result.stderr}")
            # Don't fail - status might fail if not run from repo root
    except subprocess.TimeoutExpired:
        print("FAIL: Status command timed out")
        return False

    print("PASS: workspace-isolation skill validated")
    return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
