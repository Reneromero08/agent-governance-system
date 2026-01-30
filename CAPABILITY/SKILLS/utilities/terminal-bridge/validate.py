#!/usr/bin/env python3
"""
Validation logic for terminal-bridge skill fixtures.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def validate_output(expected: Dict[str, Any], actual: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate skill output against expected fixture.

    For status checks, we verify structure rather than exact values
    since server availability may vary.
    """
    # Check required top-level keys
    required_keys = {"ok", "operation", "server"}
    missing = required_keys - set(actual.keys())
    if missing:
        return False, f"Missing required keys: {missing}"

    # Operation must match
    if expected.get("operation") != actual.get("operation"):
        return False, f"Operation mismatch: expected {expected.get('operation')}, got {actual.get('operation')}"

    # For status operation, verify structure
    if actual.get("operation") == "status":
        result = actual.get("result", {})

        if actual.get("server") == "both":
            # Both servers checked
            if "ags" not in result or "vscode" not in result:
                return False, "Status result missing ags or vscode keys"

            # Verify AGS structure
            ags = result["ags"]
            ags_required = {"server", "name", "host", "port", "reachable", "ready"}
            if not ags_required.issubset(set(ags.keys())):
                return False, f"AGS status missing keys: {ags_required - set(ags.keys())}"

            # Verify VSCode structure
            vscode = result["vscode"]
            vscode_required = {"server", "name", "host", "port", "reachable", "ready"}
            if not vscode_required.issubset(set(vscode.keys())):
                return False, f"VSCode status missing keys: {vscode_required - set(vscode.keys())}"

        elif actual.get("server") == "ags":
            ags_required = {"server", "name", "host", "port", "reachable", "ready"}
            if not ags_required.issubset(set(result.keys())):
                return False, f"AGS status missing keys: {ags_required - set(result.keys())}"

        elif actual.get("server") == "vscode":
            vscode_required = {"server", "name", "host", "port", "reachable", "ready"}
            if not vscode_required.issubset(set(result.keys())):
                return False, f"VSCode status missing keys: {vscode_required - set(result.keys())}"

    # For setup_info operation, verify structure
    elif actual.get("operation") == "setup_info":
        result = actual.get("result", {})
        if "servers" not in result:
            return False, "setup_info result missing 'servers' key"

        servers = result["servers"]
        for srv in servers.values():
            if "name" not in srv or "instructions" not in srv:
                return False, "Server info missing 'name' or 'instructions'"

    return True, "Validation passed"


def main() -> int:
    """Run validation on fixture."""
    import sys

    if len(sys.argv) != 3:
        print("Usage: validate.py <expected.json> <actual.json>")
        return 1

    expected_path = Path(sys.argv[1])
    actual_path = Path(sys.argv[2])

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))

    passed, message = validate_output(expected, actual)
    print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
