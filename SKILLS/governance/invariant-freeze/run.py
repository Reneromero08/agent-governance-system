#!/usr/bin/env python3

"""
Invariant freeze fixture check.

Verifies that all expected invariants exist in CANON/INVARIANTS.md.
This ensures invariants are not removed without a major version bump.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TOOLS.agents.skill_runtime import ensure_canon_compat

INVARIANTS_FILE = PROJECT_ROOT / "CANON" / "INVARIANTS.md"


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input: {exc}")
        return 1
    
    expected = payload.get("expected_invariants", [])
    
    # Read INVARIANTS.md and find all INV-XXX references
    if not INVARIANTS_FILE.exists():
        result = {
            **payload,
            "found_invariants": [],
            "missing": expected,
            "valid": False,
            "error": "CANON/INVARIANTS.md not found"
        }
    else:
        content = INVARIANTS_FILE.read_text(encoding="utf-8")
        found = sorted(set(re.findall(r"\[INV-\d+\]", content)))
        # Normalize to just the ID
        found_ids = [m.strip("[]") for m in found]
        
        missing = [inv for inv in expected if inv not in found_ids]
        
        result = {
            **payload,
            "found_invariants": found_ids,
            "missing": missing,
            "valid": len(missing) == 0
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    
    if result["valid"]:
        print(f"All {len(expected)} invariants present")
    else:
        print(f"Missing invariants: {result['missing']}")
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
