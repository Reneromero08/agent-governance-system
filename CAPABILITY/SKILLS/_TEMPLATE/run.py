#!/usr/bin/env python3

"""
Template skill runner.

This is a skeleton script for a skill. Replace this file with implementation
code that performs the skill's action.
"""

import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None


# Ensure writer defaults if main called without it
def main(input_path: Path, output_path: Path, writer: Optional[GuardedWriter] = None) -> int:
    if writer is None and GuardedWriter:
        writer = GuardedWriter(PROJECT_ROOT)
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    # Template: echo the input as output (replace with actual logic)
    output_data = json.dumps(payload, indent=2, sort_keys=True)
    
    # Use GuardedWriter for writes if available, otherwise fallback
    if writer:
        try:
            # Convert output_path to relative path for GuardedWriter
            rel_output_path = str(output_path.resolve().relative_to(PROJECT_ROOT))
            writer.mkdir_tmp(str(Path(rel_output_path).parent)) # Get parent directory
            writer.write_tmp(rel_output_path, output_data)
        except ValueError:
            print("Output path outside project root.")
            return 1
    else:
        # Enforce usage
        print("GuardedWriter required.")
        return 1
        
    print("[skill] Template skill executed successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
