#!/usr/bin/env python3

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


def main(input_path: Path, output_path: Path, writer: Optional[GuardedWriter] = None) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    output_data = json.dumps(payload, indent=2, sort_keys=True)
    
    # Use GuardedWriter for writes if available, otherwise fallback
    # Always use GuardedWriter for writes to enforce firewall
    writer = writer or GuardedWriter(project_root=PROJECT_ROOT)
    
    try:
        # Convert output_path to relative path for GuardedWriter
        rel_output_path = str(output_path.resolve().relative_to(PROJECT_ROOT))
        writer.mkdir_tmp(str(Path(rel_output_path).parent))
        writer.write_tmp(rel_output_path, output_data)
    except ValueError:
        # If path is not relative to PROJECT_ROOT, this is a violation of the firewall contract
        # We must fail closed rather than falling back to raw writes
        print(f"Error: Output path {output_path} is outside project root")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
