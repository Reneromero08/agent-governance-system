#!/usr/bin/env python3
"""
Canon Governance Check Skill Wrapper

Runs the canon governance check and optionally logs to Cortex provenance.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Determine repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def run_governance_check(verbose=False):
    """Run the canon governance check script."""
    cmd = ["node", str(REPO_ROOT / "TOOLS" / "check-canon-governance.js")]
    if verbose:
        cmd.append("--verbose")
    
    result = subprocess.run(
        cmd, 
        cwd=REPO_ROOT, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0
    }

def log_to_cortex(result):
    """Log governance check to Cortex provenance if CORTEX_RUN_ID is set."""
    run_id = os.environ.get("CORTEX_RUN_ID")
    if not run_id:
        return
    
    ledger_dir = REPO_ROOT / "CONTRACTS" / "_runs" / run_id
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "events.jsonl"
    
    event = {
        "type": "governance_check",
        "timestamp": os.environ.get("CORTEX_TIMESTAMP", "SENTINEL"),
        "passed": result["passed"],
        "exit_code": result["exit_code"]
    }
    
    # Append to ledger
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        f.write("\n")

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    print("🔍 Running Canon Governance Check...\n")
    result = run_governance_check(verbose)
    
    # Print output
    if result["stdout"]:
        print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"], file=sys.stderr)
    
    # Log to Cortex if configured
    try:
        log_to_cortex(result)
    except Exception as e:
        print(f"⚠️  Failed to log to Cortex: {e}", file=sys.stderr)
    
    # Exit with same code as underlying check
    sys.exit(result["exit_code"])

if __name__ == "__main__":
    main()
