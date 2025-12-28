#!/usr/bin/env python3
"""
MCP Wrapper (Phase 6.4)
Executes an MCP server as a subprocess with strict, fail-closed governance.
"""

import sys
import json
import subprocess
import hashlib
import time
import os
from pathlib import Path
from typing import Dict, Any, Tuple

# Adding REPO_ROOT to path to import primitives if needed
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def canonical_json_bytes(obj: Any) -> bytes:
    """Produce canonical JSON bytes (sorted keys, no whitespace, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def fail(code: str, message: str):
    print(f"FATAL: {code} - {message}", file=sys.stderr)
    sys.exit(1)

def run_mcp(config_path: str, output_path: str):
    config_file = Path(config_path)
    if not config_file.exists():
        fail("CONFIG_MISSING", f"Config file not found: {config_path}")

    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception as e:
        fail("CONFIG_MALFORMED", str(e))

    # Extract constraints
    cmd_vec = config.get("server_command")
    if not cmd_vec or not isinstance(cmd_vec, list):
        fail("INVALID_COMMAND", "server_command must be a non-empty list")
    
    caps = config.get("caps", {})
    # Default caps
    max_stdout = caps.get("max_stdout_bytes", 65536)
    max_stderr = caps.get("max_stderr_bytes", 0) # strict default
    timeout_ms = caps.get("timeout_ms", 10000)
    allowed_exits = caps.get("allowed_exit_codes", [0])

    request_envelope = config.get("request_envelope")
    if not request_envelope:
        fail("MISSING_ENVELOPE", "request_envelope required")

    # Serialize request
    req_bytes = canonical_json_bytes(request_envelope) + b"\n"

    # Execution
    start_time = time.time()
    try:
        # We run from REPO_ROOT to ensure consistent paths for the child if it needs them
        proc = subprocess.run(
            cmd_vec,
            input=req_bytes,
            capture_output=True,
            timeout=timeout_ms / 1000.0,
            cwd=str(REPO_ROOT) 
        )
    except subprocess.TimeoutExpired:
        fail("TIMEOUT", f"Execution exceeded {timeout_ms}ms")
    except Exception as e:
        fail("EXECUTION_ERROR", str(e))

    # Governance Checks
    if proc.returncode not in allowed_exits:
        # Provide stderr for debugging in the hard failure message
        fail("BAD_EXIT_CODE", f"Exit code {proc.returncode} not in {allowed_exits}. Child stderr: {proc.stderr.decode('utf-8', errors='replace')}")

    if len(proc.stdout) > max_stdout:
        fail("STDOUT_OVERFLOW", f"Stdout {len(proc.stdout)} > {max_stdout} bytes")

    if len(proc.stderr) > max_stderr:
        # Fail closed on stderr usage
        fail("STDERR_EMITTED", f"Stderr emitted: {proc.stderr.decode('utf-8', errors='replace')[:200]}...")

    # Output Production
    transcript_hash = hashlib.sha256(proc.stdout).hexdigest()
    
    try:
        if not proc.stdout:
             fail("INVALID_JSON_OUTPUT", "Empty stdout")
             
        # MCP Output (usually JSON-RPC response)
        # We strictly require JSON output for now.
        response_json = json.loads(proc.stdout.decode("utf-8"))
    except Exception as e:
        fail("INVALID_JSON_OUTPUT", f"MCP server did not produce valid JSON: {str(e)}. Stdout snippet: {proc.stdout[:100]!r}")

    result_artifact = {
        "status": "success",
        "transcript_hash": transcript_hash,
        "response": response_json
    }

    # Write output
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(canonical_json_bytes(result_artifact))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wrapper.py <config.json> <output.json>", file=sys.stderr)
        sys.exit(1)
    
    run_mcp(sys.argv[1], sys.argv[2])
