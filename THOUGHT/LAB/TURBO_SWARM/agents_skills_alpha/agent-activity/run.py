#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Path to the audit log
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "CONTRACTS" / "_runs" / "mcp_logs" / "audit.jsonl"

def get_file_from_args(tool, args):
    """Extract file/context information from tool arguments."""
    if not isinstance(args, dict):
        return None
    
    # Common file-related keys
    if "file" in args:
        return args["file"]
    if "path" in args:
        return args["path"]
    if "uri" in args:
        return args["uri"]
    if "pack_path" in args:
        return args["pack_path"]
    
    # Specific tool handlers
    if tool == "cortex_query" and "query" in args:
        return f"Query: {args['query']}"
    if tool == "skill_run" and "skill" in args:
        return f"Skill: {args['skill']}"
    if tool == "codebook_lookup" and "id" in args:
        return f"Codebook: {args['id']}"
    
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: run.py <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        with open(input_path, 'r') as f:
            params = json.load(f)
    except Exception:
        params = {}

    limit = params.get("limit", 10)
    active_within_seconds = params.get("active_within", 3600)  # Default 1 hour
    
    # Allow log path override for testing
    if "log_path" in params:
        log_file_path = Path(params["log_path"])
        # If relative, first try relative to input file directory (for fixtures)
        if not log_file_path.is_absolute():
            # Get the input file path from command line arguments
            input_path = Path(sys.argv[1])
            input_dir = input_path.parent
            log_file_path_from_input = input_dir / log_file_path

            if log_file_path_from_input.exists():
                log_file_path = log_file_path_from_input
            else:
                # Fall back to PROJECT_ROOT
                log_file_path = PROJECT_ROOT / log_file_path
    else:
        log_file_path = LOG_FILE

    if not log_file_path.exists():
        # No logs yet
        with open(output_path, 'w') as f:
            json.dump({"active_agents": [], "message": f"No audit logs found at {log_file_path}"}, f)
        return

    sessions = {}
    
    # Read log file (it can be large, so read line by line)
    # We want the *latest* state, so reading from end would be better, but JSONL is variable length.
    # We'll read all valid lines.
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                session_id = entry.get("session_id")
                timestamp_str = entry.get("timestamp")
                
                # If no session_id (legacy logs), skip or group under "legacy"
                if not session_id:
                    continue

                # Parse timestamp
                try:
                    ts = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    continue

                # Update session state (keep the latest)
                if session_id not in sessions or ts > sessions[session_id]["timestamp"]:
                    sessions[session_id] = {
                        "timestamp": ts,
                        "timestamp_str": timestamp_str,
                        "tool": entry.get("tool"),
                        "args": entry.get("arguments", {}),
                        "status": entry.get("status")
                    }
    except Exception as e:
        with open(output_path, 'w') as f:
            json.dump({"error": f"Failed to read audit logs: {str(e)}"}, f)
        sys.exit(1)

    # Filter and format output
    if "reference_time" in params:
        try:
            now = datetime.fromisoformat(params["reference_time"])
        except ValueError:
            now = datetime.now()
    else:
        now = datetime.now()
        
    cutoff = now - timedelta(seconds=active_within_seconds)
    
    active_agents = []
    
    for sess_id, data in sessions.items():
        if data["timestamp"] >= cutoff:
            # Extract file/context
            context = get_file_from_args(data["tool"], data["args"])
            
            active_agents.append({
                "session_id": sess_id,
                "last_seen": data["timestamp_str"],
                "seconds_ago": int((now - data["timestamp"]).total_seconds()),
                "tool": data["tool"],
                "working_on": context,
                "status": data["status"]
            })

    # Sort by most recent
    active_agents.sort(key=lambda x: x["last_seen"], reverse=True)
    active_agents = active_agents[:limit]

    output = {
        "count": len(active_agents),
        "active_agents": active_agents
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
