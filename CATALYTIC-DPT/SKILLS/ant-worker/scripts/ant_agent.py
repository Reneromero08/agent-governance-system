#!/usr/bin/env python3
import sys
import json
import time
import argparse
from pathlib import Path

# Add repo root to path to import MCP modules
# File: .../CATALYTIC-DPT/SKILLS/ant-worker/scripts/ant_agent.py
# Root: .../ (Parents[4])
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT))

# Also add CATALYTIC-DPT to path for imports
sys.path.append(str(REPO_ROOT / "CATALYTIC-DPT"))

try:
    from LAB.MCP.server import MCPTerminalServer
    # Import LFM2 runner
    sys.path.append(str(Path(__file__).parent))
    from lfm2_runner import run_model
except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr)
    # Fallback to local import if run from different CWD
    try:
        from lfm2_runner import run_model
    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Ant Worker Agent (LFM2)")
    parser.add_argument("--agent_id", default="Ant-LFM2", help="ID of this agent on the MCP bus")
    parser.add_argument("--poll_interval", type=int, default=5, help="Seconds between ledger polls")
    
    args = parser.parse_args()
    
    config_path = REPO_ROOT / "CATALYTIC-DPT" / "swarm_config.json"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return

    mcp = MCPTerminalServer()
    
    print(f"[{args.agent_id}] Online. Connected to MCP Ledger. Polling every {args.poll_interval}s...")
    print(f"[{args.agent_id}] Backend: Liquid LFM2-2.6B (via transformers)")

    # Simple dedup to avoid re-running tasks if file isn't updated (naive implementation)
    # Ideally, MCP server handles processing status. 
    # For now, we assume get_pending_tasks returns tasks that need doing.
    
    processed_tasks = set()

    try:
        while True:
            # Poll
            # This requires get_pending_tasks to be implemented in server.py
            # If not exposed, we might need to read jsonl directly or use mcp_client logic.
            # Assuming functionality exists based on previous file review.
            try:
                response = mcp.get_pending_tasks(args.agent_id)
                tasks = response.get("tasks", [])
            except Exception as e:
                # If get_pending_tasks fails/doesn't exist, log and wait
                # print(f"Polling error: {e}") 
                tasks = []

            for task in tasks:
                task_id = task.get("id")
                if task_id in processed_tasks:
                    continue
                
                print(f"[{args.agent_id}] Received Task {task_id}: {task.get('spec', {}).get('instruction', 'No instruction')}")
                
                # Execute
                spec = task.get("spec", {})
                prompt = f"TASK: {spec.get('instruction', '')}\nCONTEXT: {spec.get('context', '')}\n\nProvide the result strictly obeying the format."
                
                try:
                    result = run_model(prompt)
                    status = "success"
                except Exception as e:
                    result = f"Error: {e}"
                    status = "failure"
                
                # Report
                print(f"[{args.agent_id}] Reporting result for {task_id}...")
                mcp.report_result(task_id, args.agent_id, status, {"output": result})
                processed_tasks.add(task_id)
            
            time.sleep(args.poll_interval)
            
    except KeyboardInterrupt:
        print(f"\n[{args.agent_id}] Shutting down.")

if __name__ == "__main__":
    main()
