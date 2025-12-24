#!/usr/bin/env python3
import time
import json
import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Add the MCP directory to sys.path
sys.path.insert(0, str(Path(__file__).parent / "MCP"))
from server import mcp_server

def sanitize_prompt(text):
    """Sanitize prompt for command line safety."""
    # Remove newlines and problematic characters
    text = text.replace('"', '\\"').replace('\n', ' ').replace('\r', '')
    return text

def run_governor_logic(directive):
    """Governor (Gemini) analyzes directive and dispatches to Ants."""
    mcp_server.acknowledge_directive(directive['directive_id'])
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*60)
    print(f"MANAGER (GOVERNOR) - NEW DIRECTIVE")
    print(f"ID: {directive['directive_id']}")
    print(f"TASK: {directive['directive']}")
    print("="*60 + "\n")
    
    print("[Governor] Launching Gemini CLI...")
    
    # We use gemini.cmd on Windows if available to avoid PowerShell quoting hell
    gemini_cmd = shutil.which("gemini.cmd") or "gemini"
    
    # Construct prompt safely
    raw_prompt = f"SWARM DIRECTIVE: {directive['directive']} -- You are the GOVERNOR. Analyze this and output mechanical subtasks."
    safe_prompt = sanitize_prompt(raw_prompt)
    
    try:
        # Launch interactive process
        # We use a list args which subprocess handles better for escaping
        retcode = subprocess.call([gemini_cmd, "--prompt", safe_prompt], shell=True)
        
        if retcode != 0:
            print(f"\n[ERROR] Gemini exited with code {retcode}.")
        else:
            print(f"\n[Governor] Gemini finished. Analysis complete.")
            
        mcp_server._log_operation({
            "operation": "governor_processed",
            "directive_id": directive["directive_id"],
            "exit_code": retcode
        })
        
    except Exception as e:
        print(f"[Governor] Failed to launch Gemini: {str(e)}")

def run_ant_logic(role, task):
    """Ant (Kilo Code) executes mechanical task."""
    mcp_server.acknowledge_task(task['task_id'])
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "-"*60)
    print(f"EXECUTOR ({role}) - NEW TASK")
    print(f"ID: {task['task_id']}")
    print(f"SPEC: {json.dumps(task['task_spec'])}")
    print("-"*60 + "\n")
    
    print(f"[{role}] Launching Kilo Code...")
    
    # Kilo Code prompt (positional argument)
    raw_prompt = f"TASK: {task['task_id']} -- Execute this: {json.dumps(task['task_spec'])}"
    safe_prompt = sanitize_prompt(raw_prompt)
    
    try:
        # Usage: npx @kilocode/cli [prompt]
        # We pass just the prompt positionally
        cmd = f'npx @kilocode/cli "{safe_prompt}"'
        
        retcode = subprocess.call(cmd, shell=True)
        
        if retcode == 0:
            mcp_server.report_result(
                task_id=task["task_id"],
                from_agent=role,
                status="success",
                result={"status": "completed", "agent": role}
            )
            print(f"\n[{role}] Success. Result reported.")
        else:
            print(f"\n[{role}] Failed with code {retcode}.")
            mcp_server.report_result(
                task_id=task["task_id"],
                from_agent=role,
                status="failed",
                result={},
                errors=[f"Exit code {retcode}"]
            )
            
    except Exception as e:
        print(f"[{role}] Critical failure: {str(e)}")
        mcp_server.report_result(
            task_id=task["task_id"],
            from_agent=role,
            status="error",
            result={},
            errors=[str(e)]
        )

def main():
    parser = argparse.ArgumentParser(description="CATALYTIC-DPT REAL CLI Swarm Loop")
    parser.add_argument("--role", required=True, help="Governor, Ant-1, Ant-2")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"=== {args.role.upper()} NODE ACTIVE ===")
    print(f"Connected to MCP Ledger.")
    print(f"Polling interval: {args.interval}s\n")
    
    while True:
        try:
            if args.role == "Governor":
                res = mcp_server.get_directives(args.role)
                if res["pending_count"] > 0:
                    for d in res["directives"]:
                        run_governor_logic(d)
            else:
                res = mcp_server.get_pending_tasks(args.role)
                if res["pending_count"] > 0:
                    for t in res["tasks"]:
                        run_ant_logic(args.role, t)
                        
        except Exception as e:
            print(f"[{args.role}] Error: {str(e)}")
            time.sleep(10)

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
