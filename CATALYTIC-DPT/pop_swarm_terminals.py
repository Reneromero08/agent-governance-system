import requests
import json
import os
import sys
import time
import shutil

# Configuration
BRIDGE_URL = "http://127.0.0.1:4000/terminal"
PROJECT_ROOT = r"d:\CCC 2.0\AI\agent-governance-system"

def launch_terminal(name, command, cwd):
    payload = {
        "name": name,
        "cwd": cwd,
        "initialCommand": command
    }
    try:
        r = requests.post(BRIDGE_URL, json=payload, timeout=5)
        if r.status_code == 200:
            print(f"Launched {name}.")
        else:
            print(f"Failed {name}: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Launching Swarm Terminals...")

    # 1. GOVERNOR (Manual Gemini CLI)
    # Fix: Use ExecutionPolicy Bypass and call gemini.cmd explicitly to avoid script blocking
    gov_cmd = f'powershell -ExecutionPolicy Bypass -NoExit -Command "Get-Content CATALYTIC-DPT/GOVERNOR_CONTEXT.md; Write-Host \'\\n-- GOVERNOR ONLINE --\\n\' -ForegroundColor Green; gemini.cmd"'
    launch_terminal("AGS: Governor", gov_cmd, PROJECT_ROOT)
    
    time.sleep(1)

    # 2. ANT-1 (Automated Worker)
    ant1_cmd = f"python \"{PROJECT_ROOT}\\CATALYTIC-DPT\\agent_loop.py\" --role Ant-1 --interval 2"
    launch_terminal("AGS: Ant-1", ant1_cmd, PROJECT_ROOT)
    
    time.sleep(1)

    # 3. ANT-2 (Automated Worker)
    ant2_cmd = f"python \"{PROJECT_ROOT}\\CATALYTIC-DPT\\agent_loop.py\" --role Ant-2 --interval 2"
    launch_terminal("AGS: Ant-2", ant2_cmd, PROJECT_ROOT)

    print("Done.")
