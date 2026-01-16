#!/usr/bin/env python3

"""
SKILLS/launch-terminal/run.py

Entrypoint for the launch-terminal skill.
Strictly uses the Antigravity Bridge (Port 4000) to spawn internal terminals.
"""

import json
import sys
import requests
import os
from pathlib import Path

# Configuration
BRIDGE_URL = "http://127.0.0.1:4001/terminal"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def launch_terminal(name, command, cwd):
    payload = {
        "name": name,
        "cwd": cwd,
        "initialCommand": command
    }
    
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=2)
        if response.status_code == 200:
            return {"status": "success", "message": f"Terminal '{name}' launched."}
        else:
            return {"status": "error", "message": f"Bridge returned {response.status_code}", "detail": response.text}
    except requests.RequestException as e:
        return {"status": "error", "message": "Failed to connect to Antigravity Bridge", "detail": str(e)}

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Load Input
    data = load_json(input_path)
    
    # Extract Params
    name = data.get("name", "Agent Term")
    command = data.get("command", "")
    cwd = data.get("cwd", os.getcwd())
    
    # Execute
    result = launch_terminal(name, command, cwd)
    
    # Save Output
    save_json(output_path, result)

if __name__ == "__main__":
    main()
