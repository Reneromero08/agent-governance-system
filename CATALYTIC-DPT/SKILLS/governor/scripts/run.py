#!/usr/bin/env python3
"""
CATALYTIC-DPT/SKILLS/governor/run.py

Enables President (Claude) to delegate tasks to Gemini CLI (Governor),
which runs in the user's VSCode terminal.

Key insight:
- Claude sends: {"gemini_prompt": "analyze D:/CCC 2.0/AI/AGI/..."}
- Gemini (your terminal): gemini --prompt "analyze ..."
- Result: back to Claude with Gemini's analysis

This preserves token budget AND gives you control over what Gemini does.
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def execute_gemini_prompt(prompt, timeout=60):
    """
    Execute a Gemini prompt in YOUR VSCode terminal (via gemini CLI).

    Args:
        prompt: The prompt to send to Gemini
        timeout: Maximum seconds to wait for response

    Returns:
        dict with status, response, and metadata
    """
    try:
        # Call Gemini CLI with the prompt using --prompt flag
        result = subprocess.run(
            ["gemini", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            # Try to parse as JSON first
            try:
                response = json.loads(result.stdout)
                return {
                    "status": "success",
                    "gemini_response": response,
                    "raw_output": result.stdout
                }
            except json.JSONDecodeError:
                # If not JSON, return as text
                return {
                    "status": "success",
                    "gemini_response": result.stdout,
                    "raw_output": result.stdout,
                    "note": "Response was text, not JSON"
                }
        else:
            return {
                "status": "error",
                "message": f"Gemini CLI returned exit code {result.returncode}",
                "stderr": result.stderr,
                "stdout": result.stdout
            }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"Gemini call timed out after {timeout}s",
            "recommendation": "Increase timeout or simplify the prompt"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Gemini CLI not found in PATH",
            "installation": "npm install -g google-ai-cli",
            "or_visit": "https://github.com/google/google-ai-cli"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__
        }

def build_specialized_prompt(command_type, prompt, **kwargs):
    """
    Wrap a prompt with context based on command_type.

    Types:
    - analyze: Read/analyze files
    - execute: Run a command
    - research: Deep research
    - report: Generate a report
    """

    wrappers = {
        "analyze": lambda p: f"""Analyze the following and provide a structured response:

{p}

Return your findings in JSON format with keys: 'findings', 'files_identified', 'recommendations'""",

        "execute": lambda p: f"""Execute this task and provide step-by-step output:

{p}

Format: Return what was done, any errors, and the final result.""",

        "research": lambda p: f"""Research this topic thoroughly and provide detailed findings:

{p}

Include: background, key findings, references, and actionable recommendations.""",

        "report": lambda p: f"""Generate a comprehensive report on:

{p}

Include: executive summary, detailed findings, analysis, and recommendations."""
    }

    wrapper = wrappers.get(command_type, lambda p: p)
    return wrapper(prompt)

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load input
    input_data = load_json(input_path)

    gemini_prompt = input_data.get("gemini_prompt")
    task_id = input_data.get("task_id", "unnamed_task")
    timeout = input_data.get("timeout_seconds", 60)
    command_type = input_data.get("command_type", "analyze")
    output_format = input_data.get("output_format", "json")

    if not gemini_prompt:
        error_output = {
            "status": "error",
            "message": "Missing required field: 'gemini_prompt'",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        save_json(output_path, error_output)
        print(f"ERROR: {error_output['message']}")
        sys.exit(1)

    # Wrap prompt with context if needed
    full_prompt = build_specialized_prompt(command_type, gemini_prompt)

    print(f"[gemini-executor] Task: {task_id}")
    print(f"[gemini-executor] Type: {command_type}")
    print(f"[gemini-executor] Calling Gemini CLI in YOUR VSCode terminal...")
    print(f"[gemini-executor] Timeout: {timeout}s")

    # Execute the prompt via Gemini
    gemini_result = execute_gemini_prompt(full_prompt, timeout)

    # Build output
    output_data = {
        "status": gemini_result.get("status"),
        "task_id": task_id,
        "command_type": command_type,
        "executed_in": "YOUR_VSCode_Terminal_via_Gemini_CLI",
        "timestamp": datetime.now().isoformat(),
        "gemini_analysis": gemini_result
    }

    # Save output
    save_json(output_path, output_data)

    # Print summary
    if gemini_result.get("status") == "success":
        print(f"[gemini-executor] [OK] Success. Results saved to {output_path}")
    else:
        print(f"[gemini-executor] [ERROR] {gemini_result.get('message')}")
        print(f"[gemini-executor] Results saved to {output_path} for review")

if __name__ == "__main__":
    main()
