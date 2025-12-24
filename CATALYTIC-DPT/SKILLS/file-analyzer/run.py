#!/usr/bin/env python3
"""
CATALYTIC-DPT/SKILLS/gemini-file-analyzer/run.py

Uses Gemini CLI to analyze AGI repo and identify critical files for:
1. Swarm-governor architecture
2. Antigravity Bridge integration
3. Launch-terminal skill implementation

Outputs a structured list of files needed for integration.
"""

import json
import sys
import subprocess
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def call_gemini(task_description, repo_path):
    """
    Call Gemini CLI to analyze the repo.

    Returns: Gemini's response as string
    """
    # Prepare the full prompt for Gemini
    prompt = f"""
    Analyze this AGI repository and answer:

    {task_description}

    Repository location: {repo_path}

    Return ONLY valid JSON with this structure:
    {{
      "status": "success",
      "findings": [
        {{"file": "path/to/file", "importance": "critical|important|useful", "description": "..."}}
      ],
      "recommendations": ["...", "..."]
    }}
    """

    try:
        # Call Gemini CLI with the prompt using --prompt flag
        result = subprocess.run(
            ["gemini", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            # Try to parse the output as JSON
            try:
                response_data = json.loads(result.stdout)
                return response_data
            except json.JSONDecodeError:
                # If not valid JSON, wrap in our structure
                return {
                    "status": "success",
                    "findings": [],
                    "recommendations": [result.stdout]
                }
        else:
            return {
                "status": "error",
                "message": f"Gemini returned {result.returncode}",
                "detail": result.stderr
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Gemini call timed out after 60s"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Gemini CLI not found. Install with: npm install -g google-ai-cli"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def build_task_prompt(task_type, focus_areas, repo_path):
    """Build a specific task prompt for Gemini based on task_type."""

    prompts = {
        "analyze_swarm": f"""
            Analyze {repo_path} and identify all files related to swarm-governor.

            Focus areas: {', '.join(focus_areas)}

            For each file, provide:
            1. Path relative to repo root
            2. Importance level (critical/important/useful)
            3. 1-2 sentence description of what it does
            4. Why it's needed to understand the swarm system

            List only files that are actually critical for understanding how swarm-governor:
            - Manages parallel task execution
            - Communicates with workers
            - Handles task scheduling and timeouts
            - Returns results
        """,

        "find_gemini_config": f"""
            Search {repo_path} for any Gemini CLI configuration files or documentation.

            Look for:
            1. .gemini/ or gemini/ directories
            2. gemini.config.* files
            3. Documentation mentioning Gemini CLI
            4. References to Google AI API keys or models

            Return the path to each config file and describe what it configures.
        """,

        "identify_dependencies": f"""
            For the files in these focus areas:
            {', '.join(focus_areas)}

            Identify all dependencies:
            1. Python imports (what packages are needed)
            2. File dependencies (which files import/use which)
            3. External service dependencies (ports, APIs)
            4. Configuration dependencies (env vars, config files)

            Return as a dependency graph showing how files relate.
        """,

        "list_critical_files": f"""
            From {repo_path}, identify the MINIMUM set of files needed to understand:
            1. How swarm-governor spawns workers
            2. How Antigravity Bridge connects to VSCode
            3. How launch-terminal sends commands to user's IDE (not Claude's terminal)
            4. Integration points for Gemini CLI

            For each file, provide:
            - Relative path
            - Why it's critical
            - Size (lines of code)
            - What would break without it
        """
    }

    return prompts.get(task_type, prompts["analyze_swarm"])

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load input
    input_data = load_json(input_path)

    repo_path = input_data.get("repo_path", "D:/CCC 2.0/AI/AGI")
    task_type = input_data.get("task_type", "analyze_swarm")
    focus_areas = input_data.get("focus_areas", [
        "SKILLS/swarm-governor",
        "SKILLS/launch-terminal",
        "EXTENSIONS/antigravity-bridge"
    ])

    # Build the prompt for this task
    prompt = build_task_prompt(task_type, focus_areas, repo_path)

    # Call Gemini
    print(f"[gemini-file-analyzer] Calling Gemini CLI for task: {task_type}")
    print(f"[gemini-file-analyzer] Repo path: {repo_path}")

    response = call_gemini(prompt, repo_path)

    # Wrap response with metadata
    output_data = {
        "status": response.get("status", "unknown"),
        "task_type": task_type,
        "repo_path": repo_path,
        "gemini_analysis": response
    }

    # Save output
    save_json(output_path, output_data)

    print(f"[gemini-file-analyzer] Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
