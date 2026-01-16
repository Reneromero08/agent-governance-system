#!/usr/bin/env python3
"""Run warmup-4 test only."""

import sys
import os
import json
from datetime import datetime

scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from tool_executor_v2 import run_with_tools

OUTPUT_DIR = os.path.join(scripts_dir, "..", "nemotron-3-nano-30b-outputs", "task-benchmarks")

PROMPT = """Who is the current CEO of Microsoft? Search the web."""

def save_result(task_id: str, prompt: str, result: str, status: str, expected: str):
    """Save test result to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"{task_id}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    data = {
        "test_suite": "task_benchmarks",
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "prompt": prompt.strip(),
        "expected": expected,
        "result": result
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Saved to {filename}]")

if __name__ == "__main__":
    print("="*70)
    print("RUNNING: warmup-4 (Microsoft CEO)")
    print("="*70)

    try:
        result = run_with_tools(PROMPT)
        print(f"\nRESULT:\n{result}")
        save_result("warmup-4", PROMPT, result, "completed", "Satya Nadella")
    except Exception as e:
        print(f"ERROR: {e}")
        save_result("warmup-4", PROMPT, str(e), "error", "Satya Nadella")
