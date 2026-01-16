#!/usr/bin/env python3
"""Rerun hard-3 with scipy now available."""

import sys
import os
import json
from datetime import datetime

scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from tool_executor_v2 import run_with_tools

OUTPUT_DIR = os.path.join(scripts_dir, "..", "nemotron-3-nano-30b-outputs", "task-benchmarks")

PROMPT = """Generate 1000 random samples from a normal distribution with mean=100, std=15.
Calculate: mean, median, std, skewness, kurtosis.
Then perform a Shapiro-Wilk test to verify normality.
Report p-value and conclusion."""

def save_result(task_id: str, prompt: str, result: str, status: str, expected: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{task_id}.json")
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
    print(f"[Saved to {task_id}.json]")

if __name__ == "__main__":
    print("="*70)
    print("RERUNNING: hard-3 (Statistical Analysis) - scipy now available")
    print("="*70)

    try:
        result = run_with_tools(PROMPT)
        print(f"\nRESULT:\n{result}")
        save_result("hard-3", PROMPT, result, "completed", "p > 0.05, normally distributed")
    except Exception as e:
        print(f"ERROR: {e}")
        save_result("hard-3", PROMPT, str(e), "error", "p > 0.05, normally distributed")
