#!/usr/bin/env python3
"""
Re-run tests that timed out with the old 180s limit.
Now using 300s timeout + REPL stripping + action_input support.

Run this in a separate terminal - take all the time you need.
"""

import sys
import os
import json
from datetime import datetime

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from tool_executor_v2 import run_with_tools

OUTPUT_DIR = os.path.join(scripts_dir, "..", "nemotron-3-nano-30b-outputs", "wtf-tests")

def save_result(name: str, prompt: str, result: str, status: str):
    """Save test result to output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"rerun-{name}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    data = {
        "test_suite": "rerun_timeouts",
        "test_name": name,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "prompt": prompt.strip(),
        "result": result
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved to {filename}]")

# ============================================================================
# TESTS TO RE-RUN
# ============================================================================

TESTS = {
    "floating-point": """What is machine epsilon for 64-bit IEEE floating point?
Compute it using Python, then verify: does (1.0 + epsilon) - 1.0 == epsilon?""",

    "capability-probe": """List every tool you have access to. For each one, give an example of when you would use it vs when you would NOT use it. Then answer: what is the integral of e^(-x^2) from -infinity to infinity?""",

    "integer-partition": """Compute p(100) - the number of integer partitions of 100.
Use dynamic programming. Show your work step by step.""",

    "ambiguity": """The number 2 is both prime and even.
The number 1 is neither prime nor composite.
The number 0.999... equals 1.
sqrt(4) = +/- 2.

For each statement, determine if it's TRUE, FALSE, or CONTEXT-DEPENDENT.
Provide rigorous mathematical justification.""",

    "final-boss": """Multi-step challenge:
1. What is the half-life of Carbon-14? (Use Wikipedia if needed)
2. Starting with 1kg of C-14, compute how much remains after 50,000 years
3. Express your answer in both kg and number of atoms
4. Verify your calculation using the exponential decay formula
5. What percentage of the original sample remains?"""
}

def run_all():
    print("=" * 60)
    print("RE-RUNNING TIMEOUT TESTS")
    print("Timeout: 300s per test")
    print("=" * 60)
    print()

    for name, prompt in TESTS.items():
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print("=" * 60)
        print(f"Prompt: {prompt[:100]}...")
        print()
        print("Running (this may take a while on 30B)...")
        print()

        try:
            result = run_with_tools(prompt)
            status = "COMPLETE"
            print("\n--- RESULT ---")
            print(result[:2000] if len(result) > 2000 else result)
        except Exception as e:
            result = f"ERROR: {str(e)}"
            status = "ERROR"
            print(f"\n--- ERROR ---\n{e}")

        save_result(name, prompt, result, status)
        print()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_all()
