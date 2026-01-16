#!/usr/bin/env python3
"""
TASK BENCHMARKS - Progressive difficulty testing for tool-augmented LLMs.

Levels:
  1. WARMUP     - Single tool, obvious answer
  2. EASY       - Single tool, requires reasoning
  3. MEDIUM     - Multi-tool, sequential reasoning
  4. HARD       - Multi-tool, parallel + synthesis
  5. EXPERT     - Complex real-world scenarios
  6. WTF        - Import from wtf_tests.py

Usage:
  python task_benchmarks.py --level easy
  python task_benchmarks.py --level all
  python task_benchmarks.py --single medium-3
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from tool_executor_v2 import run_with_tools

# =============================================================================
# LEVEL 1: WARMUP - Single tool, obvious answer
# =============================================================================

WARMUP_TASKS = [
    # Python basic
    {
        "id": "warmup-1",
        "name": "Basic Arithmetic",
        "prompt": "What is 7 * 8? Use Python to compute this.",
        "expected": "56",
        "tool": "python"
    },
    # Python string
    {
        "id": "warmup-2",
        "name": "String Reverse",
        "prompt": "Reverse the string 'hello world' using Python.",
        "expected": "dlrow olleh",
        "tool": "python"
    },
    # Wikipedia lookup
    {
        "id": "warmup-3",
        "name": "Wikipedia Lookup",
        "prompt": "What year was the Eiffel Tower completed? Look it up on Wikipedia.",
        "expected": "1889",
        "tool": "wikipedia"
    },
    # Web search
    {
        "id": "warmup-4",
        "name": "Simple Search",
        "prompt": "Who is the current CEO of Microsoft? Search the web.",
        "expected": "Satya Nadella",
        "tool": "web_search"
    },
]

# =============================================================================
# LEVEL 2: EASY - Single tool, requires reasoning
# =============================================================================

EASY_TASKS = [
    # Python with logic
    {
        "id": "easy-1",
        "name": "Prime Check",
        "prompt": "Is 97 a prime number? Write Python code to check.",
        "expected": "True",
        "tool": "python"
    },
    # Python with math
    {
        "id": "easy-2",
        "name": "Factorial",
        "prompt": "What is 15 factorial? Compute it in Python and express in scientific notation.",
        "expected": "1307674368000",
        "tool": "python"
    },
    # URL fetch with inference
    {
        "id": "easy-3",
        "name": "URL Fetch Inference",
        "prompt": "Fetch https://en.wikipedia.org/wiki/Albert_Einstein and determine his age when he published the special theory of relativity.",
        "expected": "26",
        "tool": "fetch_url"
    },
    # API fetch with extraction
    {
        "id": "easy-4",
        "name": "API Fetch + Extract",
        "prompt": "Fetch the Bitcoin price from https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd and extract the current USD price.",
        "expected": "numeric",  # Dynamic, just check for number
        "tool": "fetch_url"
    },
]

# =============================================================================
# LEVEL 3: MEDIUM - Multi-tool, sequential reasoning
# =============================================================================

MEDIUM_TASKS = [
    # Grok -> Python
    {
        "id": "medium-1",
        "name": "Lookup + Calculate",
        "prompt": """Use grok to look up the speed of light (in m/s).
Then calculate how many seconds it takes light to travel from the Sun to Earth (distance: 149.6 million km).
Show your calculation in Python.""",
        "expected": "~499 seconds",
        "tools": ["grok", "python"]
    },
    # Fetch -> Python
    {
        "id": "medium-2",
        "name": "Fetch + Compute",
        "prompt": """Fetch https://www.worldometers.info/world-population/ to get current world population estimate.
Then calculate what percentage of the world lives in India (population ~1.4 billion).
Use Python for the calculation.""",
        "expected": "~17-18%",
        "tools": ["fetch_url", "python"]
    },
    # Python -> Verify -> Python
    {
        "id": "medium-3",
        "name": "Compute + Verify",
        "prompt": """Compute the first 20 Fibonacci numbers in Python.
Then compute the ratio of consecutive numbers (F(n+1)/F(n)) for the last 5 pairs.
What value does this ratio approach?""",
        "expected": "golden ratio, ~1.618",
        "tools": ["python"]
    },
    # Multi-fetch synthesis
    {
        "id": "medium-4",
        "name": "Multi-Source Synthesis",
        "prompt": """Fetch https://en.wikipedia.org/wiki/Mount_Everest and extract its height.
Fetch https://en.wikipedia.org/wiki/Mariana_Trench and extract its depth.
What is the total vertical distance between the highest and lowest points on Earth?
Use Python if needed for calculation.""",
        "expected": "~19,833 meters",
        "tools": ["fetch_url", "python"]
    },
]

# =============================================================================
# LEVEL 4: HARD - Multi-tool, parallel + synthesis
# =============================================================================

HARD_TASKS = [
    # Multi-source fact checking
    {
        "id": "hard-1",
        "name": "Fact Verification",
        "prompt": """Claim: "The Great Wall of China is visible from space."
Fetch https://en.wikipedia.org/wiki/Great_Wall_of_China to verify this claim.
Provide evidence for or against, citing the sources.""",
        "expected": "FALSE (not visible with naked eye from space)",
        "tools": ["fetch_url"]
    },
    # Complex calculation chain
    {
        "id": "hard-2",
        "name": "Compound Interest Calculation",
        "prompt": """Assume the US Federal Reserve interest rate is 4.5%.
If you invested $10,000 at this rate compounded monthly:
1. How much would you have after 5 years?
2. How long to double your money?
Use Python for calculations. Show formulas used.""",
        "expected": "calculation with 4.5% rate",
        "tools": ["python"]
    },
    # Data analysis task
    {
        "id": "hard-3",
        "name": "Statistical Analysis",
        "prompt": """Generate 1000 random samples from a normal distribution with mean=100, std=15.
Calculate: mean, median, std, skewness, kurtosis.
Then perform a Shapiro-Wilk test to verify normality.
Report p-value and conclusion.""",
        "expected": "p > 0.05, normally distributed",
        "tools": ["python"]
    },
    # Historical research
    {
        "id": "hard-4",
        "name": "Historical Research",
        "prompt": """Research the Space Race:
1. Fetch https://en.wikipedia.org/wiki/Sputnik_1 - when did it launch?
2. Fetch https://en.wikipedia.org/wiki/Apollo_11 - when did it land on the moon?
3. Calculate the time between these events in days
4. List at least 3 major milestones between these dates.""",
        "expected": "dates + calculation",
        "tools": ["fetch_url", "python"]
    },
]

# =============================================================================
# LEVEL 5: EXPERT - Complex real-world scenarios
# =============================================================================

EXPERT_TASKS = [
    # Scientific computation
    {
        "id": "expert-1",
        "name": "Orbital Mechanics",
        "prompt": """Calculate the orbital period of a satellite at altitude 400km above Earth.
Use:
- Earth's radius: 6,371 km
- Earth's mass: 5.972 x 10^24 kg
- G = 6.674 x 10^-11 N*m^2/kg^2

Apply Kepler's third law. Express answer in minutes.
Then look up the actual orbital period of the ISS to verify.""",
        "expected": "~92 minutes",
        "tools": ["python", "wikipedia"]
    },
    # Economic modeling
    {
        "id": "expert-2",
        "name": "Economic Analysis",
        "prompt": """Look up the current GDP of USA, China, and Japan.
Calculate:
1. Combined GDP as percentage of world GDP
2. If China's GDP grows at 5% and USA at 2%, when will China's GDP exceed USA's?
3. Plot a 20-year projection (just describe the trend, no actual plot needed)""",
        "expected": "analysis with projections",
        "tools": ["web_search", "python"]
    },
    # Cryptographic challenge
    {
        "id": "expert-3",
        "name": "Cryptography Basics",
        "prompt": """Implement RSA encryption from scratch:
1. Generate two small primes p=61, q=53
2. Compute n = p*q and phi(n) = (p-1)(q-1)
3. Choose e=17, verify gcd(e, phi(n)) = 1
4. Compute d such that e*d mod phi(n) = 1
5. Encrypt the message m=42
6. Decrypt to verify you get 42 back

Show all calculations in Python.""",
        "expected": "working RSA with m=42 recovered",
        "tools": ["python"]
    },
    # Multi-step research
    {
        "id": "expert-4",
        "name": "Climate Data Analysis",
        "prompt": """Research climate change:
1. What was the global average temperature anomaly in 1900? (search)
2. What is the current temperature anomaly? (search)
3. Calculate the rate of change per decade
4. If this rate continues, project the anomaly for 2050
5. What is the Paris Agreement target? How close/far are we?""",
        "expected": "data + projection + comparison",
        "tools": ["web_search", "python"]
    },
]

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.join(scripts_dir, "..", "nemotron-3-nano-30b-outputs", "task-benchmarks")


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


# =============================================================================
# TEST RUNNER
# =============================================================================

ALL_LEVELS = {
    "warmup": ("WARMUP", WARMUP_TASKS),
    "easy": ("EASY", EASY_TASKS),
    "medium": ("MEDIUM", MEDIUM_TASKS),
    "hard": ("HARD", HARD_TASKS),
    "expert": ("EXPERT", EXPERT_TASKS),
}


def run_task(task: dict) -> dict:
    """Run a single task and return result."""
    task_id = task["id"]
    name = task["name"]
    prompt = task["prompt"]
    expected = task.get("expected", "N/A")

    print(f"\n{'='*70}")
    print(f"TASK: {task_id} - {name}")
    print(f"{'='*70}")
    print(f"PROMPT: {prompt[:300]}..." if len(prompt) > 300 else f"PROMPT: {prompt}")
    print(f"EXPECTED: {expected}")
    print(f"{'='*70}")

    try:
        result = run_with_tools(prompt.strip())
        status = "completed"
        print(f"\nRESULT:\n{result[:1000]}..." if len(result) > 1000 else f"\nRESULT:\n{result}")
        save_result(task_id, prompt, result, status, expected)
        return {"task_id": task_id, "status": status, "result": result}

    except KeyboardInterrupt:
        print("\n[INTERRUPTED BY USER]")
        save_result(task_id, prompt, "INTERRUPTED", "interrupted", expected)
        return {"task_id": task_id, "status": "interrupted", "result": "INTERRUPTED"}

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        save_result(task_id, prompt, str(e), "error", expected)
        return {"task_id": task_id, "status": "error", "result": str(e)}


def run_level(level_key: str) -> list:
    """Run all tasks at a given level."""
    if level_key not in ALL_LEVELS:
        print(f"Unknown level: {level_key}")
        return []

    level_name, tasks = ALL_LEVELS[level_key]

    print(f"\n{'#'*70}")
    print(f"# LEVEL: {level_name}")
    print(f"# Tasks: {len(tasks)}")
    print(f"{'#'*70}")

    results = []
    for task in tasks:
        result = run_task(task)
        results.append(result)
        if result["status"] == "interrupted":
            break

    return results


def run_single(task_id: str) -> dict:
    """Run a single task by ID (e.g., 'easy-2')."""
    # Find the task
    for level_key, (_, tasks) in ALL_LEVELS.items():
        for task in tasks:
            if task["id"] == task_id:
                return run_task(task)

    print(f"Task not found: {task_id}")
    return {"task_id": task_id, "status": "not_found", "result": None}


def run_all() -> dict:
    """Run all levels in order."""
    all_results = {}
    for level_key in ["warmup", "easy", "medium", "hard", "expert"]:
        results = run_level(level_key)
        all_results[level_key] = results

        # Check for interrupt
        if results and results[-1]["status"] == "interrupted":
            break

    return all_results


def print_summary(results: dict):
    """Print summary of results."""
    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}")

    total = 0
    completed = 0
    errors = 0

    for level, level_results in results.items():
        level_completed = sum(1 for r in level_results if r["status"] == "completed")
        level_total = len(level_results)
        print(f"  {level.upper():10s}: {level_completed}/{level_total} completed")
        total += level_total
        completed += level_completed
        errors += sum(1 for r in level_results if r["status"] == "error")

    print(f"{'='*70}")
    print(f"  TOTAL:      {completed}/{total} completed, {errors} errors")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive task benchmarks")
    parser.add_argument("--level", "-l",
                        choices=["warmup", "easy", "medium", "hard", "expert", "all"],
                        default="warmup",
                        help="Difficulty level to run")
    parser.add_argument("--single", "-s", type=str,
                        help="Run single task by ID (e.g., 'easy-2')")
    parser.add_argument("--list", action="store_true",
                        help="List all available tasks")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable Tasks:")
        print("="*70)
        for level_key, (level_name, tasks) in ALL_LEVELS.items():
            print(f"\n{level_name}:")
            for task in tasks:
                print(f"  {task['id']:15s} - {task['name']}")
        sys.exit(0)

    print("""
    ===================================================================
                    TASK BENCHMARK SUITE

      Progressive difficulty testing for tool-augmented LLMs
      Press Ctrl+C to abort any task.
    ===================================================================
    """)

    if args.single:
        result = run_single(args.single)
        print(f"\nTask {args.single}: {result['status']}")
    elif args.level == "all":
        results = run_all()
        print_summary(results)
    else:
        results = run_level(args.level)
        print_summary({args.level: results})
