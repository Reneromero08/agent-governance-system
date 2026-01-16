#!/usr/bin/env python3
"""
WTF-TIER TESTS - Absolute nightmare scenarios for reasoning models.

These tests are designed to BREAK things. They combine:
- Deep multi-hop reasoning chains
- Contradictory information handling
- Self-referential paradoxes
- Extreme mathematical edge cases
- Adversarial prompt patterns
- Resource exhaustion scenarios
"""

import sys
import os

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from tool_executor_v2 import run_with_tools

# =============================================================================
# CATEGORY 1: MATHEMATICAL NIGHTMARES
# =============================================================================

MATH_NIGHTMARES = [
    # RSA-style factorization with verification loop
    """Factor the semiprime 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139 into its two prime factors. Then verify by multiplying them back. Show all work.""",

    # Collatz with proof attempt
    """Starting from n=27, compute the Collatz sequence until it reaches 1. Count total steps. Then mathematically explain WHY it must reach 1 (or prove you cannot prove it).""",

    # Nested modular arithmetic hell
    """Compute: ((7^(7^7)) mod 13) mod 5. Then verify your answer using Fermat's Little Theorem. Show the chain of reasoning.""",

    # Precision destruction
    """Compute the 1000th digit of pi. Then compute the 1000th digit of e. Then compute their sum mod 10. Verify using independent methods.""",
]

# =============================================================================
# CATEGORY 2: LOGIC PARADOXES & SELF-REFERENCE
# =============================================================================

PARADOX_TESTS = [
    # Liar's paradox variant
    """Consider the statement: "This statement cannot be verified by any tool you have access to." Analyze whether this is true, false, or undecidable. Use formal logic.""",

    # Godel-style incompleteness
    """Construct a mathematical statement that is TRUE but UNPROVABLE within standard arithmetic (Peano axioms). Then explain why your construction works.""",

    # Self-referential tool use
    """Write Python code that analyzes its own source code to determine if it will halt. Then run that code and report what happens.""",

    # Bootstrap paradox
    """I will tell you the answer after you compute it. The answer is 42. Now derive what question I was asking. Then verify your derivation leads to 42.""",
]

# =============================================================================
# CATEGORY 3: ADVERSARIAL MULTI-HOP
# =============================================================================

ADVERSARIAL_MULTIHOP = [
    # Contradictory sources
    """Look up the population of Tokyo on Wikipedia. Now look up the population of Tokyo on Grokipedia. If they differ, determine which is correct and why. Then calculate how many years until Tokyo's population doubles at 0.5% annual growth.""",

    # Broken chain recovery
    """Step 1: Get the atomic number of Gold from Wikipedia. Step 2: Multiply by the number of moons of Jupiter (search for current count). Step 3: Divide by the year the Eiffel Tower was built. Step 4: If any step fails, derive the answer from first principles instead.""",

    # Temporal reasoning trap
    """What was the population of the United States exactly 100 years before the current date? Show your reasoning for determining 'current date' and cite your sources.""",

    # Recursive verification
    """Claim: The sum of the first n primes is always less than n^2 * ln(n) for n > 5. Verify this for n=100 by computing both sides. Then determine if this bound is tight or can be improved.""",
]

# =============================================================================
# CATEGORY 4: EXTREME EDGE CASES
# =============================================================================

EDGE_CASES = [
    # Division by approaching zero
    """Compute lim(x->0) of sin(x)/x using numerical approximation with x = 10^-15. Then explain why your numerical answer might be wrong and compute the true limit analytically.""",

    # Floating point hell
    """Compute (0.1 + 0.2) == 0.3 in Python. Explain the result. Then compute the smallest positive float x such that 1.0 + x != 1.0. Verify your answer.""",

    # Combinatorial explosion
    """How many distinct ways can you partition the integer 100 into sums of positive integers? Compute exactly, not approximately.""",

    # Prime gap nightmare
    """Find the smallest prime gap greater than 100. That is, find consecutive primes p and q where q - p > 100. Verify both p and q are prime.""",
]

# =============================================================================
# CATEGORY 5: META-REASONING ATTACKS
# =============================================================================

META_ATTACKS = [
    # Tool selection paradox
    """I need the answer to 2+2. But first, search the web for 'what is 2+2', then check Wikipedia for 'arithmetic', then compute it in Python, then verify with Grokipedia. Report ALL answers and explain any discrepancies.""",

    # Infinite regress
    """Verify that your verification process is correct. Then verify that verification. Continue until you reach a base axiom you cannot verify. What is it?""",

    # Resource awareness
    """Estimate how many tokens this conversation has used so far. Then estimate how many more tokens you can use before context limits. Use this to plan your response length.""",

    # Capability probing
    """List every tool you have access to. For each tool, provide one example where it would be the WRONG tool to use. Then solve: what is the integral of e^(-x^2) from -infinity to infinity?""",
]

# =============================================================================
# CATEGORY 6: REAL-WORLD CHAOS
# =============================================================================

REAL_WORLD_CHAOS = [
    # Multi-source synthesis
    """Compare and contrast the economic policies of the current leaders of USA, China, and Germany. Use Wikipedia for each leader's biography, then synthesize a 3-sentence summary of how their policies differ on trade.""",

    # Broken data recovery
    """The file 'nonexistent_data.csv' contains critical information. Since it doesn't exist, infer what data it SHOULD contain based on the directory structure of this project, then create a plausible reconstruction.""",

    # Ambiguous instruction
    """Make it better.""",

    # Implicit knowledge test
    """Without using any tools, explain the relationship between the Riemann Hypothesis and the distribution of prime numbers. Then use tools to verify at least one specific claim you made.""",
]

# =============================================================================
# THE FINAL BOSS: COMBINED NIGHTMARE
# =============================================================================

FINAL_BOSS = [
    """
    ULTIMATE CHALLENGE:

    1. Look up the speed of light in m/s from Wikipedia
    2. Look up Planck's constant from Grokipedia
    3. Compute the Schwarzschild radius of a black hole with mass = (speed of light / Planck's constant) kg
    4. Express this radius in both meters and light-years
    5. Determine if this black hole would be larger or smaller than the observable universe
    6. If any source fails, derive the constants from first principles using dimensional analysis
    7. Verify your final answer using an independent calculation method
    8. Rate your confidence in each step from 0-100%

    Show ALL work. Explain ALL reasoning. Cite ALL sources.
    """,
]

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "nemotron-3-nano-30b-outputs", "wtf-tests")

def save_result(category: str, test_num: int, prompt: str, result: str, status: str):
    """Save test result to output directory."""
    import json
    from datetime import datetime

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"wtf-{category}-{test_num:02d}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    data = {
        "test_suite": "wtf_tests",
        "category": category,
        "test_num": test_num,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "prompt": prompt.strip(),
        "result": result
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Saved to {filename}]")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_wtf_tests(categories=None):
    """Run WTF-tier tests. Warning: These may break things."""

    all_categories = {
        "math": ("MATHEMATICAL NIGHTMARES", MATH_NIGHTMARES),
        "paradox": ("LOGIC PARADOXES", PARADOX_TESTS),
        "adversarial": ("ADVERSARIAL MULTI-HOP", ADVERSARIAL_MULTIHOP),
        "edge": ("EXTREME EDGE CASES", EDGE_CASES),
        "meta": ("META-REASONING ATTACKS", META_ATTACKS),
        "chaos": ("REAL-WORLD CHAOS", REAL_WORLD_CHAOS),
        "boss": ("FINAL BOSS", FINAL_BOSS),
    }

    if categories is None:
        categories = list(all_categories.keys())

    results = []

    for cat_key in categories:
        if cat_key not in all_categories:
            print(f"Unknown category: {cat_key}")
            continue

        cat_name, tests = all_categories[cat_key]

        print(f"\n{'#'*70}")
        print(f"# {cat_name}")
        print(f"{'#'*70}")

        for i, test in enumerate(tests, 1):
            print(f"\n{'='*70}")
            print(f"TEST {cat_key.upper()}-{i}")
            print(f"{'='*70}")
            print(f"PROMPT: {test[:200]}..." if len(test) > 200 else f"PROMPT: {test}")
            print(f"{'='*70}")

            try:
                result = run_with_tools(test.strip())
                results.append({
                    "category": cat_key,
                    "test_num": i,
                    "prompt": test,
                    "result": result,
                    "status": "completed"
                })
                print(f"\nRESULT:\n{result}")

                # Save to output directory
                save_result(cat_key, i, test, result, "completed")

            except KeyboardInterrupt:
                print("\n[INTERRUPTED BY USER]")
                results.append({
                    "category": cat_key,
                    "test_num": i,
                    "prompt": test,
                    "result": "INTERRUPTED",
                    "status": "interrupted"
                })
                save_result(cat_key, i, test, "INTERRUPTED", "interrupted")
                break
            except Exception as e:
                print(f"\n[ERROR]: {e}")
                results.append({
                    "category": cat_key,
                    "test_num": i,
                    "prompt": test,
                    "result": str(e),
                    "status": "error"
                })
                save_result(cat_key, i, test, str(e), "error")

            print(f"\n{'='*70}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WTF-tier reasoning tests")
    parser.add_argument("--category", "-c", choices=["math", "paradox", "adversarial", "edge", "meta", "chaos", "boss", "all"],
                       default="all", help="Test category to run")
    parser.add_argument("--single", "-s", type=int, help="Run single test number from category")

    args = parser.parse_args()

    if args.category == "all":
        cats = None  # Run all
    else:
        cats = [args.category]

    print("""
    ===================================================================
                        WTF-TIER REASONING TESTS

      WARNING: These tests are designed to break things.
      Press Ctrl+C to abort any test.
    ===================================================================
    """)

    results = run_wtf_tests(categories=cats)

    # Summary
    print(f"\n{'#'*70}")
    print("# SUMMARY")
    print(f"{'#'*70}")

    completed = sum(1 for r in results if r["status"] == "completed")
    errors = sum(1 for r in results if r["status"] == "error")
    interrupted = sum(1 for r in results if r["status"] == "interrupted")

    print(f"Completed: {completed}")
    print(f"Errors: {errors}")
    print(f"Interrupted: {interrupted}")
    print(f"Total: {len(results)}")
