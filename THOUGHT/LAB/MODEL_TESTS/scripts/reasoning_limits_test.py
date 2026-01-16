#!/usr/bin/env python3
"""
Reasoning Limits Test Suite

Goal: Find where the model breaks down, not on computation, but on REASONING:
- Multi-hop inference
- Tool selection
- Error recovery
- Ambiguity resolution
- Verification loops

These tests require practical tools (web, files) to be interesting.
"""

# Category 1: Multi-Hop Reasoning
MULTI_HOP = [
    # Requires: search -> extract -> compute -> verify
    "What is the population density of the country with the largest GDP? Use current data.",

    # Requires: wiki -> compute -> search verification
    "According to Wikipedia, what is the half-life of Carbon-14? Now compute how much remains after 10,000 years starting with 1 kg.",

    # Requires: multiple searches, comparison, computation
    "Which is more energy-dense: gasoline or lithium-ion batteries? Find MJ/kg for each, then compute how much battery mass equals 1 gallon of gas.",
]

# Category 2: Tool Selection (Right Tool for the Job)
TOOL_SELECTION = [
    # Should use computation, NOT search
    "What is the 1000th prime number?",

    # Should use search, NOT computation
    "What is the current price of Bitcoin?",

    # Should use wiki first, then compute
    "Using Planck's constant from Wikipedia, compute the energy of a photon with wavelength 500 nm.",

    # Should recognize this needs file access
    "How many Python files are in the current directory?",

    # Trick: looks like computation but needs current data
    "How old is the oldest living person right now?",
]

# Category 3: Error Recovery
ERROR_RECOVERY = [
    # First approach will timeout/fail, needs fallback
    "Factor 2^256 - 189. If sympy is too slow, try a different approach.",

    # URL might be dead, needs error handling
    "Fetch https://example.com/nonexistent and tell me what it says. If that fails, explain why.",

    # Ambiguous - should ask for clarification or state assumptions
    "What is the square root of 4?",

    # Will get no search results, needs to adapt
    "Search for 'ajklsdjfklajsdlfkj' and tell me what you find.",
]

# Category 4: Verification Loops
VERIFICATION = [
    # Should compute, then verify via Wikipedia
    "Compute the speed of light in m/s from first principles (c = 1/sqrt(epsilon_0 * mu_0)). Then verify against Wikipedia.",

    # Should search multiple sources and compare
    "What year did World War 2 end? Search for this and verify the answer is consistent.",

    # Should compute, then double-check the math
    "What is 123456789 * 987654321? Compute it, then verify by division.",
]

# Category 5: Ambiguity & Traps
AMBIGUITY = [
    # Needs to state assumptions
    "How long does it take to fly to Paris?",

    # Classic trap - 1 is NOT prime
    "List all prime numbers less than 10.",

    # Needs clarification: mean/median/mode?
    "What is the average height of humans?",

    # Multiple valid answers depending on context
    "What is 0^0?",

    # Needs to recognize impossibility
    "Factor -1 into prime numbers.",
]

# Category 6: Complex Real-World Tasks
REAL_WORLD = [
    # Multi-step with file I/O
    "List all Python files in this directory, then count total lines of code across all of them.",

    # Research task
    "What is the Riemann Hypothesis? Look it up on Wikipedia, summarize it in 2 sentences, then tell me if it's been proven.",

    # Current events + computation
    "What is the current world population? What percentage lives in urban areas? If urbanization continues at current rate, when will it hit 75%?",

    # Comparison task
    "Compare the Wikipedia entries for 'entropy' in physics vs information theory. What is the key difference?",
]

# Category 7: Meta-Reasoning
META = [
    # Should recognize it needs to break this down
    "I need to write a Python script that downloads stock data, computes moving averages, and plots them. What tools would you need that you don't currently have?",

    # Should recognize capability limits
    "Can you browse to reddit.com and tell me the top post today?",

    # Should ask clarifying questions
    "Help me optimize my code.",

    # Should recognize this is outside its scope
    "Send an email to john@example.com telling him I'll be late.",
]


def run_category(category_name: str, tests: list):
    """Run a category of tests."""
    print(f"\n{'='*70}")
    print(f"CATEGORY: {category_name}")
    print('='*70)

    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test}")

    print(f"\nTotal tests in category: {len(tests)}")


def main():
    print("REASONING LIMITS TEST SUITE")
    print("="*70)
    print("\nThese tests explore:")
    print("- Multi-hop inference")
    print("- Tool selection")
    print("- Error recovery")
    print("- Verification loops")
    print("- Ambiguity resolution")
    print("- Meta-reasoning")
    print("\nFocus: Find reasoning limits, not computational limits.")

    run_category("Multi-Hop Reasoning", MULTI_HOP)
    run_category("Tool Selection", TOOL_SELECTION)
    run_category("Error Recovery", ERROR_RECOVERY)
    run_category("Verification Loops", VERIFICATION)
    run_category("Ambiguity & Traps", AMBIGUITY)
    run_category("Complex Real-World Tasks", REAL_WORLD)
    run_category("Meta-Reasoning", META)

    print("\n" + "="*70)
    print(f"TOTAL TESTS: {len(MULTI_HOP) + len(TOOL_SELECTION) + len(ERROR_RECOVERY) + len(VERIFICATION) + len(AMBIGUITY) + len(REAL_WORLD) + len(META)}")
    print("="*70)

    print("\nTo run individual tests:")
    print("  python tool_executor_v2.py \"<test prompt>\"")
    print("\nTo run all tests (requires dependencies):")
    print("  pip install duckduckgo-search beautifulsoup4 wikipedia-api")


if __name__ == "__main__":
    main()
