#!/usr/bin/env python3
"""Test all 4 domains with visible responses."""

import sys
from pathlib import Path
import ollama
import re

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve_with_scores


def ask_model(query: str, context: list, model: str) -> str:
    """Ask model with optional context."""
    context_block = ""
    if context:
        context_block = "\n\n--- KNOWLEDGE BASE (use this to solve the problem) ---\n"
        for i, doc in enumerate(context, 1):
            preview = doc[:800] if len(doc) > 800 else doc
            context_block += f"\n[{i}] {preview}\n"
        context_block += "\n--- END KNOWLEDGE BASE ---\n\n"

    prompt = f"""{context_block}{query}

Give a clear, direct answer."""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 400}
    )
    # Handle unicode for Windows
    return result['response'].strip().encode('ascii', 'replace').decode('ascii')


# ============================================================================
# PROBLEMS - Now with harder code problem
# ============================================================================

PROBLEMS = {
    'math': {
        # Harder: requires knowing quadratic formula exactly
        'query': "Using the quadratic formula, solve 3x^2 + 14x - 37 = 0. Give both solutions as decimals rounded to 2 places.",
        'domain': 'math',
        'validator': lambda r: ('1.8' in r or '1.9' in r) and ('-6.5' in r or '-6.6' in r),
        'correct': "x = 1.88 and x = -6.55"
    },
    'code': {
        # Much harder: needs specific knowledge about maxsize parameter
        'query': """The @lru_cache decorator has a maxsize parameter. What is the DIFFERENCE between:
1. @lru_cache(maxsize=None)
2. @lru_cache(maxsize=128)

Which is better for fibonacci and WHY? Be specific about memory behavior.""",
        'domain': 'code',
        'validator': lambda r: 'unlimit' in r.lower() or 'unbounded' in r.lower() or 'no limit' in r.lower() or 'infinite' in r.lower(),
        'correct': "maxsize=None means unlimited cache (better for fib), maxsize=128 limits to 128 entries"
    },
    'logic': {
        # Keep original - it showed the gap before
        'query': "Knights tell truth, knaves lie. A says 'We are both knaves'. What are A and B?",
        'domain': 'logic',
        'validator': lambda r: bool(re.search(r'\ba\b.{0,10}knave', r.lower())) and bool(re.search(r'\bb\b.{0,10}knight', r.lower())),
        'correct': "A is knave, B is knight"
    },
    'chemistry': {
        # Even harder: requires specific stoichiometry calculation
        'query': "If 112g of iron (Fe, atomic mass=56) reacts completely with oxygen to form Fe2O3, how many grams of Fe2O3 are produced? (Fe2O3 molar mass = 160g/mol). Show just the final answer in grams.",
        'domain': 'chemistry',
        'validator': lambda r: '160' in r,
        'correct': "160g (2 mol Fe -> 1 mol Fe2O3 = 160g)"
    }
}


def test_domain(name: str, problem: dict):
    """Test a single domain."""
    print(f"\n{'='*70}")
    print(f"DOMAIN: {name.upper()}")
    print(f"{'='*70}")
    print(f"Query: {problem['query'][:100]}...")
    print(f"Expected: {problem['correct']}")

    # Get context
    context_results = retrieve_with_scores(problem['query'], k=3, threshold=0.25, domain=problem['domain'])
    context = [content for E, Df, content in context_results]

    print(f"\nContext: {len(context)} docs (E: {[f'{e:.2f}' for e,d,c in context_results]})")

    results = {}

    # Big without context
    print(f"\n[1] BIG (7b) - NO CONTEXT")
    resp = ask_model(problem['query'], [], "qwen2.5-coder:7b")
    passed = problem['validator'](resp)
    results['big_no'] = passed
    print(f"Response: {resp[:200]}...")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    # Tiny without context
    print(f"\n[2] TINY (3b) - NO CONTEXT")
    resp = ask_model(problem['query'], [], "qwen2.5-coder:3b")
    passed = problem['validator'](resp)
    results['tiny_no'] = passed
    print(f"Response: {resp[:200]}...")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    # Tiny WITH context
    print(f"\n[3] TINY (3b) - WITH CONTEXT (QUANTUM RESCUE)")
    resp = ask_model(problem['query'], context, "qwen2.5-coder:3b")
    passed = problem['validator'](resp)
    results['tiny_ctx'] = passed
    print(f"Response: {resp[:200]}...")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    # Check for rescue
    rescued = (not results['tiny_no']) and results['tiny_ctx']
    print(f"\n>>> QUANTUM RESCUE: {'YES!' if rescued else 'No'}")

    return results


if __name__ == "__main__":
    print("="*70)
    print("QUANTUM RESCUE TEST - ALL DOMAINS")
    print("="*70)

    all_results = {}
    for name, problem in PROBLEMS.items():
        all_results[name] = test_domain(name, problem)

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Domain':<12} {'Big(no)':<10} {'Tiny(no)':<10} {'Tiny(ctx)':<10} {'Rescued?':<10}")
    print("-"*52)

    rescued_count = 0
    for name, r in all_results.items():
        rescued = (not r['tiny_no']) and r['tiny_ctx']
        if rescued:
            rescued_count += 1
        print(f"{name:<12} {'PASS' if r['big_no'] else 'FAIL':<10} {'PASS' if r['tiny_no'] else 'FAIL':<10} {'PASS' if r['tiny_ctx'] else 'FAIL':<10} {'YES!' if rescued else 'No':<10}")

    print("-"*52)
    print(f"\nQuantum Rescue Success: {rescued_count}/4 domains")
