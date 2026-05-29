#!/usr/bin/env python3
"""
Test with E-WEIGHTED context delivery.

The quantum formula: E = <psi|phi> (Born rule)

Not just filtering by E threshold, but WEIGHTING context importance by E value.
High E docs appear first with explicit weight labels.
"""

import sys
from pathlib import Path
import ollama
import re

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieve import retrieve_with_scores


def ask_with_e_weighting(query: str, context_with_scores: list, model: str) -> str:
    """
    Ask model with E-WEIGHTED context.

    Context is presented in order of E (highest first) with explicit weights.
    This mirrors the quantum formula: influence = doc * E(query, doc)
    """
    context_block = ""
    if context_with_scores:
        context_block = "\n\n--- E-WEIGHTED KNOWLEDGE (higher E = more relevant) ---\n"
        for i, (E, Df, content) in enumerate(context_with_scores, 1):
            weight_label = "HIGH" if E > 0.6 else "MEDIUM" if E > 0.4 else "LOW"
            preview = content[:600] if len(content) > 600 else content
            # E-weighted emphasis in prompt
            context_block += f"\n[E={E:.3f} - {weight_label} RELEVANCE]\n{preview}\n"
        context_block += "\n--- END E-WEIGHTED KNOWLEDGE ---\n\n"
        context_block += "Use the HIGH relevance context most heavily.\n\n"

    prompt = f"""{context_block}{query}

Give a clear, direct answer."""

    result = ollama.generate(
        model=model,
        prompt=prompt,
        options={'temperature': 0.0, 'num_predict': 400}
    )
    return result['response'].strip().encode('ascii', 'replace').decode('ascii')


PROBLEMS = {
    'math': {
        'query': "Using the quadratic formula, solve 3x^2 + 14x - 37 = 0. Give both solutions as decimals.",
        'domain': 'math',
        'validator': lambda r: ('1.8' in r or '1.9' in r) and ('-6.5' in r or '-6.6' in r),
        'correct': "x = 1.88 and x = -6.55"
    },
    'code': {
        'query': "What is the DIFFERENCE between @lru_cache(maxsize=None) and @lru_cache(maxsize=128)? Be specific about memory.",
        'domain': 'code',
        'validator': lambda r: 'unlimit' in r.lower() or 'unbounded' in r.lower() or 'no limit' in r.lower() or 'infinite' in r.lower(),
        'correct': "maxsize=None means unlimited cache"
    },
    'logic': {
        'query': "Knights tell truth, knaves lie. A says 'We are both knaves'. What are A and B?",
        'domain': 'logic',
        'validator': lambda r: bool(re.search(r'\ba\b.{0,10}knave', r.lower())) and bool(re.search(r'\bb\b.{0,10}knight', r.lower())),
        'correct': "A is knave, B is knight"
    },
    'chemistry': {
        'query': "If 112g of Fe (atomic mass=56) reacts to form Fe2O3, how many grams of Fe2O3 are produced? (molar mass=160g/mol)",
        'domain': 'chemistry',
        'validator': lambda r: '160' in r,
        'correct': "160g"
    }
}


def test_domain(name: str, problem: dict):
    """Test a single domain with E-weighting."""
    print(f"\n{'='*70}")
    print(f"DOMAIN: {name.upper()}")
    print(f"{'='*70}")
    print(f"Query: {problem['query'][:80]}...")

    # Get context WITH E SCORES
    context_results = retrieve_with_scores(problem['query'], k=3, threshold=0.25, domain=problem['domain'])

    print(f"E-gated context (E = <psi|phi>):")
    for i, (E, Df, content) in enumerate(context_results, 1):
        preview = content[:60].replace('\n', ' ')
        print(f"  [{i}] E={E:.3f}, Df={Df:.1f}: {preview}...")

    results = {}

    # Big without context
    print(f"\n[1] BIG (7b) - NO CONTEXT")
    resp = ask_with_e_weighting(problem['query'], [], "qwen2.5-coder:7b")
    passed = problem['validator'](resp)
    results['big_no'] = passed
    print(f"Response: {resp[:150]}...")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    # Tiny without context
    print(f"\n[2] TINY (3b) - NO CONTEXT")
    resp = ask_with_e_weighting(problem['query'], [], "qwen2.5-coder:3b")
    passed = problem['validator'](resp)
    results['tiny_no'] = passed
    print(f"Response: {resp[:150]}...")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    # Tiny WITH E-WEIGHTED context
    print(f"\n[3] TINY (3b) - WITH E-WEIGHTED CONTEXT")
    resp = ask_with_e_weighting(problem['query'], context_results, "qwen2.5-coder:3b")
    passed = problem['validator'](resp)
    results['tiny_e_weighted'] = passed
    print(f"Response: {resp[:150]}...")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    # Check for rescue
    rescued = (not results['tiny_no']) and results['tiny_e_weighted']
    print(f"\n>>> E-WEIGHTED QUANTUM RESCUE: {'YES!' if rescued else 'No'}")

    return results


if __name__ == "__main__":
    print("="*70)
    print("E-WEIGHTED QUANTUM TEST")
    print("Formula: E = <psi|phi> (Born rule)")
    print("Context weighted by E values, high E = more influence")
    print("="*70)

    all_results = {}
    for name, problem in PROBLEMS.items():
        all_results[name] = test_domain(name, problem)

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - E-WEIGHTED QUANTUM TEST")
    print("="*70)
    print(f"{'Domain':<12} {'Big(no)':<10} {'Tiny(no)':<10} {'Tiny+E-wt':<12} {'Rescued?':<10}")
    print("-"*54)

    rescued_count = 0
    for name, r in all_results.items():
        rescued = (not r['tiny_no']) and r['tiny_e_weighted']
        if rescued:
            rescued_count += 1
        print(f"{name:<12} {'PASS' if r['big_no'] else 'FAIL':<10} {'PASS' if r['tiny_no'] else 'FAIL':<10} {'PASS' if r['tiny_e_weighted'] else 'FAIL':<12} {'YES!' if rescued else 'No':<10}")

    print("-"*54)
    print(f"\nE-Weighted Quantum Rescue: {rescued_count}/4 domains")

    if rescued_count >= 3:
        print("\n*** E = <psi|phi> HYPOTHESIS VALIDATED! ***")
