#!/usr/bin/env python3
"""
Test quantum rescue using ACTUAL geometric math from LIL_Q.

This test uses:
1. E = <psi|phi> (Born rule) for retrieval AND blending
2. blended_query = query + sum(context_i * E(query, context_i))
3. The full QuantumChat class

NOT just text concatenation!
"""

import sys
from pathlib import Path
import ollama
import numpy as np
import re

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_chat import QuantumChat
from retrieve import retrieve_with_scores


def make_llm_fn(model: str, context_as_text: list = None):
    """Create LLM function for QuantumChat."""
    def llm_fn(query: str, E: float, context: list) -> str:
        # Build prompt - context is passed separately via text for LLM
        context_block = ""
        if context_as_text:
            context_block = "\n\n--- KNOWLEDGE BASE ---\n"
            for i, doc in enumerate(context_as_text, 1):
                preview = doc[:600] if len(doc) > 600 else doc
                context_block += f"\n[{i}] {preview}\n"
            context_block += "\n--- END KNOWLEDGE BASE ---\n\n"

        prompt = f"""{context_block}[Resonance E={E:.3f}]

{query}

Give a clear, direct answer."""

        result = ollama.generate(
            model=model,
            prompt=prompt,
            options={'temperature': 0.0, 'num_predict': 400}
        )
        return result['response'].strip().encode('ascii', 'replace').decode('ascii')

    return llm_fn


PROBLEMS = {
    'math': {
        'query': "Using the quadratic formula, solve 3x^2 + 14x - 37 = 0. Give both solutions as decimals.",
        'domain': 'math',
        'validator': lambda r: ('1.8' in r or '1.9' in r) and ('-6.5' in r or '-6.6' in r),
        'correct': "x = 1.88 and x = -6.55"
    },
    'code': {
        'query': "What is the DIFFERENCE between @lru_cache(maxsize=None) and @lru_cache(maxsize=128)? Which is better for fibonacci?",
        'domain': 'code',
        'validator': lambda r: 'unlimit' in r.lower() or 'unbounded' in r.lower() or 'no limit' in r.lower(),
        'correct': "maxsize=None means unlimited cache"
    },
    'logic': {
        'query': "Knights tell truth, knaves lie. A says 'We are both knaves'. What are A and B?",
        'domain': 'logic',
        'validator': lambda r: bool(re.search(r'\ba\b.{0,10}knave', r.lower())) and bool(re.search(r'\bb\b.{0,10}knight', r.lower())),
        'correct': "A is knave, B is knight"
    },
    'chemistry': {
        'query': "If 112g of Fe (atomic mass=56) reacts to form Fe2O3, how many grams of Fe2O3? (molar mass=160)",
        'domain': 'chemistry',
        'validator': lambda r: '160' in r,
        'correct': "160g"
    }
}


def test_with_quantum_geometry():
    """Test using ACTUAL quantum geometry from QuantumChat."""

    print("="*70)
    print("QUANTUM GEOMETRIC TEST")
    print("Using: E = <psi|phi>, blended = query + sum(c * E(q,c)), FFT memory")
    print("="*70)

    results = {}

    for name, problem in PROBLEMS.items():
        print(f"\n{'='*70}")
        print(f"DOMAIN: {name.upper()}")
        print(f"{'='*70}")
        print(f"Query: {problem['query'][:80]}...")

        # Get context
        context_results = retrieve_with_scores(problem['query'], k=3, threshold=0.25, domain=problem['domain'])
        context_text = [content for E, Df, content in context_results]

        print(f"Context: {len(context_text)} docs (E: {[f'{e:.2f}' for e,d,c in context_results]})")

        domain_results = {}

        # =====================================================
        # TEST 1: Big model (7b) without context, no quantum
        # =====================================================
        print(f"\n[1] BIG (7b) - NO CONTEXT, NO QUANTUM")
        chat = QuantumChat(make_llm_fn("qwen2.5-coder:7b", None))
        response, E_val = chat.chat(problem['query'], None)
        passed = problem['validator'](response)
        domain_results['big_no_ctx'] = passed
        print(f"E={E_val:.3f}, Response: {response[:150]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # =====================================================
        # TEST 2: Tiny model (3b) without context, no quantum
        # =====================================================
        print(f"\n[2] TINY (3b) - NO CONTEXT, NO QUANTUM")
        chat = QuantumChat(make_llm_fn("qwen2.5-coder:3b", None))
        response, E_val = chat.chat(problem['query'], None)
        passed = problem['validator'](response)
        domain_results['tiny_no_ctx'] = passed
        print(f"E={E_val:.3f}, Response: {response[:150]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # =====================================================
        # TEST 3: Tiny model (3b) WITH context + QUANTUM GEOMETRY
        # =====================================================
        print(f"\n[3] TINY (3b) - WITH CONTEXT + QUANTUM GEOMETRY")
        # Pass context both as text (for LLM prompt) AND as vectors (for quantum blending)
        chat = QuantumChat(make_llm_fn("qwen2.5-coder:3b", context_text))
        response, E_val = chat.chat(problem['query'], context_text)  # context_text is embedded and blended!
        passed = problem['validator'](response)
        domain_results['tiny_quantum'] = passed
        print(f"E={E_val:.3f}, Response: {response[:150]}...")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")

        # Check for rescue
        rescued = (not domain_results['tiny_no_ctx']) and domain_results['tiny_quantum']
        print(f"\n>>> QUANTUM RESCUE: {'YES!' if rescued else 'No'}")

        results[name] = domain_results

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - QUANTUM GEOMETRIC TEST")
    print("="*70)
    print(f"{'Domain':<12} {'Big(no)':<10} {'Tiny(no)':<12} {'Tiny+Quantum':<12} {'Rescued?':<10}")
    print("-"*56)

    rescued_count = 0
    for name, r in results.items():
        rescued = (not r['tiny_no_ctx']) and r['tiny_quantum']
        if rescued:
            rescued_count += 1
        print(f"{name:<12} {'PASS' if r['big_no_ctx'] else 'FAIL':<10} {'PASS' if r['tiny_no_ctx'] else 'FAIL':<12} {'PASS' if r['tiny_quantum'] else 'FAIL':<12} {'YES!' if rescued else 'No':<10}")

    print("-"*56)
    print(f"\nQuantum Geometric Rescue: {rescued_count}/4 domains")

    if rescued_count >= 3:
        print("\n*** QUANTUM GEOMETRY HYPOTHESIS VALIDATED! ***")


if __name__ == "__main__":
    test_with_quantum_geometry()
