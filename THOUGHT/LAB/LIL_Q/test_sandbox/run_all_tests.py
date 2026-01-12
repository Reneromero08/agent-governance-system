#!/usr/bin/env python3
"""
Quantum Entanglement Test: All 4 Problems

Tests if tiny model can solve problems it normally can't when given
context via quantum entanglement across 4 domains:
1. Math - Algebraic equation
2. Code - Debugging recursive function
3. Logic - Knights and knaves puzzle
4. Chemistry - Balancing equation

Hypothesis: Context vectors blended via E-weighting on the quantum manifold
can enable tiny models to solve problems beyond their capability.
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_chat import QuantumChat
from test_sandbox.retrieve import retrieve_with_scores


# =============================================================================
# Test Problems
# =============================================================================

PROBLEMS = {
    'math': {
        'query': "Using the quadratic formula, solve 3x^2 + 14x - 37 = 0. Give both solutions as decimals rounded to 2 places.",
        'domain': 'math',
        'correct_answers': [1.88, -6.55],
        'tolerance': 0.5,
        'keywords': ['quadratic', 'formula', '-b'],
        'validator': lambda x: abs(3*x**2 + 14*x - 37) < 1
    },
    'code': {
        'query': 'The @lru_cache decorator has a maxsize parameter. What is the DIFFERENCE between @lru_cache(maxsize=None) and @lru_cache(maxsize=128)? Which is better for fibonacci and WHY?',
        'domain': 'code',
        'correct_answers': ['unlimited', 'unbounded', 'no limit', 'infinite'],
        'keywords': ['unlimited', 'unbounded', 'cache', 'memory'],
        'validator': None
    },
    'logic': {
        'query': "Knights tell truth, knaves lie. A says 'We are both knaves'. What are A and B?",
        'domain': 'logic',
        'correct_answers': [('knave', 'knight'), ('A is knave', 'B is knight')],
        'keywords': ['knave', 'knight', 'contradiction', 'false'],
        'validator': None
    },
    'chemistry': {
        'query': "If 112g of iron (Fe, atomic mass=56) reacts completely with oxygen to form Fe2O3, how many grams of Fe2O3 are produced? (Fe2O3 molar mass = 160g/mol)",
        'domain': 'chemistry',
        'correct_answers': ['160'],
        'keywords': ['mole', 'ratio', '160', 'stoichiometry'],
        'validator': None
    }
}


# =============================================================================
# LLM Generation Functions
# =============================================================================

def llm_generate_big(query: str, E: float, context: list = None) -> str:
    """Generate with big model (qwen2.5-coder:7b)."""
    try:
        import ollama

        system = "You are a helpful assistant. Show your reasoning step by step."

        context_block = ""
        if context:
            context_block = "\n\n--- REFERENCE MATERIAL ---\n"
            for i, doc in enumerate(context, 1):
                preview = doc[:600] if len(doc) > 600 else doc
                context_block += f"\n{preview}\n"
            context_block += "--- END REFERENCE ---\n"

        prompt = f"{context_block}\n\n{query}"

        result = ollama.generate(
            model='qwen2.5-coder:7b',
            prompt=prompt,
            system=system,
            options={'temperature': 0.1}
        )
        return result['response'].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def llm_generate_tiny(query: str, E: float, context: list = None) -> str:
    """Generate with tiny model (qwen2.5-coder:3b)."""
    try:
        import ollama

        system = "You are a helpful assistant. Show your reasoning step by step."

        context_block = ""
        if context:
            context_block = "\n\n--- REFERENCE MATERIAL ---\n"
            for i, doc in enumerate(context, 1):
                preview = doc[:600] if len(doc) > 600 else doc
                context_block += f"\n{preview}\n"
            context_block += "--- END REFERENCE ---\n"

        prompt = f"{context_block}\n\n{query}"

        result = ollama.generate(
            model='qwen2.5-coder:3b',
            prompt=prompt,
            system=system,
            options={'temperature': 0.1}
        )
        return result['response'].strip()
    except Exception as e:
        return f"[ERROR] {e}"


# =============================================================================
# Validation Functions
# =============================================================================

def validate_math(response: str, problem: dict) -> dict:
    """Validate math problem solution."""
    # Simple string check for expected values
    has_positive = bool(re.search(r'1\.8[0-9]?|1\.9[0-9]?', response))
    has_negative = bool(re.search(r'-6\.5[0-9]?|-6\.6[0-9]?', response))

    correct = has_positive and has_negative

    shows_work = any(kw in response.lower() for kw in problem['keywords'])

    return {
        'extracted': ['1.88, -6.55'] if correct else [],
        'correct': correct,
        'shows_work': shows_work,
        'pass': correct
    }


def validate_code(response: str, problem: dict) -> dict:
    """Validate code debugging solution."""
    response_lower = response.lower()

    # Check if any correct solution mentioned
    correct = any(ans.lower() in response_lower for ans in problem['correct_answers'])

    # Check if key concepts mentioned
    shows_work = any(kw in response_lower for kw in problem['keywords'])

    return {
        'extracted': [ans for ans in problem['correct_answers'] if ans.lower() in response_lower],
        'correct': correct,
        'shows_work': shows_work,
        'pass': correct
    }


def validate_logic(response: str, problem: dict) -> dict:
    """Validate logic puzzle solution."""
    response_lower = response.lower()

    # Check if correct answer pattern found
    correct = False
    if 'a' in response_lower and 'knave' in response_lower:
        if 'b' in response_lower and 'knight' in response_lower:
            # Check they're associated correctly
            # Look for "A is knave" or "A is a knave" and "B is knight"
            a_knave = bool(re.search(r'a\s+is\s+(?:a\s+)?knave', response_lower))
            b_knight = bool(re.search(r'b\s+is\s+(?:a\s+)?knight', response_lower))
            correct = a_knave and b_knight

    shows_work = any(kw in response_lower for kw in problem['keywords'])

    return {
        'extracted': ['A=knave, B=knight'] if correct else [],
        'correct': correct,
        'shows_work': shows_work,
        'pass': correct
    }


def validate_chemistry(response: str, problem: dict) -> dict:
    """Validate chemistry stoichiometry solution."""
    # Check for correct answer (160 grams)
    correct = '160' in response

    shows_work = any(kw.lower() in response.lower() for kw in problem['keywords'])

    return {
        'extracted': ['160g'] if correct else [],
        'correct': correct,
        'shows_work': shows_work,
        'pass': correct
    }


VALIDATORS = {
    'math': validate_math,
    'code': validate_code,
    'logic': validate_logic,
    'chemistry': validate_chemistry
}


# =============================================================================
# Test Runner
# =============================================================================

def run_single_problem(problem_name: str, problem: dict, verbose: bool = False) -> Dict:
    """Run test for a single problem."""
    print("\n" + "=" * 70)
    print(f"PROBLEM: {problem_name.upper()}")
    print("=" * 70)
    query_preview = problem['query'][:100].encode('ascii', 'replace').decode('ascii')
    print(f"\nQuery: {query_preview}{'...' if len(problem['query']) > 100 else ''}")

    results = {}

    # === CONDITION A: No Context ===
    print("\n" + "-" * 70)
    print("CONDITION A: NO CONTEXT")
    print("-" * 70)

    chat_big = QuantumChat(llm_generate_big)
    chat_tiny = QuantumChat(llm_generate_tiny)

    print("\n[1/4] Big Model (7B) - No Context...")
    response_big_nocontext, E_big_nocontext = chat_big.chat(problem['query'], context=None)
    score_big_nocontext = VALIDATORS[problem_name](response_big_nocontext, problem)

    if verbose:
        print(f"\nResponse:\n{response_big_nocontext[:300]}...\n")
    print(f"Result: {'PASS' if score_big_nocontext['pass'] else 'FAIL'}")
    print(f"E: {E_big_nocontext:.3f}")

    results['big_nocontext'] = {
        'response': response_big_nocontext,
        'E': E_big_nocontext,
        'score': score_big_nocontext
    }

    print("\n[2/4] Tiny Model (3B) - No Context...")
    response_tiny_nocontext, E_tiny_nocontext = chat_tiny.chat(problem['query'], context=None)
    score_tiny_nocontext = VALIDATORS[problem_name](response_tiny_nocontext, problem)

    if verbose:
        print(f"\nResponse:\n{response_tiny_nocontext[:300]}...\n")
    print(f"Result: {'PASS' if score_tiny_nocontext['pass'] else 'FAIL'}")
    print(f"E: {E_tiny_nocontext:.3f}")

    results['tiny_nocontext'] = {
        'response': response_tiny_nocontext,
        'E': E_tiny_nocontext,
        'score': score_tiny_nocontext
    }

    # === CONDITION B: With Context ===
    print("\n" + "-" * 70)
    print("CONDITION B: WITH CONTEXT (Quantum Rescue Attempt)")
    print("-" * 70)

    print(f"\nRetrieving context from domain '{problem['domain']}'...")
    context_with_scores = retrieve_with_scores(
        problem['query'],
        k=3,
        threshold=0.3,
        domain=problem['domain']
    )

    if not context_with_scores:
        print("[ERROR] No context retrieved!")
        return results

    print(f"\nRetrieved {len(context_with_scores)} documents:")
    for i, (E, Df, content) in enumerate(context_with_scores, 1):
        preview = content[:60].replace('\n', ' ')
        print(f"  [{i}] E={E:.3f}, Df={Df:.1f} | {preview}...")

    context = [content for E, Df, content in context_with_scores]

    # Reset minds for fair comparison
    chat_big_context = QuantumChat(llm_generate_big)
    chat_tiny_context = QuantumChat(llm_generate_tiny)

    print("\n[3/4] Big Model (7B) - With Context...")
    response_big_context, E_big_context = chat_big_context.chat(problem['query'], context=context)
    score_big_context = VALIDATORS[problem_name](response_big_context, problem)

    if verbose:
        print(f"\nResponse:\n{response_big_context[:300]}...\n")
    print(f"Result: {'PASS' if score_big_context['pass'] else 'FAIL'}")
    print(f"E: {E_big_context:.3f}")

    results['big_context'] = {
        'response': response_big_context,
        'E': E_big_context,
        'score': score_big_context
    }

    print("\n[4/4] Tiny Model (3B) - With Context...")
    response_tiny_context, E_tiny_context = chat_tiny_context.chat(problem['query'], context=context)
    score_tiny_context = VALIDATORS[problem_name](response_tiny_context, problem)

    if verbose:
        print(f"\nResponse:\n{response_tiny_context[:300]}...\n")
    print(f"Result: {'PASS' if score_tiny_context['pass'] else 'FAIL'}")
    print(f"E: {E_tiny_context:.3f}")

    results['tiny_context'] = {
        'response': response_tiny_context,
        'E': E_tiny_context,
        'score': score_tiny_context
    }

    return results


def run_all_tests(verbose: bool = False):
    """Run all 4 test problems."""
    print("=" * 70)
    print("QUANTUM ENTANGLEMENT TEST: All 4 Domains")
    print("=" * 70)
    print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nHypothesis: Context via quantum entanglement enables tiny models")
    print("to solve problems beyond their capability.")

    all_results = {}

    for problem_name, problem in PROBLEMS.items():
        results = run_single_problem(problem_name, problem, verbose=verbose)
        all_results[problem_name] = results

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n| Problem | Big (No Ctx) | Tiny (No Ctx) | Big (Ctx) | Tiny (Ctx) | Rescued? |")
    print("|---------|--------------|---------------|-----------|------------|----------|")

    hypothesis_confirmed = []

    for problem_name in PROBLEMS.keys():
        results = all_results[problem_name]

        big_no = "PASS" if results['big_nocontext']['score']['pass'] else "FAIL"
        tiny_no = "PASS" if results['tiny_nocontext']['score']['pass'] else "FAIL"
        big_yes = "PASS" if results['big_context']['score']['pass'] else "FAIL"
        tiny_yes = "PASS" if results['tiny_context']['score']['pass'] else "FAIL"

        # Check if hypothesis met for this problem
        rescued = (
            results['big_nocontext']['score']['pass'] and
            not results['tiny_nocontext']['score']['pass'] and
            results['tiny_context']['score']['pass']
        )

        hypothesis_confirmed.append(rescued)

        rescued_mark = "YES" if rescued else "No"

        print(f"| {problem_name:8s} | {big_no:12s} | {tiny_no:13s} | {big_yes:9s} | {tiny_yes:10s} | {rescued_mark:8s} |")

    # === HYPOTHESIS VALIDATION ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)

    total = len(hypothesis_confirmed)
    passed = sum(hypothesis_confirmed)

    print(f"\nProblems where quantum rescue worked: {passed}/{total}")

    if passed == total:
        print("\n*** HYPOTHESIS FULLY CONFIRMED ***")
        print("Quantum entanglement with context enabled tiny model across ALL domains!")
    elif passed >= total // 2:
        print("\n** HYPOTHESIS PARTIALLY CONFIRMED **")
        print(f"Quantum rescue worked in {passed}/{total} domains.")
    else:
        print("\n* HYPOTHESIS NOT MET *")
        print(f"Quantum rescue only worked in {passed}/{total} domains.")

    print("\n" + "=" * 70)
    print(f"Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run quantum entanglement tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show full responses')
    parser.add_argument('--problem', '-p', choices=['math', 'code', 'logic', 'chemistry'],
                       help='Run only one problem')
    args = parser.parse_args()

    if args.problem:
        # Run single problem
        results = run_single_problem(args.problem, PROBLEMS[args.problem], verbose=args.verbose)
    else:
        # Run all problems
        run_all_tests(verbose=args.verbose)
