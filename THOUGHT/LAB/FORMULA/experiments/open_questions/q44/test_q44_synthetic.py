"""
Q44: Synthetic Validation Test
==============================

Test whether R = (E / grad_S) * sigma^Df correlates with
the quantum Born rule P(psi->phi) = |<psi|phi>|^2.

This test uses synthetic embeddings for reproducibility.
No external dependencies required (just numpy).

Run: python test_q44_synthetic.py
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Local imports
from q44_core import (
    embed_synthetic,
    compute_R,
    compute_born_probability,
    compute_R_simple,
    compute_E_squared_R,
    normalize,
    pearson_correlation,
    normalize_values,
    BornValidationResult,
)
from q44_test_cases import get_all_test_cases
from q44_statistics import (
    full_correlation_analysis,
    analyze_by_category,
    check_monotonicity,
    CorrelationResult,
)


def run_single_test(test_case: Dict[str, Any], dim: int = 384) -> Dict[str, Any]:
    """
    Run a single test case comparing R to Born rule probability.

    Args:
        test_case: Dict with 'query', 'context', 'id', 'category'
        dim: Embedding dimension

    Returns:
        Result dict with R, P_born, and all components
    """
    query = test_case['query']
    context = test_case['context']

    # Generate embeddings (deterministic based on text)
    query_vec = embed_synthetic(query, dim=dim)
    context_vecs = [embed_synthetic(c, dim=dim) for c in context]

    # Compute full R
    result = compute_R(query_vec, context_vecs)

    # Also compute simplified variants
    R_simple, P_born_simple = compute_R_simple(query_vec, context_vecs)
    R_E2, P_born_E2 = compute_E_squared_R(query_vec, context_vecs)

    return {
        'id': test_case['id'],
        'query': query,
        'context': context,
        'category': test_case.get('category', 'UNKNOWN'),
        # Main results
        'R': result.R,
        'P_born': result.P_born,
        # Components
        'E': result.E,
        'grad_S': result.grad_S,
        'sigma': result.sigma,
        'Df': result.Df,
        'overlaps': result.overlaps,
        # Variants
        'R_simple': R_simple,
        'R_E2': R_E2,
    }


def run_all_tests(dim: int = 384) -> List[Dict[str, Any]]:
    """Run all 100 test cases."""
    test_cases = get_all_test_cases()
    results = []

    for test_case in test_cases:
        result = run_single_test(test_case, dim=dim)
        results.append(result)

    return results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze correlation between R and P_born across all results.
    """
    R_values = [r['R'] for r in results]
    P_values = [r['P_born'] for r in results]

    # Also analyze variants
    R_simple = [r['R_simple'] for r in results]
    R_E2 = [r['R_E2'] for r in results]

    # Full analysis for main R
    main_analysis = full_correlation_analysis(R_values, P_values)

    # Quick correlation for variants
    r_simple = pearson_correlation(np.array(R_simple), np.array(P_values))
    r_E2 = pearson_correlation(np.array(R_E2), np.array(P_values))

    # By category analysis
    category_analysis = analyze_by_category(results)

    # Monotonicity check
    monotonicity = check_monotonicity(R_values, P_values)

    return {
        'main': {
            'r': main_analysis.r,
            'r_normalized': main_analysis.r_normalized,
            'mae': main_analysis.mae,
            'ci_low': main_analysis.ci_low,
            'ci_high': main_analysis.ci_high,
            'p_value': main_analysis.p_value,
            'verdict': main_analysis.verdict,
        },
        'variants': {
            'R_simple_correlation': r_simple,
            'R_E2_correlation': r_E2,
        },
        'by_category': category_analysis,
        'monotonicity': monotonicity,
    }


def create_receipt(
    results: List[Dict[str, Any]],
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a receipt with hash for reproducibility."""

    # Hash the results
    results_json = json.dumps(results, sort_keys=True, default=str)
    results_hash = hashlib.sha256(results_json.encode()).hexdigest()

    receipt = {
        'document': 'Q44_QUANTUM_BORN_RULE_VALIDATION',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'question': 'Q44',
        'title': 'Does R compute quantum Born rule probability?',

        'summary': {
            'num_test_cases': len(results),
            'correlation_r': analysis['main']['r'],
            'correlation_r_normalized': analysis['main']['r_normalized'],
            'mean_absolute_error': analysis['main']['mae'],
            'confidence_interval_95': [
                analysis['main']['ci_low'],
                analysis['main']['ci_high']
            ],
            'p_value': analysis['main']['p_value'],
            'verdict': analysis['main']['verdict'],
        },

        'variant_analysis': analysis['variants'],
        'category_analysis': analysis['by_category'],
        'monotonicity': analysis['monotonicity'],

        'criteria': {
            'quantum_threshold': 0.9,
            'adjustment_threshold': 0.7,
            'significance_threshold': 0.01,
        },

        'interpretation': {
            'quantum': 'r > 0.9: R IS quantum projection probability',
            'needs_adjustment': '0.7 < r < 0.9: R is quantum, formula needs correction',
            'not_quantum': 'r < 0.7: R is quantum-inspired but not quantum-equivalent',
        },

        'results_hash': results_hash,
    }

    return receipt


def print_results(analysis: Dict[str, Any]):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("Q44: QUANTUM BORN RULE VALIDATION RESULTS")
    print("=" * 70)

    main = analysis['main']
    print(f"\n--- MAIN CORRELATION ANALYSIS ---")
    print(f"Pearson correlation (r):        {main['r']:.4f}")
    print(f"Normalized correlation:         {main['r_normalized']:.4f}")
    print(f"Mean Absolute Error:            {main['mae']:.4f}")
    print(f"95% Confidence Interval:        [{main['ci_low']:.4f}, {main['ci_high']:.4f}]")
    print(f"Permutation p-value:            {main['p_value']:.6f}")

    print(f"\n--- VERDICT ---")
    verdict = main['verdict']
    if verdict == "QUANTUM":
        print(f">>> {verdict}: R computes quantum Born rule! <<<")
    elif verdict == "NEEDS_ADJUSTMENT":
        print(f">>> {verdict}: R is quantum but needs formula correction <<<")
    else:
        print(f">>> {verdict}: R is quantum-inspired but not exact quantum <<<")

    print(f"\n--- VARIANT CORRELATIONS ---")
    variants = analysis['variants']
    print(f"R_simple (E/grad_S):            {variants['R_simple_correlation']:.4f}")
    print(f"R_E2 (E²/grad_S):               {variants['R_E2_correlation']:.4f}")

    print(f"\n--- BY CATEGORY ---")
    for cat, stats in analysis['by_category'].items():
        print(f"  {cat:10} (n={stats['n']:3}): r={stats['r']:.4f}, "
              f"R_mean={stats['R_mean']:.4f}, P_mean={stats['P_mean']:.4f}")

    print(f"\n--- MONOTONICITY CHECK ---")
    mono = analysis['monotonicity']
    print(f"Spearman rank correlation:      {mono['spearman_rho']:.4f}")
    print(f"Low R quartile mean P_born:     {mono['low_R_mean_P']:.4f}")
    print(f"High R quartile mean P_born:    {mono['high_R_mean_P']:.4f}")
    print(f"Monotonic relationship:         {mono['monotonic']}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    print("Q44: Quantum Born Rule Validation")
    print("Testing R = (E / grad_S) * sigma^Df vs P = |<psi|phi>|²")
    print("-" * 50)

    # Run all tests
    print("\nRunning 100 test cases...")
    results = run_all_tests(dim=384)
    print(f"Completed {len(results)} test cases.")

    # Analyze
    print("\nAnalyzing correlations...")
    analysis = analyze_results(results)

    # Print results
    print_results(analysis)

    # Create and save receipt
    receipt = create_receipt(results, analysis)

    output_dir = Path(__file__).parent
    results_path = output_dir / 'q44_results.json'
    receipt_path = output_dir / 'q44_receipt.json'

    # Save detailed results
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'analysis': analysis,
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {results_path}")

    # Save receipt
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)
    print(f"Receipt saved to: {receipt_path}")

    # Return verdict
    return analysis['main']['verdict']


if __name__ == "__main__":
    verdict = main()
    print(f"\n>>> FINAL VERDICT: {verdict} <<<")
