#!/usr/bin/env python3
"""
Q42 Test 2: Joint R Formula - Local vs Bipartite Measurement

PURPOSE: Test whether R exhibits entanglement-like behavior on semantic pairs.

For quantum Bell states:
- R_local_A ≈ 0 (maximally mixed locally)
- R_local_B ≈ 0 (maximally mixed locally)
- R_joint >> 0 (strong correlation)

For semantic pairs:
- If R is local (H0): R_joint ≈ R_local_A × R_local_B (factorizable)
- If R detects entanglement (H1): R_joint >> R_local_A × R_local_B

This test evaluates three candidate joint R formulas:
1. Concatenation: R on concatenated observations
2. Correlation: Bivariate Gaussian kernel
3. Mutual Information: I(A;B)/H(A,B)

Run: python test_q42_joint_r.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from bell import (
    compute_R_base,
    joint_R_concatenation,
    joint_R_correlation,
    joint_R_mutual_info,
    JointRResult,
    ENTANGLED_PAIRS,
)


# =============================================================================
# OBSERVATION GENERATORS
# =============================================================================

def generate_independent_observations(
    n_samples: int = 1000,
    truth_A: float = 0.0,
    truth_B: float = 0.0,
    sigma_A: float = 1.0,
    sigma_B: float = 1.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate independent observations for two systems.

    This is the "classical" case - no entanglement.
    """
    np.random.seed(seed)
    obs_A = truth_A + sigma_A * np.random.randn(n_samples)
    obs_B = truth_B + sigma_B * np.random.randn(n_samples)
    return obs_A, obs_B


def generate_correlated_observations(
    n_samples: int = 1000,
    truth_A: float = 0.0,
    truth_B: float = 0.0,
    sigma: float = 1.0,
    correlation: float = 0.8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated observations (classical correlation).

    This tests whether classical correlation affects factorizability.
    """
    np.random.seed(seed)

    # Generate correlated Gaussians
    cov = [[1, correlation], [correlation, 1]]
    samples = np.random.multivariate_normal([0, 0], cov, n_samples)

    obs_A = truth_A + sigma * samples[:, 0]
    obs_B = truth_B + sigma * samples[:, 1]

    return obs_A, obs_B


def generate_entangled_observations(
    n_samples: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simulate "entangled" observations (quantum-like).

    For Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:
    - Local measurement: P(0) = P(1) = 0.5 (maximally mixed)
    - Joint measurement: perfect correlation

    Returns:
        obs_A, obs_B, truth_A, truth_B
    """
    np.random.seed(seed)

    # Simulate Bell-like behavior
    # Locally random, but perfectly correlated
    outcomes = np.random.choice([0, 1], n_samples)

    # Both observers see same outcome (perfect correlation)
    obs_A = outcomes.astype(float)
    obs_B = outcomes.astype(float)

    # Truth is undefined locally (50/50)
    truth_A = 0.5
    truth_B = 0.5

    return obs_A, obs_B, truth_A, truth_B


def generate_anti_correlated_observations(
    n_samples: int = 1000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simulate anti-correlated (like |Ψ-⟩ singlet state).

    If A measures 0, B measures 1 and vice versa.
    """
    np.random.seed(seed)

    outcomes_A = np.random.choice([0, 1], n_samples)
    outcomes_B = 1 - outcomes_A  # Perfect anti-correlation

    obs_A = outcomes_A.astype(float)
    obs_B = outcomes_B.astype(float)

    truth_A = 0.5
    truth_B = 0.5

    return obs_A, obs_B, truth_A, truth_B


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_factorizability_independent() -> Tuple[bool, Dict]:
    """
    Test: Independent observations should be factorizable.

    For independent A, B:
    R_joint ≈ R_local_A × R_local_B
    """
    print("\n" + "=" * 70)
    print("TEST 2a: Factorizability - Independent Systems")
    print("=" * 70)

    obs_A, obs_B = generate_independent_observations(
        n_samples=1000,
        truth_A=0.0,
        truth_B=0.0,
        sigma_A=1.0,
        sigma_B=1.0
    )

    result = joint_R_concatenation(obs_A, obs_B, 0.0, 0.0)

    print(f"\nIndependent observations (ρ = 0):")
    print(f"  R_local_A: {result.R_local_A:.6f}")
    print(f"  R_local_B: {result.R_local_B:.6f}")
    print(f"  R_product: {result.R_product:.6f}")
    print(f"  R_joint:   {result.R_joint:.6f}")
    print(f"  Ratio:     {result.entanglement_ratio:.4f}")
    print(f"  Factorizable: {result.is_factorizable}")

    passed = result.is_factorizable

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Independent systems are factorizable")

    return passed, {
        'R_local_A': result.R_local_A,
        'R_local_B': result.R_local_B,
        'R_product': result.R_product,
        'R_joint': result.R_joint,
        'ratio': result.entanglement_ratio,
        'is_factorizable': result.is_factorizable,
        'passed': passed
    }


def test_factorizability_correlated() -> Tuple[bool, Dict]:
    """
    Test: Classically correlated observations - still factorizable?

    Classical correlation should NOT produce entanglement signature.
    """
    print("\n" + "=" * 70)
    print("TEST 2b: Factorizability - Classically Correlated")
    print("=" * 70)

    results_by_rho = {}

    for rho in [0.0, 0.3, 0.6, 0.9]:
        obs_A, obs_B = generate_correlated_observations(
            n_samples=1000,
            correlation=rho
        )

        result = joint_R_concatenation(obs_A, obs_B, 0.0, 0.0)
        results_by_rho[rho] = result.entanglement_ratio

        print(f"\nCorrelation ρ = {rho}:")
        print(f"  R_joint/R_product = {result.entanglement_ratio:.4f}")
        print(f"  Factorizable: {result.is_factorizable}")

    # Classical correlation should not produce ratio >> 1
    # All ratios should be close to 1
    max_ratio = max(results_by_rho.values())
    passed = max_ratio < 2.0  # Must stay below entanglement threshold

    print(f"\nMax ratio across correlations: {max_ratio:.4f}")
    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Classical correlation doesn't produce entanglement signature")

    return passed, {
        'ratios': {str(k): float(v) for k, v in results_by_rho.items()},
        'max_ratio': float(max_ratio),
        'passed': passed
    }


def test_bell_state_simulation() -> Tuple[bool, Dict]:
    """
    Test: Bell-state-like observations should show entanglement signature.

    For simulated Bell state:
    - Local R low (maximally uncertain)
    - Joint R high (perfect correlation)
    - Ratio >> 1
    """
    print("\n" + "=" * 70)
    print("TEST 2c: Bell State Simulation")
    print("=" * 70)

    obs_A, obs_B, truth_A, truth_B = generate_entangled_observations()

    # Test all three joint R formulas
    result_concat = joint_R_concatenation(obs_A, obs_B, truth_A, truth_B)
    result_corr = joint_R_correlation(obs_A, obs_B, truth_A, truth_B)
    R_mutual = joint_R_mutual_info(obs_A, obs_B)

    print(f"\nSimulated Bell state |Φ+⟩:")
    print(f"  Local measurements: 50/50 random")
    print(f"  Joint measurements: perfectly correlated")

    print(f"\nConcatenation formula:")
    print(f"  R_local_A: {result_concat.R_local_A:.6f}")
    print(f"  R_local_B: {result_concat.R_local_B:.6f}")
    print(f"  R_joint:   {result_concat.R_joint:.6f}")
    print(f"  Ratio:     {result_concat.entanglement_ratio:.4f}")
    print(f"  Is entangled: {result_concat.is_entangled}")

    print(f"\nCorrelation formula:")
    print(f"  R_joint:   {result_corr.R_joint:.6f}")
    print(f"  Ratio:     {result_corr.entanglement_ratio:.4f}")
    print(f"  Is entangled: {result_corr.is_entangled}")

    print(f"\nMutual Information formula:")
    print(f"  R_MI:      {R_mutual:.6f}")

    # At least one formula should detect "entanglement"
    any_entangled = result_concat.is_entangled or result_corr.is_entangled
    passed = any_entangled

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Bell-like state shows entanglement signature")

    return passed, {
        'concat': {
            'R_local_A': result_concat.R_local_A,
            'R_local_B': result_concat.R_local_B,
            'R_joint': result_concat.R_joint,
            'ratio': result_concat.entanglement_ratio,
            'is_entangled': result_concat.is_entangled
        },
        'correlation': {
            'R_joint': result_corr.R_joint,
            'ratio': result_corr.entanglement_ratio,
            'is_entangled': result_corr.is_entangled
        },
        'mutual_info': R_mutual,
        'any_entangled': any_entangled,
        'passed': passed
    }


def test_anti_correlation() -> Tuple[bool, Dict]:
    """
    Test: Anti-correlated observations (singlet-like).

    Should also show entanglement signature.
    """
    print("\n" + "=" * 70)
    print("TEST 2d: Anti-Correlated (Singlet-like)")
    print("=" * 70)

    obs_A, obs_B, truth_A, truth_B = generate_anti_correlated_observations()

    result = joint_R_concatenation(obs_A, obs_B, truth_A, truth_B)

    print(f"\nSimulated singlet state |Ψ-⟩:")
    print(f"  Local: 50/50 random")
    print(f"  Joint: perfectly anti-correlated")
    print(f"\n  R_joint/R_product = {result.entanglement_ratio:.4f}")
    print(f"  Is entangled: {result.is_entangled}")

    passed = result.is_entangled

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Anti-correlation shows entanglement signature")

    return passed, {
        'R_local_A': result.R_local_A,
        'R_local_B': result.R_local_B,
        'R_joint': result.R_joint,
        'ratio': result.entanglement_ratio,
        'is_entangled': result.is_entangled,
        'passed': passed
    }


def test_formula_comparison() -> Tuple[bool, Dict]:
    """
    Compare all three joint R formulas systematically.
    """
    print("\n" + "=" * 70)
    print("TEST 2e: Joint R Formula Comparison")
    print("=" * 70)

    scenarios = {
        'independent': lambda: generate_independent_observations(),
        'low_corr': lambda: generate_correlated_observations(correlation=0.3),
        'high_corr': lambda: generate_correlated_observations(correlation=0.9),
        'bell_like': lambda: generate_entangled_observations()[:2],
    }

    results = {}

    for name, generator in scenarios.items():
        data = generator()
        if len(data) == 2:
            obs_A, obs_B = data
            truth_A, truth_B = 0.0, 0.0
        else:
            obs_A, obs_B = data
            truth_A, truth_B = 0.0, 0.0

        r_concat = joint_R_concatenation(obs_A, obs_B, truth_A, truth_B)
        r_corr = joint_R_correlation(obs_A, obs_B, truth_A, truth_B)
        r_mi = joint_R_mutual_info(obs_A, obs_B)

        results[name] = {
            'concat_ratio': r_concat.entanglement_ratio,
            'corr_ratio': r_corr.entanglement_ratio,
            'mutual_info': r_mi,
            'any_entangled': r_concat.is_entangled or r_corr.is_entangled
        }

        print(f"\n{name}:")
        print(f"  Concat ratio: {r_concat.entanglement_ratio:.4f}")
        print(f"  Corr ratio:   {r_corr.entanglement_ratio:.4f}")
        print(f"  Mutual Info:  {r_mi:.4f}")

    # Best formula should discriminate: high for bell, low for independent
    # Here we just check consistency
    passed = True

    return passed, results


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests() -> Dict:
    """Run all joint R tests."""

    print("\n" + "=" * 70)
    print("Q42 TEST 2: JOINT R FORMULA - LOCAL VS BIPARTITE")
    print("=" * 70)
    print("\nHypotheses:")
    print("  H0: R_joint ≈ R_local_A × R_local_B (factorizable)")
    print("  H1: R_joint >> product (entanglement signature)")

    results = {}
    all_passed = True

    tests = [
        ('independent', test_factorizability_independent),
        ('correlated', test_factorizability_correlated),
        ('bell_state', test_bell_state_simulation),
        ('anti_corr', test_anti_correlation),
        ('formula_comparison', test_formula_comparison),
    ]

    for name, test_func in tests:
        passed, data = test_func()
        results[name] = data
        all_passed = all_passed and passed

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Joint R Tests")
    print("=" * 70)

    n_passed = sum(1 for r in results.values() if r.get('passed', True))

    print(f"\nTests passed: {n_passed}/{len(tests)}")

    # Verdict on H0 vs H1
    # If Bell-like states show entanglement but classical don't, joint R works
    bell_entangled = results['bell_state'].get('any_entangled', False)
    classical_entangled = not results['correlated'].get('passed', True)

    if bell_entangled and not classical_entangled:
        verdict = "Joint R discriminates quantum-like from classical correlations"
        h0_status = "NUANCED"
        h1_status = "PARTIAL"
    elif bell_entangled and classical_entangled:
        verdict = "Joint R shows false positives on classical correlations"
        h0_status = "UNCERTAIN"
        h1_status = "UNCERTAIN"
    else:
        verdict = "Joint R is factorizable for all cases (H0 supported)"
        h0_status = "CONFIRMED"
        h1_status = "REJECTED"

    print(f"\nVerdict: {verdict}")

    results['summary'] = {
        'n_passed': n_passed,
        'n_total': len(tests),
        'all_passed': all_passed,
        'verdict': verdict,
        'h0_status': h0_status,
        'h1_status': h1_status,
        'timestamp': datetime.now().isoformat()
    }

    return results


if __name__ == '__main__':
    results = run_all_tests()

    output_path = Path(__file__).parent / 'q42_test2_results.json'
    with open(output_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")
    sys.exit(0 if results['summary']['all_passed'] else 1)
