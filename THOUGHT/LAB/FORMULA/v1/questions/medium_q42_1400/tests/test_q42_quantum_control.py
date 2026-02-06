#!/usr/bin/env python3
"""
Q42 Test 0: Quantum Control - Apparatus Validation

PURPOSE: Prove the CHSH machinery works before applying to semantics.

This test validates:
1. Quantum Bell state gives S = 2√2 ≈ 2.828 (Tsirelson bound)
2. Classical hidden variable gives S ≤ 2 (Bell's theorem)
3. The difference is statistically significant

If this test fails, the CHSH apparatus is broken and semantic tests are meaningless.

Run: python test_q42_quantum_control.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# Add path for local imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    QUTIP_AVAILABLE = False
    print("Warning: QuTiP not available, using analytical quantum mechanics")

from bell import (
    CLASSICAL_BOUND,
    QUANTUM_BOUND,
    compute_chsh,
    optimal_chsh_angles,
    quantum_correlation,
    simulate_quantum_chsh,
    simulate_classical_chsh,
    bootstrap_chsh_confidence,
    is_violation_significant,
)


# =============================================================================
# QUANTUM TESTS (using QuTiP if available)
# =============================================================================

def create_bell_state():
    """
    Create Bell state |Phi+> = (|00> + |11>)/√2 using QuTiP.
    """
    if not QUTIP_AVAILABLE:
        return None

    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)

    # |Phi+> = (|00> + |11>)/sqrt(2)
    bell_state = (qt.tensor(zero, zero) + qt.tensor(one, one)).unit()

    return bell_state


def measure_correlation_qutip(
    state,
    theta_A: float,
    theta_B: float,
    n_samples: int = 10000
) -> float:
    """
    Measure correlation between two qubits using QuTiP.

    For state |psi>, measure σ(θ_A) on qubit A and σ(θ_B) on qubit B.

    Args:
        state: QuTiP quantum state
        theta_A: Measurement angle for Alice
        theta_B: Measurement angle for Bob
        n_samples: Number of measurement samples

    Returns:
        Correlation coefficient
    """
    if not QUTIP_AVAILABLE:
        # Fall back to analytical
        return quantum_correlation(theta_A, theta_B)

    # Create measurement operators
    # σ(θ) = cos(θ)σ_z + sin(θ)σ_x
    sigma_A = np.cos(theta_A) * qt.sigmaz() + np.sin(theta_A) * qt.sigmax()
    sigma_B = np.cos(theta_B) * qt.sigmaz() + np.sin(theta_B) * qt.sigmax()

    # Joint measurement operator
    joint_op = qt.tensor(sigma_A, sigma_B)

    # Expectation value
    rho = state * state.dag()
    E = qt.expect(joint_op, rho)

    return E


def test_quantum_chsh_qutip() -> Tuple[bool, Dict]:
    """
    Test CHSH on quantum Bell state using QuTiP.

    Expected: S = 2√2 ≈ 2.828
    """
    print("\n" + "=" * 70)
    print("TEST 0a: Quantum CHSH (QuTiP)")
    print("=" * 70)

    if QUTIP_AVAILABLE:
        bell_state = create_bell_state()
        a, a_prime, b, b_prime = optimal_chsh_angles()

        E_ab = measure_correlation_qutip(bell_state, a, b)
        E_ab_prime = measure_correlation_qutip(bell_state, a, b_prime)
        E_a_prime_b = measure_correlation_qutip(bell_state, a_prime, b)
        E_a_prime_b_prime = measure_correlation_qutip(bell_state, a_prime, b_prime)

        result = compute_chsh(E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)
    else:
        result = simulate_quantum_chsh()

    print(f"\nBell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"Measurement angles: a=0°, a'=90°, b=45°, b'=135°")
    print(f"\nCorrelations:")
    print(f"  E(a,b)    = {result.E_ab:.6f}")
    print(f"  E(a,b')   = {result.E_ab_prime:.6f}")
    print(f"  E(a',b)   = {result.E_a_prime_b:.6f}")
    print(f"  E(a',b')  = {result.E_a_prime_b_prime:.6f}")
    print(f"\nCHSH statistic: S = {result.S:.6f}")
    print(f"Expected (Tsirelson bound): 2√2 = {QUANTUM_BOUND:.6f}")
    print(f"Deviation: {abs(result.S - QUANTUM_BOUND):.6f}")

    passed = abs(result.S - QUANTUM_BOUND) < 0.01

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Quantum control")

    return passed, {
        'S': result.S,
        'expected': QUANTUM_BOUND,
        'deviation': abs(result.S - QUANTUM_BOUND),
        'E_ab': result.E_ab,
        'E_ab_prime': result.E_ab_prime,
        'E_a_prime_b': result.E_a_prime_b,
        'E_a_prime_b_prime': result.E_a_prime_b_prime,
        'passed': passed
    }


def test_classical_chsh() -> Tuple[bool, Dict]:
    """
    Test CHSH on classical hidden variable model.

    Expected: S ≤ 2 (Bell's theorem)
    """
    print("\n" + "=" * 70)
    print("TEST 0b: Classical CHSH (Hidden Variable)")
    print("=" * 70)

    result = simulate_classical_chsh()

    print(f"\nClassical strategy: deterministic hidden variable λ")
    print(f"A(θ,λ) = sign(cos(θ-λ)), B(θ,λ) = sign(cos(θ-λ))")
    print(f"\nCorrelations:")
    print(f"  E(a,b)    = {result.E_ab:.6f}")
    print(f"  E(a,b')   = {result.E_ab_prime:.6f}")
    print(f"  E(a',b)   = {result.E_a_prime_b:.6f}")
    print(f"  E(a',b')  = {result.E_a_prime_b_prime:.6f}")
    print(f"\nCHSH statistic: S = {result.S:.6f}")
    print(f"Classical bound: {CLASSICAL_BOUND:.6f}")

    passed = result.S <= CLASSICAL_BOUND

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Classical control")

    return passed, {
        'S': result.S,
        'bound': CLASSICAL_BOUND,
        'E_ab': result.E_ab,
        'E_ab_prime': result.E_ab_prime,
        'E_a_prime_b': result.E_a_prime_b,
        'E_a_prime_b_prime': result.E_a_prime_b_prime,
        'passed': passed
    }


def test_analytical_quantum() -> Tuple[bool, Dict]:
    """
    Test analytical quantum correlation formula.

    For |Φ+⟩: E(a,b) = -cos(a-b)
    """
    print("\n" + "=" * 70)
    print("TEST 0c: Analytical Quantum Formula")
    print("=" * 70)

    a, a_prime, b, b_prime = optimal_chsh_angles()

    # Analytical predictions
    E_ab = quantum_correlation(a, b)
    E_ab_prime = quantum_correlation(a, b_prime)
    E_a_prime_b = quantum_correlation(a_prime, b)
    E_a_prime_b_prime = quantum_correlation(a_prime, b_prime)

    print(f"\nFormula: E(θ_A, θ_B) = -cos(θ_A - θ_B)")
    print(f"\nAngles (radians):")
    print(f"  a = {a:.4f} (0°)")
    print(f"  a' = {a_prime:.4f} (90°)")
    print(f"  b = {b:.4f} (45°)")
    print(f"  b' = {b_prime:.4f} (135°)")

    print(f"\nCorrelations:")
    print(f"  E(a,b)    = -cos({a-b:.4f}) = {E_ab:.6f}")
    print(f"  E(a,b')   = -cos({a-b_prime:.4f}) = {E_ab_prime:.6f}")
    print(f"  E(a',b)   = -cos({a_prime-b:.4f}) = {E_a_prime_b:.6f}")
    print(f"  E(a',b')  = -cos({a_prime-b_prime:.4f}) = {E_a_prime_b_prime:.6f}")

    result = compute_chsh(E_ab, E_ab_prime, E_a_prime_b, E_a_prime_b_prime)

    print(f"\nS = |{E_ab:.4f} - ({E_ab_prime:.4f}) + {E_a_prime_b:.4f} + {E_a_prime_b_prime:.4f}|")
    print(f"S = |{E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime:.4f}|")
    print(f"S = {result.S:.6f}")

    # Verify it equals 2√2
    passed = abs(result.S - QUANTUM_BOUND) < 1e-10

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Analytical formula gives exact 2√2")

    return passed, {
        'S': result.S,
        'expected': QUANTUM_BOUND,
        'error': abs(result.S - QUANTUM_BOUND),
        'passed': passed
    }


def test_separation_ratio() -> Tuple[bool, Dict]:
    """
    Test that quantum/classical separation is sufficient.

    For irrefutable tests, we need clear separation between hypotheses.
    """
    print("\n" + "=" * 70)
    print("TEST 0d: Quantum/Classical Separation")
    print("=" * 70)

    quantum_result = simulate_quantum_chsh()
    classical_result = simulate_classical_chsh()

    separation = quantum_result.S - classical_result.S
    ratio = quantum_result.S / classical_result.S if classical_result.S > 0 else float('inf')

    print(f"\nQuantum S: {quantum_result.S:.6f}")
    print(f"Classical S: {classical_result.S:.6f}")
    print(f"Separation: {separation:.6f}")
    print(f"Ratio: {ratio:.4f}x")

    # For irrefutable tests, need clear separation
    # Quantum exceeds classical by ~0.83 (2.83 - 2.0)
    passed = separation > 0.8 and ratio > 1.4

    print(f"\n{'✓ PASS' if passed else '✗ FAIL'}: Sufficient quantum/classical separation")

    return passed, {
        'quantum_S': quantum_result.S,
        'classical_S': classical_result.S,
        'separation': separation,
        'ratio': ratio,
        'passed': passed
    }


def test_statistical_significance() -> Tuple[bool, Dict]:
    """
    Test that quantum violation is statistically significant.

    Run multiple trials and check that S > 2 with high confidence.
    """
    print("\n" + "=" * 70)
    print("TEST 0e: Statistical Significance")
    print("=" * 70)

    n_trials = 100

    # Bootstrap quantum CHSH
    quantum_S_values = []
    for _ in range(n_trials):
        result = simulate_quantum_chsh()
        quantum_S_values.append(result.S)

    mean_S = np.mean(quantum_S_values)
    std_S = np.std(quantum_S_values)
    ci_95 = 1.96 * std_S / np.sqrt(n_trials)

    print(f"\nQuantum CHSH over {n_trials} trials:")
    print(f"  Mean S: {mean_S:.6f}")
    print(f"  Std S: {std_S:.6f}")
    print(f"  95% CI: [{mean_S - ci_95:.6f}, {mean_S + ci_95:.6f}]")

    # Check lower bound exceeds classical
    lower_bound = mean_S - 3 * std_S
    print(f"  Lower bound (3σ): {lower_bound:.6f}")

    significant = is_violation_significant(mean_S, std_S, n_sigma=3.0)

    print(f"\n{'✓ PASS' if significant else '✗ FAIL'}: Violation significant at 3σ level")

    return significant, {
        'mean_S': mean_S,
        'std_S': std_S,
        'ci_95': ci_95,
        'lower_bound_3sigma': lower_bound,
        'significant': significant
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> Dict:
    """Run all quantum control tests."""

    print("\n" + "=" * 70)
    print("Q42 TEST 0: QUANTUM CONTROL - APPARATUS VALIDATION")
    print("=" * 70)
    print(f"\nPurpose: Validate CHSH machinery before semantic tests")
    print(f"QuTiP available: {QUTIP_AVAILABLE}")

    results = {}
    all_passed = True

    # Run each test
    tests = [
        ('quantum_qutip', test_quantum_chsh_qutip),
        ('classical', test_classical_chsh),
        ('analytical', test_analytical_quantum),
        ('separation', test_separation_ratio),
        ('significance', test_statistical_significance),
    ]

    for name, test_func in tests:
        passed, data = test_func()
        results[name] = data
        all_passed = all_passed and passed

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Quantum Control Tests")
    print("=" * 70)

    n_passed = sum(1 for r in results.values() if r.get('passed', False) or r.get('significant', False))
    n_total = len(results)

    print(f"\nTests passed: {n_passed}/{n_total}")
    print(f"Apparatus valid: {all_passed}")

    for name, data in results.items():
        status = '✓' if data.get('passed', data.get('significant', False)) else '✗'
        print(f"  {status} {name}")

    results['summary'] = {
        'n_passed': n_passed,
        'n_total': n_total,
        'all_passed': all_passed,
        'apparatus_valid': all_passed,
        'timestamp': datetime.now().isoformat()
    }

    return results


if __name__ == '__main__':
    results = run_all_tests()

    # Save results
    output_path = Path(__file__).parent / 'q42_test0_results.json'
    with open(output_path, 'w') as f:
        # Convert numpy types for JSON
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

    # Exit code based on pass/fail
    sys.exit(0 if results['summary']['all_passed'] else 1)
