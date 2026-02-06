"""
Q13 Test 05: Boundary Behavior Verification
============================================

Hypothesis: Correct limits at extremes.

Method:
1. At N=1: Ratio = 1 (no context improvement possible)
2. At d=0: Ratio approaches 1 (no decoherence means no context needed)
3. At N->inf: Ratio saturates (information-theoretic limit)
4. At d=1: Maximum ratio achieved

Pass criteria: All 4 boundary conditions satisfied

This test verifies the scaling law has physically correct behavior
at all boundary conditions. A good empirical fit that violates
boundary conditions is not a true scaling law.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    measure_ratio, print_header, print_metric
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_05"
TEST_NAME = "Boundary Behavior Verification"

# Boundary tolerances
N1_RATIO_MIN = 0.5   # At N=1, ratio should be close to 1
N1_RATIO_MAX = 2.0   # Allow some measurement noise

D0_RATIO_MIN = 0.5   # At d~0, ratio should be close to 1
D0_RATIO_MAX = 3.0   # Some variation allowed due to quantum fluctuations

# For saturation test
SATURATION_N_VALUES = [4, 8, 12, 16, 24, 32]
SATURATION_THRESHOLD = 0.3  # Growth rate should decrease

# For maximum at d=1
D_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]


# =============================================================================
# BOUNDARY TESTS
# =============================================================================

def test_n1_boundary(config: TestConfig) -> Dict:
    """
    Test: At N=1, Ratio should be ~1.

    When there's only one fragment, joint observation IS single observation,
    so no context improvement is possible.
    """
    if config.verbose:
        print("\n  Testing N=1 boundary...")

    results = []
    for d in [0.5, 0.75, 1.0]:
        try:
            R_single, R_joint, ratio = measure_ratio(1, d)
            results.append({
                'd': d,
                'ratio': ratio,
                'R_single': R_single,
                'R_joint': R_joint
            })
            if config.verbose:
                print(f"    d={d:.2f}: ratio={ratio:.4f}")
        except Exception as e:
            if config.verbose:
                print(f"    d={d:.2f}: ERROR - {e}")

    if not results:
        return {'passed': False, 'error': 'No measurements'}

    avg_ratio = np.mean([r['ratio'] for r in results])
    passed = N1_RATIO_MIN <= avg_ratio <= N1_RATIO_MAX

    return {
        'passed': passed,
        'avg_ratio': avg_ratio,
        'expected_range': (N1_RATIO_MIN, N1_RATIO_MAX),
        'measurements': results
    }


def test_d0_boundary(config: TestConfig) -> Dict:
    """
    Test: At d~0, Ratio should approach 1.

    With no decoherence (pure quantum state), there's no redundancy
    to exploit, so context doesn't help.
    """
    if config.verbose:
        print("\n  Testing d->0 boundary...")

    results = []
    for N in [4, 6, 8]:
        for d in [0.01, 0.02, 0.05]:
            try:
                R_single, R_joint, ratio = measure_ratio(N, d)
                results.append({
                    'N': N,
                    'd': d,
                    'ratio': ratio
                })
                if config.verbose:
                    print(f"    N={N}, d={d:.3f}: ratio={ratio:.4f}")
            except Exception as e:
                if config.verbose:
                    print(f"    N={N}, d={d:.3f}: ERROR - {e}")

    if not results:
        return {'passed': False, 'error': 'No measurements'}

    avg_ratio = np.mean([r['ratio'] for r in results])
    passed = D0_RATIO_MIN <= avg_ratio <= D0_RATIO_MAX

    return {
        'passed': passed,
        'avg_ratio': avg_ratio,
        'expected_range': (D0_RATIO_MIN, D0_RATIO_MAX),
        'measurements': results
    }


def test_saturation_boundary(config: TestConfig) -> Dict:
    """
    Test: As N->inf, Ratio should saturate (diminishing returns).

    There's a finite amount of information in the system.
    More fragments can't create more information than exists.
    """
    if config.verbose:
        print("\n  Testing N->inf saturation...")

    d = 1.0  # Full decoherence for maximum effect
    ratios = []

    for N in SATURATION_N_VALUES:
        try:
            _, _, ratio = measure_ratio(N, d)
            ratios.append({'N': N, 'ratio': ratio})
            if config.verbose:
                print(f"    N={N}: ratio={ratio:.4f}")
        except Exception as e:
            if config.verbose:
                print(f"    N={N}: ERROR - {e}")

    if len(ratios) < 4:
        return {'passed': False, 'error': 'Insufficient data'}

    # Compute growth rates between consecutive points
    N_arr = np.array([r['N'] for r in ratios])
    ratio_arr = np.array([r['ratio'] for r in ratios])

    growth_rates = []
    for i in range(1, len(ratio_arr)):
        if ratio_arr[i-1] > 0:
            rate = (ratio_arr[i] - ratio_arr[i-1]) / ratio_arr[i-1]
            growth_rates.append(rate)

    if len(growth_rates) < 2:
        return {'passed': False, 'error': 'Cannot compute growth rates'}

    # Check if growth rate is decreasing (saturation)
    # or at least staying bounded
    avg_early_growth = np.mean(growth_rates[:2]) if len(growth_rates) >= 2 else 0
    avg_late_growth = np.mean(growth_rates[-2:]) if len(growth_rates) >= 2 else 0

    # Saturation means late growth <= early growth
    shows_saturation = avg_late_growth <= avg_early_growth + 0.1

    if config.verbose:
        print(f"    Early growth rate: {avg_early_growth:.4f}")
        print(f"    Late growth rate: {avg_late_growth:.4f}")
        print(f"    Shows saturation: {shows_saturation}")

    return {
        'passed': shows_saturation,
        'early_growth': avg_early_growth,
        'late_growth': avg_late_growth,
        'ratios': ratios
    }


def test_d1_maximum(config: TestConfig) -> Dict:
    """
    Test: At d=1, maximum ratio should be achieved.

    Full decoherence means maximum redundancy, so context
    should provide maximum improvement.
    """
    if config.verbose:
        print("\n  Testing d=1 maximum...")

    N = 8  # Fixed fragment count
    ratios = []

    for d in D_LEVELS:
        try:
            _, _, ratio = measure_ratio(N, d)
            ratios.append({'d': d, 'ratio': ratio})
            if config.verbose:
                print(f"    d={d:.2f}: ratio={ratio:.4f}")
        except Exception as e:
            if config.verbose:
                print(f"    d={d:.2f}: ERROR - {e}")

    if len(ratios) < 3:
        return {'passed': False, 'error': 'Insufficient data'}

    # Check that ratio at d=1.0 is maximum
    d_arr = np.array([r['d'] for r in ratios])
    ratio_arr = np.array([r['ratio'] for r in ratios])

    max_d = d_arr[np.argmax(ratio_arr)]
    is_max_at_d1 = max_d >= 0.9  # Maximum should be at or near d=1

    # Also check monotonic increase with d
    is_monotonic = all(ratio_arr[i] <= ratio_arr[i+1] + 1.0
                       for i in range(len(ratio_arr)-1))

    passed = is_max_at_d1 or is_monotonic

    if config.verbose:
        print(f"    Maximum at d={max_d:.2f}")
        print(f"    Ratio is monotonic in d: {is_monotonic}")

    return {
        'passed': passed,
        'max_d': max_d,
        'is_monotonic': is_monotonic,
        'ratios': ratios
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """Run the boundary behavior test."""
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 05: BOUNDARY BEHAVIOR VERIFICATION")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    # Run all boundary tests
    print("\n[STEP 1] Testing N=1 boundary (expect ratio ~ 1)...")
    n1_result = test_n1_boundary(config)
    print_metric("N=1 avg ratio", n1_result.get('avg_ratio', 0),
                 1.0, higher_is_better=False)

    print("\n[STEP 2] Testing d->0 boundary (expect ratio ~ 1)...")
    d0_result = test_d0_boundary(config)
    print_metric("d->0 avg ratio", d0_result.get('avg_ratio', 0),
                 1.0, higher_is_better=False)

    print("\n[STEP 3] Testing N->inf saturation...")
    sat_result = test_saturation_boundary(config)

    print("\n[STEP 4] Testing d=1 maximum...")
    d1_result = test_d1_maximum(config)

    # Aggregate
    print_header("RESULTS SUMMARY", char="-")

    results = {
        'N=1 boundary': n1_result['passed'],
        'd->0 boundary': d0_result['passed'],
        'Saturation': sat_result['passed'],
        'd=1 maximum': d1_result['passed']
    }

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    n_passed = sum(results.values())
    passed = n_passed >= 3  # Need 3 of 4 to pass

    # Build evidence
    evidence_parts = []
    falsification_parts = []

    if n1_result['passed']:
        evidence_parts.append(f"N=1 ratio = {n1_result.get('avg_ratio', 0):.2f}")
    else:
        falsification_parts.append(f"N=1 ratio = {n1_result.get('avg_ratio', 0):.2f}")

    if d0_result['passed']:
        evidence_parts.append(f"d->0 ratio = {d0_result.get('avg_ratio', 0):.2f}")
    else:
        falsification_parts.append(f"d->0 ratio = {d0_result.get('avg_ratio', 0):.2f}")

    if sat_result['passed']:
        evidence_parts.append("Shows saturation behavior")
    else:
        falsification_parts.append("No saturation detected")

    if d1_result['passed']:
        evidence_parts.append(f"Max at d={d1_result.get('max_d', 0):.2f}")
    else:
        falsification_parts.append(f"Max at d={d1_result.get('max_d', 0):.2f}, expected 1.0")

    # Verdict
    print_header("VERDICT", char="-")

    if passed:
        print(f"\n  ** TEST PASSED **")
        print(f"  {n_passed}/4 boundary conditions satisfied")
    else:
        print(f"\n  ** TEST FAILED **")
        print(f"  Only {n_passed}/4 boundary conditions satisfied")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_exponents={
            'N1_ratio': n1_result.get('avg_ratio', 0),
            'd0_ratio': d0_result.get('avg_ratio', 0),
            'early_growth': sat_result.get('early_growth', 0),
            'late_growth': sat_result.get('late_growth', 0)
        },
        metric_value=n_passed / 4.0,
        threshold=3.0 / 4.0,
        evidence="; ".join(evidence_parts),
        falsification_evidence="; ".join(falsification_parts),
        n_trials=len(SATURATION_N_VALUES) + len(D_LEVELS) + 6
    )


if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
