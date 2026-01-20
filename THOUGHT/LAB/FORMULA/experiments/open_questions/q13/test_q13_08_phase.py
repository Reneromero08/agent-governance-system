"""
Q13 Test 08: Phase Transition Detection
========================================

Hypothesis: There exists a critical fragment count N_c where resolution "crystallizes."

Method:
1. Measure Ratio(N) for N = 1 to 20 at fixed d=1.0
2. Compute susceptibility: chi = d(Ratio)/dN
3. Look for divergence/peak at N_c
4. Verify Binder cumulant crossing if exists

Pass criteria: Either confirm phase transition with critical exponents
               OR prove continuous (no sharp transition)

This test determines whether context improvement is a SUDDEN transition
or a GRADUAL process.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    measure_ratio, print_header, print_metric
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_08"
TEST_NAME = "Phase Transition Detection"

# Fragment range for sweep
N_MIN = 1
N_MAX = 20

# Decoherence level
DECOHERENCE = 1.0

# Thresholds
SUSCEPTIBILITY_PEAK_RATIO = 3.0  # Peak should be 3x background
GRADIENT_SMOOTHING_SIGMA = 1.5  # Gaussian smoothing for gradient


# =============================================================================
# PHASE TRANSITION ANALYSIS
# =============================================================================

def collect_ratio_sweep(config: TestConfig) -> Dict:
    """Sweep N from 1 to N_MAX at fixed decoherence."""
    if config.verbose:
        print(f"\n  Sweeping N from {N_MIN} to {N_MAX} at d={DECOHERENCE}...")

    results = {'N': [], 'ratio': []}

    for N in range(N_MIN, N_MAX + 1):
        try:
            _, _, ratio = measure_ratio(N, DECOHERENCE)
            results['N'].append(N)
            results['ratio'].append(ratio)
            if config.verbose:
                print(f"    N={N}: ratio = {ratio:.4f}")
        except Exception as e:
            if config.verbose:
                print(f"    N={N}: ERROR - {e}")

    for key in results:
        results[key] = np.array(results[key])

    return results


def compute_susceptibility(N: np.ndarray, ratio: np.ndarray) -> Dict:
    """
    Compute susceptibility chi = d(Ratio)/dN.

    In phase transitions, susceptibility diverges at critical point.
    """
    if len(N) < 5:
        return {'chi': np.array([]), 'N_chi': np.array([]), 'peak_found': False}

    # Smooth the ratio first
    ratio_smooth = gaussian_filter1d(ratio.astype(float), GRADIENT_SMOOTHING_SIGMA)

    # Compute gradient (susceptibility)
    chi = np.gradient(ratio_smooth, N.astype(float))

    # Smooth susceptibility
    chi_smooth = gaussian_filter1d(chi, GRADIENT_SMOOTHING_SIGMA)

    # Find peak
    peak_idx = np.argmax(np.abs(chi_smooth))
    peak_value = np.abs(chi_smooth[peak_idx])
    background = np.median(np.abs(chi_smooth))

    peak_ratio = peak_value / background if background > 0 else 0
    peak_found = peak_ratio > SUSCEPTIBILITY_PEAK_RATIO

    return {
        'chi': chi_smooth,
        'N_chi': N,
        'peak_idx': peak_idx,
        'peak_N': N[peak_idx],
        'peak_value': peak_value,
        'background': background,
        'peak_ratio': peak_ratio,
        'peak_found': peak_found
    }


def compute_binder_cumulant(ratio: np.ndarray) -> Dict:
    """
    Compute Binder cumulant proxy for finite-size effects.

    For true phase transitions, Binder cumulant shows characteristic crossing.
    Here we compute a simplified version using ratio moments.
    """
    if len(ratio) < 5:
        return {'U': 0, 'crossing_found': False}

    # Compute moments
    mean_r = np.mean(ratio)
    mean_r2 = np.mean(ratio ** 2)
    mean_r4 = np.mean(ratio ** 4)

    # Binder cumulant: U = 1 - <r^4> / (3 * <r^2>^2)
    if mean_r2 > 0:
        U = 1 - mean_r4 / (3 * mean_r2 ** 2)
    else:
        U = 0

    # For phase transition, U should be between 0 and 2/3
    is_valid = 0 < U < 0.67

    return {
        'U': U,
        'mean_r': mean_r,
        'mean_r2': mean_r2,
        'mean_r4': mean_r4,
        'is_valid': is_valid
    }


def fit_power_law_transition(N: np.ndarray, ratio: np.ndarray) -> Dict:
    """
    Fit ratio to power law and check for smooth vs sharp transition.

    Sharp transition: exponent changes abruptly at critical point.
    Smooth transition: single power law fits entire range.
    """
    if len(N) < 5:
        return {'smooth': False, 'r_squared': 0}

    # Filter valid data
    mask = (ratio > 0) & (N > 0)
    N_v = N[mask].astype(float)
    r_v = ratio[mask]

    if len(r_v) < 5:
        return {'smooth': False, 'r_squared': 0}

    # Fit log-log
    log_N = np.log(N_v)
    log_r = np.log(r_v)

    try:
        coeffs = np.polyfit(log_N, log_r, 1)
        predicted = np.polyval(coeffs, log_N)

        ss_res = np.sum((log_r - predicted) ** 2)
        ss_tot = np.sum((log_r - np.mean(log_r)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # High R^2 = smooth power law, no phase transition
        smooth = r_squared > 0.95

        return {
            'smooth': smooth,
            'r_squared': r_squared,
            'exponent': coeffs[0],
            'intercept': coeffs[1]
        }
    except:
        return {'smooth': False, 'r_squared': 0}


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """Run the phase transition detection test."""
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 08: PHASE TRANSITION DETECTION")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    # Collect ratio sweep
    print("\n[STEP 1] Collecting ratio vs N sweep...")
    data = collect_ratio_sweep(config)

    if len(data['ratio']) < 10:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            falsification_evidence="Insufficient data for phase analysis"
        )

    # Compute susceptibility
    print("\n[STEP 2] Computing susceptibility chi = dRatio/dN...")
    chi_result = compute_susceptibility(data['N'], data['ratio'])

    if config.verbose:
        print(f"    Peak at N = {chi_result.get('peak_N', 0)}")
        print(f"    Peak/background ratio = {chi_result.get('peak_ratio', 0):.2f}")
        print(f"    Sharp transition: {chi_result.get('peak_found', False)}")

    # Compute Binder cumulant
    print("\n[STEP 3] Computing Binder cumulant proxy...")
    binder_result = compute_binder_cumulant(data['ratio'])

    if config.verbose:
        print(f"    Binder U = {binder_result.get('U', 0):.4f}")
        print(f"    Valid range (0 < U < 0.67): {binder_result.get('is_valid', False)}")

    # Fit power law
    print("\n[STEP 4] Testing for smooth power law...")
    power_result = fit_power_law_transition(data['N'], data['ratio'])

    if config.verbose:
        print(f"    Power law R^2 = {power_result.get('r_squared', 0):.4f}")
        print(f"    Exponent = {power_result.get('exponent', 0):.3f}")
        print(f"    Is smooth (no transition): {power_result.get('smooth', False)}")

    # Determine result
    print_header("ANALYSIS", char="-")

    phase_transition_detected = chi_result.get('peak_found', False)
    smooth_power_law = power_result.get('smooth', False)

    if phase_transition_detected:
        conclusion = "PHASE_TRANSITION"
        print(f"\n  Sharp phase transition detected at N_c ~ {chi_result.get('peak_N', 0)}")
    elif smooth_power_law:
        conclusion = "SMOOTH_POWER_LAW"
        print(f"\n  Smooth power law (exponent = {power_result.get('exponent', 0):.3f})")
        print("  No phase transition - context improvement is CONTINUOUS")
    else:
        conclusion = "INCONCLUSIVE"
        print("\n  Neither sharp transition nor smooth power law")

    # The test PASSES if we can definitively categorize the behavior
    # Either as phase transition OR as continuous
    passed = phase_transition_detected or smooth_power_law

    # Build evidence
    if phase_transition_detected:
        evidence = f"Phase transition at N_c = {chi_result.get('peak_N', 0)}, "
        evidence += f"susceptibility peak ratio = {chi_result.get('peak_ratio', 0):.1f}"
        falsification = ""
    elif smooth_power_law:
        evidence = f"Smooth power law: exponent = {power_result.get('exponent', 0):.3f}, "
        evidence += f"R^2 = {power_result.get('r_squared', 0):.4f}"
        falsification = ""
    else:
        evidence = "Data collected but pattern unclear"
        falsification = f"R^2 = {power_result.get('r_squared', 0):.3f}, "
        falsification += f"peak ratio = {chi_result.get('peak_ratio', 0):.2f}"

    # Verdict
    print_header("VERDICT", char="-")

    if passed:
        print("\n  ** TEST PASSED **")
        print(f"  Behavior classified as: {conclusion}")
    else:
        print("\n  ** TEST FAILED **")
        print("  Cannot definitively classify transition behavior")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_law="critical" if phase_transition_detected else "power",
        scaling_exponents={
            'power_exponent': power_result.get('exponent', 0),
            'critical_N': chi_result.get('peak_N', 0) if phase_transition_detected else 0,
            'susceptibility_peak': chi_result.get('peak_ratio', 0),
            'binder_U': binder_result.get('U', 0)
        },
        fit_quality=power_result.get('r_squared', 0),
        metric_value=chi_result.get('peak_ratio', 0),
        threshold=SUSCEPTIBILITY_PEAK_RATIO,
        evidence=evidence,
        falsification_evidence=falsification,
        n_trials=len(data['N'])
    )


if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
    print(f"Scaling law: {result.scaling_law}")
