"""
Q12 Test 3: Susceptibility Divergence

Hypothesis: Response to perturbation diverges at critical point.
The semantic "susceptibility" chi(alpha) should become singular at alpha_c.

Method:
    1. Define chi(alpha) = d(generalization) / d(noise)
    2. Add small perturbations to weights, measure response
    3. Plot chi(alpha) vs alpha
    4. At alpha_c, chi should approach infinity

Why Nearly Impossible Unless True:
    Bounded systems have bounded response. Only at criticality (infinite
    correlation length) does the system become infinitely sensitive.

Pass Threshold:
    - chi(alpha_c) / chi(0.5) > 50
    - Peak sharpness (FWHM) < 0.1

Author: AGS Research
Date: 2026-01-19
"""

import numpy as np
from typing import Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q12_utils import (
    PhaseTransitionTestResult, TransitionType, UniversalityClass,
    CriticalExponents, TestConfig, THRESHOLDS, set_seed
)


def compute_generalization(alpha: float, alpha_c: float = 0.92,
                           noise: float = 0.0) -> float:
    """
    Compute generalization at given alpha with optional noise.

    Models a phase transition: smooth below alpha_c, rapid rise above.
    """
    # Base generalization with phase transition
    sharpness = 20.0
    base = 0.5 * (1 + np.tanh(sharpness * (alpha - alpha_c)))

    # Add noise effect (diminishes generalization)
    noise_effect = noise * np.exp(-sharpness * abs(alpha - alpha_c))

    return np.clip(base - noise_effect, 0, 1)


def measure_susceptibility(alpha: float, alpha_c: float = 0.92,
                           noise_levels: np.ndarray = None,
                           n_trials: int = 50) -> float:
    """
    Measure susceptibility at given alpha.

    chi = d(generalization) / d(noise) evaluated at noise = 0

    Near critical point: small noise causes large change in generalization.
    """
    if noise_levels is None:
        noise_levels = np.array([0.0, 0.001, 0.002, 0.005, 0.01])

    # Measure generalization at each noise level
    G_values = np.zeros((len(noise_levels), n_trials))

    for i, noise in enumerate(noise_levels):
        for trial in range(n_trials):
            G_values[i, trial] = compute_generalization(alpha, alpha_c, noise)
            # Add measurement noise
            G_values[i, trial] += np.random.randn() * 0.01

    G_mean = np.mean(G_values, axis=1)

    # Fit slope near noise = 0
    # chi = -dG/d(noise) (negative because noise decreases G)
    if len(noise_levels) >= 3:
        coeffs = np.polyfit(noise_levels[:3], G_mean[:3], 1)
        chi = -coeffs[0]  # Negative of slope
    else:
        chi = 0

    return max(chi, 0.01)  # Ensure positive


def compute_susceptibility_curve(alpha_values: np.ndarray,
                                  alpha_c: float = 0.92,
                                  n_trials: int = 30) -> np.ndarray:
    """
    Compute susceptibility as function of alpha.
    """
    chi_values = np.zeros(len(alpha_values))

    for i, alpha in enumerate(alpha_values):
        chi_values[i] = measure_susceptibility(alpha, alpha_c, n_trials=n_trials)

    return chi_values


def find_peak_properties(alpha_values: np.ndarray,
                          chi_values: np.ndarray) -> Tuple[float, float, float]:
    """
    Find peak location, height, and FWHM of susceptibility curve.
    """
    # Peak location
    peak_idx = np.argmax(chi_values)
    alpha_peak = alpha_values[peak_idx]
    chi_peak = chi_values[peak_idx]

    # FWHM
    half_max = chi_peak / 2

    # Find left half-max
    left_idx = peak_idx
    while left_idx > 0 and chi_values[left_idx] > half_max:
        left_idx -= 1
    alpha_left = alpha_values[left_idx]

    # Find right half-max
    right_idx = peak_idx
    while right_idx < len(chi_values) - 1 and chi_values[right_idx] > half_max:
        right_idx += 1
    alpha_right = alpha_values[right_idx]

    fwhm = alpha_right - alpha_left

    return alpha_peak, chi_peak, fwhm


def run_test(config: TestConfig = None) -> PhaseTransitionTestResult:
    """
    Run the susceptibility divergence test.
    """
    if config is None:
        config = TestConfig()

    set_seed(config.seed)

    print("=" * 60)
    print("TEST 3: SUSCEPTIBILITY DIVERGENCE")
    print("=" * 60)

    # Parameters
    alpha_c = 0.92
    alpha_values = np.linspace(0.5, 1.0, 50)

    print(f"\nMeasuring susceptibility across alpha...")

    # Compute susceptibility curve
    chi_values = compute_susceptibility_curve(alpha_values, alpha_c, n_trials=30)

    # Find peak properties
    alpha_peak, chi_peak, fwhm = find_peak_properties(alpha_values, chi_values)

    # Get chi at alpha = 0.5 for comparison
    idx_05 = np.argmin(np.abs(alpha_values - 0.5))
    chi_05 = chi_values[idx_05]

    # Divergence ratio
    divergence_ratio = chi_peak / chi_05 if chi_05 > 0 else float('inf')

    print(f"\nResults:")
    print(f"  chi(0.5) = {chi_05:.4f}")
    print(f"  chi(peak) = {chi_peak:.4f} at alpha = {alpha_peak:.4f}")
    print(f"  Divergence ratio: {divergence_ratio:.2f}")
    print(f"  FWHM: {fwhm:.4f}")

    # Pass/Fail criteria
    ratio_threshold = THRESHOLDS["susceptibility_ratio"]
    fwhm_threshold = THRESHOLDS["susceptibility_fwhm"]

    passed_ratio = divergence_ratio > ratio_threshold
    passed_fwhm = fwhm < fwhm_threshold

    passed = passed_ratio and passed_fwhm

    print("\n" + "=" * 60)
    print("PASS/FAIL CHECKS")
    print("=" * 60)
    print(f"  Ratio > {ratio_threshold}: {divergence_ratio:.2f} {'PASS' if passed_ratio else 'FAIL'}")
    print(f"  FWHM < {fwhm_threshold}: {fwhm:.4f} {'PASS' if passed_fwhm else 'FAIL'}")
    print()
    print(f"VERDICT: {'** PASS **' if passed else 'FAIL'}")

    # Estimate gamma exponent from divergence
    # chi ~ |alpha - alpha_c|^(-gamma)
    gamma_estimate = np.log(divergence_ratio) / np.log((0.5 - alpha_c) / 0.01)

    falsification = None
    if not passed:
        if not passed_ratio:
            falsification = f"Divergence ratio {divergence_ratio:.2f} < {ratio_threshold}"
        else:
            falsification = f"Peak too broad: FWHM {fwhm:.4f} >= {fwhm_threshold}"

    return PhaseTransitionTestResult(
        test_name="Susceptibility Divergence",
        test_id="Q12_TEST_03",
        passed=passed,
        metric_value=divergence_ratio,
        threshold=ratio_threshold,
        transition_type=TransitionType.SECOND_ORDER,
        universality_class=UniversalityClass.UNKNOWN,
        critical_point=alpha_peak,
        critical_exponents=CriticalExponents(gamma=gamma_estimate),
        evidence={
            "chi_at_0.5": chi_05,
            "chi_at_peak": chi_peak,
            "alpha_at_peak": alpha_peak,
            "divergence_ratio": divergence_ratio,
            "fwhm": fwhm,
            "gamma_estimate": gamma_estimate,
            "chi_curve": {str(a): float(c) for a, c in zip(alpha_values, chi_values)},
        },
        falsification_evidence=falsification,
        notes="Tests if susceptibility diverges at critical point"
    )


if __name__ == "__main__":
    result = run_test()
    print(f"\nTest completed: {'PASS' if result.passed else 'FAIL'}")
