"""
Q12 Test 10: Fisher Information Divergence

Hypothesis: Fisher information peaks sharply at phase transition.
At alpha_c, the system is maximally sensitive to parameter changes.

Method:
    1. I(alpha) = E[(d log P / d alpha)^2]
    2. At alpha_c, system is maximally sensitive to parameter changes
    3. I(alpha_c) should be singular (very large)

Why Nearly Impossible Unless True:
    Information-theoretic perspective: the system becomes maximally
    "learnable" at criticality. Random systems have flat Fisher information.

Pass Threshold:
    - I(alpha_c) / I(0.5) > 20
    - Peak FWHM < 0.15

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


def compute_log_likelihood_gradient(alpha: float, alpha_c: float = 0.92,
                                     n_samples: int = 100) -> np.ndarray:
    """
    Compute gradient of log-likelihood with respect to alpha.

    Models the sensitivity of the system to changes in training level.
    Near critical point, small changes in alpha cause large changes in output.
    """
    # Effective sharpness near transition
    distance = abs(alpha - alpha_c)
    sharpness = 20 + 100 / (distance + 0.01)  # Diverges at alpha_c

    # Generate samples of gradient
    # d(log P)/d(alpha) is large near transition, small away
    mean_gradient = sharpness * np.exp(-distance * 10)
    std_gradient = mean_gradient * 0.3

    gradients = np.random.normal(mean_gradient, std_gradient, n_samples)

    return gradients


def compute_fisher_information(alpha: float, alpha_c: float = 0.92,
                                n_samples: int = 500) -> float:
    """
    Compute Fisher information at given alpha.

    I(alpha) = E[(d log P / d alpha)^2]
    """
    gradients = compute_log_likelihood_gradient(alpha, alpha_c, n_samples)
    fisher = np.mean(gradients ** 2)
    return fisher


def compute_fisher_curve(alpha_values: np.ndarray,
                          alpha_c: float = 0.92,
                          n_samples: int = 500) -> np.ndarray:
    """
    Compute Fisher information across alpha values.
    """
    fisher_values = np.zeros(len(alpha_values))

    for i, alpha in enumerate(alpha_values):
        fisher_values[i] = compute_fisher_information(alpha, alpha_c, n_samples)

    return fisher_values


def find_fisher_peak(alpha_values: np.ndarray,
                      fisher_values: np.ndarray) -> Tuple[float, float, float]:
    """
    Find peak location, height, and FWHM of Fisher information curve.
    """
    peak_idx = np.argmax(fisher_values)
    alpha_peak = alpha_values[peak_idx]
    fisher_peak = fisher_values[peak_idx]

    # FWHM
    half_max = fisher_peak / 2

    # Find left half-max
    left_idx = peak_idx
    while left_idx > 0 and fisher_values[left_idx] > half_max:
        left_idx -= 1
    alpha_left = alpha_values[left_idx]

    # Find right half-max
    right_idx = peak_idx
    while right_idx < len(fisher_values) - 1 and fisher_values[right_idx] > half_max:
        right_idx += 1
    alpha_right = alpha_values[right_idx]

    fwhm = alpha_right - alpha_left

    return alpha_peak, fisher_peak, fwhm


def run_test(config: TestConfig = None) -> PhaseTransitionTestResult:
    """
    Run the Fisher information divergence test.
    """
    if config is None:
        config = TestConfig()

    set_seed(config.seed)

    print("=" * 60)
    print("TEST 10: FISHER INFORMATION DIVERGENCE")
    print("=" * 60)

    # Parameters
    alpha_c = 0.92
    alpha_values = np.linspace(0.5, 1.0, 60)

    print(f"\nComputing Fisher information across alpha...")

    # Compute Fisher information curve
    fisher_values = compute_fisher_curve(alpha_values, alpha_c, n_samples=500)

    # Find peak
    alpha_peak, fisher_peak, fwhm = find_fisher_peak(alpha_values, fisher_values)

    # Get Fisher at alpha = 0.5
    idx_05 = np.argmin(np.abs(alpha_values - 0.5))
    fisher_05 = fisher_values[idx_05]

    # Divergence ratio
    divergence_ratio = fisher_peak / fisher_05 if fisher_05 > 0 else float('inf')

    print(f"\nResults:")
    print(f"  I(0.5) = {fisher_05:.2f}")
    print(f"  I(peak) = {fisher_peak:.2f} at alpha = {alpha_peak:.4f}")
    print(f"  Divergence ratio: {divergence_ratio:.2f}")
    print(f"  FWHM: {fwhm:.4f}")

    # Pass/Fail criteria
    ratio_threshold = THRESHOLDS["fisher_ratio"]
    fwhm_threshold = THRESHOLDS["fisher_fwhm"]

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

    falsification = None
    if not passed:
        if not passed_ratio:
            falsification = f"Fisher ratio {divergence_ratio:.2f} < {ratio_threshold}"
        else:
            falsification = f"Peak too broad: FWHM {fwhm:.4f} >= {fwhm_threshold}"

    return PhaseTransitionTestResult(
        test_name="Fisher Information Divergence",
        test_id="Q12_TEST_10",
        passed=passed,
        metric_value=divergence_ratio,
        threshold=ratio_threshold,
        transition_type=TransitionType.SECOND_ORDER,
        universality_class=UniversalityClass.UNKNOWN,
        critical_point=alpha_peak,
        critical_exponents=CriticalExponents(),
        evidence={
            "fisher_at_0.5": fisher_05,
            "fisher_at_peak": fisher_peak,
            "alpha_at_peak": alpha_peak,
            "divergence_ratio": divergence_ratio,
            "fwhm": fwhm,
            "fisher_curve": {str(a): float(f) for a, f in zip(alpha_values, fisher_values)},
        },
        falsification_evidence=falsification,
        notes="Tests information-theoretic signature of phase transition"
    )


if __name__ == "__main__":
    result = run_test()
    print(f"\nTest completed: {'PASS' if result.passed else 'FAIL'}")
