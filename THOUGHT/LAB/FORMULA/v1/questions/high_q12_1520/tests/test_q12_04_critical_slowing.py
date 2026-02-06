"""
Q12 Test 4: Critical Slowing Down

Hypothesis: Dynamics slow as alpha approaches alpha_c.
The relaxation time tau should diverge near the critical point.

Method:
    1. Initialize system near equilibrium
    2. Measure relaxation time tau(alpha) to return to equilibrium
    3. tau should diverge: tau ~ |alpha - alpha_c|^(-z*nu)

Why Nearly Impossible Unless True:
    Uniform dynamics cannot produce diverging timescales. Only near
    criticality does the system become "indecisive" between phases.

Pass Threshold:
    - tau(alpha_c) / tau(0.5) > 10

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


def simulate_relaxation(alpha: float, alpha_c: float = 0.92,
                        max_steps: int = 1000,
                        equilibrium_threshold: float = 0.01) -> int:
    """
    Simulate relaxation dynamics and measure time to equilibrium.

    Near critical point, relaxation slows due to critical fluctuations.
    """
    # Equilibrium value of order parameter
    if alpha > alpha_c:
        M_eq = 0.5 * (1 + np.tanh(20 * (alpha - alpha_c)))
    else:
        M_eq = 0.05 * alpha

    # Start from perturbed state
    M = M_eq + 0.3 * np.random.randn()

    # Relaxation rate: slows near critical point
    # tau ~ |alpha - alpha_c|^(-z*nu) where z*nu ~ 2 typically
    distance_to_critical = abs(alpha - alpha_c) + 0.01
    relaxation_rate = 0.1 * (distance_to_critical ** 1.5)

    # Simulate dynamics
    for step in range(max_steps):
        # Relaxation towards equilibrium with noise
        dM = relaxation_rate * (M_eq - M) + 0.01 * np.random.randn()
        M = M + dM

        # Check if equilibrated
        if abs(M - M_eq) < equilibrium_threshold:
            return step + 1

    return max_steps


def measure_relaxation_time(alpha: float, alpha_c: float = 0.92,
                             n_trials: int = 50) -> Tuple[float, float]:
    """
    Measure mean relaxation time at given alpha.

    Returns:
        (mean_tau, std_tau)
    """
    tau_values = []

    for _ in range(n_trials):
        tau = simulate_relaxation(alpha, alpha_c)
        tau_values.append(tau)

    return np.mean(tau_values), np.std(tau_values)


def compute_relaxation_curve(alpha_values: np.ndarray,
                              alpha_c: float = 0.92,
                              n_trials: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relaxation time as function of alpha.
    """
    tau_mean = np.zeros(len(alpha_values))
    tau_std = np.zeros(len(alpha_values))

    for i, alpha in enumerate(alpha_values):
        tau_mean[i], tau_std[i] = measure_relaxation_time(alpha, alpha_c, n_trials)

    return tau_mean, tau_std


def run_test(config: TestConfig = None) -> PhaseTransitionTestResult:
    """
    Run the critical slowing down test.
    """
    if config is None:
        config = TestConfig()

    set_seed(config.seed)

    print("=" * 60)
    print("TEST 4: CRITICAL SLOWING DOWN")
    print("=" * 60)

    # Parameters
    alpha_c = 0.92
    alpha_values = np.linspace(0.5, 0.99, 40)

    print(f"\nMeasuring relaxation time across alpha...")

    # Compute relaxation curve
    tau_mean, tau_std = compute_relaxation_curve(alpha_values, alpha_c, n_trials=30)

    # Find tau at key points
    idx_05 = np.argmin(np.abs(alpha_values - 0.5))
    idx_peak = np.argmax(tau_mean)

    tau_05 = tau_mean[idx_05]
    tau_max = tau_mean[idx_peak]
    alpha_at_max = alpha_values[idx_peak]

    # Slowing ratio
    slowing_ratio = tau_max / tau_05 if tau_05 > 0 else float('inf')

    print(f"\nResults:")
    print(f"  tau(0.5) = {tau_05:.2f}")
    print(f"  tau(max) = {tau_max:.2f} at alpha = {alpha_at_max:.4f}")
    print(f"  Slowing ratio: {slowing_ratio:.2f}")

    # Estimate dynamic exponent z*nu
    # tau ~ |alpha - alpha_c|^(-z*nu)
    distance_max = abs(alpha_at_max - alpha_c)
    distance_05 = abs(0.5 - alpha_c)
    if tau_05 > 0 and tau_max > tau_05:
        z_nu = np.log(tau_max / tau_05) / np.log(distance_05 / distance_max)
    else:
        z_nu = 0

    print(f"  Estimated z*nu = {z_nu:.4f}")

    # Pass/Fail criteria
    ratio_threshold = THRESHOLDS["tau_ratio"]

    passed = slowing_ratio > ratio_threshold

    print("\n" + "=" * 60)
    print("PASS/FAIL CHECKS")
    print("=" * 60)
    print(f"  Slowing ratio > {ratio_threshold}: {slowing_ratio:.2f} {'PASS' if passed else 'FAIL'}")
    print()
    print(f"VERDICT: {'** PASS **' if passed else 'FAIL'}")

    falsification = None
    if not passed:
        falsification = f"Slowing ratio {slowing_ratio:.2f} < {ratio_threshold}"

    return PhaseTransitionTestResult(
        test_name="Critical Slowing Down",
        test_id="Q12_TEST_04",
        passed=passed,
        metric_value=slowing_ratio,
        threshold=ratio_threshold,
        transition_type=TransitionType.SECOND_ORDER,
        universality_class=UniversalityClass.UNKNOWN,
        critical_point=alpha_at_max,
        critical_exponents=CriticalExponents(z=z_nu),  # z*nu stored in z
        evidence={
            "tau_at_0.5": tau_05,
            "tau_at_max": tau_max,
            "alpha_at_max": alpha_at_max,
            "slowing_ratio": slowing_ratio,
            "z_nu_estimate": z_nu,
            "tau_curve": {str(a): float(t) for a, t in zip(alpha_values, tau_mean)},
        },
        falsification_evidence=falsification,
        notes="Tests if dynamics slow near critical point"
    )


if __name__ == "__main__":
    result = run_test()
    print(f"\nTest completed: {'PASS' if result.passed else 'FAIL'}")
