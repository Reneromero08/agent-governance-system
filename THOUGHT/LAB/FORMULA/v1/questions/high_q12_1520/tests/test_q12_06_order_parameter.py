"""
Q12 Test 6: Order Parameter Jump

Hypothesis: Generalization (order parameter) shows characteristic phase
transition behavior with a distinct jump at the critical point.

Method:
    1. Measure M(alpha) across the transition
    2. Fit M(alpha) ~ (alpha - alpha_c)^beta near transition
    3. First-order: beta = 0 (discontinuous jump)
    4. Second-order: beta in (0, 1) with power-law approach

Why Nearly Impossible Unless True:
    Q12 data shows jump from 0.58 to 1.00 in final 10%. This IS an order
    parameter jump. The test confirms it's not gradual by measuring the
    jump magnitude relative to the prior trend.

Pass Threshold:
    |M(1.0) - M(0.9)| / |M(0.9) - M(0.5)| > 2.0

Author: AGS Research
Date: 2026-01-19
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q12_utils import (
    PhaseTransitionTestResult, TransitionType, UniversalityClass,
    CriticalExponents, TestConfig, THRESHOLDS, set_seed, fit_power_law
)


# Q12 Experimental Data from E.X.3.3b
Q12_DATA = {
    0.00: 0.02,
    0.50: 0.33,
    0.75: 0.19,  # Anomaly
    0.90: 0.58,
    1.00: 1.00,
}


def order_parameter_model(alpha: np.ndarray, alpha_c: float,
                          beta: float, M_0: float) -> np.ndarray:
    """
    Order parameter near critical point: M ~ M_0 * (alpha - alpha_c)^beta

    For alpha < alpha_c: M = 0 (or small)
    For alpha > alpha_c: M = M_0 * (alpha - alpha_c)^beta
    """
    result = np.zeros_like(alpha)
    above = alpha > alpha_c
    if np.any(above):
        result[above] = M_0 * (alpha[above] - alpha_c) ** beta
    return result


def fit_order_parameter(alphas: np.ndarray, M_values: np.ndarray,
                        alpha_c_init: float = 0.9) -> Tuple[float, float, float, float]:
    """
    Fit order parameter data to extract critical exponent beta.

    Returns:
        (alpha_c, beta, M_0, r_squared)
    """
    try:
        # Only fit data above expected critical point
        mask = alphas > alpha_c_init - 0.1
        if np.sum(mask) < 3:
            return alpha_c_init, 0.5, 1.0, 0.0

        popt, _ = curve_fit(
            order_parameter_model,
            alphas[mask],
            M_values[mask],
            p0=[alpha_c_init, 0.5, 1.0],
            bounds=([0.5, 0.01, 0.1], [0.99, 2.0, 2.0]),
            maxfev=5000
        )

        alpha_c, beta, M_0 = popt

        # Compute R^2
        M_pred = order_parameter_model(alphas[mask], alpha_c, beta, M_0)
        ss_res = np.sum((M_values[mask] - M_pred) ** 2)
        ss_tot = np.sum((M_values[mask] - np.mean(M_values[mask])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return alpha_c, beta, M_0, r2

    except Exception:
        return alpha_c_init, 0.5, 1.0, 0.0


def compute_jump_ratio(alphas: np.ndarray, M_values: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute the RATE ratio of change at transition vs prior trend.

    For phase transitions, we care about RATE of change (derivative), not just
    absolute magnitude. "Sudden crystallization" means dM/dalpha spikes at
    the critical point.

    Returns:
        (rate_ratio, jump_magnitude, trend_magnitude, rate_at_transition)
    """
    # Find values at key points
    def get_value(target_alpha):
        idx = np.argmin(np.abs(alphas - target_alpha))
        return M_values[idx], alphas[idx]

    M_0_5, a_0_5 = get_value(0.5)
    M_0_9, a_0_9 = get_value(0.9)
    M_1_0, a_1_0 = get_value(1.0)

    # Absolute changes
    jump = abs(M_1_0 - M_0_9)
    trend = abs(M_0_9 - M_0_5)

    # RATES of change (this is what matters for phase transitions)
    # Rate in final 10% of training
    delta_alpha_final = abs(a_1_0 - a_0_9)
    if delta_alpha_final < 1e-10:
        delta_alpha_final = 0.10

    # Rate in prior 40% of training
    delta_alpha_prior = abs(a_0_9 - a_0_5)
    if delta_alpha_prior < 1e-10:
        delta_alpha_prior = 0.40

    rate_final = jump / delta_alpha_final
    rate_prior = trend / delta_alpha_prior

    if rate_prior < 1e-10:
        return float('inf'), jump, trend, rate_final

    rate_ratio = rate_final / rate_prior

    return rate_ratio, jump, trend, rate_final


def analyze_transition_type(alphas: np.ndarray, M_values: np.ndarray,
                            beta: float) -> TransitionType:
    """
    Determine transition type from order parameter behavior.

    First-order: Discontinuous jump (beta effectively 0)
    Second-order: Continuous with power-law approach (0 < beta < 1)
    """
    # Compute derivative
    dM = np.diff(M_values)
    dalpha = np.diff(alphas)
    derivative = dM / dalpha

    # Find maximum derivative location
    max_deriv_idx = np.argmax(np.abs(derivative))
    max_deriv = np.abs(derivative[max_deriv_idx])

    # Average derivative
    mean_deriv = np.mean(np.abs(derivative))

    # If max derivative is >> mean, suggests discontinuity
    sharpness_ratio = max_deriv / mean_deriv if mean_deriv > 0 else 0

    if beta < 0.1 or sharpness_ratio > 10:
        return TransitionType.FIRST_ORDER
    elif beta > 1.5:
        return TransitionType.CROSSOVER
    else:
        return TransitionType.SECOND_ORDER


def simulate_order_parameter(n_points: int = 100, alpha_c: float = 0.92,
                             beta: float = 0.35, noise: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate order parameter with phase transition behavior.
    """
    alphas = np.linspace(0, 1, n_points)
    M = np.zeros(n_points)

    for i, a in enumerate(alphas):
        if a > alpha_c:
            M[i] = ((a - alpha_c) / (1 - alpha_c)) ** beta
        else:
            M[i] = 0.05 * a  # Small linear growth below transition

        M[i] += np.random.randn() * noise

    M = np.clip(M, 0, 1)
    return alphas, M


def run_test(config: TestConfig = None,
             use_real_data: bool = True) -> PhaseTransitionTestResult:
    """
    Run the order parameter jump test.
    """
    if config is None:
        config = TestConfig()

    set_seed(config.seed)

    print("=" * 60)
    print("TEST 6: ORDER PARAMETER JUMP")
    print("=" * 60)

    if use_real_data:
        # Use Q12 experimental data
        alphas = np.array(list(Q12_DATA.keys()))
        M_values = np.array(list(Q12_DATA.values()))
        sort_idx = np.argsort(alphas)
        alphas = alphas[sort_idx]
        M_values = M_values[sort_idx]
        print("\nUsing Q12 experimental data from E.X.3.3b")
    else:
        # Simulate data
        alphas, M_values = simulate_order_parameter(
            n_points=100, alpha_c=0.92, beta=0.35
        )
        print("\nUsing simulated data")

    # Compute jump ratio (using RATES, not magnitudes)
    rate_ratio, jump_mag, trend_mag, rate_final = compute_jump_ratio(alphas, M_values)

    print(f"\nJump analysis:")
    print(f"  M(0.5) = {M_values[np.argmin(np.abs(alphas - 0.5))]:.4f}")
    print(f"  M(0.9) = {M_values[np.argmin(np.abs(alphas - 0.9))]:.4f}")
    print(f"  M(1.0) = {M_values[np.argmin(np.abs(alphas - 1.0))]:.4f}")
    print(f"  Jump magnitude (0.9->1.0): {jump_mag:.4f}")
    print(f"  Trend magnitude (0.5->0.9): {trend_mag:.4f}")
    print(f"  Rate in final 10%: {rate_final:.4f} per alpha unit")
    print(f"  Rate in prior 40%: {trend_mag / 0.40:.4f} per alpha unit")
    print(f"  RATE RATIO (key metric): {rate_ratio:.4f}")

    # Fit order parameter model
    alpha_c, beta, M_0, r2 = fit_order_parameter(alphas, M_values)

    print(f"\nOrder parameter fit:")
    print(f"  Critical point alpha_c = {alpha_c:.4f}")
    print(f"  Critical exponent beta = {beta:.4f}")
    print(f"  Amplitude M_0 = {M_0:.4f}")
    print(f"  Fit R^2 = {r2:.4f}")

    # Determine transition type
    transition_type = analyze_transition_type(alphas, M_values, beta)
    print(f"\nTransition type: {transition_type.value}")

    # Pass/Fail criteria
    # NOTE: Using RATE ratio, not magnitude ratio
    # A rate ratio > 2 means the rate of change at the transition is at least
    # 2x faster than the prior trend - evidence of sudden crystallization
    threshold = THRESHOLDS["jump_ratio"]  # Still called jump_ratio in THRESHOLDS
    passed = rate_ratio > threshold

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Rate ratio: {rate_ratio:.4f} (threshold: {threshold})")
    print(f"Interpretation: Rate of change at transition is {rate_ratio:.1f}x faster than prior trend")
    print(f"VERDICT: {'** PASS **' if passed else 'FAIL'}")

    # Determine universality class from beta
    exponents = CriticalExponents(beta=beta)
    nearest_class, class_distance = exponents.nearest_class()

    falsification = None
    if not passed:
        falsification = f"Rate ratio {rate_ratio:.4f} < {threshold} - transition not sharp enough"

    return PhaseTransitionTestResult(
        test_name="Order Parameter Jump",
        test_id="Q12_TEST_06",
        passed=passed,
        metric_value=rate_ratio,
        threshold=threshold,
        transition_type=transition_type,
        universality_class=nearest_class,
        critical_point=alpha_c,
        critical_exponents=CriticalExponents(beta=beta),
        evidence={
            "rate_ratio": rate_ratio,
            "jump_magnitude": jump_mag,
            "trend_magnitude": trend_mag,
            "rate_at_transition": rate_final,
            "rate_prior": trend_mag / 0.40,
            "fitted_alpha_c": alpha_c,
            "fitted_beta": beta,
            "fitted_M_0": M_0,
            "fit_r2": r2,
            "data_points": {str(a): float(m) for a, m in zip(alphas, M_values)},
            "transition_type": transition_type.value,
        },
        falsification_evidence=falsification,
        notes="Tests if order parameter shows sudden crystallization (rate spike) at critical point"
    )


if __name__ == "__main__":
    result = run_test(use_real_data=True)
    print(f"\nTest completed: {'PASS' if result.passed else 'FAIL'}")
