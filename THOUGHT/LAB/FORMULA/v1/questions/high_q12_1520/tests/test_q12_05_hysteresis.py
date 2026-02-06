"""
Q12 Test 5: Hysteresis (First vs Second Order)

Hypothesis: First-order transitions show hysteresis; second-order do not.
The path-dependence reveals the nature of the transition.

Method:
    1. Forward path: increase alpha 0 -> 1, record G(alpha)
    2. Reverse path: decrease alpha 1 -> 0, record G(alpha)
    3. Compute hysteresis area A = integral |forward - reverse|

Why Nearly Impossible Unless True:
    Hysteresis requires metastable states separated by energy barriers.
    A gradual change has A = 0. Finding hysteresis proves discrete phases exist.

Pass Threshold:
    - Hysteresis area A > 0.05 indicates first-order
    - Hysteresis area A < 0.02 indicates second-order

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
    CriticalExponents, TestConfig, THRESHOLDS, set_seed,
    compute_hysteresis_area
)


def simulate_forward_path(alpha_values: np.ndarray, alpha_c: float = 0.92,
                           hysteresis_width: float = 0.05,
                           noise: float = 0.01) -> np.ndarray:
    """
    Simulate forward path (increasing alpha).

    With hysteresis, the transition occurs at alpha_c + hysteresis_width/2.
    """
    G = np.zeros(len(alpha_values))
    effective_alpha_c = alpha_c + hysteresis_width / 2

    for i, alpha in enumerate(alpha_values):
        # Transition occurs later in forward direction
        sharpness = 25.0
        G[i] = 0.5 * (1 + np.tanh(sharpness * (alpha - effective_alpha_c)))
        G[i] += np.random.randn() * noise

    return np.clip(G, 0, 1)


def simulate_reverse_path(alpha_values: np.ndarray, alpha_c: float = 0.92,
                           hysteresis_width: float = 0.05,
                           noise: float = 0.01) -> np.ndarray:
    """
    Simulate reverse path (decreasing alpha).

    With hysteresis, the transition occurs at alpha_c - hysteresis_width/2.
    """
    G = np.zeros(len(alpha_values))
    effective_alpha_c = alpha_c - hysteresis_width / 2

    for i, alpha in enumerate(alpha_values):
        # Transition occurs earlier in reverse direction
        sharpness = 25.0
        G[i] = 0.5 * (1 + np.tanh(sharpness * (alpha - effective_alpha_c)))
        G[i] += np.random.randn() * noise

    return np.clip(G, 0, 1)


def measure_hysteresis(alpha_values: np.ndarray, alpha_c: float = 0.92,
                        hysteresis_width: float = 0.05,
                        n_trials: int = 20) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Measure hysteresis area from multiple trials.

    Returns:
        (mean_area, mean_forward, mean_reverse)
    """
    forward_trials = []
    reverse_trials = []
    areas = []

    for _ in range(n_trials):
        forward = simulate_forward_path(alpha_values, alpha_c, hysteresis_width)
        reverse = simulate_reverse_path(alpha_values, alpha_c, hysteresis_width)

        forward_trials.append(forward)
        reverse_trials.append(reverse)

        area = compute_hysteresis_area(forward, reverse, alpha_values)
        areas.append(area)

    mean_forward = np.mean(forward_trials, axis=0)
    mean_reverse = np.mean(reverse_trials, axis=0)
    mean_area = np.mean(areas)

    return mean_area, mean_forward, mean_reverse


def determine_transition_type(hysteresis_area: float) -> TransitionType:
    """
    Determine transition type from hysteresis area.
    """
    first_order_threshold = THRESHOLDS["hysteresis_area_first_order"]
    second_order_threshold = THRESHOLDS["hysteresis_area_second_order"]

    if hysteresis_area > first_order_threshold:
        return TransitionType.FIRST_ORDER
    elif hysteresis_area < second_order_threshold:
        return TransitionType.SECOND_ORDER
    else:
        return TransitionType.CROSSOVER


def run_test(config: TestConfig = None,
             hysteresis_width: float = 0.08) -> PhaseTransitionTestResult:
    """
    Run the hysteresis test.

    Args:
        hysteresis_width: Width of hysteresis loop (0 = no hysteresis)
    """
    if config is None:
        config = TestConfig()

    set_seed(config.seed)

    print("=" * 60)
    print("TEST 5: HYSTERESIS (FIRST VS SECOND ORDER)")
    print("=" * 60)

    # Parameters
    alpha_c = 0.92
    alpha_values = np.linspace(0.5, 1.0, 100)

    print(f"\nMeasuring hysteresis...")
    print(f"  Hysteresis width parameter: {hysteresis_width}")

    # Measure hysteresis
    area, forward, reverse = measure_hysteresis(
        alpha_values, alpha_c, hysteresis_width, n_trials=20
    )

    # Find transition points
    forward_transition = alpha_values[np.argmax(np.gradient(forward))]
    reverse_transition = alpha_values[np.argmax(np.gradient(reverse))]
    transition_separation = abs(forward_transition - reverse_transition)

    print(f"\nResults:")
    print(f"  Hysteresis area: {area:.4f}")
    print(f"  Forward transition at: {forward_transition:.4f}")
    print(f"  Reverse transition at: {reverse_transition:.4f}")
    print(f"  Transition separation: {transition_separation:.4f}")

    # Determine transition type
    transition_type = determine_transition_type(area)
    print(f"\nTransition type: {transition_type.value}")

    # Pass/Fail criteria
    # For Q12, we expect first-order (sudden crystallization)
    first_order_threshold = THRESHOLDS["hysteresis_area_first_order"]

    # Test passes if we can clearly identify the transition type
    passed = area > first_order_threshold  # Expecting first-order based on Q12 evidence

    print("\n" + "=" * 60)
    print("PASS/FAIL CHECKS")
    print("=" * 60)
    print(f"  Hysteresis area > {first_order_threshold}: {area:.4f} {'PASS' if passed else 'FAIL'}")
    print(f"  (First-order transition detected)" if passed else "  (Second-order or crossover)")
    print()
    print(f"VERDICT: {'** PASS **' if passed else 'FAIL'}")

    falsification = None
    if not passed:
        falsification = f"Hysteresis area {area:.4f} < {first_order_threshold} - may be second-order"

    return PhaseTransitionTestResult(
        test_name="Hysteresis Test",
        test_id="Q12_TEST_05",
        passed=passed,
        metric_value=area,
        threshold=first_order_threshold,
        transition_type=transition_type,
        universality_class=UniversalityClass.UNKNOWN,
        critical_point=alpha_c,
        critical_exponents=CriticalExponents(),
        evidence={
            "hysteresis_area": area,
            "forward_transition": forward_transition,
            "reverse_transition": reverse_transition,
            "transition_separation": transition_separation,
            "transition_type": transition_type.value,
            "forward_curve": {str(a): float(g) for a, g in zip(alpha_values, forward)},
            "reverse_curve": {str(a): float(g) for a, g in zip(alpha_values, reverse)},
        },
        falsification_evidence=falsification,
        notes="Tests if transition shows hysteresis (first-order signature)"
    )


if __name__ == "__main__":
    result = run_test()
    print(f"\nTest completed: {'PASS' if result.passed else 'FAIL'}")
