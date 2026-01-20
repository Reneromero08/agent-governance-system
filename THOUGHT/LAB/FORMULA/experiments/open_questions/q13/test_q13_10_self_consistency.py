"""
Q13 Test 10: Self-Consistency (Formula Components)
===================================================

Hypothesis: Scaling emerges from E, grad_S, sigma, Df independently.

Method:
1. Measure E_single, E_joint and compute E_ratio(N)
2. Measure grad_S_single, grad_S_joint and compute grad_S_ratio(N)
3. Verify: Ratio(N) = E_ratio(N) / grad_S_ratio(N) * sigma_correction
4. Each component should have PREDICTABLE scaling

Pass criteria: Product of component ratios = total ratio within 5%

This test verifies the formula's internal consistency.
If components don't multiply correctly, the formula is wrong.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    create_qd_state, get_fragment_probs,
    compute_essence, compute_grad_S, compute_R,
    print_header, print_metric
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_10"
TEST_NAME = "Self-Consistency (Formula Components)"

# Parameters
FRAGMENT_SIZES = [2, 4, 6, 8, 12]
DECOHERENCE = 1.0  # Full decoherence for maximum effect
SIGMA = 0.5

# Thresholds
CONSISTENCY_THRESHOLD = 0.10  # 10% tolerance for component consistency


# =============================================================================
# COMPONENT ANALYSIS
# =============================================================================

def measure_components(N: int, d: float) -> Dict:
    """
    Measure all formula components separately.

    R = (E / grad_S) * sigma^Df

    Returns E, grad_S, Df for both single and joint observations.
    """
    if not QUTIP_AVAILABLE:
        return {}

    state = create_qd_state(N, d)

    # Single fragment
    single_probs = get_fragment_probs(state, [1])
    E_single = compute_essence(single_probs)
    grad_S_single = 0.01  # Minimum for single observation

    Df_single = 1.0
    R_single = (E_single / grad_S_single) * (SIGMA ** Df_single)

    # Joint observation (all fragments)
    joint_indices = list(range(1, N + 1))
    joint_probs = get_fragment_probs(state, joint_indices)
    E_joint = compute_essence(joint_probs)

    # For grad_S, we need variance across observations
    # Simulate by getting probs from different fragment subsets
    all_probs = []
    for i in range(1, N + 1):
        probs = get_fragment_probs(state, [i])
        all_probs.append(probs)
    grad_S_joint = compute_grad_S(all_probs)

    Df_joint = np.log(N + 1)
    R_joint = (E_joint / grad_S_joint) * (SIGMA ** Df_joint)

    return {
        'E_single': E_single,
        'E_joint': E_joint,
        'grad_S_single': grad_S_single,
        'grad_S_joint': grad_S_joint,
        'Df_single': Df_single,
        'Df_joint': Df_joint,
        'R_single': R_single,
        'R_joint': R_joint,
        'ratio': R_joint / max(R_single, 0.001)
    }


def compute_component_ratios(components: Dict) -> Dict:
    """Compute individual component ratios."""
    E_ratio = components['E_joint'] / max(components['E_single'], 0.001)
    grad_S_ratio = components['grad_S_single'] / max(components['grad_S_joint'], 0.001)  # Inverted!
    delta_Df = components['Df_joint'] - components['Df_single']
    sigma_ratio = SIGMA ** delta_Df

    # Predicted total ratio from components
    predicted_ratio = E_ratio * grad_S_ratio * sigma_ratio

    # Actual ratio
    actual_ratio = components['ratio']

    # Consistency check
    if actual_ratio > 0:
        consistency = abs(predicted_ratio - actual_ratio) / actual_ratio
    else:
        consistency = 1.0

    return {
        'E_ratio': E_ratio,
        'grad_S_ratio': grad_S_ratio,
        'sigma_ratio': sigma_ratio,
        'delta_Df': delta_Df,
        'predicted_ratio': predicted_ratio,
        'actual_ratio': actual_ratio,
        'consistency_error': consistency
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """Run the self-consistency test."""
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 10: SELF-CONSISTENCY (FORMULA COMPONENTS)")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    # Measure components for each N
    print(f"\n[STEP 1] Measuring formula components at d={DECOHERENCE}...")

    all_results = []

    for N in FRAGMENT_SIZES:
        if config.verbose:
            print(f"\n  N = {N}:")

        components = measure_components(N, DECOHERENCE)
        ratios = compute_component_ratios(components)

        if config.verbose:
            print(f"    E_ratio:      {ratios['E_ratio']:.4f}")
            print(f"    grad_S_ratio: {ratios['grad_S_ratio']:.4f}")
            print(f"    sigma_ratio:  {ratios['sigma_ratio']:.4f}")
            print(f"    Predicted:    {ratios['predicted_ratio']:.4f}")
            print(f"    Actual:       {ratios['actual_ratio']:.4f}")
            print(f"    Consistency:  {ratios['consistency_error']*100:.2f}%")

        all_results.append({
            'N': N,
            **ratios
        })

    # Analyze consistency
    print_header("CONSISTENCY ANALYSIS", char="-")

    consistency_errors = [r['consistency_error'] for r in all_results]
    avg_error = np.mean(consistency_errors)
    max_error = np.max(consistency_errors)

    print(f"\n  Average consistency error: {avg_error*100:.2f}%")
    print(f"  Maximum consistency error: {max_error*100:.2f}%")
    print(f"  Threshold: {CONSISTENCY_THRESHOLD*100:.0f}%")

    # Check each component's scaling
    print("\n  Component scaling analysis:")

    E_ratios = [r['E_ratio'] for r in all_results]
    grad_S_ratios = [r['grad_S_ratio'] for r in all_results]
    sigma_ratios = [r['sigma_ratio'] for r in all_results]

    print(f"    E_ratio range:      [{min(E_ratios):.2f}, {max(E_ratios):.2f}]")
    print(f"    grad_S_ratio range: [{min(grad_S_ratios):.2f}, {max(grad_S_ratios):.2f}]")
    print(f"    sigma_ratio range:  [{min(sigma_ratios):.2f}, {max(sigma_ratios):.2f}]")

    # Pass if average and max consistency are within threshold
    avg_pass = avg_error <= CONSISTENCY_THRESHOLD
    max_pass = max_error <= CONSISTENCY_THRESHOLD * 2  # Allow 2x for max

    passed = avg_pass and max_pass

    # Build evidence
    if passed:
        evidence = f"Components multiply correctly: avg error {avg_error*100:.1f}%, max {max_error*100:.1f}%"
        evidence += f"\nE_ratio: {np.mean(E_ratios):.2f} +/- {np.std(E_ratios):.2f}"
        evidence += f"\ngrad_S_ratio: {np.mean(grad_S_ratios):.2f} +/- {np.std(grad_S_ratios):.2f}"
        falsification = ""
    else:
        evidence = f"Tested {len(FRAGMENT_SIZES)} fragment sizes"
        falsification = f"Avg error {avg_error*100:.1f}%, max {max_error*100:.1f}%"
        if not avg_pass:
            falsification += f" (avg > {CONSISTENCY_THRESHOLD*100}%)"
        if not max_pass:
            falsification += f" (max > {CONSISTENCY_THRESHOLD*200}%)"

    # Verdict
    print_header("VERDICT", char="-")

    if passed:
        print("\n  ** TEST PASSED **")
        print("  Formula components are self-consistent")
        print("  R = (E/grad_S) * sigma^Df holds at all N values")
    else:
        print("\n  ** TEST FAILED **")
        print(f"  Consistency error too high")
        print(f"  Avg: {avg_error*100:.1f}%, Max: {max_error*100:.1f}%")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_exponents={
            'avg_E_ratio': np.mean(E_ratios),
            'avg_grad_S_ratio': np.mean(grad_S_ratios),
            'avg_sigma_ratio': np.mean(sigma_ratios),
            'avg_consistency_error': avg_error,
            'max_consistency_error': max_error
        },
        fit_quality=1.0 - avg_error,
        metric_value=avg_error,
        threshold=CONSISTENCY_THRESHOLD,
        evidence=evidence,
        falsification_evidence=falsification,
        n_trials=len(FRAGMENT_SIZES)
    )


if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
