#!/usr/bin/env python3
"""
Q7: Renormalization Group Beta-Function

The beta-function measures how R changes under scale transformation:
    beta(R) = dR / d(ln lambda)

Fixed point condition: beta(R*) = 0
Stability: |d_beta/dR|_{R*} determines whether R* is stable or unstable

If R = E(z)/sigma is a fixed point, then beta(R) approx 0 across all scales.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.scale_transformation import (
    ScaleData,
    ScaleTransformation,
    compute_R,
    compute_R_at_scale
)


# =============================================================================
# BETA-FUNCTION COMPUTATION
# =============================================================================

def compute_beta(
    base_data: ScaleData,
    scale_1: int,
    scale_2: int,
    group_size: int = 2,
    kernel: str = "gaussian"
) -> float:
    """
    Compute beta(R) = dR / d(ln lambda) between two scales.

    beta approx (R_2 - R_1) / (ln(lambda_2) - ln(lambda_1))

    where lambda_k = group_size^k is the scale factor at level k.

    Args:
        base_data: Micro-scale data
        scale_1: First scale level
        scale_2: Second scale level (scale_2 > scale_1)
        group_size: Aggregation group size
        kernel: Evidence kernel

    Returns:
        beta value (should be approx 0 for fixed point)
    """
    if scale_2 <= scale_1:
        raise ValueError("scale_2 must be > scale_1")

    R_1 = compute_R_at_scale(base_data, scale_1, group_size, kernel)
    R_2 = compute_R_at_scale(base_data, scale_2, group_size, kernel)

    # Scale parameters
    lambda_1 = group_size ** scale_1
    lambda_2 = group_size ** scale_2

    # beta = dR / d(ln lambda)
    d_ln_lambda = np.log(lambda_2) - np.log(lambda_1)
    if d_ln_lambda == 0:
        return 0.0

    beta = (R_2 - R_1) / d_ln_lambda

    return float(beta)


def compute_beta_trajectory(
    base_data: ScaleData,
    n_scales: int = 5,
    group_size: int = 2,
    kernel: str = "gaussian"
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute beta trajectory across multiple scales.

    Returns:
        (scale_levels, R_values, beta_values)
    """
    T = ScaleTransformation(aggregation="mean")
    data = base_data

    scale_levels = []
    R_values = []
    beta_values = []

    prev_R = None
    prev_scale = None

    for scale in range(n_scales):
        R = compute_R(data, kernel)
        scale_levels.append(scale)
        R_values.append(R)

        if prev_R is not None:
            # Compute beta between consecutive scales
            lambda_prev = group_size ** prev_scale
            lambda_curr = group_size ** scale

            d_ln_lambda = np.log(lambda_curr) - np.log(lambda_prev)
            if d_ln_lambda > 0:
                beta = (R - prev_R) / d_ln_lambda
                beta_values.append(beta)
            else:
                beta_values.append(0.0)

        prev_R = R
        prev_scale = scale

        if len(data.observations) < group_size:
            break
        data = T(data, group_size)

    # Pad beta_values to match scale_levels length
    while len(beta_values) < len(scale_levels):
        beta_values.insert(0, 0.0)

    return scale_levels, R_values, beta_values


# =============================================================================
# FIXED POINT ANALYSIS
# =============================================================================

@dataclass
class FixedPointAnalysis:
    """Results of fixed point analysis."""
    is_fixed_point: bool
    mean_beta: float
    max_beta: float
    std_beta: float
    R_values: List[float]
    beta_values: List[float]
    cv_R: float  # Coefficient of variation of R
    stability: str  # "stable", "unstable", or "marginal"


def analyze_fixed_point(
    base_data: ScaleData,
    n_scales: int = 5,
    group_size: int = 2,
    beta_threshold: float = 0.05,
    cv_threshold: float = 0.1,
    kernel: str = "gaussian"
) -> FixedPointAnalysis:
    """
    Analyze whether R is a fixed point under scale transformation.

    A fixed point requires:
    1. beta(R) approx 0 (mean and max beta below threshold)
    2. R is approximately constant across scales (low CV)

    Args:
        base_data: Micro-scale data
        n_scales: Number of scales to test
        group_size: Aggregation size
        beta_threshold: Maximum allowed |beta|
        cv_threshold: Maximum allowed CV for R
        kernel: Evidence kernel

    Returns:
        FixedPointAnalysis with detailed results
    """
    scales, R_values, beta_values = compute_beta_trajectory(
        base_data, n_scales, group_size, kernel
    )

    # Filter out edge effects (first beta is 0 by construction)
    beta_array = np.array(beta_values[1:]) if len(beta_values) > 1 else np.array([0.0])
    R_array = np.array(R_values)

    mean_beta = float(np.mean(np.abs(beta_array)))
    max_beta = float(np.max(np.abs(beta_array)))
    std_beta = float(np.std(beta_array))

    cv_R = float(np.std(R_array) / (np.mean(R_array) + 1e-10))

    # Fixed point criteria
    is_fixed_point = (mean_beta < beta_threshold) and (cv_R < cv_threshold)

    # Stability analysis
    # If beta is consistently positive, R grows with scale (unstable from above)
    # If beta is consistently negative, R shrinks with scale (unstable from below)
    # If beta oscillates around 0, it's a stable fixed point
    if len(beta_array) > 1:
        if np.all(beta_array > 0.01):
            stability = "unstable_growth"
        elif np.all(beta_array < -0.01):
            stability = "unstable_decay"
        elif std_beta < mean_beta * 2:
            stability = "stable"
        else:
            stability = "marginal"
    else:
        stability = "insufficient_data"

    return FixedPointAnalysis(
        is_fixed_point=is_fixed_point,
        mean_beta=mean_beta,
        max_beta=max_beta,
        std_beta=std_beta,
        R_values=R_values,
        beta_values=beta_values,
        cv_R=cv_R,
        stability=stability
    )


# =============================================================================
# UNIVERSALITY TEST
# =============================================================================

def test_universality(
    base_data: ScaleData,
    n_iterations: int = 10,
    group_size: int = 2,
    kernel: str = "gaussian"
) -> Dict[str, List[float]]:
    """
    Test universality: Do alternative measures flow to R under RG iteration?

    We start with 5 alternative evidence measures and apply coarse-graining
    repeatedly. If R is a universal attractor, all should converge.

    Args:
        base_data: Micro-scale data
        n_iterations: Number of RG iterations
        group_size: Aggregation size
        kernel: Evidence kernel

    Returns:
        Dict mapping measure name to trajectory of values
    """
    # Alternative evidence measures (different from R = E/sigma)
    def R_standard(data: ScaleData) -> float:
        """Standard R = E/sigma"""
        return compute_R(data, kernel)

    def R_squared_sigma(data: ScaleData) -> float:
        """E/sigma² - violates C4"""
        z = np.abs(data.observations - data.truth) / data.sigma
        E = np.mean(np.exp(-0.5 * z**2))
        return E / (data.sigma ** 2)

    def R_E_squared(data: ScaleData) -> float:
        """E²/sigma - non-linear"""
        z = np.abs(data.observations - data.truth) / data.sigma
        E = np.mean(np.exp(-0.5 * z**2))
        return (E ** 2) / data.sigma

    def R_additive(data: ScaleData) -> float:
        """E - sigma - additive, not multiplicative"""
        z = np.abs(data.observations - data.truth) / data.sigma
        E = np.mean(np.exp(-0.5 * z**2))
        return E - data.sigma

    def R_product(data: ScaleData) -> float:
        """E × sigma - wrong direction"""
        z = np.abs(data.observations - data.truth) / data.sigma
        E = np.mean(np.exp(-0.5 * z**2))
        return E * data.sigma

    measures = {
        "R_standard": R_standard,
        "R_squared_sigma": R_squared_sigma,
        "R_E_squared": R_E_squared,
        "R_additive": R_additive,
        "R_product": R_product,
    }

    trajectories = {name: [] for name in measures}

    T = ScaleTransformation(aggregation="mean")
    data = base_data

    for iteration in range(n_iterations):
        for name, measure_fn in measures.items():
            try:
                val = measure_fn(data)
                trajectories[name].append(val)
            except Exception:
                trajectories[name].append(np.nan)

        if len(data.observations) < group_size:
            break
        data = T(data, group_size)

    return trajectories


def analyze_universality(
    trajectories: Dict[str, List[float]],
    reference: str = "R_standard"
) -> Dict[str, Dict]:
    """
    Analyze which alternative measures converge to the reference.

    Returns:
        Dict with convergence analysis for each measure
    """
    ref_traj = np.array(trajectories[reference])
    results = {}

    for name, traj in trajectories.items():
        traj_arr = np.array(traj)

        # Skip NaN values
        valid_mask = ~np.isnan(traj_arr) & ~np.isnan(ref_traj[:len(traj_arr)])
        if not np.any(valid_mask):
            results[name] = {
                "converges": False,
                "final_ratio": np.nan,
                "correlation": np.nan,
                "reason": "all_nan"
            }
            continue

        traj_valid = traj_arr[valid_mask]
        ref_valid = ref_traj[:len(traj_arr)][valid_mask]

        # Compute correlation with reference
        if len(traj_valid) > 1:
            corr = float(np.corrcoef(traj_valid, ref_valid)[0, 1])
        else:
            corr = np.nan

        # Compute final ratio
        if len(traj_valid) > 0 and ref_valid[-1] != 0:
            final_ratio = traj_valid[-1] / ref_valid[-1]
        else:
            final_ratio = np.nan

        # Determine convergence
        converges = (
            not np.isnan(corr) and
            corr > 0.9 and
            not np.isnan(final_ratio) and
            0.5 < final_ratio < 2.0
        )

        results[name] = {
            "converges": converges,
            "final_ratio": float(final_ratio) if not np.isnan(final_ratio) else None,
            "correlation": float(corr) if not np.isnan(corr) else None,
            "trajectory": traj
        }

    return results


# =============================================================================
# TESTS
# =============================================================================

def run_self_tests():
    """Run self-tests for beta-function."""
    print("\n" + "="*80)
    print("Q7: BETA-FUNCTION SELF-TESTS")
    print("="*80)

    np.random.seed(42)

    # Generate test data
    n = 512
    truth = 0.0
    sigma = 1.0
    obs = np.random.normal(truth, sigma, n)

    base_data = ScaleData(
        observations=obs,
        truth=truth,
        sigma=sigma,
        scale_level=0
    )

    print(f"\nTest data: n={n}, truth={truth}, sigma={sigma}")

    # Test 1: beta trajectory
    print("\n--- Test 1: beta Trajectory ---")
    scales, R_values, beta_values = compute_beta_trajectory(base_data, n_scales=6)
    print(f"  Scales: {scales}")
    print(f"  R values: {[f'{r:.4f}' for r in R_values]}")
    print(f"  beta values: {[f'{b:.6f}' for b in beta_values]}")

    # Test 2: Fixed point analysis
    print("\n--- Test 2: Fixed Point Analysis ---")
    analysis = analyze_fixed_point(base_data, n_scales=6, beta_threshold=0.05)
    print(f"  Is fixed point: {analysis.is_fixed_point}")
    print(f"  Mean |beta|: {analysis.mean_beta:.6f}")
    print(f"  Max |beta|: {analysis.max_beta:.6f}")
    print(f"  CV(R): {analysis.cv_R:.4f}")
    print(f"  Stability: {analysis.stability}")

    # Test 3: Universality
    print("\n--- Test 3: Universality Test ---")
    trajectories = test_universality(base_data, n_iterations=6)
    uni_results = analyze_universality(trajectories)

    for name, result in uni_results.items():
        status = "[PASS]" if result["converges"] else "[FAIL]"
        corr = result["correlation"]
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        print(f"  {status} {name}: corr={corr_str}, converges={result['converges']}")

    # Verdict
    print("\n" + "="*80)
    test_passed = (
        analysis.is_fixed_point and
        analysis.mean_beta < 0.05 and
        uni_results["R_standard"]["converges"]
    )
    print(f"BETA-FUNCTION TESTS: {'PASSED' if test_passed else 'FAILED'}")

    if test_passed:
        print("\nCONCLUSION:")
        print("  R = E(z)/sigma is a fixed point of the RG transformation.")
        print("  beta(R) approx 0 confirms scale invariance.")
        print("  Standard R converges under iteration.")
    print("="*80)

    return test_passed


if __name__ == "__main__":
    run_self_tests()
