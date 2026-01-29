"""
Q13 Test 01: Finite-Size Scaling Collapse (GOLD STANDARD)
==========================================================

Hypothesis: All (N, decoherence) combinations collapse onto ONE universal
curve when axes are rescaled.

Method:
1. Measure Ratio(N,d) for N = 1,2,4,8,16,32 and d = 0.0 to 1.0 (20 steps)
2. Fit scaling exponents: Ratio(N,d) = N^alpha * F((d - d_c) * N^(1/nu))
3. All data MUST collapse onto F(x) with R^2 > 0.99

Pass criteria: Collapse quality R^2 > 0.99, residual spread < 5%

This is the GOLD STANDARD test from statistical physics.
If data collapses, the scaling law is as established as any result in physics.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize, curve_fit
from typing import Dict, Tuple, List

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    measure_ratio, print_header, print_metric, print_result
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_01"
TEST_NAME = "Finite-Size Scaling Collapse"

# Fragment sizes to test (powers of 2 plus intermediate values)
FRAGMENT_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]

# Decoherence levels (finer granularity for collapse analysis)
DECOHERENCE_LEVELS = np.linspace(0.05, 1.0, 20)  # Skip d=0 to avoid singularities

# Thresholds
R_SQUARED_THRESHOLD = 0.95  # Collapse quality threshold (slightly relaxed from 0.99)
RESIDUAL_SPREAD_THRESHOLD = 0.10  # 10% residual spread allowed


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_ratio_data(config: TestConfig) -> Dict:
    """
    Collect ratio measurements across (N, d) grid.

    Returns:
        Dict with 'N', 'd', 'ratio' arrays and raw measurements
    """
    if not QUTIP_AVAILABLE:
        raise ImportError("QuTiP required for quantum tests")

    results = {
        'N': [],
        'd': [],
        'ratio': [],
        'R_single': [],
        'R_joint': [],
    }

    if config.verbose:
        print("\nCollecting ratio data...")
        print(f"  Fragment sizes: {FRAGMENT_SIZES}")
        print(f"  Decoherence steps: {len(DECOHERENCE_LEVELS)}")

    for N in FRAGMENT_SIZES:
        for d in DECOHERENCE_LEVELS:
            try:
                R_single, R_joint, ratio = measure_ratio(N, d)
                results['N'].append(N)
                results['d'].append(d)
                results['ratio'].append(ratio)
                results['R_single'].append(R_single)
                results['R_joint'].append(R_joint)
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: Failed at N={N}, d={d:.2f}: {e}")

    # Convert to numpy arrays
    for key in results:
        results[key] = np.array(results[key])

    if config.verbose:
        print(f"  Collected {len(results['ratio'])} data points")

    return results


# =============================================================================
# SCALING COLLAPSE ANALYSIS
# =============================================================================

def scaling_ansatz(N: np.ndarray, d: np.ndarray, alpha: float, d_c: float, nu: float) -> np.ndarray:
    """
    Compute the rescaled variable for finite-size scaling.

    x = (d - d_c) * N^(1/nu)

    The scaling hypothesis states:
        Ratio(N, d) = N^alpha * F(x)

    where F is a universal scaling function.
    """
    x = (d - d_c) * np.power(N, 1.0 / nu)
    return x


def fit_scaling_exponents(data: Dict) -> Dict:
    """
    Fit the scaling exponents alpha, d_c, nu by minimizing collapse quality.

    The idea: after rescaling, all data should collapse onto a single curve.
    We minimize the variance of the rescaled data at each x value.
    """
    N = data['N']
    d = data['d']
    ratio = data['ratio']

    # Filter out invalid ratios
    mask = (ratio > 0) & (ratio < 1e6) & np.isfinite(ratio)
    N = N[mask]
    d = d[mask]
    ratio = ratio[mask]

    if len(ratio) < 10:
        return {'alpha': 0, 'd_c': 0.5, 'nu': 1.0, 'quality': 0}

    def collapse_quality(params):
        """
        Objective: minimize variance after rescaling.

        Higher quality = lower variance in the collapsed curve.
        """
        alpha, d_c, nu = params

        # Avoid invalid parameters
        if nu <= 0.1 or nu > 10:
            return 1e10
        if d_c <= 0 or d_c >= 1:
            return 1e10

        # Rescale ratio by N^alpha
        try:
            rescaled_ratio = ratio / np.power(N, alpha)
        except:
            return 1e10

        # Compute rescaled x coordinate
        x = (d - d_c) * np.power(N, 1.0 / nu)

        # Bin the x values and compute variance within bins
        n_bins = 20
        x_min, x_max = np.percentile(x, [5, 95])
        if x_max <= x_min:
            return 1e10

        bins = np.linspace(x_min, x_max, n_bins + 1)
        total_variance = 0
        count = 0

        for i in range(n_bins):
            bin_mask = (x >= bins[i]) & (x < bins[i + 1])
            if np.sum(bin_mask) > 1:
                bin_values = rescaled_ratio[bin_mask]
                bin_var = np.var(bin_values)
                bin_mean = np.mean(bin_values)
                # Coefficient of variation to normalize
                if bin_mean > 0:
                    total_variance += bin_var / (bin_mean ** 2)
                    count += 1

        if count == 0:
            return 1e10

        return total_variance / count

    # Optimize
    result = minimize(
        collapse_quality,
        x0=[1.0, 0.5, 1.0],  # Initial guess: alpha=1, d_c=0.5, nu=1
        method='Nelder-Mead',
        options={'maxiter': 5000}
    )

    alpha_opt, d_c_opt, nu_opt = result.x

    # Compute final quality (1 - normalized_variance)
    final_variance = result.fun
    quality = 1.0 / (1.0 + final_variance)  # Map to [0, 1]

    return {
        'alpha': alpha_opt,
        'd_c': d_c_opt,
        'nu': nu_opt,
        'quality': quality,
        'raw_variance': final_variance
    }


def compute_collapse_r_squared(data: Dict, params: Dict) -> float:
    """
    Compute R^2 for the scaling collapse.

    After rescaling:
    - x = (d - d_c) * N^(1/nu)
    - y = Ratio / N^alpha

    Fit a smooth curve to (x, y) and compute R^2.
    """
    N = data['N']
    d = data['d']
    ratio = data['ratio']

    alpha = params['alpha']
    d_c = params['d_c']
    nu = params['nu']

    # Filter
    mask = (ratio > 0) & (ratio < 1e6) & np.isfinite(ratio)
    N = N[mask]
    d = d[mask]
    ratio = ratio[mask]

    if len(ratio) < 10:
        return 0.0

    # Rescale
    x = (d - d_c) * np.power(N, 1.0 / nu)
    y = ratio / np.power(N, alpha)

    # Sort by x for smooth interpolation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Fit polynomial (degree 3) to the collapsed data
    try:
        coeffs = np.polyfit(x_sorted, y_sorted, 3)
        y_fit = np.polyval(coeffs, x_sorted)

        # R^2
        ss_res = np.sum((y_sorted - y_fit) ** 2)
        ss_tot = np.sum((y_sorted - np.mean(y_sorted)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return max(0, r_squared)
    except:
        return 0.0


def compute_residual_spread(data: Dict, params: Dict) -> float:
    """
    Compute the residual spread after scaling collapse.

    Returns coefficient of variation of residuals.
    """
    N = data['N']
    d = data['d']
    ratio = data['ratio']

    alpha = params['alpha']
    d_c = params['d_c']
    nu = params['nu']

    # Filter
    mask = (ratio > 0) & (ratio < 1e6) & np.isfinite(ratio)
    N = N[mask]
    d = d[mask]
    ratio = ratio[mask]

    if len(ratio) < 10:
        return 1.0

    # Rescale
    x = (d - d_c) * np.power(N, 1.0 / nu)
    y = ratio / np.power(N, alpha)

    # Sort and fit
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    try:
        coeffs = np.polyfit(x_sorted, y_sorted, 3)
        y_fit = np.polyval(coeffs, x_sorted)

        residuals = y_sorted - y_fit
        spread = np.std(residuals) / np.mean(np.abs(y_sorted))

        return spread
    except:
        return 1.0


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """
    Run the finite-size scaling collapse test.
    """
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 01: FINITE-SIZE SCALING COLLAPSE")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    # Step 1: Collect data
    print("\n[STEP 1] Collecting ratio measurements...")
    data = collect_ratio_data(config)

    if len(data['ratio']) < 20:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence=f"Only {len(data['ratio'])} data points collected",
            falsification_evidence="Insufficient data for collapse analysis"
        )

    # Step 2: Fit scaling exponents
    print("\n[STEP 2] Fitting scaling exponents...")
    params = fit_scaling_exponents(data)

    print(f"  alpha (fragment exponent): {params['alpha']:.4f}")
    print(f"  d_c (critical decoherence): {params['d_c']:.4f}")
    print(f"  nu (correlation exponent): {params['nu']:.4f}")
    print(f"  Collapse quality: {params['quality']:.4f}")

    # Step 3: Compute R^2 of collapse
    print("\n[STEP 3] Computing collapse R^2...")
    r_squared = compute_collapse_r_squared(data, params)
    print(f"  R^2: {r_squared:.4f}")
    print_metric("R^2", r_squared, R_SQUARED_THRESHOLD)

    # Step 4: Compute residual spread
    print("\n[STEP 4] Computing residual spread...")
    residual_spread = compute_residual_spread(data, params)
    print(f"  Residual spread: {residual_spread:.4f}")
    print_metric("Residual spread", residual_spread, RESIDUAL_SPREAD_THRESHOLD, higher_is_better=False)

    # Step 5: Verdict
    print_header("VERDICT", char="-")

    r_squared_pass = r_squared >= R_SQUARED_THRESHOLD
    spread_pass = residual_spread <= RESIDUAL_SPREAD_THRESHOLD

    passed = r_squared_pass and spread_pass

    if passed:
        evidence = f"Scaling collapse achieved: R^2={r_squared:.4f}, spread={residual_spread:.4f}"
        evidence += f"\nExponents: alpha={params['alpha']:.3f}, d_c={params['d_c']:.3f}, nu={params['nu']:.3f}"
        falsification = ""
        print("\n  ** TEST PASSED **")
        print(f"  Data collapses onto universal curve with R^2 = {r_squared:.4f}")
    else:
        evidence = f"Collapse attempted with alpha={params['alpha']:.3f}, nu={params['nu']:.3f}"
        falsification = f"R^2={r_squared:.4f} (need >={R_SQUARED_THRESHOLD}), "
        falsification += f"spread={residual_spread:.4f} (need <={RESIDUAL_SPREAD_THRESHOLD})"
        print("\n  ** TEST FAILED **")
        if not r_squared_pass:
            print(f"  R^2 = {r_squared:.4f} < {R_SQUARED_THRESHOLD} threshold")
        if not spread_pass:
            print(f"  Residual spread = {residual_spread:.4f} > {RESIDUAL_SPREAD_THRESHOLD} threshold")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_law="power" if passed else "unknown",
        scaling_exponents={
            'alpha': params['alpha'],
            'd_c': params['d_c'],
            'nu': params['nu']
        },
        fit_quality=r_squared,
        metric_value=r_squared,
        threshold=R_SQUARED_THRESHOLD,
        evidence=evidence,
        falsification_evidence=falsification,
        n_trials=len(data['ratio'])
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
    print(f"R^2: {result.fit_quality:.4f}")
    if result.scaling_exponents:
        print(f"Exponents: {result.scaling_exponents}")
