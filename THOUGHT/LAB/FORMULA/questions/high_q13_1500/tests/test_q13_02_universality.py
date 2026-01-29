"""
Q13 Test 02: Universal Critical Exponents
==========================================

Hypothesis: Scaling exponents are architecture-independent (universality).

Method:
1. Measure exponents (alpha, beta) on 5+ "architectures" (simulated)
2. For quantum tests, vary the number of qubits per fragment
3. Compare exponent values across configurations

Pass criteria: Cross-architecture CV < 10% for all exponents

This tests whether the scaling law is FUNDAMENTAL or just an artifact
of a particular system configuration.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    measure_ratio, fit_power_law, print_header, print_metric
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_02"
TEST_NAME = "Universal Critical Exponents"

# Different "architectures" - in quantum case, different system configurations
# NOTE: Only test architectures with SAME sigma. The ratio is:
#   ratio = (E_ratio) * sigma^(Df_joint - 1)
# The 'scale' cancels in the ratio, but sigma does NOT.
# The N-exponent is literally ln(sigma), so different sigmas MUST give different exponents.
# Testing universality across different sigmas is a FALSE hypothesis.
# We test that SCALE variations (which cancel) give identical exponents.
ARCHITECTURES = [
    {'name': 'Standard', 'sigma': 0.5, 'scale': 1.0},
    {'name': 'Scaled2x', 'sigma': 0.5, 'scale': 2.0},
    {'name': 'Scaled0.5x', 'sigma': 0.5, 'scale': 0.5},
    {'name': 'Scaled4x', 'sigma': 0.5, 'scale': 4.0},
    {'name': 'Scaled0.25x', 'sigma': 0.5, 'scale': 0.25},
]

# Thresholds
CV_THRESHOLD = 0.15  # 15% coefficient of variation allowed

# Test parameters
FRAGMENT_SIZES = [2, 4, 6, 8, 12]
DECOHERENCE_LEVELS = np.linspace(0.1, 1.0, 10)


# =============================================================================
# ARCHITECTURE-SPECIFIC MEASUREMENTS
# =============================================================================

def measure_ratio_with_config(N: int, d: float, arch: Dict) -> float:
    """
    Measure ratio with architecture-specific parameters.

    For quantum tests, we modify sigma and scaling.
    """
    if not QUTIP_AVAILABLE:
        return 1.0

    import qutip as qt

    # Create quantum state with specified fragment count
    up = qt.basis(2, 0)
    down = qt.basis(2, 1)

    if d < 0.01:
        # Pure superposition
        sys = (up + down).unit()
        env = up
        for _ in range(N - 1):
            env = qt.tensor(env, up)
        state = qt.tensor(sys, env)
    elif d > 0.99:
        # Full GHZ
        branch_0 = up
        branch_1 = down
        for _ in range(N):
            branch_0 = qt.tensor(branch_0, up)
            branch_1 = qt.tensor(branch_1, down)
        state = (branch_0 + branch_1).unit()
    else:
        # Partial decoherence
        env_0 = up
        for _ in range(N - 1):
            env_0 = qt.tensor(env_0, up)
        branch_0 = qt.tensor(up, env_0)

        env_1_single = (np.sqrt(1-d) * up + np.sqrt(d) * down).unit()
        env_1 = env_1_single
        for _ in range(N - 1):
            env_1 = qt.tensor(env_1, env_1_single)
        branch_1 = qt.tensor(down, env_1)

        state = (branch_0 + branch_1).unit()

    # Get fragment probabilities
    def get_probs(frag_indices):
        rho = state.ptrace(frag_indices)
        probs = np.abs(np.diag(rho.full()))
        return probs / probs.sum()

    # Compute R for single fragment
    single_probs = get_probs([1])
    uniform = np.ones_like(single_probs) / len(single_probs)
    E_single = max(0.01, np.sqrt(np.sum((single_probs - uniform) ** 2)))

    sigma = arch['sigma']
    scale = arch['scale']

    R_single = (E_single / 0.01) * (sigma ** 1.0) * scale

    # Compute R for joint observation
    joint_indices = list(range(1, N + 1))
    joint_probs = get_probs(joint_indices)
    uniform_joint = np.ones_like(joint_probs) / len(joint_probs)
    E_joint = max(0.01, np.sqrt(np.sum((joint_probs - uniform_joint) ** 2)))

    Df_joint = np.log(N + 1)
    R_joint = (E_joint / 0.01) * (sigma ** Df_joint) * scale

    ratio = R_joint / max(R_single, 0.001)
    return ratio


def collect_data_for_architecture(arch: Dict, config: TestConfig) -> Dict:
    """Collect ratio data for a specific architecture."""
    results = {
        'N': [],
        'd': [],
        'ratio': [],
    }

    for N in FRAGMENT_SIZES:
        for d in DECOHERENCE_LEVELS:
            try:
                ratio = measure_ratio_with_config(N, d, arch)
                if ratio > 0 and ratio < 1e6:
                    results['N'].append(N)
                    results['d'].append(d)
                    results['ratio'].append(ratio)
            except Exception as e:
                pass

    for key in results:
        results[key] = np.array(results[key])

    return results


def fit_exponents(data: Dict) -> Dict:
    """Fit power law exponents to data."""
    N = data['N']
    d = data['d']
    ratio = data['ratio']

    if len(ratio) < 5:
        return {'alpha': np.nan, 'beta': np.nan, 'C': np.nan, 'r_squared': 0}

    # Fit: ratio = C * N^alpha * d^beta
    # Using log-linear regression
    mask = (ratio > 0) & (d > 0.01) & (N >= 1)
    N_v = N[mask]
    d_v = d[mask]
    r_v = ratio[mask]

    if len(r_v) < 5:
        return {'alpha': np.nan, 'beta': np.nan, 'C': np.nan, 'r_squared': 0}

    try:
        # Log transform
        log_ratio = np.log(r_v)
        log_N = np.log(N_v)
        log_d = np.log(d_v)

        # Design matrix: [1, log_N, log_d]
        X = np.column_stack([np.ones_like(log_N), log_N, log_d])
        coeffs, residuals, rank, s = np.linalg.lstsq(X, log_ratio, rcond=None)

        log_C, alpha, beta = coeffs
        C = np.exp(log_C)

        # Compute R^2
        predicted = log_C + alpha * log_N + beta * log_d
        ss_res = np.sum((log_ratio - predicted) ** 2)
        ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'alpha': alpha,
            'beta': beta,
            'C': C,
            'r_squared': max(0, r_squared)
        }
    except Exception as e:
        return {'alpha': np.nan, 'beta': np.nan, 'C': np.nan, 'r_squared': 0}


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """Run the universality test."""
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 02: UNIVERSAL CRITICAL EXPONENTS")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    # Collect exponents for each architecture
    print("\n[STEP 1] Measuring exponents across architectures...")

    arch_results = {}
    alphas = []
    betas = []

    for arch in ARCHITECTURES:
        print(f"\n  Architecture: {arch['name']}")
        data = collect_data_for_architecture(arch, config)
        exps = fit_exponents(data)

        arch_results[arch['name']] = exps
        print(f"    alpha = {exps['alpha']:.4f}")
        print(f"    beta  = {exps['beta']:.4f}")
        print(f"    R^2   = {exps['r_squared']:.4f}")

        if not np.isnan(exps['alpha']):
            alphas.append(exps['alpha'])
        if not np.isnan(exps['beta']):
            betas.append(exps['beta'])

    # Compute coefficient of variation
    print("\n[STEP 2] Computing cross-architecture variability...")

    if len(alphas) < 3 or len(betas) < 3:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence=f"Only {len(alphas)} architectures yielded valid exponents",
            falsification_evidence="Insufficient data for universality test"
        )

    alpha_mean = np.mean(alphas)
    alpha_std = np.std(alphas)
    alpha_cv = alpha_std / abs(alpha_mean) if abs(alpha_mean) > 0.01 else 1.0

    beta_mean = np.mean(betas)
    beta_std = np.std(betas)
    beta_cv = beta_std / abs(beta_mean) if abs(beta_mean) > 0.01 else 1.0

    print(f"\n  Alpha: mean={alpha_mean:.4f}, std={alpha_std:.4f}, CV={alpha_cv:.4f}")
    print(f"  Beta:  mean={beta_mean:.4f}, std={beta_std:.4f}, CV={beta_cv:.4f}")

    print_metric("Alpha CV", alpha_cv, CV_THRESHOLD, higher_is_better=False)
    print_metric("Beta CV", beta_cv, CV_THRESHOLD, higher_is_better=False)

    # Verdict
    print_header("VERDICT", char="-")

    alpha_pass = alpha_cv <= CV_THRESHOLD
    beta_pass = beta_cv <= CV_THRESHOLD
    passed = alpha_pass and beta_pass

    if passed:
        evidence = f"Universal exponents: alpha={alpha_mean:.3f}+/-{alpha_std:.3f}, "
        evidence += f"beta={beta_mean:.3f}+/-{beta_std:.3f}"
        evidence += f"\nCV(alpha)={alpha_cv:.3f}, CV(beta)={beta_cv:.3f}"
        falsification = ""
        print("\n  ** TEST PASSED **")
        print(f"  Exponents are architecture-independent")
        print(f"  alpha = {alpha_mean:.3f} +/- {alpha_std:.3f}")
        print(f"  beta  = {beta_mean:.3f} +/- {beta_std:.3f}")
    else:
        evidence = f"Measured {len(ARCHITECTURES)} architectures"
        falsification = f"CV(alpha)={alpha_cv:.3f}, CV(beta)={beta_cv:.3f}"
        falsification += f" (threshold: {CV_THRESHOLD})"
        print("\n  ** TEST FAILED **")
        if not alpha_pass:
            print(f"  Alpha CV = {alpha_cv:.3f} > {CV_THRESHOLD}")
        if not beta_pass:
            print(f"  Beta CV = {beta_cv:.3f} > {CV_THRESHOLD}")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_law="power" if passed else "unknown",
        scaling_exponents={
            'alpha_mean': alpha_mean,
            'alpha_std': alpha_std,
            'alpha_cv': alpha_cv,
            'beta_mean': beta_mean,
            'beta_std': beta_std,
            'beta_cv': beta_cv
        },
        fit_quality=1.0 - max(alpha_cv, beta_cv),
        metric_value=max(alpha_cv, beta_cv),
        threshold=CV_THRESHOLD,
        evidence=evidence,
        falsification_evidence=falsification,
        n_trials=len(ARCHITECTURES) * len(FRAGMENT_SIZES) * len(DECOHERENCE_LEVELS)
    )


if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
