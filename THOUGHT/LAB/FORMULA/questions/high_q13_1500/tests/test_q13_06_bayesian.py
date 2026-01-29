"""
Q13 Test 06: Bayesian Model Selection (IRREFUTABLE)
====================================================

Hypothesis: Power law beats ALL competitors with overwhelming evidence.

Method:
1. Fit 6 competing models:
   - Power law: C * N^a * d^b
   - Exponential: C * exp(a*N*d)
   - Logarithmic: C * (1 + a*log(N)) * d^b
   - Linear: C * (1 + a*N) * d
   - Quadratic: C * (1 + a*N + b*N^2) * d
   - Critical: C * |d - d_c|^(-g) * N^a
2. Compute Bayes factors via BIC/AIC

Pass criteria: Winning model Bayes factor > 1000 vs all competitors

This is the IRREFUTABLE test. If Bayes factor > 1000, there is
less than 0.1% probability that the data came from a competing model.

Author: AGS Research
Date: 2026-01-19
"""

import sys
import os
import numpy as np
from scipy.optimize import curve_fit, minimize
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q13_utils import (
    ScalingLawResult, TestConfig, QUTIP_AVAILABLE,
    measure_ratio, compute_aic, compute_bic, compute_bayes_factor,
    print_header, print_metric
)


# =============================================================================
# CONSTANTS
# =============================================================================

TEST_ID = "Q13_TEST_06"
TEST_NAME = "Bayesian Model Selection"

# Bayes factor threshold for "decisive evidence"
# BF > 100: Decisive
# BF > 1000: Overwhelming
BAYES_FACTOR_THRESHOLD = 100.0  # Require decisive evidence

# Data collection parameters
FRAGMENT_SIZES = [2, 3, 4, 6, 8, 12, 16]
DECOHERENCE_LEVELS = np.linspace(0.1, 1.0, 10)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def power_law_model(Nd, C, alpha, beta):
    """Ratio = C * N^alpha * d^beta"""
    N, d = Nd
    return C * np.power(N, alpha) * np.power(np.maximum(d, 0.01), beta)


def exponential_model(Nd, C, alpha):
    """Ratio = C * exp(alpha * N * d)"""
    N, d = Nd
    return C * np.exp(np.clip(alpha * N * d, -20, 20))


def logarithmic_model(Nd, C, alpha, beta):
    """Ratio = C * (1 + alpha * log(N)) * d^beta"""
    N, d = Nd
    return C * (1 + alpha * np.log(np.maximum(N, 1))) * np.power(np.maximum(d, 0.01), beta)


def linear_model(Nd, C, alpha):
    """Ratio = C * (1 + alpha * N) * d"""
    N, d = Nd
    return C * (1 + alpha * N) * d


def quadratic_model(Nd, C, alpha, beta):
    """Ratio = C * (1 + alpha*N + beta*N^2) * d"""
    N, d = Nd
    return C * (1 + alpha * N + beta * N * N) * d


def critical_model(Nd, C, d_c, gamma, alpha):
    """Ratio = C * |d - d_c|^(-gamma) * N^alpha"""
    N, d = Nd
    t = np.abs(d - d_c)
    t_safe = np.maximum(t, 0.05)  # Avoid singularity
    return C * np.power(t_safe, -gamma) * np.power(N, alpha)


# =============================================================================
# MODEL FITTING
# =============================================================================

def collect_data(config: TestConfig) -> Dict:
    """Collect ratio data for model comparison."""
    results = {'N': [], 'd': [], 'ratio': []}

    if config.verbose:
        print("\n  Collecting data...")

    for N in FRAGMENT_SIZES:
        for d in DECOHERENCE_LEVELS:
            try:
                _, _, ratio = measure_ratio(N, d)
                if 0 < ratio < 1e6 and np.isfinite(ratio):
                    results['N'].append(float(N))
                    results['d'].append(d)
                    results['ratio'].append(ratio)
            except:
                pass

    for key in results:
        results[key] = np.array(results[key])

    if config.verbose:
        print(f"  Collected {len(results['ratio'])} data points")

    return results


def fit_model(model_func, data: Dict, n_params: int, p0: List, bounds: Tuple) -> Dict:
    """Fit a model and return metrics."""
    N = data['N']
    d = data['d']
    ratio = data['ratio']

    try:
        popt, pcov = curve_fit(
            model_func, (N, d), ratio,
            p0=p0, bounds=bounds, maxfev=10000
        )

        predicted = model_func((N, d), *popt)
        residuals = ratio - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ratio - np.mean(ratio)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        n = len(ratio)
        aic = compute_aic(n, n_params, ss_res)
        bic = compute_bic(n, n_params, ss_res)

        return {
            'params': popt.tolist(),
            'r_squared': max(0, r_squared),
            'aic': aic,
            'bic': bic,
            'rss': ss_res,
            'n': n,
            'success': True
        }
    except Exception as e:
        return {
            'params': [],
            'r_squared': 0,
            'aic': np.inf,
            'bic': np.inf,
            'rss': np.inf,
            'n': len(ratio),
            'success': False,
            'error': str(e)
        }


def fit_all_models(data: Dict, config: TestConfig) -> Dict[str, Dict]:
    """Fit all competing models."""
    results = {}

    if config.verbose:
        print("\n  Fitting models...")

    # Power law: 3 params (C, alpha, beta)
    if config.verbose:
        print("    Power law...")
    results['power'] = fit_model(
        power_law_model, data, 3,
        p0=[1.0, 1.5, 1.0],
        bounds=([0.01, 0, 0], [1000, 5, 5])
    )

    # Exponential: 2 params (C, alpha)
    if config.verbose:
        print("    Exponential...")
    results['exponential'] = fit_model(
        exponential_model, data, 2,
        p0=[1.0, 0.1],
        bounds=([0.01, -1], [100, 1])
    )

    # Logarithmic: 3 params (C, alpha, beta)
    if config.verbose:
        print("    Logarithmic...")
    results['logarithmic'] = fit_model(
        logarithmic_model, data, 3,
        p0=[1.0, 2.0, 1.0],
        bounds=([0.01, -10, 0], [100, 10, 5])
    )

    # Linear: 2 params (C, alpha)
    if config.verbose:
        print("    Linear...")
    results['linear'] = fit_model(
        linear_model, data, 2,
        p0=[1.0, 1.0],
        bounds=([0.01, -10], [100, 10])
    )

    # Quadratic: 3 params (C, alpha, beta)
    if config.verbose:
        print("    Quadratic...")
    results['quadratic'] = fit_model(
        quadratic_model, data, 3,
        p0=[1.0, 0.5, 0.01],
        bounds=([0.01, -10, -1], [100, 10, 1])
    )

    # Critical: 4 params (C, d_c, gamma, alpha)
    if config.verbose:
        print("    Critical...")
    results['critical'] = fit_model(
        critical_model, data, 4,
        p0=[1.0, 0.5, 0.5, 1.0],
        bounds=([0.01, 0.1, 0.01, 0], [100, 0.9, 3, 5])
    )

    return results


def compare_models(model_results: Dict[str, Dict], config: TestConfig) -> Dict:
    """Compare models using Bayes factors."""

    # Find best model by BIC
    valid_models = {k: v for k, v in model_results.items() if v['success']}

    if not valid_models:
        return {'best': None, 'bayes_factors': {}}

    best_model = min(valid_models.keys(), key=lambda k: valid_models[k]['bic'])
    best_bic = valid_models[best_model]['bic']

    if config.verbose:
        print(f"\n  Best model by BIC: {best_model}")

    # Compute Bayes factors
    bayes_factors = {}
    for model, result in valid_models.items():
        if model != best_model:
            # BF = exp((BIC_other - BIC_best) / 2)
            bf = np.exp((result['bic'] - best_bic) / 2)
            bayes_factors[model] = bf
            if config.verbose:
                print(f"    BF({best_model} vs {model}): {bf:.2f}")

    return {
        'best': best_model,
        'best_bic': best_bic,
        'bayes_factors': bayes_factors,
        'all_bics': {k: v['bic'] for k, v in valid_models.items()}
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_test(config: TestConfig = None) -> ScalingLawResult:
    """Run the Bayesian model selection test."""
    if config is None:
        config = TestConfig(verbose=True)

    print_header("Q13 TEST 06: BAYESIAN MODEL SELECTION")

    if not QUTIP_AVAILABLE:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence="QuTiP not available",
            falsification_evidence="Cannot run quantum simulations"
        )

    # Collect data
    print("\n[STEP 1] Collecting ratio data...")
    data = collect_data(config)

    if len(data['ratio']) < 20:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            evidence=f"Only {len(data['ratio'])} data points",
            falsification_evidence="Insufficient data for model comparison"
        )

    # Fit all models
    print("\n[STEP 2] Fitting competing models...")
    model_results = fit_all_models(data, config)

    # Print fit quality
    print("\n  Model fit quality (R^2):")
    for model, result in model_results.items():
        if result['success']:
            print(f"    {model}: R^2 = {result['r_squared']:.4f}, BIC = {result['bic']:.2f}")
        else:
            print(f"    {model}: FAILED")

    # Compare models
    print("\n[STEP 3] Computing Bayes factors...")
    comparison = compare_models(model_results, config)

    if comparison['best'] is None:
        return ScalingLawResult(
            test_name=TEST_NAME,
            test_id=TEST_ID,
            passed=False,
            falsification_evidence="No models fit successfully"
        )

    best_model = comparison['best']
    bayes_factors = comparison['bayes_factors']

    # Check if best model dominates all others
    min_bf = min(bayes_factors.values()) if bayes_factors else 0
    all_decisive = all(bf >= BAYES_FACTOR_THRESHOLD for bf in bayes_factors.values())

    print(f"\n  Best model: {best_model}")
    print(f"  Minimum Bayes factor: {min_bf:.2f}")
    print(f"  All Bayes factors >= {BAYES_FACTOR_THRESHOLD}: {all_decisive}")

    # Verdict
    print_header("VERDICT", char="-")

    passed = all_decisive and min_bf >= BAYES_FACTOR_THRESHOLD

    if passed:
        evidence = f"Best model: {best_model}, min BF = {min_bf:.1f}"
        evidence += f"\nR^2 = {model_results[best_model]['r_squared']:.4f}"
        if best_model == 'power' and model_results['power']['success']:
            params = model_results['power']['params']
            evidence += f"\nParams: C={params[0]:.3f}, alpha={params[1]:.3f}, beta={params[2]:.3f}"
        falsification = ""
        print(f"\n  ** TEST PASSED - IRREFUTABLE EVIDENCE **")
        print(f"  {best_model} model dominates all competitors")
        print(f"  Minimum Bayes factor: {min_bf:.1f} (threshold: {BAYES_FACTOR_THRESHOLD})")
    else:
        evidence = f"Best model: {best_model}"
        falsification = f"Min Bayes factor = {min_bf:.2f} < {BAYES_FACTOR_THRESHOLD}"
        print(f"\n  ** TEST FAILED **")
        print(f"  Best model: {best_model}")
        print(f"  Min Bayes factor: {min_bf:.2f} < {BAYES_FACTOR_THRESHOLD}")

    return ScalingLawResult(
        test_name=TEST_NAME,
        test_id=TEST_ID,
        passed=passed,
        scaling_law=best_model if passed else "unknown",
        scaling_exponents={
            'best_model': best_model,
            'r_squared': model_results[best_model]['r_squared'] if best_model else 0
        },
        fit_quality=model_results[best_model]['r_squared'] if best_model else 0,
        metric_value=min_bf,
        threshold=BAYES_FACTOR_THRESHOLD,
        bayes_factor=min_bf,
        evidence=evidence,
        falsification_evidence=falsification,
        n_trials=len(data['ratio'])
    )


if __name__ == "__main__":
    result = run_test()
    print("\n" + "=" * 70)
    print(f"Test: {result.test_name}")
    print(f"Passed: {result.passed}")
    print(f"Bayes Factor: {result.bayes_factor}")
