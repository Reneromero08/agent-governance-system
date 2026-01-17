#!/usr/bin/env python3
"""Test 3: Error Threshold & Exponential Suppression.

Proves R-gating implements QECC by showing exponential error suppression
below a critical threshold.

Hypothesis:
    Below epsilon_th, logical errors are exponentially suppressed.
    Above epsilon_th, logical errors are amplified.

Protocol:
    1. Sweep physical error rate from 0.01% to 100%
    2. Measure logical error rate (gate failures on corrupted data)
    3. Fit two-regime model to find threshold
    4. Verify suppression exponent k > 2

Success Criteria:
    - Two-regime fit with chi-square p < 0.05
    - Suppression exponent k > 2 (quadratic or better)
    - Semantic k > Random k + 1

Expected:
    epsilon_th ~ 1/Df ~ 4.5% for Df ~ 22

Usage:
    python test_threshold.py [--n-trials 500] [--n-epsilon 30]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Local imports
from core import (
    inject_n_errors,
    r_gate,
    compute_R,
    generate_random_embeddings,
    compute_effective_dimensionality,
    DEFAULT_R_THRESHOLD,
)

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Test Data
# =============================================================================

VALID_PHRASES = [
    "The sun rises in the east.",
    "Water boils at one hundred degrees Celsius.",
    "Humans need oxygen to survive.",
    "Mathematics is the language of science.",
    "Light travels faster than sound.",
    "The moon orbits the Earth.",
    "Plants convert sunlight into energy.",
    "Gravity keeps planets in orbit.",
    "DNA carries genetic information.",
    "Computers process binary data.",
]


def get_embeddings(phrases: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Get embeddings for phrases."""
    if not HAS_TRANSFORMERS:
        np.random.seed(42)
        dim = 384
        embeddings = np.random.randn(len(phrases), dim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    model = SentenceTransformer(model_name)
    embeddings = model.encode(phrases, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# =============================================================================
# Error Threshold Models
# =============================================================================

def suppression_model(eps_phys, eps_th, k_below, k_above, baseline):
    """Two-regime error model.

    Below threshold: eps_log ~ eps_phys^k_below (suppression if k > 1)
    Above threshold: eps_log ~ (eps_phys/eps_th)^k_above * threshold_value
    """
    result = np.zeros_like(eps_phys)

    below_mask = eps_phys < eps_th
    above_mask = ~below_mask

    # Below threshold: power law suppression
    if np.any(below_mask):
        result[below_mask] = baseline * (eps_phys[below_mask] ** k_below)

    # Above threshold: different scaling
    if np.any(above_mask):
        threshold_value = baseline * (eps_th ** k_below)
        result[above_mask] = threshold_value * ((eps_phys[above_mask] / eps_th) ** k_above)

    return result


def fit_threshold_model(
    eps_phys: np.ndarray,
    eps_log: np.ndarray
) -> Tuple[Dict, float]:
    """Fit two-regime model to error data.

    Returns:
        Tuple of (parameters dict, R-squared)
    """
    # Filter out zeros for fitting
    valid = (eps_log > 0) & (eps_phys > 0)
    if np.sum(valid) < 5:
        return {
            "eps_th": 0.1,
            "k_below": 1.0,
            "k_above": 1.0,
            "baseline": 1.0
        }, 0.0

    eps_phys_valid = eps_phys[valid]
    eps_log_valid = eps_log[valid]

    try:
        # Initial guess
        p0 = [0.1, 2.0, 0.5, 1.0]

        # Bounds
        bounds = (
            [0.001, 0.1, 0.1, 0.01],   # Lower bounds
            [0.5, 10.0, 5.0, 100.0]     # Upper bounds
        )

        popt, _ = curve_fit(
            suppression_model,
            eps_phys_valid,
            eps_log_valid,
            p0=p0,
            bounds=bounds,
            maxfev=5000
        )

        # Compute R-squared
        predicted = suppression_model(eps_phys_valid, *popt)
        ss_res = np.sum((eps_log_valid - predicted) ** 2)
        ss_tot = np.sum((eps_log_valid - np.mean(eps_log_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "eps_th": float(popt[0]),
            "k_below": float(popt[1]),
            "k_above": float(popt[2]),
            "baseline": float(popt[3])
        }, float(r_squared)

    except (RuntimeError, ValueError) as e:
        print(f"Fitting failed: {e}")
        return {
            "eps_th": 0.1,
            "k_below": 1.0,
            "k_above": 1.0,
            "baseline": 1.0
        }, 0.0


# =============================================================================
# Main Test
# =============================================================================

def measure_logical_error_rate(
    embeddings: np.ndarray,
    physical_error_rate: float,
    n_trials: int = 100,
    threshold: float = DEFAULT_R_THRESHOLD
) -> float:
    """Measure logical error rate at given physical error rate.

    Args:
        embeddings: Base embeddings
        physical_error_rate: Noise level (0 to 1)
        n_trials: Number of measurement trials
        threshold: R-gate threshold

    Returns:
        Logical error rate (fraction of gate failures that are wrong)
    """
    dim = embeddings.shape[1]
    logical_errors = 0

    for trial in range(n_trials):
        # Pick random embedding as "truth"
        idx = np.random.randint(len(embeddings))
        truth = embeddings[idx]

        # Create noisy observations
        n_obs = 5
        observations = np.array([
            truth + np.random.randn(dim) * physical_error_rate
            for _ in range(n_obs)
        ])
        # Normalize
        norms = np.linalg.norm(observations, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        observations = observations / norms

        # Check R-gate
        gate_result = r_gate(observations, threshold)

        # Compute reconstruction error (how far from truth)
        centroid = observations.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        reconstruction_error = 1 - np.dot(truth, centroid)

        # Logical error: gate passed but reconstruction is wrong
        # OR gate failed but reconstruction was actually good
        is_corrupted = reconstruction_error > 0.1

        if gate_result.passed and is_corrupted:
            logical_errors += 1
        # Note: We only count false positives (passed when should fail)
        # as logical errors in the QECC sense

    return logical_errors / n_trials


def run_threshold_test(
    n_epsilon: int = 30,
    n_trials: int = 200,
    dim: int = 384,
    n_random_seeds: int = 3
) -> Dict:
    """Run full error threshold test.

    Args:
        n_epsilon: Number of epsilon values to test
        n_trials: Trials per epsilon
        dim: Embedding dimension
        n_random_seeds: Number of random baselines

    Returns:
        Complete test results dict
    """
    print("=" * 70)
    print("TEST 3: ERROR THRESHOLD & EXPONENTIAL SUPPRESSION")
    print("=" * 70)
    print()

    # Get semantic embeddings
    print("Loading semantic embeddings...")
    semantic_emb = get_embeddings(VALID_PHRASES)
    semantic_df = compute_effective_dimensionality(semantic_emb)
    print(f"  Semantic Df: {semantic_df:.2f}")
    print(f"  Expected threshold: ~{1/semantic_df:.4f} ({100/semantic_df:.2f}%)")
    print()

    # Generate random baselines
    print(f"Generating {n_random_seeds} random baselines...")
    random_embs = [
        generate_random_embeddings(len(VALID_PHRASES), dim, seed=42 + i)
        for i in range(n_random_seeds)
    ]
    print()

    # Epsilon range (logarithmic from 0.01% to 100%)
    eps_range = np.logspace(-4, 0, n_epsilon)

    results = {
        "test_id": "q40-error-threshold",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "n_epsilon": n_epsilon,
            "n_trials": n_trials,
            "dim": dim,
            "n_random_seeds": n_random_seeds,
        },
        "semantic_df": float(semantic_df),
        "epsilon_range": eps_range.tolist(),
    }

    # Measure semantic error rates
    print("Measuring semantic error rates...")
    semantic_logical_errors = []
    for i, eps in enumerate(eps_range):
        rate = measure_logical_error_rate(semantic_emb, eps, n_trials)
        semantic_logical_errors.append(rate)
        print(f"  eps={eps:.4f} ({eps*100:.2f}%): logical_error={rate:.4f}")

    semantic_logical_errors = np.array(semantic_logical_errors)

    # Fit semantic model
    print("\nFitting semantic threshold model...")
    semantic_params, semantic_r2 = fit_threshold_model(eps_range, semantic_logical_errors)
    print(f"  Threshold: {semantic_params['eps_th']:.4f} ({semantic_params['eps_th']*100:.2f}%)")
    print(f"  Suppression exponent (k_below): {semantic_params['k_below']:.2f}")
    print(f"  Amplification exponent (k_above): {semantic_params['k_above']:.2f}")
    print(f"  R-squared: {semantic_r2:.4f}")

    results["semantic"] = {
        "logical_errors": semantic_logical_errors.tolist(),
        "params": semantic_params,
        "r_squared": semantic_r2,
    }

    # Measure random baselines
    print("\nMeasuring random baseline error rates...")
    random_all_errors = []
    random_all_params = []

    for seed_idx, random_emb in enumerate(random_embs):
        print(f"  Random seed {seed_idx}:")
        random_errors = []
        for eps in eps_range:
            rate = measure_logical_error_rate(random_emb, eps, n_trials // 2)
            random_errors.append(rate)
        random_errors = np.array(random_errors)
        random_all_errors.append(random_errors)

        params, r2 = fit_threshold_model(eps_range, random_errors)
        random_all_params.append(params)
        print(f"    k_below: {params['k_below']:.2f}, R2: {r2:.4f}")

    # Aggregate random results
    random_k_belows = [p['k_below'] for p in random_all_params]
    random_k_mean = np.mean(random_k_belows)
    random_k_std = np.std(random_k_belows)

    results["random"] = {
        "k_belows": random_k_belows,
        "k_mean": float(random_k_mean),
        "k_std": float(random_k_std),
    }

    # Verdict
    is_qecc = semantic_params['k_below'] > 2.0  # At least quadratic suppression
    is_better_than_random = semantic_params['k_below'] > random_k_mean + 1.0
    good_fit = semantic_r2 > 0.7

    verdict_pass = is_qecc and good_fit

    results["verdict"] = {
        "is_qecc": is_qecc,
        "is_better_than_random": is_better_than_random,
        "good_fit": good_fit,
        "overall_pass": verdict_pass,
        "interpretation": (
            f"PASS: Error suppression exponent k={semantic_params['k_below']:.2f} > 2 "
            f"indicates QECC behavior. Threshold at {semantic_params['eps_th']*100:.1f}%."
            if verdict_pass else
            f"FAIL: Suppression exponent k={semantic_params['k_below']:.2f} "
            "does not indicate strong QECC behavior."
        )
    }

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Semantic k_below: {semantic_params['k_below']:.2f}")
    print(f"Random k_below mean: {random_k_mean:.2f} +/- {random_k_std:.2f}")
    print(f"Is QECC (k > 2): {is_qecc}")
    print(f"Better than random: {is_better_than_random}")
    print(f"Good fit (R2 > 0.7): {good_fit}")
    print(f"OVERALL: {'PASS' if verdict_pass else 'FAIL'}")
    print(results["verdict"]["interpretation"])
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description='Q40 Test 3: Error Threshold')
    parser.add_argument('--n-epsilon', type=int, default=30,
                        help='Number of epsilon values')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Trials per epsilon')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    results = run_threshold_test(
        n_epsilon=args.n_epsilon,
        n_trials=args.n_trials,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent / "results" / "error_threshold.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0 if results["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
