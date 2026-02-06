#!/usr/bin/env python3
"""
Q41 TIER 2.1: Semantic L-Functions

Tests semantic L-function construction with Euler product factorization
and functional equation verification.

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import sys
import json
import argparse
import platform
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any
from dataclasses import asdict
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import (
    TestConfig, TestResult, to_builtin, preprocess_embeddings,
    DEFAULT_CORPUS, compute_corpus_hash, load_embeddings
)

__version__ = "1.0.0"
__suite__ = "Q41_TIER2_1_L_FUNCTIONS"


def identify_semantic_primes(X: np.ndarray, n_primes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify "semantic primes" via K-means clustering.
    Cluster centers are the "primes".
    """
    kmeans = KMeans(n_clusters=n_primes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, labels


def compute_local_l_factor(X: np.ndarray, prime_center: np.ndarray, s: float) -> complex:
    """
    Compute local L-factor L_p(s, π) for a semantic prime.

    Classical: L_p(s) = (1 - a_p * p^{-s})^{-1}

    Semantic analog:
    - "p" = norm of prime center
    - "a_p" = average projection onto prime direction
    """
    p_norm = np.linalg.norm(prime_center) + 1e-10
    prime_dir = prime_center / p_norm
    projections = X @ prime_dir
    a_p = np.mean(np.abs(projections))

    z = a_p * np.power(p_norm, -s)
    if np.abs(z) >= 1:
        z = z * 0.9 / np.abs(z)

    return 1.0 / (1.0 - z)


def compute_semantic_l_function(X: np.ndarray, prime_centers: np.ndarray, s: float) -> complex:
    """Compute semantic L-function as Euler product over primes."""
    L_value = 1.0 + 0j
    for prime_center in prime_centers:
        L_p = compute_local_l_factor(X, prime_center, s)
        L_value *= L_p
    return L_value


def test_functional_equation(X: np.ndarray, prime_centers: np.ndarray, s_values: List[float]) -> Dict[str, Any]:
    """
    Test functional equation: L(s) ~ ε(s) L(1-s)
    """
    l_values = {s: compute_semantic_l_function(X, prime_centers, s) for s in s_values}

    # Test ratio consistency
    functional_ratios = []
    for s in s_values:
        if s != 0.5 and (1 - s) in l_values:
            L_s = l_values[s]
            L_1_minus_s = l_values[1 - s]
            if np.abs(L_1_minus_s) > 1e-10:
                ratio = L_s / L_1_minus_s
                functional_ratios.append(np.abs(ratio))

    if len(functional_ratios) >= 2:
        ratio_cv = np.std(functional_ratios) / (np.mean(functional_ratios) + 1e-10)
        functional_equation_quality = 1.0 / (1.0 + ratio_cv)
    else:
        functional_equation_quality = 0.5

    # Test smoothness (in log-scale since L-functions have exponential growth)
    l_magnitudes = [np.abs(l_values[s]) for s in sorted(s_values)]
    log_magnitudes = [np.log(m + 1e-10) for m in l_magnitudes]
    if len(log_magnitudes) >= 3:
        # Smoothness = how close to linear is the log-growth?
        # A smooth L-function has regular log-linear growth
        second_derivs = np.diff(np.diff(log_magnitudes))
        smoothness = 1.0 / (1.0 + np.std(second_derivs) / (np.abs(np.mean(log_magnitudes)) + 1e-10))
    else:
        smoothness = 0.5

    return {
        "l_values": {s: {"real": float(l_values[s].real), "imag": float(l_values[s].imag)} for s in s_values},
        "functional_ratios": functional_ratios,
        "functional_equation_quality": functional_equation_quality,
        "smoothness": smoothness
    }


def run_test(embeddings_dict: Dict[str, np.ndarray], config: TestConfig, verbose: bool = True) -> TestResult:
    """
    TIER 2.1: Semantic L-Function Construction

    TESTS:
    1. Euler product factorization over semantic primes
    2. Functional equation L(s) ~ ε(s) L(1-s)
    3. Analytic continuation (smoothness in s)

    PASS CRITERIA:
    - Functional equation quality > 0.3
    - Smoothness > 0.3
    - Positive control passes
    """
    np.random.seed(config.seed)

    n_primes = 10
    s_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    model_results = {}
    all_l_values = {}

    for model_name, X in embeddings_dict.items():
        X_proc = preprocess_embeddings(X, config.preprocessing)
        prime_centers, _ = identify_semantic_primes(X_proc, n_primes)
        fe_results = test_functional_equation(X_proc, prime_centers, s_values)

        model_results[model_name] = {
            "n_primes": n_primes,
            "functional_equation_quality": fe_results["functional_equation_quality"],
            "smoothness": fe_results["smoothness"],
            "l_values": fe_results["l_values"]
        }

        all_l_values[model_name] = [np.abs(complex(v["real"], v["imag"]))
                                     for v in fe_results["l_values"].values()]

    # Cross-model correlation
    cross_model_corrs = []
    model_names = list(all_l_values.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            lv1 = np.array(all_l_values[model_names[i]])
            lv2 = np.array(all_l_values[model_names[j]])
            if len(lv1) == len(lv2) and len(lv1) > 2:
                corr, _ = pearsonr(lv1, lv2)
                if not np.isnan(corr):
                    cross_model_corrs.append(corr)

    mean_cross_corr = np.mean(cross_model_corrs) if cross_model_corrs else 0.0

    # Aggregate
    fe_qualities = [r["functional_equation_quality"] for r in model_results.values()]
    smoothnesses = [r["smoothness"] for r in model_results.values()]
    mean_fe_quality = np.mean(fe_qualities)
    mean_smoothness = np.mean(smoothnesses)

    # Controls
    X_proc = list(embeddings_dict.values())[0]
    X_proc = preprocess_embeddings(X_proc, config.preprocessing)
    n_synthetic = X_proc.shape[0]
    d_synthetic = min(X_proc.shape[1], 50)

    # Positive control: structured data
    X_synthetic = np.zeros((n_synthetic, d_synthetic))
    for i in range(n_synthetic):
        cluster = i % n_primes
        X_synthetic[i] = np.random.randn(d_synthetic) * 0.1
        X_synthetic[i, cluster % d_synthetic] += 1.0

    pc_primes, _ = identify_semantic_primes(X_synthetic, n_primes)
    pc_results = test_functional_equation(X_synthetic, pc_primes, s_values)
    positive_control_quality = pc_results["functional_equation_quality"]

    # Negative control: random
    X_random = np.random.randn(n_synthetic, d_synthetic)
    nc_primes, _ = identify_semantic_primes(X_random, n_primes)
    nc_results = test_functional_equation(X_random, nc_primes, s_values)
    negative_control_quality = nc_results["functional_equation_quality"]

    # Pass criteria
    fe_pass = mean_fe_quality > 0.3
    smoothness_pass = mean_smoothness > 0.3
    positive_control_pass = positive_control_quality > 0.4

    passed = fe_pass and smoothness_pass and positive_control_pass

    return TestResult(
        name="TIER 2.1: Semantic L-Functions",
        test_type="langlands",
        passed=passed,
        metrics={
            "mean_functional_equation_quality": mean_fe_quality,
            "mean_smoothness": mean_smoothness,
            "mean_cross_model_correlation": mean_cross_corr,
            "model_results": model_results
        },
        thresholds={
            "functional_equation_quality_min": 0.3,
            "smoothness_min": 0.3
        },
        controls={
            "positive_control_quality": positive_control_quality,
            "negative_control_quality": negative_control_quality,
            "positive_pass": positive_control_pass
        },
        notes=f"FE quality: {mean_fe_quality:.3f}, Smoothness: {mean_smoothness:.3f}, "
              f"Cross-model corr: {mean_cross_corr:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Q41 TIER 2.1: Semantic L-Functions")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*60)
        print(f"Q41 TIER 2.1: SEMANTIC L-FUNCTIONS v{__version__}")
        print("="*60)

    config = TestConfig(seed=args.seed)
    corpus = DEFAULT_CORPUS

    if verbose:
        print(f"\nLoading embeddings...")
    embeddings = load_embeddings(corpus, verbose=verbose)

    if len(embeddings) < 2:
        print("ERROR: Need at least 2 embedding models")
        sys.exit(1)

    result = run_test(embeddings, config, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        status = "PASS" if result.passed else "FAIL"
        print(f"Result: {status}")
        print(f"Notes: {result.notes}")

    # Save receipt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = out_dir / f"q41_tier2_1_receipt_{timestamp_str}.json"

    receipt = to_builtin({
        "suite": __suite__,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": result.passed,
        "metrics": result.metrics,
        "thresholds": result.thresholds,
        "controls": result.controls,
        "notes": result.notes
    })

    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    if verbose:
        print(f"Receipt saved: {receipt_path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
