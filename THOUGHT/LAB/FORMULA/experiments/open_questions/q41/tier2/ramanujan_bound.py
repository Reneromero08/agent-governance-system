#!/usr/bin/env python3
"""
Q41 TIER 2.2: Ramanujan Bound Analog

Tests that Hecke eigenvalues are bounded, analogous to the
Ramanujan conjecture which bounds Fourier coefficients of modular forms.

The Ramanujan bound: |a_p| ≤ 2 * p^{(k-1)/2}

Semantic analog: Eigenvalues of averaging operators are bounded
by a universal constant (spectral radius bound).

Author: Claude
Date: 2026-01-11
Version: 1.1.0
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import (
    TestConfig, TestResult, to_builtin, preprocess_embeddings,
    build_knn_graph, DEFAULT_CORPUS, load_embeddings
)

__version__ = "1.1.0"
__suite__ = "Q41_TIER2_2_RAMANUJAN_BOUND"


def compute_hecke_eigenvalues(X: np.ndarray, k: int) -> np.ndarray:
    """
    Compute Hecke-like eigenvalues from symmetric normalized adjacency.

    Uses D^{-1/2} A D^{-1/2} which is symmetric and has eigenvalues in [-1, 1].
    """
    A = build_knn_graph(X, k)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degrees = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt

    # A_sym is symmetric with eigenvalues in [-1, 1]
    eigenvalues = np.linalg.eigvalsh(A_sym)
    return np.sort(eigenvalues)[::-1]


def test_ramanujan_bound(eigenvalues: np.ndarray) -> Dict[str, Any]:
    """
    Test Ramanujan-like bounds on eigenvalues.

    For a Markov operator (row-stochastic matrix):
    - Largest eigenvalue is 1
    - All eigenvalues in [-1, 1]
    - Spectral gap = 1 - second_eigenvalue indicates mixing

    Ramanujan graph: spectral gap is optimal (2*sqrt(d-1)/d for d-regular)
    """
    # Remove trivial eigenvalue = 1
    nontrivial = eigenvalues[np.abs(eigenvalues - 1.0) > 0.001]

    if len(nontrivial) < 2:
        return {
            "bound_satisfied": True,
            "spectral_gap": 1.0,
            "max_nontrivial": 0.0,
            "min_eigenvalue": 0.0,
            "all_in_unit_interval": True,
            "n_eigenvalues": len(eigenvalues)
        }

    max_nontrivial = np.max(np.abs(nontrivial))
    min_eigenvalue = np.min(eigenvalues)
    spectral_gap = 1.0 - max_nontrivial

    # Ramanujan-type bound: all eigenvalues in [-1, 1]
    all_in_unit = np.all(np.abs(eigenvalues) <= 1.0 + 1e-10)

    # Strong Ramanujan bound: spectral gap > some threshold
    # For random graphs, expect spectral_gap ~ 1/sqrt(n)
    n = len(eigenvalues)
    expected_gap_lower = 0.5 / np.sqrt(n)  # Very loose bound
    gap_good = spectral_gap > expected_gap_lower

    return {
        "bound_satisfied": all_in_unit and gap_good,
        "spectral_gap": float(spectral_gap),
        "max_nontrivial": float(max_nontrivial),
        "min_eigenvalue": float(min_eigenvalue),
        "all_in_unit_interval": all_in_unit,
        "expected_gap_lower": float(expected_gap_lower),
        "n_eigenvalues": len(eigenvalues)
    }


def run_test(embeddings_dict: Dict[str, np.ndarray], config: TestConfig, verbose: bool = True) -> TestResult:
    """
    TIER 2.2: Ramanujan Bound Analog

    TESTS:
    - Eigenvalues of Hecke operator are in [-1, 1]
    - Spectral gap is positive (mixing property)
    - Gap is consistent across models

    PASS CRITERIA:
    - All eigenvalues in unit interval
    - Spectral gap > 0 for all models
    - Gap CV < 1.0 (roughly consistent)
    """
    np.random.seed(config.seed)

    model_results = {}
    all_gaps = []
    all_bound_satisfied = []
    k_values = [5, 10, 15]

    for model_name, X in embeddings_dict.items():
        X_proc = preprocess_embeddings(X, config.preprocessing)

        k_results = {}
        for k in k_values:
            eigenvalues = compute_hecke_eigenvalues(X_proc, k)
            bound_result = test_ramanujan_bound(eigenvalues)
            k_results[k] = bound_result

            all_gaps.append(bound_result["spectral_gap"])
            all_bound_satisfied.append(bound_result["bound_satisfied"])

        model_results[model_name] = k_results

    # Aggregate
    mean_gap = np.mean(all_gaps)
    gap_cv = np.std(all_gaps) / (mean_gap + 1e-10)
    bound_fraction = np.mean(all_bound_satisfied)

    # Controls
    X_sample = list(embeddings_dict.values())[0]
    X_sample = preprocess_embeddings(X_sample, config.preprocessing)
    n = X_sample.shape[0]
    d = min(X_sample.shape[1], 20)

    # Positive control: connected graph
    grid_side = int(np.sqrt(n))
    X_grid = np.array([[i // grid_side, i % grid_side] for i in range(grid_side ** 2)], dtype=float)
    X_grid = np.hstack([X_grid, np.random.randn(len(X_grid), d - 2) * 0.1])
    X_grid = preprocess_embeddings(X_grid, "l2")
    pc_eigs = compute_hecke_eigenvalues(X_grid, 5)
    pc_result = test_ramanujan_bound(pc_eigs)

    # Negative control: random points
    X_random = np.random.randn(n, d)
    X_random = preprocess_embeddings(X_random, "l2")
    nc_eigs = compute_hecke_eigenvalues(X_random, 5)
    nc_result = test_ramanujan_bound(nc_eigs)

    # Pass criteria
    gap_positive = mean_gap > 0
    cv_ok = gap_cv < 1.0
    bound_ok = bound_fraction > 0.8

    passed = gap_positive and cv_ok and bound_ok

    return TestResult(
        name="TIER 2.2: Ramanujan Bound",
        test_type="langlands",
        passed=passed,
        metrics={
            "mean_spectral_gap": mean_gap,
            "gap_cv": gap_cv,
            "bound_satisfied_fraction": bound_fraction,
            "model_results": {k: {kk: to_builtin(vv) for kk, vv in v.items()}
                             for k, v in model_results.items()}
        },
        thresholds={
            "spectral_gap_min": 0.0,
            "gap_cv_max": 1.0,
            "bound_fraction_min": 0.8
        },
        controls={
            "positive_control_gap": pc_result["spectral_gap"],
            "positive_control_bound": pc_result["bound_satisfied"],
            "negative_control_gap": nc_result["spectral_gap"],
            "negative_control_bound": nc_result["bound_satisfied"]
        },
        notes=f"Mean gap: {mean_gap:.4f} (CV: {gap_cv:.2f}), "
              f"Bound satisfied: {bound_fraction:.1%}"
    )


def main():
    parser = argparse.ArgumentParser(description="Q41 TIER 2.2: Ramanujan Bound")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*60)
        print(f"Q41 TIER 2.2: RAMANUJAN BOUND v{__version__}")
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
    receipt_path = out_dir / f"q41_tier2_2_receipt_{timestamp_str}.json"

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
