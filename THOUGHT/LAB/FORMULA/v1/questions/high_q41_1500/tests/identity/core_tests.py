#!/usr/bin/env python3
"""
Q41 Identity Tests: Mathematical Truths

These tests verify that our implementations are mathematically correct.
They test fundamental identities that MUST hold regardless of the data.

Identity Tests:
1. Kernel Trace Identity: trace(K) = sum(eigenvalues(K))
2. Laplacian Properties: symmetric, PSD, eigenvalues in [0,2]
3. Heat Trace Consistency: trace(exp(-tL)) via matrix exp = eigendecomposition
4. Rotation Invariance: distance-based constructions unchanged under rotation

Author: Claude
Date: 2026-01-11
Version: 1.0.0
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import (
    TestConfig, TestResult, to_builtin,
    DEFAULT_CORPUS, load_embeddings, preprocess_embeddings
)

__version__ = "1.0.0"
__suite__ = "Q41_IDENTITY_TESTS"


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix."""
    if metric == "cosine":
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        sim = X_norm @ X_norm.T
        D = 1.0 - sim
        np.fill_diagonal(D, 0)
    else:
        D = squareform(pdist(X, metric="euclidean"))
    return D


def build_mutual_knn_graph(D: np.ndarray, k: int) -> np.ndarray:
    """Build mutual k-NN graph (symmetric adjacency)."""
    n = len(D)
    k = min(k, n - 1)
    knn_idx = np.argsort(D, axis=1)[:, 1:k+1]
    A_asym = np.zeros((n, n), dtype=int)
    for i in range(n):
        A_asym[i, knn_idx[i]] = 1
    A = (A_asym * A_asym.T).astype(float)
    return A


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Compute normalized graph Laplacian."""
    n = len(A)
    degrees = np.sum(A, axis=1)
    degrees[degrees == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    L = (L + L.T) / 2.0
    return L


def heat_trace_from_laplacian(L: np.ndarray, t_grid: List[float]) -> np.ndarray:
    """Compute heat trace for multiple t values."""
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.maximum(eigenvalues, 0)
    traces = np.array([np.sum(np.exp(-t * eigenvalues)) for t in t_grid])
    return traces


def random_orthogonal_matrix(d: int, seed: int) -> np.ndarray:
    """Generate random orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def test_kernel_trace_identity(X: np.ndarray, config: TestConfig) -> TestResult:
    """
    IDENTITY TEST 1: trace(K) = sum(eigenvalues(K))

    This MUST pass - it's a mathematical identity.
    """
    D = pairwise_distances(X, config.distance_metric)
    t = 1.0
    K = np.exp(-t * D**2)

    trace_direct = np.trace(K)
    eigenvalues = np.linalg.eigvalsh(K)
    trace_eigen = np.sum(eigenvalues)

    error = abs(trace_direct - trace_eigen)
    rel_error = error / (abs(trace_direct) + 1e-10)

    passed = rel_error < config.identity_tolerance

    return TestResult(
        name="kernel_trace_identity",
        test_type="identity",
        passed=passed,
        metrics={
            "trace_direct": float(trace_direct),
            "trace_eigensum": float(trace_eigen),
            "absolute_error": float(error),
            "relative_error": float(rel_error)
        },
        thresholds={"relative_error": config.identity_tolerance},
        controls={},
        notes="trace(K) = sum(eigenvalues(K)) - basic linear algebra identity"
    )


def test_laplacian_properties(X: np.ndarray, config: TestConfig) -> TestResult:
    """
    IDENTITY TEST 2: Laplacian Properties

    Verifies: symmetric, PSD, eigenvalues in [0,2]
    """
    D = pairwise_distances(X, config.distance_metric)
    A = build_mutual_knn_graph(D, config.k_neighbors)
    L = normalized_laplacian(A)

    symmetry_error = np.max(np.abs(L - L.T))
    is_symmetric = symmetry_error < config.identity_tolerance

    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues_sorted = np.sort(eigenvalues)

    min_eig = eigenvalues_sorted[0]
    max_eig = eigenvalues_sorted[-1]

    is_psd = min_eig >= -config.identity_tolerance
    eigs_in_range = max_eig <= 2.0 + config.identity_tolerance

    n_components, _ = connected_components(A > 0, directed=False)
    near_zero_eigs = np.sum(np.abs(eigenvalues) < 0.01)

    passed = is_symmetric and is_psd and eigs_in_range

    return TestResult(
        name="laplacian_properties",
        test_type="identity",
        passed=passed,
        metrics={
            "symmetry_error": float(symmetry_error),
            "min_eigenvalue": float(min_eig),
            "max_eigenvalue": float(max_eig),
            "n_components": int(n_components),
            "near_zero_eigenvalues": int(near_zero_eigs)
        },
        thresholds={
            "symmetry_error": config.identity_tolerance,
            "min_eigenvalue": -config.identity_tolerance,
            "max_eigenvalue": 2.0 + config.identity_tolerance
        },
        controls={},
        notes="Normalized Laplacian must be symmetric PSD with eigenvalues in [0,2]"
    )


def test_heat_trace_consistency(X: np.ndarray, config: TestConfig) -> TestResult:
    """
    IDENTITY TEST 3: Heat Trace Consistency

    trace(exp(-tL)) via matrix exp == sum(exp(-t*lambda_i))
    """
    D = pairwise_distances(X, config.distance_metric)
    A = build_mutual_knn_graph(D, config.k_neighbors)
    L = normalized_laplacian(A)

    t_test = 1.0

    K = linalg.expm(-t_test * L)
    trace_matrix = np.trace(K)

    eigenvalues = np.linalg.eigvalsh(L)
    trace_eigen = np.sum(np.exp(-t_test * eigenvalues))

    error = abs(trace_matrix - trace_eigen)
    rel_error = error / (abs(trace_matrix) + 1e-10)

    passed = rel_error < config.identity_tolerance * 100

    return TestResult(
        name="heat_trace_consistency",
        test_type="identity",
        passed=passed,
        metrics={
            "trace_matrix_exp": float(trace_matrix),
            "trace_eigensum": float(trace_eigen),
            "absolute_error": float(error),
            "relative_error": float(rel_error),
            "t_value": t_test
        },
        thresholds={"relative_error": config.identity_tolerance * 100},
        controls={},
        notes="trace(exp(-tL)) = sum(exp(-t*lambda_i)) - eigendecomposition identity"
    )


def test_rotation_invariance(X: np.ndarray, config: TestConfig) -> TestResult:
    """
    IDENTITY TEST 4: Rotation Invariance

    Distance-based constructions must be unchanged under rotation.
    """
    heat_t_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    D_orig = pairwise_distances(X, "euclidean")
    A_orig = build_mutual_knn_graph(D_orig, config.k_neighbors)
    L_orig = normalized_laplacian(A_orig)
    eigs_orig = np.sort(np.linalg.eigvalsh(L_orig))
    heat_orig = heat_trace_from_laplacian(L_orig, heat_t_grid)

    Q = random_orthogonal_matrix(X.shape[1], config.seed + 1000)
    X_rot = X @ Q

    D_rot = pairwise_distances(X_rot, "euclidean")
    A_rot = build_mutual_knn_graph(D_rot, config.k_neighbors)
    L_rot = normalized_laplacian(A_rot)
    eigs_rot = np.sort(np.linalg.eigvalsh(L_rot))
    heat_rot = heat_trace_from_laplacian(L_rot, heat_t_grid)

    dist_error = np.max(np.abs(D_orig - D_rot))
    eig_error = np.max(np.abs(eigs_orig - eigs_rot))
    heat_error = np.max(np.abs(heat_orig - heat_rot))

    passed = (dist_error < config.identity_tolerance and
              eig_error < config.identity_tolerance * 100 and
              heat_error < config.identity_tolerance * 100)

    return TestResult(
        name="rotation_invariance",
        test_type="identity",
        passed=passed,
        metrics={
            "distance_matrix_error": float(dist_error),
            "eigenvalue_error": float(eig_error),
            "heat_trace_error": float(heat_error)
        },
        thresholds={
            "distance_matrix_error": config.identity_tolerance,
            "eigenvalue_error": config.identity_tolerance * 100,
            "heat_trace_error": config.identity_tolerance * 100
        },
        controls={"rotation_seed": config.seed + 1000},
        notes="Euclidean distance-based constructions must be rotation-invariant"
    )


def run_identity_tests(X: np.ndarray, config: TestConfig, verbose: bool = True) -> List[TestResult]:
    """Run all identity tests."""
    results = []

    tests = [
        ("Kernel Trace Identity", test_kernel_trace_identity),
        ("Laplacian Properties", test_laplacian_properties),
        ("Heat Trace Consistency", test_heat_trace_consistency),
        ("Rotation Invariance", test_rotation_invariance),
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("IDENTITY TESTS (Mathematical Truths)")
        print("=" * 60)

    for test_name, test_fn in tests:
        if verbose:
            print(f"\n  Running: {test_name}...")

        result = test_fn(X, config)
        results.append(result)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"    {test_name}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Q41 Identity Tests")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent.parent / "receipts"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print(f"Q41 IDENTITY TESTS v{__version__}")
        print("=" * 60)

    config = TestConfig(seed=args.seed)
    corpus = DEFAULT_CORPUS

    if verbose:
        print(f"\nLoading embeddings...")
    embeddings = load_embeddings(corpus, verbose=verbose)

    if not embeddings:
        print("ERROR: No embedding models available")
        sys.exit(1)

    # Use first model for identity tests
    first_model = list(embeddings.keys())[0]
    X = preprocess_embeddings(embeddings[first_model], config.preprocessing)

    results = run_identity_tests(X, config, verbose=verbose)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Identity Tests: {passed}/{total} passed")

    # Save receipt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = out_dir / f"q41_identity_receipt_{timestamp_str}.json"

    receipt = to_builtin({
        "suite": __suite__,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "total": total,
        "all_pass": passed == total,
        "model_used": first_model,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "metrics": r.metrics,
                "notes": r.notes
            }
            for r in results
        ]
    })

    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    if verbose:
        print(f"Receipt saved: {receipt_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
