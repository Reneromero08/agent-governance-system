#!/usr/bin/env python3
"""
Q41 TIER 3: Hecke Operators

Tests whether embedding spaces admit Hecke-like averaging operators
with the algebraic properties expected from the Langlands program.

The Langlands connection: Hecke operators are central to the theory
of automorphic forms and their connection to Galois representations.

Semantic analogs:
- Averaging operators T_k over k-neighborhoods
- Commutativity: T_k T_l = T_l T_k
- Self-adjointness and eigenvalue structure

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

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.utils import (
    TestConfig, TestResult, to_builtin, preprocess_embeddings,
    DEFAULT_CORPUS, load_embeddings
)

__version__ = "1.0.0"
__suite__ = "Q41_TIER3_HECKE_OPERATORS"


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix."""
    from scipy.spatial.distance import pdist, squareform
    if metric == "cosine":
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        sim = X_norm @ X_norm.T
        D = 1.0 - sim
        np.fill_diagonal(D, 0)
    else:
        D = squareform(pdist(X, metric="euclidean"))
    return D


def normalized_l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute normalized L2 distance."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    dist = np.linalg.norm(x - y)
    scale = np.linalg.norm(x) + np.linalg.norm(y) + 1e-10
    return float(dist / scale)


def safe_float(val: Any) -> float:
    """Convert to float safely."""
    import math
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except:
        return 0.0


def random_orthogonal_matrix(d: int, seed: int) -> np.ndarray:
    """Generate random orthogonal matrix via QR decomposition."""
    rng = np.random.RandomState(seed)
    H = rng.randn(d, d)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


def generate_controls(X: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    """Generate control embeddings for testing."""
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Rotated (orthogonal transformation)
    Q = random_orthogonal_matrix(d, seed)
    rotated = X @ Q

    # Noise corrupted
    noise_scale = np.std(X) * 2.0
    noise_corrupted = X + rng.randn(n, d) * noise_scale

    return {
        "rotated": rotated,
        "noise_corrupted": noise_corrupted
    }


def build_hecke_operator(D: np.ndarray, k: int) -> np.ndarray:
    """
    Build Hecke-like averaging operator T_k.

    T_k averages over k-nearest neighbors.
    """
    n = len(D)
    T = np.zeros((n, n))
    knn_idx = np.argsort(D, axis=1)[:, 1:k+1]
    for i in range(n):
        T[i, knn_idx[i]] = 1.0 / k
    return T


def run_test(embeddings_dict: Dict[str, np.ndarray], config: TestConfig, verbose: bool = True) -> TestResult:
    """
    TIER 3: Hecke Operators

    TESTS:
    - 3.1: Commutativity - T_k T_l ≈ T_l T_k
    - 3.2: Self-adjointness - T close to symmetric after symmetrization
    - 3.3: Eigenvalue structure - bounded eigenvalues

    PASS CRITERIA:
    - Mean commutativity error < 0.3
    - Controls pass (rotated preserves, random breaks)
    """
    np.random.seed(config.seed)

    first_model = list(embeddings_dict.keys())[0]
    X = preprocess_embeddings(embeddings_dict[first_model], config.preprocessing)
    n = len(X)

    D = pairwise_distances(X, config.distance_metric)

    # Construct Hecke-like operators for different neighborhood sizes
    k_values = [3, 5, 7, 10]
    hecke_ops = {}

    if verbose:
        print("\n  Constructing Hecke operators...")

    for k in k_values:
        hecke_ops[k] = build_hecke_operator(D, k)

    # Test 1: Commutativity
    if verbose:
        print("  Testing commutativity T_k T_l = T_l T_k...")

    commutativity_errors = []
    for i, k1 in enumerate(k_values):
        for k2 in k_values[i+1:]:
            T1, T2 = hecke_ops[k1], hecke_ops[k2]
            comm_error = np.linalg.norm(T1 @ T2 - T2 @ T1, 'fro') / (n + 1e-10)
            commutativity_errors.append((k1, k2, safe_float(comm_error)))

    mean_comm_error = safe_float(np.mean([e[2] for e in commutativity_errors]))

    if verbose:
        print(f"    Mean commutativity error: {mean_comm_error:.4f}")

    # Test 2: Self-adjointness
    if verbose:
        print("  Testing self-adjointness...")

    symmetry_errors = []
    for k, T in hecke_ops.items():
        T_sym = (T + T.T) / 2
        sym_error = np.linalg.norm(T - T_sym, 'fro') / (np.linalg.norm(T, 'fro') + 1e-10)
        symmetry_errors.append((k, safe_float(sym_error)))

    mean_sym_error = safe_float(np.mean([e[1] for e in symmetry_errors]))

    if verbose:
        print(f"    Mean symmetry error: {mean_sym_error:.4f}")

    # Test 3: Eigenvalue structure
    if verbose:
        print("  Analyzing eigenvalue structure...")

    eigenvalue_stats = {}
    for k, T in hecke_ops.items():
        T_sym = (T + T.T) / 2
        eigs = np.sort(np.linalg.eigvalsh(T_sym))[::-1]

        spectral_gap = safe_float(eigs[0] - eigs[1]) if len(eigs) > 1 else 0
        max_eig = safe_float(eigs[0])
        min_eig = safe_float(eigs[-1])

        eigenvalue_stats[k] = {
            "spectral_gap": spectral_gap,
            "max_eigenvalue": max_eig,
            "min_eigenvalue": min_eig,
            "eigenvalue_range": safe_float(max_eig - min_eig)
        }

        if verbose:
            print(f"    k={k}: gap={spectral_gap:.4f}, range=[{min_eig:.3f}, {max_eig:.3f}]")

    # Cross-model comparison
    model_names = list(embeddings_dict.keys())
    cross_model_comm = []

    for name in model_names[1:]:
        X_other = preprocess_embeddings(embeddings_dict[name], config.preprocessing)
        D_other = pairwise_distances(X_other, config.distance_metric)

        k = 5
        T_other = build_hecke_operator(D_other, k)

        T_sym_first = (hecke_ops[k] + hecke_ops[k].T) / 2
        T_sym_other = (T_other + T_other.T) / 2

        eigs_first = np.sort(np.linalg.eigvalsh(T_sym_first))[::-1]
        eigs_other = np.sort(np.linalg.eigvalsh(T_sym_other))[::-1]

        min_len = min(len(eigs_first), len(eigs_other))
        spec_dist = normalized_l2_distance(eigs_first[:min_len], eigs_other[:min_len])
        cross_model_comm.append((first_model, name, safe_float(spec_dist)))

    mean_cross_model_dist = safe_float(np.mean([c[2] for c in cross_model_comm])) if cross_model_comm else 0

    # Controls
    controls_data = generate_controls(X, config.seed)

    # Positive control: rotated should preserve Hecke structure
    X_rot = controls_data["rotated"]
    D_rot = pairwise_distances(X_rot, config.distance_metric)
    T_rot = build_hecke_operator(D_rot, 5)

    T_sym_orig = (hecke_ops[5] + hecke_ops[5].T) / 2
    T_sym_rot = (T_rot + T_rot.T) / 2

    eigs_orig = np.sort(np.linalg.eigvalsh(T_sym_orig))[::-1]
    eigs_rot = np.sort(np.linalg.eigvalsh(T_sym_rot))[::-1]

    pos_error = float(np.max(np.abs(eigs_orig - eigs_rot)))
    positive_control_pass = bool(pos_error < 1e-6)

    # Negative control: random graph should have different structure
    rng = np.random.RandomState(config.seed)
    T_random = np.zeros((n, n))
    for i in range(n):
        random_neighbors = rng.choice(n, size=5, replace=False)
        random_neighbors = random_neighbors[random_neighbors != i][:5]
        if len(random_neighbors) > 0:
            T_random[i, random_neighbors] = 1.0 / len(random_neighbors)

    T_sym_random = (T_random + T_random.T) / 2
    eigs_random = np.sort(np.linalg.eigvalsh(T_sym_random))[::-1]

    neg_dist = normalized_l2_distance(eigs_orig, eigs_random)
    negative_control_pass = bool(neg_dist > 0.05)

    # Pass criteria
    passed = bool(mean_comm_error < 0.3 and positive_control_pass and negative_control_pass)

    if verbose:
        print(f"\n  Controls: positive={positive_control_pass}, negative={negative_control_pass}")

    return TestResult(
        name="TIER 3: Hecke Operators",
        test_type="langlands",
        passed=passed,
        metrics={
            "commutativity_errors": commutativity_errors,
            "mean_commutativity_error": mean_comm_error,
            "symmetry_errors": symmetry_errors,
            "mean_symmetry_error": mean_sym_error,
            "eigenvalue_stats": to_builtin(eigenvalue_stats),
            "cross_model_spectral_distances": cross_model_comm,
            "mean_cross_model_distance": mean_cross_model_dist
        },
        thresholds={
            "commutativity_threshold": 0.3,
            "positive_control_threshold": 1e-6,
            "negative_control_threshold": 0.05
        },
        controls={
            "positive_rotated_error": safe_float(pos_error),
            "positive_control_pass": positive_control_pass,
            "negative_random_distance": safe_float(neg_dist),
            "negative_control_pass": negative_control_pass
        },
        notes=f"Commutativity: {mean_comm_error:.3f}, Symmetry: {mean_sym_error:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Q41 TIER 3: Hecke Operators")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent.parent / "receipts"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print(f"Q41 TIER 3: HECKE OPERATORS v{__version__}")
        print("=" * 60)

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
        print(f"\n{'=' * 60}")
        status = "PASS" if result.passed else "FAIL"
        print(f"Result: {status}")
        print(f"Notes: {result.notes}")

    # Save receipt
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = out_dir / f"q41_tier3_receipt_{timestamp_str}.json"

    receipt = {
        "suite": __suite__,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": result.passed,
        "metrics": to_builtin(result.metrics),
        "thresholds": to_builtin(result.thresholds),
        "controls": to_builtin(result.controls),
        "notes": result.notes
    }

    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    if verbose:
        print(f"Receipt saved: {receipt_path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
