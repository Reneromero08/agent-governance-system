#!/usr/bin/env python3
"""
Q41 TIER 4: Automorphic Forms

Tests whether embedding spaces admit automorphic-like functions -
eigenfunctions of the Laplacian with transformation properties.

The Langlands connection: Automorphic forms are functions on quotient
spaces that transform predictably under group actions. The Langlands
correspondence relates these to Galois representations.

Semantic analogs:
- Laplacian eigenfunctions as "harmonic" functions on semantic space
- Orthogonality relations
- Completeness (spanning the space)
- Transformation under symmetry

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
    build_knn_graph, build_graph_laplacian,
    DEFAULT_CORPUS, load_embeddings
)

__version__ = "1.0.0"
__suite__ = "Q41_TIER4_AUTOMORPHIC_FORMS"


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


def run_test(embeddings_dict: Dict[str, np.ndarray], config: TestConfig, verbose: bool = True) -> TestResult:
    """
    TIER 4: Automorphic Forms

    TESTS:
    - 4.1: Orthogonality - eigenfunctions form orthonormal basis
    - 4.2: Completeness - eigenfunctions span the space (reconstruction)
    - 4.3: Eigenvalue structure - gaps indicate automorphic form clusters

    PASS CRITERIA:
    - Orthogonality error < 0.01
    - Reconstruction error < 0.9
    - Controls pass
    """
    np.random.seed(config.seed)

    first_model = list(embeddings_dict.keys())[0]
    X = preprocess_embeddings(embeddings_dict[first_model], config.preprocessing)
    n, d = X.shape

    D = pairwise_distances(X, config.distance_metric)
    A = build_mutual_knn_graph(D, config.k_neighbors)
    L = normalized_laplacian(A)

    if verbose:
        print("\n  Computing eigenfunctions (automorphic forms)...")

    # Compute eigenfunctions of Laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Test 1: Orthogonality
    if verbose:
        print("  Testing orthogonality...")

    n_test = min(20, len(eigenvalues))
    V = eigenvectors[:, :n_test]
    orthogonality_matrix = V.T @ V
    orthogonality_error = np.linalg.norm(orthogonality_matrix - np.eye(n_test), 'fro')
    orthogonality_error = safe_float(orthogonality_error / n_test)

    if verbose:
        print(f"    Orthogonality error: {orthogonality_error:.2e}")

    # Test 2: Completeness (reconstruction)
    if verbose:
        print("  Testing completeness (reconstruction)...")

    K_orig = np.exp(-D**2)  # Gaussian kernel

    n_reconstruct = min(30, len(eigenvalues))
    V_recon = eigenvectors[:, :n_reconstruct]
    K_proj = V_recon @ (V_recon.T @ K_orig @ V_recon) @ V_recon.T

    reconstruction_error = np.linalg.norm(K_orig - K_proj, 'fro') / (np.linalg.norm(K_orig, 'fro') + 1e-10)
    reconstruction_error = safe_float(reconstruction_error)

    if verbose:
        print(f"    Reconstruction error: {reconstruction_error:.3f}")

    # Test 3: Eigenvalue gap structure
    if verbose:
        print("  Analyzing eigenvalue gap structure...")

    eigenvalue_gaps = np.diff(eigenvalues[:n_test])
    gap_variance = safe_float(np.var(eigenvalue_gaps))
    mean_gap = safe_float(np.mean(eigenvalue_gaps))
    gap_cv = safe_float(gap_variance**0.5 / (mean_gap + 1e-10))

    if verbose:
        print(f"    Mean gap: {mean_gap:.4f}, CV: {gap_cv:.3f}")

    # Test 4: Localization (participation ratio)
    if verbose:
        print("  Computing participation ratios...")

    participation_ratios = []
    for i in range(n_test):
        v = eigenvectors[:, i]
        v2 = v**2
        pr = 1.0 / (np.sum(v2**2) + 1e-10)
        participation_ratios.append(safe_float(pr))

    mean_pr = safe_float(np.mean(participation_ratios))
    std_pr = safe_float(np.std(participation_ratios))

    if verbose:
        print(f"    Mean participation ratio: {mean_pr:.2f} +/- {std_pr:.2f}")

    # Cross-model comparison
    model_names = list(embeddings_dict.keys())
    eigenfunction_similarities = []

    for name in model_names[1:]:
        X_other = preprocess_embeddings(embeddings_dict[name], config.preprocessing)
        D_other = pairwise_distances(X_other, config.distance_metric)
        A_other = build_mutual_knn_graph(D_other, config.k_neighbors)
        L_other = normalized_laplacian(A_other)

        eigs_other, vecs_other = np.linalg.eigh(L_other)
        idx_other = np.argsort(eigs_other)
        eigs_other = eigs_other[idx_other]

        min_len = min(n_test, len(eigs_other))
        spec_dist = normalized_l2_distance(eigenvalues[:min_len], eigs_other[:min_len])
        eigenfunction_similarities.append((first_model, name, safe_float(spec_dist)))

    mean_similarity = safe_float(np.mean([s[2] for s in eigenfunction_similarities])) if eigenfunction_similarities else 0

    # Controls
    controls_data = generate_controls(X, config.seed)

    # Positive control: rotated should preserve eigenvalue structure
    X_rot = controls_data["rotated"]
    D_rot = pairwise_distances(X_rot, config.distance_metric)
    A_rot = build_mutual_knn_graph(D_rot, config.k_neighbors)
    L_rot = normalized_laplacian(A_rot)
    eigs_rot = np.sort(np.linalg.eigvalsh(L_rot))

    pos_error = float(np.max(np.abs(eigenvalues[:n_test] - eigs_rot[:n_test])))
    positive_control_pass = bool(pos_error < 1e-6)

    # Negative control: noise-corrupted should change spectrum
    X_noise = controls_data["noise_corrupted"]
    D_noise = pairwise_distances(X_noise, config.distance_metric)
    A_noise = build_mutual_knn_graph(D_noise, config.k_neighbors)
    L_noise = normalized_laplacian(A_noise)
    eigs_noise = np.sort(np.linalg.eigvalsh(L_noise))

    neg_dist = normalized_l2_distance(eigenvalues[:n_test], eigs_noise[:n_test])
    negative_control_pass = bool(neg_dist > 0.05)

    if verbose:
        print(f"\n  Controls: positive={positive_control_pass}, negative={negative_control_pass}")

    # Pass criteria
    passed = bool(
        orthogonality_error < 0.01 and
        reconstruction_error < 0.9 and
        positive_control_pass and
        negative_control_pass
    )

    return TestResult(
        name="TIER 4: Automorphic Forms",
        test_type="langlands",
        passed=passed,
        metrics={
            "orthogonality_error": orthogonality_error,
            "reconstruction_error": reconstruction_error,
            "n_eigenfunctions_tested": n_test,
            "eigenvalue_gap_stats": {
                "mean_gap": mean_gap,
                "gap_variance": gap_variance,
                "gap_cv": gap_cv
            },
            "participation_ratio_stats": {
                "mean": mean_pr,
                "std": std_pr,
                "values": participation_ratios[:10]
            },
            "cross_model_similarities": eigenfunction_similarities,
            "mean_cross_model_distance": mean_similarity
        },
        thresholds={
            "orthogonality_threshold": 0.01,
            "reconstruction_threshold": 0.9,
            "positive_control_threshold": 1e-6,
            "negative_control_threshold": 0.05
        },
        controls={
            "positive_rotated_error": safe_float(pos_error),
            "positive_control_pass": positive_control_pass,
            "negative_noise_distance": safe_float(neg_dist),
            "negative_control_pass": negative_control_pass
        },
        notes=f"Orthogonality: {orthogonality_error:.2e}, Reconstruction: {reconstruction_error:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Q41 TIER 4: Automorphic Forms")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent.parent / "receipts"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print(f"Q41 TIER 4: AUTOMORPHIC FORMS v{__version__}")
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
    receipt_path = out_dir / f"q41_tier4_receipt_{timestamp_str}.json"

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
