#!/usr/bin/env python3
"""
Eigen-Spectrum Alignment Proof of Concept

Quick validation that eigenvalue-based alignment works cross-model.
Uses proper MDS (double-centered Gram matrix) and Procrustes rotation.

Result: Does mapping via eigenvectors improve cross-model similarity?
"""

import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
from typing import Dict, List, Tuple

# Models to test
MODELS = {
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'e5-large-v2': 'intfloat/e5-large-v2',
}

ANCHORS = ['dog', 'love', 'up', 'true', 'king', 'water', 'run', 'happy']
TEST_WORDS = ['cat', 'hate', 'down', 'false', 'queen', 'fire', 'walk', 'sad']

_model_cache = {}


def load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {name}...", end=" ", flush=True)
    model = SentenceTransformer(MODELS[name])
    print("done")
    _model_cache[name] = model
    return model


def get_embeddings(name: str, words: List[str]) -> np.ndarray:
    model = load_model(name)
    prefix = 'query: ' if name == 'e5-large-v2' else ''
    texts = [prefix + w for w in words]
    return model.encode(texts, normalize_embeddings=True)


def compute_squared_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distance matrix."""
    # For normalized embeddings: d² = 2(1 - cos_sim)
    cos_sim = embeddings @ embeddings.T
    return 2 * (1 - cos_sim)


def classical_mds(D2: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical MDS via double-centered Gram matrix.

    B = -1/2 * J * D² * J  where J = I - (1/n)*11^T

    Returns: (coordinates X, eigenvalues Λ, eigenvectors V)
    """
    n = D2.shape[0]

    # Centering matrix J = I - (1/n) * 1 * 1^T
    H = np.eye(n) - np.ones((n, n)) / n

    # Double-centered Gram matrix
    B = -0.5 * H @ D2 @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues
    pos_mask = eigenvalues > 1e-10
    if k is None:
        k = np.sum(pos_mask)
    k = min(k, np.sum(pos_mask))

    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]

    # MDS coordinates: X = V * sqrt(Λ)
    X = eigenvectors * np.sqrt(eigenvalues)

    return X, eigenvalues, eigenvectors


def procrustes_align(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find orthogonal rotation R that minimizes ||X_source @ R - X_target||
    """
    # Ensure same dimensionality
    k = min(X_source.shape[1], X_target.shape[1])
    X_s = X_source[:, :k]
    X_t = X_target[:, :k]

    R, scale = orthogonal_procrustes(X_s, X_t)

    # Compute residual
    aligned = X_s @ R
    residual = np.linalg.norm(aligned - X_t, 'fro')

    return R, residual


def out_of_sample_mds(d2_to_anchors: np.ndarray, D2_anchors: np.ndarray,
                       V: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """
    Project new points into MDS space using Gower's formula.

    d2_to_anchors: (m, n) squared distances from m new points to n anchors
    """
    n = D2_anchors.shape[0]

    # Row means of anchor distance matrix
    r = D2_anchors.mean(axis=1)  # (n,)
    r_bar = r.mean()

    # For each new point
    m = d2_to_anchors.shape[0]
    coords = []

    for i in range(m):
        d2 = d2_to_anchors[i]  # (n,)
        d_bar = d2.mean()

        # Gower's formula: b = -0.5 * (d² - d_bar - r + r_bar)
        b = -0.5 * (d2 - d_bar - r + r_bar)

        # Project: y = Λ^(-1/2) * V^T * b
        y = (V.T @ b) / np.sqrt(eigenvalues)
        coords.append(y)

    return np.array(coords)


def test_alignment():
    """Main test: does eigenvector alignment improve cross-model similarity?"""
    print("=" * 60)
    print("EIGEN-SPECTRUM ALIGNMENT PROOF OF CONCEPT")
    print("=" * 60)

    # Get anchor embeddings
    print("\n1. Loading embeddings...")
    emb_mini_anchors = get_embeddings('all-MiniLM-L6-v2', ANCHORS)
    emb_e5_anchors = get_embeddings('e5-large-v2', ANCHORS)
    emb_mini_test = get_embeddings('all-MiniLM-L6-v2', TEST_WORDS)
    emb_e5_test = get_embeddings('e5-large-v2', TEST_WORDS)

    # Compute distance matrices
    print("\n2. Computing MDS...")
    D2_mini = compute_squared_distance_matrix(emb_mini_anchors)
    D2_e5 = compute_squared_distance_matrix(emb_e5_anchors)

    # MDS
    X_mini, λ_mini, V_mini = classical_mds(D2_mini)
    X_e5, λ_e5, V_e5 = classical_mds(D2_e5)

    print(f"   MiniLM eigenvalues: {λ_mini[:5].round(4)}")
    print(f"   E5 eigenvalues:     {λ_e5[:5].round(4)}")

    # Eigenvalue correlation
    k = min(len(λ_mini), len(λ_e5))
    eig_corr = np.corrcoef(λ_mini[:k], λ_e5[:k])[0, 1]
    print(f"   Eigenvalue correlation: {eig_corr:.4f}")

    # Procrustes alignment
    print("\n3. Computing Procrustes alignment...")
    R, residual = procrustes_align(X_mini, X_e5)
    print(f"   Rotation matrix R: {R.shape}")
    print(f"   Alignment residual: {residual:.4f}")

    # Project test words
    print("\n4. Projecting test words...")

    # Distances from test words to anchors
    d2_mini_test = cdist(emb_mini_test, emb_mini_anchors, 'sqeuclidean')
    d2_e5_test = cdist(emb_e5_test, emb_e5_anchors, 'sqeuclidean')

    # Out-of-sample MDS coordinates
    Y_mini = out_of_sample_mds(d2_mini_test, D2_mini, V_mini, λ_mini)
    Y_e5 = out_of_sample_mds(d2_e5_test, D2_e5, V_e5, λ_e5)

    # Align MiniLM test coords to E5 space
    k = min(Y_mini.shape[1], R.shape[0])
    Y_mini_aligned = Y_mini[:, :k] @ R[:k, :k]

    # Compare
    print("\n5. Results:")
    print("-" * 40)

    # Before alignment: raw MDS coord similarity
    raw_sims = []
    for i in range(len(TEST_WORDS)):
        k_min = min(Y_mini.shape[1], Y_e5.shape[1])
        sim = np.dot(Y_mini[i, :k_min], Y_e5[i, :k_min]) / (
            np.linalg.norm(Y_mini[i, :k_min]) * np.linalg.norm(Y_e5[i, :k_min]) + 1e-10
        )
        raw_sims.append(sim)

    # After alignment
    aligned_sims = []
    for i in range(len(TEST_WORDS)):
        k_min = min(Y_mini_aligned.shape[1], Y_e5.shape[1])
        sim = np.dot(Y_mini_aligned[i, :k_min], Y_e5[i, :k_min]) / (
            np.linalg.norm(Y_mini_aligned[i, :k_min]) * np.linalg.norm(Y_e5[i, :k_min]) + 1e-10
        )
        aligned_sims.append(sim)

    print(f"{'Word':<10} {'Raw MDS sim':<15} {'Aligned sim':<15} {'Δ':<10}")
    print("-" * 50)
    for i, word in enumerate(TEST_WORDS):
        delta = aligned_sims[i] - raw_sims[i]
        print(f"{word:<10} {raw_sims[i]:<15.4f} {aligned_sims[i]:<15.4f} {delta:+.4f}")

    print("-" * 50)
    print(f"{'MEAN':<10} {np.mean(raw_sims):<15.4f} {np.mean(aligned_sims):<15.4f} {np.mean(aligned_sims) - np.mean(raw_sims):+.4f}")

    # Verdict
    print("\n" + "=" * 60)
    improvement = np.mean(aligned_sims) - np.mean(raw_sims)
    if improvement > 0.05:
        print("VERDICT: ALIGNMENT IMPROVES CROSS-MODEL SIMILARITY")
        print(f"Mean improvement: {improvement:+.4f}")
    elif improvement > 0:
        print("VERDICT: MARGINAL IMPROVEMENT")
        print(f"Mean improvement: {improvement:+.4f}")
    else:
        print("VERDICT: NO IMPROVEMENT (needs investigation)")
        print(f"Mean change: {improvement:+.4f}")
    print("=" * 60)

    return {
        'eigenvalue_correlation': float(eig_corr),
        'procrustes_residual': float(residual),
        'mean_raw_similarity': float(np.mean(raw_sims)),
        'mean_aligned_similarity': float(np.mean(aligned_sims)),
        'improvement': float(improvement),
    }


if __name__ == "__main__":
    results = test_alignment()

    # Save results
    with open('eigen_alignment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: eigen_alignment_results.json")
