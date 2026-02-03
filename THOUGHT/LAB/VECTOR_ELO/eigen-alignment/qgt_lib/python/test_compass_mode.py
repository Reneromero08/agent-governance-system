#!/usr/bin/env python3
"""
Test: QGT Principal Directions = Compass Mode (E.X MDS Eigenvectors)

Hypothesis: The natural gradient directions from QGT (Fubini-Study metric
eigenvectors) should match the MDS eigenvectors used in E.X alignment.

Mathematical basis:
- QGT metric = covariance matrix C = X^T X / n
- MDS Gram matrix = X X^T (for centered data)
- Same non-zero eigenvalues (related by n factor)
- Eigenvectors are related: if Cv = lambda*v, then Xv is Gram eigenvector
"""

import sys
import numpy as np
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'benchmarks' / 'validation'))

# Import QGT
from qgt_lib.python import qgt

# Import E.X modules
from lib.mds import classical_mds, squared_distance_matrix
from untrained_transformer import (
    get_trained_bert_embeddings,
    ANCHOR_WORDS,
    HELD_OUT_WORDS,
)


def subspace_alignment(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute alignment between two subspaces via principal angles.

    Returns average cosine of principal angles (1.0 = identical subspaces).
    """
    # Ensure column vectors
    if U1.shape[0] < U1.shape[1]:
        U1 = U1.T
    if U2.shape[0] < U2.shape[1]:
        U2 = U2.T

    # Orthonormalize
    U1, _ = np.linalg.qr(U1)
    U2, _ = np.linalg.qr(U2)

    # SVD of U1^T @ U2 gives cosines of principal angles
    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)

    return float(np.mean(s))


def main():
    print("=" * 70)
    print("TEST: QGT Principal Directions = Compass Mode (MDS Eigenvectors)")
    print("=" * 70)
    print()

    # Get words
    all_words = list(set(ANCHOR_WORDS + HELD_OUT_WORDS))
    print(f"Using {len(all_words)} words")
    print()

    # Get embeddings
    print("Loading trained BERT embeddings...")
    trained_emb, _ = get_trained_bert_embeddings(all_words)
    emb_matrix = np.array([trained_emb[w] for w in all_words])
    print(f"  Shape: {emb_matrix.shape}")
    print()

    # Normalize
    emb_normalized = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

    # === Method 1: QGT Principal Directions ===
    print("-" * 70)
    print("Method 1: QGT (Fubini-Study Metric Eigenvectors)")
    print("-" * 70)

    qgt_eigenvalues, qgt_eigenvectors = qgt.metric_eigenspectrum(emb_normalized, normalize=False)
    print(f"  Top 5 eigenvalues: {qgt_eigenvalues[:5]}")
    print(f"  QGT eigenvector shape: {qgt_eigenvectors.shape}")
    print()

    # === Method 2: MDS Eigenvectors ===
    print("-" * 70)
    print("Method 2: MDS (Double-Centered Gram Matrix Eigenvectors)")
    print("-" * 70)

    # Compute MDS
    D2 = squared_distance_matrix(emb_normalized)
    mds_coords, mds_eigenvalues, mds_eigenvectors = classical_mds(D2, k=50)

    print(f"  Top 5 eigenvalues: {mds_eigenvalues[:5]}")
    print(f"  MDS eigenvector shape: {mds_eigenvectors.shape}")
    print()

    # === Compare Subspaces ===
    print("-" * 70)
    print("Subspace Comparison")
    print("-" * 70)

    # The QGT eigenvectors are in embedding space (768D)
    # The MDS eigenvectors are in sample space (115D)
    # They're related by: mds_eigenvector = X @ qgt_eigenvector / sqrt(n * lambda)

    # Transform QGT eigenvectors to sample space for comparison
    # v_sample = X @ v_embedding (normalized)
    n_compare = 22  # Compare top 22 dimensions

    # Project embeddings onto QGT principal components
    qgt_projections = emb_normalized @ qgt_eigenvectors[:, :n_compare]
    qgt_projections = qgt_projections / np.linalg.norm(qgt_projections, axis=0, keepdims=True)

    # MDS eigenvectors (already in sample space)
    mds_vecs = mds_eigenvectors[:, :n_compare]
    mds_vecs = mds_vecs / np.linalg.norm(mds_vecs, axis=0, keepdims=True)

    # Compute alignment
    alignment = subspace_alignment(qgt_projections, mds_vecs)
    print(f"  Subspace alignment (22D): {alignment:.4f}")
    print()

    # Individual direction alignments
    print("  Per-direction alignment (|cos|):")
    for i in range(min(10, n_compare)):
        qgt_proj = qgt_projections[:, i]

        # Find best matching MDS direction
        best_cos = 0
        best_j = 0
        for j in range(n_compare):
            cos_sim = abs(np.dot(qgt_proj, mds_vecs[:, j]))
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_j = j

        print(f"    QGT[{i+1}] <-> MDS[{best_j+1}]: {best_cos:.4f}")
    print()

    # === Eigenvalue Comparison ===
    print("-" * 70)
    print("Eigenvalue Spectrum Comparison")
    print("-" * 70)

    # Scale MDS eigenvalues to match QGT scale
    # MDS eigenvalues are from Gram matrix, QGT from covariance
    # Relation: lambda_mds = n * lambda_qgt (approximately)
    n_samples = len(all_words)
    scaled_mds = mds_eigenvalues / n_samples

    print("  First 10 eigenvalues:")
    print(f"  {'Rank':<6}{'QGT':<15}{'MDS (scaled)':<15}{'Ratio':<10}")
    for i in range(10):
        ratio = qgt_eigenvalues[i] / scaled_mds[i] if scaled_mds[i] > 1e-10 else float('inf')
        print(f"  {i+1:<6}{qgt_eigenvalues[i]:<15.6f}{scaled_mds[i]:<15.6f}{ratio:<10.2f}")
    print()

    # Correlation of eigenvalue spectra
    k = min(50, len(qgt_eigenvalues), len(scaled_mds))
    corr = np.corrcoef(qgt_eigenvalues[:k], scaled_mds[:k])[0, 1]
    print(f"  Eigenvalue correlation (top 50): {corr:.4f}")
    print()

    # === Verdict ===
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    if alignment > 0.8:
        print(f"  [PASS] Subspace alignment = {alignment:.4f} (>0.8)")
        print("  QGT principal directions match MDS eigenvectors!")
        print("  Natural gradient = Compass mode CONFIRMED")
        status = "CONFIRMED"
    elif alignment > 0.5:
        print(f"  [PARTIAL] Subspace alignment = {alignment:.4f} (0.5-0.8)")
        print("  Moderate agreement between QGT and MDS directions")
        status = "PARTIAL"
    else:
        print(f"  [FAIL] Subspace alignment = {alignment:.4f} (<0.5)")
        print("  QGT and MDS directions don't match well")
        status = "FAILED"
    print()

    if corr > 0.95:
        print(f"  [PASS] Eigenvalue correlation = {corr:.4f} (>0.95)")
        print("  Same spectral structure!")
    else:
        print(f"  [INFO] Eigenvalue correlation = {corr:.4f}")
    print()

    return {
        'subspace_alignment': alignment,
        'eigenvalue_correlation': corr,
        'status': status,
    }


if __name__ == '__main__':
    main()
