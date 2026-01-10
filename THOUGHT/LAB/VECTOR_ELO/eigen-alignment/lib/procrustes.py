"""Procrustes Alignment and Out-of-Sample Projection.

Implements orthogonal Procrustes analysis for aligning coordinate
systems between embedding models, plus Gower's formula for
projecting new points into the aligned space.

The key insight: Different models have the same underlying
"shape" (eigenvalue spectrum) but different orientations.
Procrustes finds the optimal rotation to align them.

References:
    - Schönemann (1966): A generalized solution of the orthogonal Procrustes problem
    - Gower (1968): Adding a point to a principal coordinate analysis
"""

from typing import Tuple
import numpy as np
from scipy.linalg import orthogonal_procrustes


def procrustes_align(
    X_source: np.ndarray,
    X_target: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Find optimal orthogonal rotation to align source to target.

    Finds rotation matrix R that minimizes:
        ||X_source @ R - X_target||_F

    Uses SVD-based orthogonal Procrustes solution.

    Args:
        X_source: (n, k) source coordinates (to be rotated)
        X_target: (n, k) target coordinates (reference)

    Returns:
        Tuple of:
            R: (k, k) orthogonal rotation matrix
            residual: Frobenius norm of alignment error
    """
    # Ensure same dimensionality
    k = min(X_source.shape[1], X_target.shape[1])
    X_s = X_source[:, :k]
    X_t = X_target[:, :k]

    # Orthogonal Procrustes: find R minimizing ||X_s R - X_t||
    R, scale = orthogonal_procrustes(X_s, X_t)

    # Compute residual
    aligned = X_s @ R
    residual = np.linalg.norm(aligned - X_t, 'fro')

    return R, residual


def out_of_sample_mds(
    d2_to_anchors: np.ndarray,
    D2_anchors: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray
) -> np.ndarray:
    """Project new points into MDS space using Gower's formula.

    Given squared distances from new points to the anchor set,
    computes their coordinates in the MDS space without
    recomputing the full MDS.

    Gower's formula:
        b = -1/2 * (d² - d̄ - r + r̄)
        y = Λ^(-1/2) * V^T * b

    Args:
        d2_to_anchors: (m, n) squared distances from m new points to n anchors
        D2_anchors: (n, n) squared distance matrix of anchors
        eigenvectors: (n, k) eigenvectors from anchor MDS
        eigenvalues: (k,) eigenvalues from anchor MDS

    Returns:
        (m, k) MDS coordinates for the new points
    """
    n = D2_anchors.shape[0]

    # Row means of anchor distance matrix: r_i = mean_j D²[i,j]
    r = D2_anchors.mean(axis=1)  # (n,)

    # Grand mean: r̄ = mean_i r_i
    r_bar = r.mean()

    # Process each new point
    m = d2_to_anchors.shape[0]
    coords = []

    for i in range(m):
        d2 = d2_to_anchors[i]  # (n,) distances to anchors

        # Mean of new point's distances: d̄ = mean_i d²_i
        d_bar = d2.mean()

        # Gower's formula: b = -1/2 * (d² - d̄ - r + r̄)
        b = -0.5 * (d2 - d_bar - r + r_bar)

        # Project: y = Λ^(-1/2) * V^T * b
        y = (eigenvectors.T @ b) / np.sqrt(eigenvalues)
        coords.append(y)

    return np.array(coords)


def align_points(
    points: np.ndarray,
    rotation: np.ndarray
) -> np.ndarray:
    """Apply rotation matrix to align points.

    Args:
        points: (m, k) points in source coordinate system
        rotation: (k, k) rotation matrix from procrustes_align

    Returns:
        (m, k) aligned points in target coordinate system
    """
    k = min(points.shape[1], rotation.shape[0])
    return points[:, :k] @ rotation[:k, :k]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def alignment_quality(
    X_source: np.ndarray,
    X_target: np.ndarray,
    R: np.ndarray
) -> dict:
    """Compute alignment quality metrics.

    Args:
        X_source: (n, k) source coordinates
        X_target: (n, k) target coordinates
        R: (k, k) rotation matrix

    Returns:
        Dict with quality metrics:
            - residual: Frobenius norm of error
            - relative_error: residual / ||X_target||
            - mean_cosine_sim: mean cosine similarity after alignment
            - per_point_sims: per-point cosine similarities
    """
    k = min(X_source.shape[1], X_target.shape[1], R.shape[0])
    X_s = X_source[:, :k]
    X_t = X_target[:, :k]
    R_k = R[:k, :k]

    aligned = X_s @ R_k
    residual = np.linalg.norm(aligned - X_t, 'fro')
    target_norm = np.linalg.norm(X_t, 'fro')

    # Per-point cosine similarities
    sims = []
    for i in range(aligned.shape[0]):
        sim = cosine_similarity(aligned[i], X_t[i])
        sims.append(sim)

    return {
        'residual': float(residual),
        'relative_error': float(residual / target_norm) if target_norm > 0 else 0.0,
        'mean_cosine_sim': float(np.mean(sims)),
        'per_point_sims': sims
    }
