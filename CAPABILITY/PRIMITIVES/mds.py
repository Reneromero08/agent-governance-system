"""Classical Multidimensional Scaling (MDS).

Implements classical MDS via double-centered Gram matrix for
deriving coordinates from distance matrices.

Key insight: The eigenvalue spectrum of the Gram matrix is
INVARIANT across embedding models (r > 0.99), even when raw
distance matrices are uncorrelated.

This enables cross-model vector communication: different models
see the same geometric "shape" but with different orientations.

References:
    - Torgerson (1952): Classical MDS
    - Gower (1966): Adding a point to a principal coordinate analysis
"""

from typing import Tuple, Union
import numpy as np


def squared_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distance matrix from embeddings.

    For L2-normalized embeddings, uses the identity:
        d^2(a, b) = 2(1 - cos_sim(a, b))

    Args:
        embeddings: (n, d) array of L2-normalized embeddings

    Returns:
        (n, n) squared distance matrix D^2 where D^2[i,j] = ||e_i - e_j||^2
    """
    # For normalized embeddings: ||a - b||^2 = 2(1 - a.b)
    cos_sim = embeddings @ embeddings.T
    return 2 * (1 - cos_sim)


def centering_matrix(n: int) -> np.ndarray:
    """Compute the centering matrix J = I - (1/n)11^T.

    The centering matrix removes the mean from each row/column
    when used as J @ X @ J.

    Args:
        n: Size of the matrix

    Returns:
        (n, n) centering matrix
    """
    return np.eye(n) - np.ones((n, n)) / n


def classical_mds(
    D2: np.ndarray,
    k: Union[int, None] = None,
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical MDS via double-centered Gram matrix.

    Computes coordinates X such that the pairwise distances
    in X approximate the input distance matrix D.

    The key formula:
        B = -1/2 * J * D^2 * J  (double-centered Gram matrix)
        B = V L V^T             (eigendecomposition)
        X = V sqrt(L)           (MDS coordinates)

    Args:
        D2: (n, n) squared distance matrix
        k: Number of dimensions to retain (default: all positive eigenvalues)
        epsilon: Threshold for positive eigenvalues

    Returns:
        Tuple of:
            X: (n, k) MDS coordinates
            eigenvalues: (k,) positive eigenvalues (sorted descending)
            eigenvectors: (n, k) corresponding eigenvectors
    """
    n = D2.shape[0]

    # Centering matrix: J = I - (1/n) * 1 * 1^T
    J = centering_matrix(n)

    # Double-centered Gram matrix: B = -1/2 * J * D^2 * J
    B = -0.5 * J @ D2 @ J

    # Eigendecomposition (symmetric, so use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort descending by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep only positive eigenvalues
    pos_mask = eigenvalues > epsilon
    n_positive = np.sum(pos_mask)

    if k is None:
        k = n_positive
    else:
        k = min(k, n_positive)

    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]

    # MDS coordinates: X = V * sqrt(L)
    X = eigenvectors * np.sqrt(eigenvalues)

    return X, eigenvalues, eigenvectors


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Compute effective rank (participation ratio) of eigenvalue spectrum.

    Effective rank measures how many eigenvalues contribute significantly.
    For a uniform spectrum, effective_rank = n.
    For a single dominant eigenvalue, effective_rank ~ 1.

    Formula: (sum(L))^2 / sum(L^2)

    Args:
        eigenvalues: Array of eigenvalues (need not be normalized)

    Returns:
        Effective rank as a float
    """
    eigenvalues = np.abs(eigenvalues)
    if np.sum(eigenvalues) == 0:
        return 0.0
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def stress(D_original: np.ndarray, D_reconstructed: np.ndarray) -> float:
    """Compute Kruskal's stress for MDS quality assessment.

    Stress measures how well the MDS coordinates reproduce
    the original distances. Lower is better.

    Formula: sqrt(sum((d_ij - d_hat_ij)^2) / sum(d_ij^2))

    Args:
        D_original: Original distance matrix
        D_reconstructed: Distance matrix from MDS coordinates

    Returns:
        Stress value (0 = perfect, >0.2 = poor)
    """
    # Use upper triangle only (avoid counting twice)
    n = D_original.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    d_orig = D_original[mask]
    d_recon = D_reconstructed[mask]

    numerator = np.sum((d_orig - d_recon) ** 2)
    denominator = np.sum(d_orig ** 2)

    if denominator == 0:
        return 0.0

    return np.sqrt(numerator / denominator)
