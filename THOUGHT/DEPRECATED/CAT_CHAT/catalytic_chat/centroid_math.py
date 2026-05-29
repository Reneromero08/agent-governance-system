#!/usr/bin/env python3
"""
Centroid Math (Phase J.1)

Mathematical operations for centroid hierarchy and E-score computation.

Key Functions:
- compute_centroid: Compute mean centroid from a list of vectors
- update_centroid_incremental: Update centroid without recomputing from scratch
- compute_E: Compute Born rule E-score (quantum-inspired relevance measure)

The E-score uses the Born rule from quantum mechanics: E = |<q|i>|^2
This is the squared cosine similarity (squared inner product of unit vectors).

Part of Phase J.1: Centroid Hierarchy Schema.
"""

from typing import List, Optional, Tuple
import numpy as np


# Embedding dimensions (must match hierarchy_schema.py and vector_persistence.py)
EMBEDDING_DIM = 384


def compute_centroid(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute the centroid (mean) of a list of vectors.

    The centroid is the element-wise arithmetic mean of all input vectors.
    This provides a representative vector for a cluster of embeddings.

    Args:
        vectors: List of numpy arrays, each of shape (384,)

    Returns:
        Centroid vector of shape (384,) with float32 dtype

    Raises:
        ValueError: If vectors list is empty or vectors have wrong shape

    Example:
        >>> v1 = np.array([1.0, 0.0], dtype=np.float32)
        >>> v2 = np.array([0.0, 1.0], dtype=np.float32)
        >>> centroid = compute_centroid([v1, v2])
        >>> print(centroid)  # [0.5, 0.5]
    """
    if not vectors:
        raise ValueError("Cannot compute centroid of empty vector list")

    # Validate shapes
    for i, v in enumerate(vectors):
        if v.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Vector {i} has shape {v.shape}, expected ({EMBEDDING_DIM},)"
            )

    # Stack and compute mean
    stacked = np.stack(vectors, axis=0)
    centroid = np.mean(stacked, axis=0)

    return centroid.astype(np.float32)


def update_centroid_incremental(
    old_centroid: np.ndarray,
    old_count: int,
    new_vec: np.ndarray
) -> np.ndarray:
    """Update centroid incrementally with a new vector.

    Uses the running mean formula: new_centroid = (old * n + new) / (n + 1)

    This avoids recomputing the full mean when adding a single vector,
    which is O(1) instead of O(n).

    Args:
        old_centroid: Current centroid vector of shape (384,)
        old_count: Number of vectors in the current centroid
        new_vec: New vector to add of shape (384,)

    Returns:
        Updated centroid vector of shape (384,) with float32 dtype

    Raises:
        ValueError: If old_count is negative or shapes are wrong

    Example:
        >>> centroid = np.array([1.0, 1.0], dtype=np.float32)
        >>> new_vec = np.array([4.0, 4.0], dtype=np.float32)
        >>> updated = update_centroid_incremental(centroid, 2, new_vec)
        >>> print(updated)  # [2.0, 2.0] = (1*2 + 4) / 3
    """
    if old_count < 0:
        raise ValueError(f"old_count must be non-negative, got {old_count}")

    if old_centroid.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"old_centroid has shape {old_centroid.shape}, "
            f"expected ({EMBEDDING_DIM},)"
        )

    if new_vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"new_vec has shape {new_vec.shape}, expected ({EMBEDDING_DIM},)"
        )

    # Handle edge case of first vector
    if old_count == 0:
        return new_vec.astype(np.float32)

    # Running mean formula: new_mean = (old_mean * n + new_value) / (n + 1)
    new_centroid = (old_centroid * old_count + new_vec) / (old_count + 1)

    return new_centroid.astype(np.float32)


def compute_E(query_vec: np.ndarray, item_vec: np.ndarray) -> float:
    """Compute E-score using Born rule: |<q|i>|^2 (cosine similarity squared).

    The E-score is inspired by the Born rule in quantum mechanics, where
    the probability of measuring a state is the squared magnitude of
    the inner product. For unit vectors, this equals cosine^2.

    Properties:
    - Range: [0, 1]
    - E = 1 when vectors are identical (or anti-parallel)
    - E = 0 when vectors are orthogonal
    - E is symmetric: E(q, i) = E(i, q)

    Args:
        query_vec: Query vector of shape (384,)
        item_vec: Item vector of shape (384,)

    Returns:
        E-score in range [0, 1]

    Raises:
        ValueError: If vectors have wrong shape

    Example:
        >>> q = np.array([1.0, 0.0], dtype=np.float32)
        >>> i = np.array([0.707, 0.707], dtype=np.float32)
        >>> E = compute_E(q, i)
        >>> print(f"E = {E:.4f}")  # E = 0.5 (45 degree angle)
    """
    if query_vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"query_vec has shape {query_vec.shape}, expected ({EMBEDDING_DIM},)"
        )

    if item_vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"item_vec has shape {item_vec.shape}, expected ({EMBEDDING_DIM},)"
        )

    # Compute norms
    query_norm = np.linalg.norm(query_vec)
    item_norm = np.linalg.norm(item_vec)

    # Handle zero vectors
    if query_norm == 0 or item_norm == 0:
        return 0.0

    # Cosine similarity: dot(q, i) / (||q|| * ||i||)
    cosine = np.dot(query_vec, item_vec) / (query_norm * item_norm)

    # Born rule: E = |<q|i>|^2 = cos^2(theta)
    E = cosine ** 2

    # Clamp to [0, 1] to handle floating point errors
    return float(max(0.0, min(1.0, E)))


def batch_compute_E(
    query_vec: np.ndarray,
    item_vecs: np.ndarray
) -> np.ndarray:
    """Compute E-scores for a query against multiple items efficiently.

    Vectorized implementation for computing E-scores against many items at once.

    Args:
        query_vec: Query vector of shape (384,)
        item_vecs: Item vectors of shape (N, 384)

    Returns:
        E-scores array of shape (N,) with values in [0, 1]

    Raises:
        ValueError: If shapes are incompatible
    """
    if query_vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"query_vec has shape {query_vec.shape}, expected ({EMBEDDING_DIM},)"
        )

    if len(item_vecs.shape) != 2 or item_vecs.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"item_vecs has shape {item_vecs.shape}, "
            f"expected (N, {EMBEDDING_DIM})"
        )

    # Handle empty item_vecs
    if item_vecs.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # Normalize query
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(item_vecs), dtype=np.float32)
    query_unit = query_vec / query_norm

    # Normalize items (with small epsilon to avoid division by zero)
    item_norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
    # Avoid division by zero
    item_norms = np.maximum(item_norms, 1e-10)
    items_unit = item_vecs / item_norms

    # Compute cosine similarities (dot products of unit vectors)
    cosines = np.dot(items_unit, query_unit)

    # Born rule: E = cos^2
    E_scores = cosines ** 2

    # Clamp to [0, 1]
    E_scores = np.clip(E_scores, 0.0, 1.0)

    return E_scores.astype(np.float32)


def merge_centroids(
    centroids: List[Tuple[np.ndarray, int]]
) -> Tuple[np.ndarray, int]:
    """Merge multiple centroids into one, weighted by their turn counts.

    Used when combining multiple child centroids into a parent centroid.

    Args:
        centroids: List of (centroid, turn_count) tuples

    Returns:
        Tuple of (merged_centroid, total_turn_count)

    Raises:
        ValueError: If centroids list is empty

    Example:
        >>> c1 = (np.array([1.0, 0.0]), 100)
        >>> c2 = (np.array([0.0, 1.0]), 200)
        >>> merged, total = merge_centroids([c1, c2])
        >>> # merged = (100*[1,0] + 200*[0,1]) / 300 = [0.333, 0.667]
    """
    if not centroids:
        raise ValueError("Cannot merge empty list of centroids")

    total_count = sum(count for _, count in centroids)
    if total_count == 0:
        raise ValueError("Total turn count is zero")

    weighted_sum = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for centroid, count in centroids:
        if centroid.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Centroid has shape {centroid.shape}, "
                f"expected ({EMBEDDING_DIM},)"
            )
        weighted_sum += centroid.astype(np.float64) * count

    merged = (weighted_sum / total_count).astype(np.float32)
    return merged, total_count


def compute_variance(
    centroid: np.ndarray,
    vectors: List[np.ndarray]
) -> float:
    """Compute the average squared distance from centroid to vectors.

    Useful for understanding how well a centroid represents its children.
    Lower variance means the centroid is more representative.

    Args:
        centroid: Centroid vector of shape (384,)
        vectors: List of vectors of shape (384,)

    Returns:
        Average squared Euclidean distance
    """
    if not vectors:
        return 0.0

    total_sq_dist = 0.0
    for v in vectors:
        diff = centroid - v
        total_sq_dist += float(np.dot(diff, diff))

    return total_sq_dist / len(vectors)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length.

    Args:
        vec: Input vector of shape (384,)

    Returns:
        Unit vector of shape (384,), or zero vector if input is zero
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


if __name__ == "__main__":
    # Self-test
    print("Testing centroid math functions...")

    # Use full 384-dim for all tests
    v1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    v2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    v3 = np.random.randn(EMBEDDING_DIM).astype(np.float32)

    centroid = compute_centroid([v1, v2, v3])
    expected = (v1 + v2 + v3) / 3
    assert np.allclose(centroid, expected)
    print("compute_centroid: PASSED")

    # Test update_centroid_incremental
    centroid_12 = compute_centroid([v1, v2])
    updated = update_centroid_incremental(centroid_12, 2, v3)
    assert np.allclose(updated, centroid)
    print("update_centroid_incremental: PASSED")

    # Test compute_E
    # Identical vectors should have E = 1
    E_same = compute_E(v1, v1)
    assert abs(E_same - 1.0) < 1e-5
    print(f"E(v, v) = {E_same:.6f} (expected 1.0): PASSED")

    # Orthogonal vectors should have E = 0
    ortho1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    ortho2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    ortho1[0] = 1.0
    ortho2[1] = 1.0
    E_ortho = compute_E(ortho1, ortho2)
    assert abs(E_ortho) < 1e-5
    print(f"E(ortho1, ortho2) = {E_ortho:.6f} (expected 0.0): PASSED")

    # Test batch_compute_E
    items = np.stack([v1, v2, v3], axis=0)
    E_batch = batch_compute_E(v1, items)
    assert len(E_batch) == 3
    assert abs(E_batch[0] - 1.0) < 1e-5  # v1 vs v1
    print("batch_compute_E: PASSED")

    # Test merge_centroids
    c1 = (v1, 100)
    c2 = (v2, 200)
    merged, total = merge_centroids([c1, c2])
    expected_merged = (v1 * 100 + v2 * 200) / 300
    assert total == 300
    assert np.allclose(merged, expected_merged)
    print("merge_centroids: PASSED")

    # Test compute_variance
    variance = compute_variance(centroid, [v1, v2, v3])
    assert variance >= 0
    print(f"compute_variance: {variance:.6f} PASSED")

    # Test normalize_vector
    unnorm = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 5
    normed = normalize_vector(unnorm)
    assert abs(np.linalg.norm(normed) - 1.0) < 1e-5
    print("normalize_vector: PASSED")

    print("\nAll tests passed!")
