"""Tests for Procrustes module."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import procrustes


class TestProcrustesAlign:
    """Tests for procrustes_align."""

    def test_identity_alignment(self):
        """Identical matrices should give identity rotation."""
        np.random.seed(42)
        X = np.random.randn(5, 3)

        R, residual = procrustes.procrustes_align(X, X)

        # Rotation should be close to identity
        np.testing.assert_array_almost_equal(R, np.eye(3), decimal=5)

        # Residual should be zero
        assert residual < 1e-10

    def test_orthogonal_rotation(self):
        """Should recover known rotation."""
        np.random.seed(42)
        X = np.random.randn(5, 3)

        # Apply known rotation
        theta = np.pi / 4  # 45 degrees around z-axis
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        Y = X @ R_true

        R, residual = procrustes.procrustes_align(X, Y)

        # Should recover the rotation
        np.testing.assert_array_almost_equal(R, R_true, decimal=5)
        assert residual < 1e-10

    def test_orthogonality(self):
        """Returned matrix should be orthogonal."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        Y = np.random.randn(5, 3)

        R, residual = procrustes.procrustes_align(X, Y)

        # R @ R.T should be identity
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)


class TestOutOfSampleMDS:
    """Tests for out_of_sample_mds."""

    def test_anchor_points_recovered(self):
        """Anchor points projected should match original MDS coordinates."""
        from lib import mds

        np.random.seed(42)
        n = 5
        d = 10

        # Random embeddings
        embeddings = np.random.randn(n, d)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute MDS
        D2 = mds.squared_distance_matrix(embeddings)
        X, eigenvalues, eigenvectors = mds.classical_mds(D2)

        # Project anchor points (should recover X)
        Y = procrustes.out_of_sample_mds(D2, D2, eigenvectors, eigenvalues)

        np.testing.assert_array_almost_equal(X, Y, decimal=5)

    def test_new_point_projection(self):
        """New point should be correctly projected."""
        from lib import mds

        np.random.seed(42)
        n = 5
        d = 10

        # Random embeddings (anchors + new point)
        all_embeddings = np.random.randn(n + 1, d)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

        anchor_embeddings = all_embeddings[:n]
        new_embedding = all_embeddings[n:n+1]

        # Compute anchor MDS
        D2_anchors = mds.squared_distance_matrix(anchor_embeddings)
        X_anchors, eigenvalues, eigenvectors = mds.classical_mds(D2_anchors)

        # Compute distances from new point to anchors
        d2_new = np.sum((new_embedding - anchor_embeddings) ** 2, axis=1, keepdims=True).T

        # Project new point
        Y_new = procrustes.out_of_sample_mds(d2_new, D2_anchors, eigenvectors, eigenvalues)

        # Verify by checking distance consistency
        # Distance in MDS space should approximately match original
        for i in range(n):
            mds_dist = np.sqrt(np.sum((Y_new[0] - X_anchors[i]) ** 2))
            orig_dist = np.sqrt(d2_new[0, i])
            # Allow some tolerance due to dimensionality reduction
            assert abs(mds_dist - orig_dist) < 1.0  # Loose bound


class TestAlignPoints:
    """Tests for align_points."""

    def test_identity_rotation(self):
        """Identity rotation should not change points."""
        np.random.seed(42)
        points = np.random.randn(5, 3)
        R = np.eye(3)

        aligned = procrustes.align_points(points, R)

        np.testing.assert_array_almost_equal(aligned, points)

    def test_rotation_applied(self):
        """Rotation should be correctly applied."""
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=float)

        # 90 degree counter-clockwise rotation around z-axis
        # For points @ R: [1,0,0] @ R = [0,-1,0], [0,1,0] @ R = [1,0,0]
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)

        aligned = procrustes.align_points(points, R)

        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
        ])

        np.testing.assert_array_almost_equal(aligned, expected)


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = np.array([1, 2, 3])
        sim = procrustes.cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        sim = procrustes.cosine_similarity(a, b)
        assert abs(sim) < 1e-10

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        a = np.array([1, 0, 0])
        b = np.array([-1, 0, 0])
        sim = procrustes.cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-10

    def test_zero_vector(self):
        """Zero vector should give similarity 0."""
        a = np.array([1, 2, 3])
        b = np.array([0, 0, 0])
        sim = procrustes.cosine_similarity(a, b)
        assert sim == 0.0


class TestAlignmentQuality:
    """Tests for alignment_quality."""

    def test_perfect_alignment(self):
        """Perfect alignment should have zero residual and sim 1."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        R = np.eye(3)

        quality = procrustes.alignment_quality(X, X, R)

        assert quality['residual'] < 1e-10
        assert quality['relative_error'] < 1e-10
        assert abs(quality['mean_cosine_sim'] - 1.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
