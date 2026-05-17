"""Tests for MDS module."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import mds


class TestSquaredDistanceMatrix:
    """Tests for squared_distance_matrix."""

    def test_identity_diagonal(self):
        """Diagonal should be zero (distance to self)."""
        embeddings = np.random.randn(5, 10)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        D2 = mds.squared_distance_matrix(embeddings)

        np.testing.assert_array_almost_equal(np.diag(D2), 0.0)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        embeddings = np.random.randn(5, 10)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        D2 = mds.squared_distance_matrix(embeddings)

        np.testing.assert_array_almost_equal(D2, D2.T)

    def test_non_negative(self):
        """All distances should be non-negative."""
        embeddings = np.random.randn(5, 10)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        D2 = mds.squared_distance_matrix(embeddings)

        assert np.all(D2 >= -1e-10)

    def test_identical_vectors_zero_distance(self):
        """Identical vectors should have zero distance."""
        v = np.array([[1.0, 0.0, 0.0]])
        embeddings = np.vstack([v, v, v])

        D2 = mds.squared_distance_matrix(embeddings)

        np.testing.assert_array_almost_equal(D2, 0.0)


class TestCenteringMatrix:
    """Tests for centering_matrix."""

    def test_shape(self):
        """Should return n x n matrix."""
        J = mds.centering_matrix(5)
        assert J.shape == (5, 5)

    def test_idempotent(self):
        """J @ J = J (centering is idempotent)."""
        J = mds.centering_matrix(5)
        J2 = J @ J
        np.testing.assert_array_almost_equal(J, J2)

    def test_symmetric(self):
        """Centering matrix should be symmetric."""
        J = mds.centering_matrix(5)
        np.testing.assert_array_almost_equal(J, J.T)


class TestClassicalMDS:
    """Tests for classical_mds."""

    def test_known_configuration(self):
        """Test with a known simple configuration."""
        # Create a simple 2D configuration (square)
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=float)

        # Compute squared distance matrix
        n = len(points)
        D2 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D2[i, j] = np.sum((points[i] - points[j]) ** 2)

        # Run MDS
        X, eigenvalues, eigenvectors = mds.classical_mds(D2)

        # Should recover 2D (2 positive eigenvalues)
        assert len(eigenvalues) == 2

        # Eigenvalues should be positive
        assert np.all(eigenvalues > 0)

    def test_distance_preservation(self):
        """MDS coordinates should preserve pairwise distances."""
        # Random points
        np.random.seed(42)
        original = np.random.randn(5, 3)

        # Compute original distances
        n = len(original)
        D2_original = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D2_original[i, j] = np.sum((original[i] - original[j]) ** 2)

        # Run MDS
        X, eigenvalues, eigenvectors = mds.classical_mds(D2_original)

        # Compute distances in MDS space
        D2_reconstructed = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D2_reconstructed[i, j] = np.sum((X[i] - X[j]) ** 2)

        # Should be approximately equal
        np.testing.assert_array_almost_equal(D2_original, D2_reconstructed, decimal=5)

    def test_k_truncation(self):
        """Should truncate to k dimensions when specified."""
        np.random.seed(42)
        D2 = np.random.rand(10, 10)
        D2 = (D2 + D2.T) / 2  # Make symmetric
        np.fill_diagonal(D2, 0)

        X, eigenvalues, eigenvectors = mds.classical_mds(D2, k=3)

        assert X.shape[1] == 3
        assert len(eigenvalues) == 3


class TestEffectiveRank:
    """Tests for effective_rank."""

    def test_uniform_spectrum(self):
        """Uniform eigenvalues should give n."""
        eigenvalues = np.array([1, 1, 1, 1, 1])
        rank = mds.effective_rank(eigenvalues)
        assert abs(rank - 5.0) < 1e-10

    def test_single_dominant(self):
        """Single dominant eigenvalue should give ~1."""
        eigenvalues = np.array([1000, 0.01, 0.01, 0.01])
        rank = mds.effective_rank(eigenvalues)
        assert rank < 1.1

    def test_empty(self):
        """Empty eigenvalues should give 0."""
        eigenvalues = np.array([0, 0, 0])
        rank = mds.effective_rank(eigenvalues)
        assert rank == 0.0


class TestStress:
    """Tests for stress computation."""

    def test_perfect_reconstruction(self):
        """Perfect reconstruction should have zero stress."""
        D = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ], dtype=float)

        stress = mds.stress(D, D)
        assert stress < 1e-10

    def test_stress_bounds(self):
        """Stress should be non-negative."""
        D1 = np.random.rand(5, 5)
        D1 = (D1 + D1.T) / 2
        np.fill_diagonal(D1, 0)

        D2 = np.random.rand(5, 5)
        D2 = (D2 + D2.T) / 2
        np.fill_diagonal(D2, 0)

        stress = mds.stress(D1, D2)
        assert stress >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
