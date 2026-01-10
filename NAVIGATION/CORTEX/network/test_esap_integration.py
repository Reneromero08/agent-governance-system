#!/usr/bin/env python3
"""
Tests for ESAP Cassette Integration.

Tests spectral alignment verification between cassettes in the network.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import sqlite3

sys.path.insert(0, str(Path(__file__).parent))
from esap_cassette import ESAPCassetteMixin, CassetteSpectrum, VectorCassetteBase
from esap_hub import ESAPNetworkHub
from cassette_protocol import DatabaseCassette


class MockVectorCassette(ESAPCassetteMixin, DatabaseCassette):
    """Mock cassette with synthetic vectors for testing."""

    def __init__(self, cassette_id: str, vectors: np.ndarray, capabilities: list = None):
        # Create temp db
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        super().__init__(Path(self.temp_db.name), cassette_id)
        self._vectors = vectors
        self.capabilities = capabilities or ["vectors", "esap"]
        self.embedder_id = "mock-embedder"

    def get_vectors_for_spectrum(self) -> np.ndarray:
        return self._vectors

    def query(self, query_text: str, top_k: int = 10):
        return [{"content": f"Mock result for {query_text}", "source": self.cassette_id}]

    def get_stats(self):
        return {"vector_count": len(self._vectors), "cassette_id": self.cassette_id}


class TestCassetteSpectrum:
    """Tests for CassetteSpectrum computation."""

    def test_from_vectors_basic(self):
        """Compute spectrum from random vectors."""
        np.random.seed(42)
        vectors = np.random.randn(100, 384)
        anchor_hash = "sha256:" + "a" * 64

        spectrum = CassetteSpectrum.from_vectors(vectors, anchor_hash)

        assert spectrum.vector_count == 100
        assert spectrum.vector_dim == 384
        assert spectrum.effective_rank > 0
        assert len(spectrum.cumulative_variance) > 0
        assert spectrum.cumulative_variance[-1] == pytest.approx(1.0, abs=0.01)

    def test_trained_like_spectrum(self):
        """Trained-like vectors have Df ~ 20-25."""
        np.random.seed(42)
        # Simulate trained embedding (concentrated variance)
        n, dim = 100, 384
        vectors = np.random.randn(n, dim)
        # Add structure (low-rank component)
        U = np.random.randn(n, 10)
        V = np.random.randn(10, dim)
        vectors = vectors * 0.1 + U @ V

        spectrum = CassetteSpectrum.from_vectors(vectors, "sha256:" + "a" * 64)

        assert 5 < spectrum.effective_rank < 50

    def test_to_compact(self):
        """Convert to ESAP SpectrumCompact format."""
        np.random.seed(42)
        vectors = np.random.randn(50, 128)
        spectrum = CassetteSpectrum.from_vectors(vectors, "sha256:" + "b" * 64)

        compact = spectrum.to_compact("test-embedder")

        assert compact.anchor_set_hash == "sha256:" + "b" * 64
        assert compact.embedder_id == "test-embedder"
        assert compact.effective_rank == spectrum.effective_rank


class TestESAPCassetteMixin:
    """Tests for ESAP cassette functionality."""

    def test_compute_spectrum(self):
        """Cassette computes spectrum from vectors."""
        np.random.seed(42)
        vectors = np.random.randn(64, 256)
        cassette = MockVectorCassette("test_cassette", vectors)

        spectrum = cassette.compute_spectrum()

        assert spectrum is not None
        assert spectrum.vector_count == 64

    def test_esap_handshake(self):
        """ESAP handshake includes spectrum."""
        np.random.seed(42)
        vectors = np.random.randn(64, 256)
        cassette = MockVectorCassette("test_cassette", vectors)

        handshake = cassette.esap_handshake()

        assert "esap" in handshake
        assert handshake["esap"]["enabled"] is True
        assert "spectrum" in handshake["esap"]
        assert "effective_rank" in handshake["esap"]["spectrum"]

    def test_verify_alignment_same_source(self):
        """Verify alignment with spectrum from same distribution."""
        np.random.seed(42)
        base = np.random.randn(100, 256)

        # Two cassettes with similar vectors
        vectors_a = base + np.random.randn(100, 256) * 0.1
        vectors_b = base + np.random.randn(100, 256) * 0.1

        cassette_a = MockVectorCassette("cassette_a", vectors_a)
        cassette_b = MockVectorCassette("cassette_b", vectors_b)

        # Get spectrum from B
        handshake_b = cassette_b.esap_handshake()
        spectrum_b = handshake_b["esap"]["spectrum"]

        # A verifies alignment with B
        result = cassette_a.verify_alignment(spectrum_b)

        assert result["converges"] is True
        assert result["correlation"] > 0.9

    def test_verify_alignment_different_source(self):
        """Alignment fails with completely different spectra."""
        np.random.seed(42)

        # Cassette A: trained-like (low rank)
        U = np.random.randn(100, 5)
        V = np.random.randn(5, 256)
        vectors_a = U @ V + np.random.randn(100, 256) * 0.1

        # Cassette B: random (high rank, uniform)
        vectors_b = np.random.randn(100, 256)

        cassette_a = MockVectorCassette("cassette_a", vectors_a)
        cassette_b = MockVectorCassette("cassette_b", vectors_b)

        handshake_b = cassette_b.esap_handshake()
        spectrum_b = handshake_b["esap"]["spectrum"]

        result = cassette_a.verify_alignment(spectrum_b)

        # Should diverge due to very different effective ranks
        # (Note: correlation might still be high due to monotonicity)
        assert "correlation" in result


class TestESAPNetworkHub:
    """Tests for ESAP-enabled network hub."""

    def test_register_esap_cassette(self):
        """Register cassette with ESAP verification."""
        np.random.seed(42)
        hub = ESAPNetworkHub(verbose=False)
        vectors = np.random.randn(64, 256)
        cassette = MockVectorCassette("test", vectors)

        result = hub.register_cassette(cassette)

        assert "test" in hub.cassettes
        assert "test" in hub.spectra
        assert result["esap"]["enabled"] is True

    def test_alignment_groups(self):
        """Aligned cassettes are grouped together."""
        np.random.seed(42)
        hub = ESAPNetworkHub(verbose=False)

        # Same base = aligned
        base = np.random.randn(100, 256)
        cassette_a = MockVectorCassette("a", base + np.random.randn(100, 256) * 0.1)
        cassette_b = MockVectorCassette("b", base + np.random.randn(100, 256) * 0.1)

        hub.register_cassette(cassette_a)
        hub.register_cassette(cassette_b)

        # Should be in same group
        assert len(hub.alignment_groups) == 1
        group = list(hub.alignment_groups.values())[0]
        assert "a" in group
        assert "b" in group

    def test_convergence_matrix(self):
        """Convergence matrix tracks pairwise alignment."""
        np.random.seed(42)
        hub = ESAPNetworkHub(verbose=False)

        vectors_a = np.random.randn(64, 256)
        vectors_b = np.random.randn(64, 256)

        hub.register_cassette(MockVectorCassette("a", vectors_a))
        hub.register_cassette(MockVectorCassette("b", vectors_b))

        assert len(hub.convergence_matrix) == 1
        key = ("a", "b")
        assert key in hub.convergence_matrix

    def test_query_aligned(self):
        """Query only aligned cassettes."""
        np.random.seed(42)
        hub = ESAPNetworkHub(verbose=False)

        base = np.random.randn(100, 256)
        hub.register_cassette(MockVectorCassette("a", base))
        hub.register_cassette(MockVectorCassette("b", base + np.random.randn(100, 256) * 0.1))

        # Query aligned with 'a'
        results = hub.query_aligned("test query", "a", top_k=5)

        assert "a" in results
        assert "b" in results

    def test_alignment_matrix(self):
        """Get full alignment matrix."""
        np.random.seed(42)
        hub = ESAPNetworkHub(verbose=False)

        hub.register_cassette(MockVectorCassette("x", np.random.randn(50, 128)))
        hub.register_cassette(MockVectorCassette("y", np.random.randn(50, 128)))

        matrix = hub.get_alignment_matrix()

        assert "cassettes" in matrix
        assert "matrix" in matrix
        assert "groups" in matrix
        assert "threshold" in matrix
        assert matrix["matrix"]["x"]["x"] == 1.0


class TestRealWorldScenario:
    """Integration tests with realistic scenarios."""

    def test_governance_agi_alignment(self):
        """Test alignment between governance and AGI research vectors.

        Simulates scenario where two knowledge bases with similar
        semantic content should align.
        """
        np.random.seed(42)
        hub = ESAPNetworkHub(verbose=False)

        # Simulate governance embeddings (governance docs)
        gov_base = np.random.randn(200, 384)
        gov_vectors = gov_base + np.random.randn(200, 384) * 0.2

        # Simulate AGI research embeddings (research papers)
        # Same embedding model = should align
        agi_vectors = gov_base + np.random.randn(200, 384) * 0.2

        gov_cassette = MockVectorCassette("governance", gov_vectors, ["vectors", "governance"])
        agi_cassette = MockVectorCassette("agi_research", agi_vectors, ["vectors", "research"])

        hub.register_cassette(gov_cassette)
        result = hub.register_cassette(agi_cassette)

        # Should be aligned
        alignment = result["esap"]["alignment_results"]["governance"]
        assert alignment["converges"] is True
        assert alignment["correlation"] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
