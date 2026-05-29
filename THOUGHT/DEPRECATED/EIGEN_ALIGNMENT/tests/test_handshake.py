"""Tests for ESAP Handshake Protocol."""

import pytest
import numpy as np
from lib.handshake import (
    ESAPHandshake,
    SpectrumCompact,
    compute_cumulative_variance,
    compute_effective_rank,
    check_convergence,
    create_handshake_from_embeddings,
    CONVERGENCE_THRESHOLD,
)
from lib.protocol import SpectrumMismatchError


class TestCumulativeVariance:
    """Tests for cumulative variance computation."""

    def test_basic(self):
        """Cumulative variance sums to 1."""
        ev = np.array([0.5, 0.3, 0.2])
        cv = compute_cumulative_variance(ev)
        assert np.isclose(cv[-1], 1.0)

    def test_ordering(self):
        """Cumulative variance is monotonically increasing."""
        ev = np.array([10, 5, 3, 2, 1])
        cv = compute_cumulative_variance(ev)
        assert all(cv[i] <= cv[i+1] for i in range(len(cv)-1))

    def test_zero_handling(self):
        """Handles zero eigenvalues gracefully."""
        ev = np.array([0, 0, 0])
        cv = compute_cumulative_variance(ev)
        assert all(v == 0 for v in cv)


class TestEffectiveRank:
    """Tests for effective rank (Df) computation."""

    def test_uniform_spectrum(self):
        """Uniform spectrum has max effective rank."""
        ev = np.ones(10)
        df = compute_effective_rank(ev)
        assert np.isclose(df, 10.0)

    def test_single_component(self):
        """Single non-zero eigenvalue has Df=1."""
        ev = np.array([1, 0, 0, 0])
        df = compute_effective_rank(ev)
        assert np.isclose(df, 1.0)

    def test_trained_model_range(self):
        """Trained models typically have Df ~ 20-25."""
        # Simulate trained spectrum (exponential decay)
        ev = np.exp(-np.arange(50) / 10)
        df = compute_effective_rank(ev)
        assert 5 < df < 30  # Reasonable range


class TestConvergenceCheck:
    """Tests for Spectral Convergence Theorem verification."""

    def test_identical_converges(self):
        """Identical spectra converge."""
        cv = np.cumsum(np.ones(10) / 10)
        result = check_convergence(cv, cv, 10.0, 10.0)
        assert result["converges"] is True
        assert result["correlation"] >= 0.99

    def test_different_diverges(self):
        """Very different spectra diverge."""
        cv_a = np.array([0.9, 0.95, 0.98, 1.0])  # Concentrated
        cv_b = np.array([0.25, 0.5, 0.75, 1.0])  # Uniform
        result = check_convergence(cv_a, cv_b, 1.5, 4.0)
        assert result["converges"] is False

    def test_threshold_exact(self):
        """Exact threshold behavior."""
        cv = np.cumsum(np.ones(10) / 10)
        result = check_convergence(cv, cv, 10.0, 10.0)
        assert result["correlation"] >= CONVERGENCE_THRESHOLD


class TestESAPHandshake:
    """Tests for the handshake protocol."""

    @pytest.fixture
    def spectrum_a(self):
        """Spectrum for agent A."""
        ev = np.exp(-np.arange(20) / 5)
        return SpectrumCompact.from_eigenvalues(
            ev, "sha256:" + "a" * 64, "model_a"
        )

    @pytest.fixture
    def spectrum_b(self):
        """Similar spectrum for agent B (should converge)."""
        ev = np.exp(-np.arange(20) / 5.1)  # Slightly different
        return SpectrumCompact.from_eigenvalues(
            ev, "sha256:" + "a" * 64, "model_b"  # Same anchor hash
        )

    @pytest.fixture
    def spectrum_incompatible(self):
        """Incompatible spectrum (different anchors)."""
        ev = np.exp(-np.arange(20) / 5)
        return SpectrumCompact.from_eigenvalues(
            ev, "sha256:" + "b" * 64, "model_c"  # Different anchor hash
        )

    def test_successful_handshake(self, spectrum_a, spectrum_b):
        """Complete successful handshake."""
        handler_a = ESAPHandshake("agent_a", spectrum_a)
        handler_b = ESAPHandshake("agent_b", spectrum_b)

        # A initiates
        hello = handler_a.create_hello()
        assert hello["type"] == "ESAP_HELLO"
        assert hello["sender_id"] == "agent_a"

        # B responds
        response = handler_b.process_hello(hello)
        assert response["type"] == "ESAP_ACK"
        assert response["convergence"]["converges"] is True

        # A confirms
        success = handler_a.process_ack(response)
        assert success is True

    def test_anchor_mismatch_rejection(self, spectrum_a, spectrum_incompatible):
        """Handshake fails with mismatched anchors."""
        handler_a = ESAPHandshake("agent_a", spectrum_a)
        handler_b = ESAPHandshake("agent_b", spectrum_incompatible)

        hello = handler_a.create_hello()
        response = handler_b.process_hello(hello)

        assert response["type"] == "ESAP_REJECT"
        assert response["reason"] == "ANCHOR_MISMATCH"

    def test_divergent_spectrum_rejection(self, spectrum_a):
        """Handshake fails with divergent spectra."""
        # Create radically different spectrum
        ev_divergent = np.ones(20)  # Uniform (Df=20, very different curve)
        spectrum_divergent = SpectrumCompact.from_eigenvalues(
            ev_divergent, "sha256:" + "a" * 64, "model_divergent"
        )

        handler_a = ESAPHandshake("agent_a", spectrum_a)
        handler_b = ESAPHandshake("agent_b", spectrum_divergent)

        hello = handler_a.create_hello()
        response = handler_b.process_hello(hello)

        assert response["type"] == "ESAP_REJECT"
        assert response["reason"] == "SPECTRUM_DIVERGENCE"

    def test_replay_protection(self, spectrum_a, spectrum_b):
        """Cannot reuse nonce."""
        handler_a = ESAPHandshake("agent_a", spectrum_a)
        handler_b = ESAPHandshake("agent_b", spectrum_b)

        hello = handler_a.create_hello()
        response = handler_b.process_hello(hello)

        # First ACK should succeed
        handler_a.process_ack(response)

        # Second ACK with same nonce should fail
        with pytest.raises(SpectrumMismatchError, match="Invalid nonce"):
            handler_a.process_ack(response)

    def test_capabilities_exchange(self, spectrum_a, spectrum_b):
        """Capabilities are exchanged correctly."""
        caps_a = ["symbol_resolution", "cross_lingual"]
        caps_b = ["symbol_resolution", "governance"]

        handler_a = ESAPHandshake("agent_a", spectrum_a, caps_a)
        handler_b = ESAPHandshake("agent_b", spectrum_b, caps_b)

        hello = handler_a.create_hello()
        assert hello["capabilities"] == caps_a

        response = handler_b.process_hello(hello)
        assert response["capabilities"] == caps_b


class TestCreateFromEmbeddings:
    """Tests for convenience function."""

    def test_from_random_embeddings(self):
        """Create handler from random embeddings."""
        embeddings = np.random.randn(64, 384)  # 64 anchors, 384 dim
        anchor_hash = "sha256:" + "a" * 64

        handler = create_handshake_from_embeddings(
            agent_id="test_agent",
            embeddings=embeddings,
            anchor_set_hash=anchor_hash,
            embedder_id="test_model",
            capabilities=["test"]
        )

        assert handler.agent_id == "test_agent"
        assert handler.spectrum.anchor_set_hash == anchor_hash
        assert handler.spectrum.effective_rank > 0
        assert len(handler.spectrum.cumulative_variance) > 0

    def test_similar_embeddings_converge(self):
        """Embeddings from similar models converge."""
        np.random.seed(42)
        base = np.random.randn(64, 384)

        # Add small noise (simulating different models trained on same data)
        embeddings_a = base + np.random.randn(64, 384) * 0.1
        embeddings_b = base + np.random.randn(64, 384) * 0.1

        anchor_hash = "sha256:" + "a" * 64

        handler_a = create_handshake_from_embeddings(
            "agent_a", embeddings_a, anchor_hash
        )
        handler_b = create_handshake_from_embeddings(
            "agent_b", embeddings_b, anchor_hash
        )

        hello = handler_a.create_hello()
        response = handler_b.process_hello(hello)

        assert response["type"] == "ESAP_ACK"
        assert response["convergence"]["converges"] is True
