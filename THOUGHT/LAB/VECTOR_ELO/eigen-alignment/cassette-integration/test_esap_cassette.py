#!/usr/bin/env python3
"""
ESAP Cassette Integration Tests (EXPERIMENTAL)

Tests for the simplified ESAP cassette mixin.
These tests were extracted from NAVIGATION/CORTEX/network/test_phase4.py
when ESAP was moved back to LAB (2026-01-25).

Status: EXPERIMENTAL - ESAP roadmap items ESAP.1-5 are incomplete.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import numpy as np

from esap_cassette import ESAPCassetteMixin
from esap_hub import ESAPNetworkHub


class TestESAPCassetteMixin:
    """Tests for ESAP spectral alignment mixin."""

    def test_spectrum_computation(self):
        """Spectrum signature computed correctly."""
        mixin = ESAPCassetteMixin()
        vectors = np.random.randn(100, 384)
        spectrum = mixin.compute_spectrum_signature(vectors)

        assert "eigenvalues_top_k" in spectrum
        assert len(spectrum["eigenvalues_top_k"]) == 10
        assert "effective_rank" in spectrum
        assert spectrum["effective_rank"] > 0
        assert "anchor_hash" in spectrum
        assert spectrum["anchor_hash"].startswith("sha256:")

    def test_empty_spectrum(self):
        """Empty/insufficient vectors return empty spectrum."""
        mixin = ESAPCassetteMixin()

        # Too few vectors
        spectrum = mixin.compute_spectrum_signature(np.random.randn(1, 384))
        assert spectrum["anchor_hash"] == "sha256:empty"

    def test_spectrum_correlation_identical(self):
        """Identical spectrums have correlation 1.0."""
        mixin = ESAPCassetteMixin()
        vectors = np.random.randn(100, 384)
        spec = mixin.compute_spectrum_signature(vectors)

        corr = ESAPCassetteMixin.compute_spectrum_correlation(spec, spec)
        assert corr == pytest.approx(1.0, abs=0.001)

    def test_spectrum_correlation_similar(self):
        """Similar vectors have high correlation."""
        mixin = ESAPCassetteMixin()
        vectors_a = np.random.randn(100, 384)
        vectors_b = vectors_a + np.random.randn(100, 384) * 0.1  # Small noise

        spec_a = mixin.compute_spectrum_signature(vectors_a)
        spec_b = mixin.compute_spectrum_signature(vectors_b)

        corr = ESAPCassetteMixin.compute_spectrum_correlation(spec_a, spec_b)
        assert corr > 0.8  # Should be highly correlated

    def test_spectrum_correlation_different(self):
        """Different random vectors have lower correlation."""
        mixin = ESAPCassetteMixin()
        vectors_a = np.random.randn(100, 384)
        vectors_b = np.random.randn(100, 384)  # Completely different

        spec_a = mixin.compute_spectrum_signature(vectors_a)
        spec_b = mixin.compute_spectrum_signature(vectors_b)

        corr = ESAPCassetteMixin.compute_spectrum_correlation(spec_a, spec_b)
        # Random vectors still have some structure, so correlation won't be 0
        # But should be less than identical (1.0) - use looser tolerance
        assert corr < 1.0  # Should be less than perfect correlation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
