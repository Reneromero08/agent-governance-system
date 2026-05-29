"""Tests for Protocol module."""

import json
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import protocol


class TestAnchorSet:
    """Tests for AnchorSet."""

    def test_from_words(self):
        """Should create anchor set from word list."""
        words = ['dog', 'cat', 'bird']
        anchor_set = protocol.AnchorSet.from_words(words)

        assert len(anchor_set.anchors) == 3
        assert anchor_set.anchors[0]['text'] == 'dog'
        assert anchor_set.anchors[0]['id'] == 'a000'

    def test_hash_deterministic(self):
        """Same words should produce same hash."""
        words = ['dog', 'cat', 'bird']
        a1 = protocol.AnchorSet.from_words(words)
        a2 = protocol.AnchorSet.from_words(words)

        assert a1.anchor_hash == a2.anchor_hash

    def test_hash_different_for_different_words(self):
        """Different words should produce different hash."""
        a1 = protocol.AnchorSet.from_words(['dog', 'cat'])
        a2 = protocol.AnchorSet.from_words(['dog', 'bird'])

        assert a1.anchor_hash != a2.anchor_hash

    def test_to_dict(self):
        """Should produce valid dict."""
        anchor_set = protocol.AnchorSet.from_words(['dog', 'cat'])
        d = anchor_set.to_dict()

        assert d['type'] == 'ANCHOR_SET'
        assert d['version'] == protocol.PROTOCOL_VERSION
        assert len(d['anchors']) == 2


class TestEmbedderDescriptor:
    """Tests for EmbedderDescriptor."""

    def test_creation(self):
        """Should create descriptor."""
        desc = protocol.EmbedderDescriptor(
            embedder_id='all-MiniLM-L6-v2',
            dimension=384,
            weights_hash='sha256:abc123'
        )

        assert desc.embedder_id == 'all-MiniLM-L6-v2'
        assert desc.dimension == 384

    def test_to_dict(self):
        """Should produce valid dict."""
        desc = protocol.EmbedderDescriptor(
            embedder_id='all-MiniLM-L6-v2',
            dimension=384
        )
        d = desc.to_dict()

        assert d['type'] == 'EMBEDDER_DESCRIPTOR'
        assert d['embedder_id'] == 'all-MiniLM-L6-v2'


class TestSpectrumSignature:
    """Tests for SpectrumSignature."""

    def test_creation(self):
        """Should create signature."""
        sig = protocol.SpectrumSignature(
            eigenvalues=[0.5, 0.3, 0.2],
            anchor_set_hash='sha256:abc123',
            embedder_id='all-MiniLM-L6-v2'
        )

        assert sig.k == 3
        assert sig.effective_rank > 0

    def test_hash_deterministic(self):
        """Same eigenvalues should produce same hash."""
        eigenvalues = [0.5, 0.3, 0.2]

        s1 = protocol.SpectrumSignature(
            eigenvalues=eigenvalues,
            anchor_set_hash='sha256:abc',
            embedder_id='test'
        )
        s2 = protocol.SpectrumSignature(
            eigenvalues=eigenvalues,
            anchor_set_hash='sha256:abc',
            embedder_id='test'
        )

        assert s1.spectrum_hash == s2.spectrum_hash

    def test_to_dict(self):
        """Should produce valid dict."""
        sig = protocol.SpectrumSignature(
            eigenvalues=[0.5, 0.3, 0.2],
            anchor_set_hash='sha256:abc123',
            embedder_id='all-MiniLM-L6-v2'
        )
        d = sig.to_dict()

        assert d['type'] == 'SPECTRUM_SIGNATURE'
        assert d['k'] == 3


class TestAlignmentMap:
    """Tests for AlignmentMap."""

    def test_creation(self):
        """Should create alignment map."""
        R = [[1, 0], [0, 1]]
        amap = protocol.AlignmentMap(
            rotation_matrix=R,
            source_embedder='model_a',
            target_embedder='model_b',
            anchor_set_hash='sha256:abc'
        )

        assert amap.k == 2
        assert amap.source_embedder == 'model_a'

    def test_as_numpy(self):
        """Should convert to numpy array."""
        import numpy as np

        R = [[1, 0], [0, 1]]
        amap = protocol.AlignmentMap(
            rotation_matrix=R,
            source_embedder='model_a',
            target_embedder='model_b',
            anchor_set_hash='sha256:abc'
        )

        R_np = amap.as_numpy()
        np.testing.assert_array_equal(R_np, np.eye(2))


class TestCanonicalJson:
    """Tests for canonical_json."""

    def test_sorted_keys(self):
        """Keys should be sorted."""
        obj = {'z': 1, 'a': 2, 'm': 3}
        result = protocol.canonical_json(obj)

        assert result == '{"a":2,"m":3,"z":1}'

    def test_no_whitespace(self):
        """Should have no extra whitespace."""
        obj = {'a': [1, 2, 3]}
        result = protocol.canonical_json(obj)

        assert result == '{"a":[1,2,3]}'


class TestComputeHash:
    """Tests for compute_hash."""

    def test_deterministic(self):
        """Same input should produce same hash."""
        data = "hello world"
        h1 = protocol.compute_hash(data)
        h2 = protocol.compute_hash(data)

        assert h1 == h2

    def test_prefix(self):
        """Hash should have sha256: prefix."""
        h = protocol.compute_hash("test")
        assert h.startswith("sha256:")

    def test_bytes_and_str(self):
        """Bytes and str should produce same hash."""
        h1 = protocol.compute_hash("test")
        h2 = protocol.compute_hash(b"test")

        assert h1 == h2


class TestErrors:
    """Tests for error classes."""

    def test_error_code(self):
        """Errors should have code."""
        err = protocol.AnchorMismatchError()
        assert err.code == "E001"

    def test_error_message(self):
        """Errors should have message."""
        err = protocol.AnchorMismatchError("custom message")
        assert "custom message" in str(err)

    def test_all_error_codes_unique(self):
        """All error codes should be unique."""
        errors = [
            protocol.AnchorMismatchError,
            protocol.EmbedderMismatchError,
            protocol.MetricMismatchError,
            protocol.SpectrumMismatchError,
            protocol.InsufficientRankError,
            protocol.AlignmentFailedError,
            protocol.SchemaInvalidError,
            protocol.VersionUnsupportedError,
        ]

        codes = [e.code for e in errors]
        assert len(codes) == len(set(codes))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
