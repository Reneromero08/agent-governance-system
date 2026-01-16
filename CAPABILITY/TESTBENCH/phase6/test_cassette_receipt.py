#!/usr/bin/env python3
"""
Phase 6 Tests: CassetteReceipt

Tests for receipt creation, hashing, chain verification, and Merkle root computation.
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "CAPABILITY" / "PRIMITIVES"))

import pytest
from cassette_receipt import (
    CassetteReceipt,
    create_receipt,
    receipt_from_dict,
    verify_receipt,
    verify_receipt_chain,
    compute_session_merkle_root,
    canonical_json,
    canonical_json_bytes,
)


class TestCassetteReceipt:
    """Test CassetteReceipt dataclass."""

    def test_create_receipt_basic(self):
        """Test basic receipt creation."""
        receipt = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
        )

        assert receipt.cassette_id == "resident"
        assert receipt.operation == "SAVE"
        assert receipt.record_id == "a" * 64
        assert receipt.record_hash == "b" * 64
        assert receipt.receipt_hash is not None
        assert len(receipt.receipt_hash) == 64

    def test_create_receipt_with_chain(self):
        """Test receipt creation with chain linkage."""
        receipt1 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            receipt_index=0,
        )

        receipt2 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="c" * 64,
            record_hash="d" * 64,
            parent_receipt_hash=receipt1.receipt_hash,
            receipt_index=1,
        )

        assert receipt2.parent_receipt_hash == receipt1.receipt_hash
        assert receipt2.receipt_index == 1

    def test_invalid_operation_raises(self):
        """Test that invalid operation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operation"):
            create_receipt(
                cassette_id="resident",
                operation="INVALID",
                record_id="a" * 64,
                record_hash="b" * 64,
            )

    def test_invalid_record_id_raises(self):
        """Test that invalid record_id raises ValueError."""
        with pytest.raises(ValueError, match="record_id must be 64 hex chars"):
            create_receipt(
                cassette_id="resident",
                operation="SAVE",
                record_id="short",
                record_hash="b" * 64,
            )


class TestReceiptHashDeterminism:
    """Test receipt hash determinism."""

    def test_same_inputs_same_hash(self):
        """Test that identical inputs produce identical hashes."""
        timestamp = "2025-01-16T12:00:00+00:00"

        receipt1 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            timestamp_utc=timestamp,
        )

        receipt2 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            timestamp_utc=timestamp,
        )

        assert receipt1.receipt_hash == receipt2.receipt_hash

    def test_different_inputs_different_hash(self):
        """Test that different inputs produce different hashes."""
        receipt1 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
        )

        receipt2 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="c" * 64,  # Different record_id
            record_hash="b" * 64,
        )

        assert receipt1.receipt_hash != receipt2.receipt_hash

    def test_timestamp_excluded_from_hash(self):
        """Test that timestamp is excluded from hash computation."""
        receipt1 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            timestamp_utc="2025-01-01T00:00:00+00:00",
        )

        receipt2 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            timestamp_utc="2025-12-31T23:59:59+00:00",
        )

        assert receipt1.receipt_hash == receipt2.receipt_hash


class TestReceiptVerification:
    """Test receipt verification."""

    def test_verify_valid_receipt(self):
        """Test that valid receipt verifies correctly."""
        receipt = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
        )

        assert verify_receipt(receipt) is True

    def test_verify_from_dict(self):
        """Test verification from dictionary."""
        receipt = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
        )

        data = receipt.to_dict()
        restored = receipt_from_dict(data)

        assert verify_receipt(restored) is True
        assert restored.receipt_hash == receipt.receipt_hash


class TestReceiptChainVerification:
    """Test receipt chain verification."""

    def test_verify_valid_chain(self):
        """Test that valid chain verifies correctly."""
        receipt1 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            receipt_index=0,
        )

        receipt2 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="c" * 64,
            record_hash="d" * 64,
            parent_receipt_hash=receipt1.receipt_hash,
            receipt_index=1,
        )

        receipt3 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="e" * 64,
            record_hash="f" * 64,
            parent_receipt_hash=receipt2.receipt_hash,
            receipt_index=2,
        )

        result = verify_receipt_chain([receipt1, receipt2, receipt3])

        assert result["valid"] is True
        assert result["chain_length"] == 3
        assert result["merkle_root"] is not None
        assert len(result["merkle_root"]) == 64

    def test_broken_chain_fails(self):
        """Test that broken chain fails verification."""
        receipt1 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            receipt_index=0,
        )

        # Create receipt2 with wrong parent
        receipt2 = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="c" * 64,
            record_hash="d" * 64,
            parent_receipt_hash="x" * 64,  # Wrong parent
            receipt_index=1,
        )

        result = verify_receipt_chain([receipt1, receipt2])

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_first_receipt_must_have_null_parent(self):
        """Test that first receipt must have null parent."""
        receipt = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            parent_receipt_hash="x" * 64,  # Should be None
            receipt_index=0,
        )

        result = verify_receipt_chain([receipt])

        assert result["valid"] is False


class TestMerkleRoot:
    """Test Merkle root computation."""

    def test_single_hash(self):
        """Test Merkle root of single hash."""
        hashes = ["a" * 64]
        root = compute_session_merkle_root(hashes)

        assert root is not None
        assert len(root) == 64

    def test_two_hashes(self):
        """Test Merkle root of two hashes."""
        hashes = ["a" * 64, "b" * 64]
        root = compute_session_merkle_root(hashes)

        assert root is not None
        assert len(root) == 64

    def test_deterministic(self):
        """Test that Merkle root is deterministic."""
        hashes = ["a" * 64, "b" * 64, "c" * 64]

        root1 = compute_session_merkle_root(hashes)
        root2 = compute_session_merkle_root(hashes)

        assert root1 == root2

    def test_order_matters(self):
        """Test that hash order matters."""
        hashes1 = ["a" * 64, "b" * 64]
        hashes2 = ["b" * 64, "a" * 64]

        root1 = compute_session_merkle_root(hashes1)
        root2 = compute_session_merkle_root(hashes2)

        assert root1 != root2

    def test_empty_list_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="Cannot compute Merkle root from empty list"):
            compute_session_merkle_root([])


class TestCanonicalJson:
    """Test canonical JSON functions."""

    def test_sorted_keys(self):
        """Test that keys are sorted."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)

        assert result == '{"a":2,"m":3,"z":1}'

    def test_minimal_separators(self):
        """Test minimal separators."""
        obj = {"key": "value"}
        result = canonical_json(obj)

        assert result == '{"key":"value"}'
        assert " " not in result

    def test_bytes_with_newline(self):
        """Test that bytes include trailing newline."""
        obj = {"key": "value"}
        result = canonical_json_bytes(obj)

        assert result.endswith(b"\n")


class TestReceiptDisplay:
    """Test receipt display formats."""

    def test_compact_format(self):
        """Test compact display format."""
        receipt = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
        )

        compact = receipt.compact()

        assert "[RECEIPT]" in compact
        assert "SAVE" in compact
        assert "resident" in compact

    def test_verbose_format(self):
        """Test verbose display format."""
        receipt = create_receipt(
            cassette_id="resident",
            operation="SAVE",
            record_id="a" * 64,
            record_hash="b" * 64,
            agent_id="test_agent",
        )

        verbose = receipt.verbose()

        assert "CASSETTE RECEIPT" in verbose
        assert "resident" in verbose
        assert "SAVE" in verbose
        assert "test_agent" in verbose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
