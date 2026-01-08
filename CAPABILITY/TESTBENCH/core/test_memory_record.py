#!/usr/bin/env python3
"""
Tests for MemoryRecord primitive - Phase 5.0 Foundation

Tests:
1. Record creation with minimal args
2. Record creation with all args
3. Hash determinism (same input -> same hash)
4. Hash consistency (id matches computed hash)
5. Validation of valid records
6. Validation catches missing required fields
7. Validation catches hash mismatch
8. Embedding addition
9. JSON serialization determinism
10. JSON round-trip
11. Invalid score values
12. Schema validation (if jsonschema available)
13. Edge cases (empty embeddings, special characters)
"""

import json
import sys
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.memory_record import (
    create_record,
    validate_record,
    hash_record,
    add_embedding,
    to_json,
    from_json,
    canonical_bytes,
    full_hash,
    MemoryRecord,
)


class TestRecordCreation:
    """Tests for create_record function."""

    def test_create_minimal(self):
        """Test record creation with minimal arguments."""
        record = create_record("Hello, world!")

        assert record["text"] == "Hello, world!"
        assert len(record["id"]) == 64  # SHA-256 hex
        assert record["embeddings"] == {}
        assert "created_at" in record["payload"]
        assert record["scores"]["trust"] == 1.0
        assert record["lineage"]["version"] == 1
        assert record["receipts"]["content_hash"] == record["id"]

    def test_create_with_all_args(self):
        """Test record creation with all optional arguments."""
        record = create_record(
            "Full record content",
            doc_path="LAW/CANON/CONSTITUTION/CONTRACT.md",
            doc_id="contract-v1",
            tags=["canon", "constitution", "law"],
            roles=["admin", "auditor"],
            chunk_index=0,
            chunk_count=5,
            content_type="text/markdown",
            language="en",
            extra={"custom_field": "custom_value"},
            elo=1500.0,
            trust=0.95,
            created_by="test_suite",
            tool_version="1.0.0",
        )

        assert record["payload"]["doc_path"] == "LAW/CANON/CONSTITUTION/CONTRACT.md"
        assert record["payload"]["tags"] == ["canon", "constitution", "law"]
        assert record["payload"]["chunk_index"] == 0
        assert record["payload"]["chunk_count"] == 5
        assert record["scores"]["elo"] == 1500.0
        assert record["scores"]["trust"] == 0.95
        assert record["receipts"]["created_by"] == "test_suite"
        assert record["receipts"]["tool_version"] == "1.0.0"


class TestHashDeterminism:
    """Tests for hash determinism."""

    def test_same_input_same_hash(self):
        """Same text content produces same hash."""
        text = "Deterministic content for hashing"

        record1 = create_record(text)
        record2 = create_record(text)

        assert record1["id"] == record2["id"]
        assert hash_record(record1) == hash_record(record2)

    def test_different_input_different_hash(self):
        """Different text content produces different hash."""
        record1 = create_record("First content")
        record2 = create_record("Second content")

        assert record1["id"] != record2["id"]

    def test_hash_consistency(self):
        """Record id matches computed hash."""
        record = create_record("Content for hash consistency test")

        computed = hash_record(record)
        assert record["id"] == computed
        assert record["receipts"]["content_hash"] == computed


class TestValidation:
    """Tests for validate_record function."""

    def test_valid_record_passes(self):
        """Valid record passes validation."""
        record = create_record("Valid content")
        verdict = validate_record(record)

        assert verdict["valid"] is True
        assert len(verdict["errors"]) == 0

    def test_missing_required_fields(self):
        """Missing required fields fail validation."""
        incomplete = {"text": "Missing fields"}
        verdict = validate_record(incomplete)

        assert verdict["valid"] is False
        assert any("Missing required field" in e for e in verdict["errors"])

    def test_hash_mismatch_detected(self):
        """Hash mismatch is detected."""
        record = create_record("Original content")
        record["id"] = "0" * 64  # Wrong hash

        verdict = validate_record(record)

        assert verdict["valid"] is False
        assert any("Hash mismatch" in e for e in verdict["errors"])

    def test_receipt_hash_mismatch_detected(self):
        """Receipt content_hash mismatch is detected."""
        record = create_record("Content for receipt test")
        record["receipts"]["content_hash"] = "1" * 64  # Wrong hash

        verdict = validate_record(record)

        assert verdict["valid"] is False
        assert any("Receipt hash mismatch" in e for e in verdict["errors"])

    def test_empty_text_fails(self):
        """Empty text fails validation."""
        record = create_record("placeholder")
        record["text"] = ""
        record["id"] = hash_record({"text": ""})  # Recalc for empty

        verdict = validate_record(record, check_hash=False)

        assert verdict["valid"] is False
        assert any("non-empty" in e for e in verdict["errors"])

    def test_invalid_score_values(self):
        """Invalid score values fail validation."""
        record = create_record("Score test content")
        record["scores"]["trust"] = 2.0  # Invalid: > 1

        verdict = validate_record(record)

        assert verdict["valid"] is False
        assert any("trust" in e and "between 0 and 1" in e for e in verdict["errors"])


class TestEmbeddings:
    """Tests for embedding operations."""

    def test_add_embedding(self):
        """Adding embedding updates record correctly."""
        record = create_record("Text for embedding")
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        add_embedding(record, "test-model", vector, "text-embedding-test")

        assert "test-model" in record["embeddings"]
        emb = record["embeddings"]["test-model"]
        assert emb["vector"] == vector
        assert emb["model"] == "text-embedding-test"
        assert emb["dimensions"] == 5
        assert "test-model" in record["receipts"]["embedding_receipts"]

    def test_multiple_embeddings(self):
        """Multiple embeddings can coexist."""
        record = create_record("Multi-embed content")

        add_embedding(record, "model-a", [0.1, 0.2], "model-a-v1")
        add_embedding(record, "model-b", [0.3, 0.4, 0.5], "model-b-v1")

        assert len(record["embeddings"]) == 2
        assert record["embeddings"]["model-a"]["dimensions"] == 2
        assert record["embeddings"]["model-b"]["dimensions"] == 3


class TestSerialization:
    """Tests for JSON serialization."""

    def test_to_json_deterministic(self):
        """JSON serialization is deterministic."""
        record = create_record("Serialization test")

        json1 = to_json(record)
        json2 = to_json(record)

        assert json1 == json2

    def test_json_round_trip(self):
        """JSON round-trip preserves record."""
        original = create_record(
            "Round trip content",
            tags=["test", "round-trip"],
            doc_path="test/path.md",
        )

        json_str = to_json(original)
        restored = from_json(json_str)

        assert restored["id"] == original["id"]
        assert restored["text"] == original["text"]
        assert restored["payload"]["tags"] == original["payload"]["tags"]

    def test_canonical_bytes_deterministic(self):
        """Canonical bytes are deterministic."""
        record = create_record("Canonical test")

        bytes1 = canonical_bytes(record)
        bytes2 = canonical_bytes(record)

        assert bytes1 == bytes2

    def test_full_hash_changes_on_modification(self):
        """Full hash changes when any field changes."""
        record = create_record("Hash change test")
        hash1 = full_hash(record)

        record["scores"]["elo"] = 1500.0
        hash2 = full_hash(record)

        assert hash1 != hash2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_embeddings_valid(self):
        """Empty embeddings dict is valid."""
        record = create_record("No embeddings yet")
        assert record["embeddings"] == {}

        verdict = validate_record(record)
        assert verdict["valid"] is True

    def test_unicode_content(self):
        """Unicode content is handled correctly."""
        record = create_record("日本語テスト 中文测试 한국어테스트 🚀")

        assert record["text"] == "日本語テスト 中文测试 한국어테스트 🚀"
        verdict = validate_record(record)
        assert verdict["valid"] is True

    def test_special_characters(self):
        """Special characters in content are handled."""
        content = 'Test with "quotes", \\backslashes\\, and\nnewlines'
        record = create_record(content)

        verdict = validate_record(record)
        assert verdict["valid"] is True

        # Round-trip preserves content
        restored = from_json(to_json(record))
        assert restored["text"] == content

    def test_large_content(self):
        """Large content is handled correctly."""
        large_text = "x" * 100000  # 100KB
        record = create_record(large_text)

        assert len(record["text"]) == 100000
        verdict = validate_record(record)
        assert verdict["valid"] is True


class TestSchemaValidation:
    """Tests for JSON schema validation."""

    def test_schema_validation_if_available(self):
        """Schema validation runs if jsonschema is installed."""
        record = create_record("Schema validation test")
        verdict = validate_record(record, check_schema=True)

        # Should pass or warn (if jsonschema not installed)
        assert verdict["valid"] is True

    def test_invalid_embeddings_structure(self):
        """Invalid embeddings structure fails validation."""
        record = create_record("Embeddings structure test")
        record["embeddings"]["bad"] = "not an object"

        verdict = validate_record(record, check_schema=False)

        assert verdict["valid"] is False
        assert any("must be an object" in e for e in verdict["errors"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
