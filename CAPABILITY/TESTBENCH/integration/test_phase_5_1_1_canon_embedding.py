#!/usr/bin/env python3
"""
Tests for Canon Embedding - Phase 5.1.1

Tests:
1. Canon inventory creation
2. Manifest hash determinism
3. Canon embedding with MemoryRecord
4. Search functionality
5. Index rebuild determinism
6. Index verification
7. Statistics reporting
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.canon_index import (
    inventory_canon,
    embed_canon,
    search_canon,
    rebuild_index,
    get_index_stats,
    verify_index,
    _hash_text,
)


@pytest.fixture
def temp_canon_dir(tmp_path):
    """Create a temporary canon directory with test files."""
    canon_dir = tmp_path / "LAW" / "CANON"
    canon_dir.mkdir(parents=True)

    # Create test files
    (canon_dir / "CONSTITUTION").mkdir()
    (canon_dir / "CONSTITUTION" / "CONTRACT.md").write_text(
        "# Contract\n\nThis is the governance contract."
    )
    (canon_dir / "CONSTITUTION" / "INTEGRITY.md").write_text(
        "# Integrity\n\nIntegrity rules for the system."
    )

    (canon_dir / "GOVERNANCE").mkdir()
    (canon_dir / "GOVERNANCE" / "VERIFICATION.md").write_text(
        "# Verification Protocol\n\nAll operations must be verified."
    )

    (canon_dir / "META").mkdir()
    (canon_dir / "META" / "INDEX.md").write_text(
        "# Index\n\nThis is the canon index file."
    )

    return canon_dir


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_canon_index.db"


class TestCanonInventory:
    """Tests for inventory_canon function."""

    def test_inventory_creation(self, temp_canon_dir):
        """Test that inventory creates correct manifest."""
        manifest = inventory_canon(temp_canon_dir)

        assert manifest["total_files"] == 4
        assert manifest["total_bytes"] > 0
        assert len(manifest["files"]) == 4
        assert "manifest_hash" in manifest

    def test_inventory_file_paths(self, temp_canon_dir):
        """Test that file paths are normalized."""
        manifest = inventory_canon(temp_canon_dir)

        # Check normalized paths (forward slashes)
        assert "CONSTITUTION/CONTRACT.md" in manifest["files"]
        assert "GOVERNANCE/VERIFICATION.md" in manifest["files"]

    def test_inventory_content_hashes(self, temp_canon_dir):
        """Test that content hashes are correct."""
        manifest = inventory_canon(temp_canon_dir)

        # Read file and compute expected hash
        content = (temp_canon_dir / "CONSTITUTION" / "CONTRACT.md").read_bytes()
        expected_hash = _hash_text(content.decode("utf-8"))

        # Note: _hash_text uses text encoding, but inventory uses bytes
        # So we need to use the same method as inventory
        import hashlib
        expected_hash = hashlib.sha256(content).hexdigest()

        assert manifest["files"]["CONSTITUTION/CONTRACT.md"]["sha256"] == expected_hash

    def test_inventory_hash_determinism(self, temp_canon_dir):
        """Test that same content produces same manifest hash."""
        manifest1 = inventory_canon(temp_canon_dir)
        manifest2 = inventory_canon(temp_canon_dir)

        assert manifest1["manifest_hash"] == manifest2["manifest_hash"]

    def test_inventory_receipt(self, temp_canon_dir):
        """Test that receipt is emitted correctly."""
        manifest = inventory_canon(temp_canon_dir, emit_receipt=True)

        assert "receipt" in manifest
        assert manifest["receipt"]["operation"] == "inventory_canon"
        assert manifest["receipt"]["total_files"] == 4


class TestCanonEmbedding:
    """Tests for embed_canon function."""

    def test_embed_creates_database(self, temp_canon_dir, temp_db_path):
        """Test that embedding creates the database."""
        result = embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        assert temp_db_path.exists()
        assert result["embedded"] == 4
        assert result["errors"] == 0

    def test_embed_stores_records(self, temp_canon_dir, temp_db_path):
        """Test that embedding stores MemoryRecord-compatible entries."""
        import sqlite3

        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT COUNT(*) as count FROM canon_records")
        count = cursor.fetchone()["count"]

        assert count == 4

        # Check that records have expected fields
        cursor = conn.execute("SELECT * FROM canon_records LIMIT 1")
        row = cursor.fetchone()

        assert row["id"] is not None
        assert len(row["id"]) == 64  # SHA-256 hex
        assert row["text"] is not None
        assert row["embedding"] is not None
        assert row["trust"] == 1.0

        conn.close()

    def test_embed_skips_existing(self, temp_canon_dir, temp_db_path):
        """Test that existing embeddings are skipped."""
        # First embed
        result1 = embed_canon(temp_canon_dir, temp_db_path, verbose=False)
        assert result1["embedded"] == 4

        # Second embed without force
        result2 = embed_canon(temp_canon_dir, temp_db_path, verbose=False)
        assert result2["embedded"] == 0
        assert result2["skipped"] == 4

    def test_embed_force_reembeds(self, temp_canon_dir, temp_db_path):
        """Test that force flag re-embeds all files."""
        # First embed
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        # Force re-embed
        result = embed_canon(temp_canon_dir, temp_db_path, force=True, verbose=False)
        assert result["embedded"] == 4
        assert result["skipped"] == 0

    def test_embed_receipt(self, temp_canon_dir, temp_db_path):
        """Test that receipt is created correctly."""
        result = embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        assert "receipt" in result
        assert result["receipt"]["operation"] == "embed_canon"
        assert result["receipt"]["model_id"] == "all-MiniLM-L6-v2"
        assert result["receipt"]["dimensions"] == 384
        assert "receipt_hash" in result["receipt"]


class TestCanonSearch:
    """Tests for search_canon function."""

    def test_search_returns_results(self, temp_canon_dir, temp_db_path):
        """Test that search returns relevant results."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        results = search_canon("verification protocol", db_path=temp_db_path, top_k=3)

        assert len(results) > 0
        assert all("similarity" in r for r in results)
        assert all("file_path" in r for r in results)

    def test_search_ranking(self, temp_canon_dir, temp_db_path):
        """Test that results are ranked by similarity."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        results = search_canon("verification", db_path=temp_db_path, top_k=5)

        # Results should be sorted by similarity descending
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_top_k_limit(self, temp_canon_dir, temp_db_path):
        """Test that top_k limits results."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        results = search_canon("governance", db_path=temp_db_path, top_k=2)

        assert len(results) <= 2

    def test_search_min_similarity(self, temp_canon_dir, temp_db_path):
        """Test that min_similarity filters results."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        # Very high threshold should filter most results
        results = search_canon(
            "random unrelated query xyz",
            db_path=temp_db_path,
            min_similarity=0.9,
        )

        # Most results should be filtered out
        assert len(results) <= 1


class TestRebuildIndex:
    """Tests for rebuild_index function."""

    def test_rebuild_deletes_and_recreates(self, temp_canon_dir, temp_db_path):
        """Test that rebuild deletes existing and recreates."""
        # Initial embed
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        # Rebuild
        result = rebuild_index(temp_canon_dir, temp_db_path, verbose=False)

        assert result["embedded"] == 4
        assert result["receipt"]["operation"] == "rebuild_index"

    def test_rebuild_determinism(self, temp_canon_dir, temp_db_path):
        """Test that rebuilds produce consistent results."""
        # First build
        result1 = rebuild_index(temp_canon_dir, temp_db_path, verbose=False)
        hash1 = result1["receipt"]["manifest_hash"]

        # Second build
        result2 = rebuild_index(temp_canon_dir, temp_db_path, verbose=False)
        hash2 = result2["receipt"]["manifest_hash"]

        # Manifest hashes should match
        assert hash1 == hash2


class TestIndexStats:
    """Tests for get_index_stats function."""

    def test_stats_on_empty(self, temp_db_path):
        """Test stats on non-existent index."""
        stats = get_index_stats(temp_db_path)
        assert stats["exists"] is False

    def test_stats_after_embed(self, temp_canon_dir, temp_db_path):
        """Test stats after embedding."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        stats = get_index_stats(temp_db_path)

        assert stats["exists"] is True
        assert stats["total_records"] == 4
        assert stats["embedded_records"] == 4
        assert stats["receipt_count"] >= 1


class TestIndexVerification:
    """Tests for verify_index function."""

    def test_verify_valid_index(self, temp_canon_dir, temp_db_path):
        """Test verification of valid index."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        result = verify_index(temp_db_path, temp_canon_dir)

        assert result["valid"] is True
        assert len(result["mismatches"]) == 0
        assert len(result["missing_files"]) == 0

    def test_verify_detects_missing_file(self, temp_canon_dir, temp_db_path):
        """Test that verification detects missing indexed file."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        # Add a new file (will be missing from index)
        (temp_canon_dir / "NEW_FILE.md").write_text("New content")

        result = verify_index(temp_db_path, temp_canon_dir)

        assert result["valid"] is False
        assert "NEW_FILE.md" in result["missing_files"]

    def test_verify_detects_changed_file(self, temp_canon_dir, temp_db_path):
        """Test that verification detects changed file content."""
        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        # Modify a file
        (temp_canon_dir / "CONSTITUTION" / "CONTRACT.md").write_text("Modified content")

        result = verify_index(temp_db_path, temp_canon_dir)

        assert result["valid"] is False
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["file"] == "CONSTITUTION/CONTRACT.md"


class TestIntegrationWithMemoryRecord:
    """Tests for integration with MemoryRecord primitive."""

    def test_stored_id_matches_hash(self, temp_canon_dir, temp_db_path):
        """Test that stored id is deterministic content hash."""
        import sqlite3

        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT id, text FROM canon_records")
        for row in cursor.fetchall():
            expected_id = _hash_text(row["text"])
            assert row["id"] == expected_id

        conn.close()

    def test_embedding_dimensions(self, temp_canon_dir, temp_db_path):
        """Test that embeddings have correct dimensions."""
        import sqlite3
        import numpy as np

        embed_canon(temp_canon_dir, temp_db_path, verbose=False)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT embedding, dimensions FROM canon_records")
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            assert len(embedding) == 384
            assert row["dimensions"] == 384

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
