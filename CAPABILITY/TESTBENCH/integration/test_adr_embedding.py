#!/usr/bin/env python3
"""
Tests for ADR Embedding - Phase 5.1.2

Tests:
1. ADR inventory creation with metadata parsing
2. YAML frontmatter extraction
3. Manifest hash determinism
4. ADR embedding with MemoryRecord
5. Search functionality with metadata
6. Cross-reference to canon files
7. Index rebuild determinism
8. Index verification
9. Statistics reporting
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.adr_index import (
    inventory_adrs,
    embed_adrs,
    search_adrs,
    rebuild_adr_index,
    get_adr_stats,
    verify_adr_index,
    get_related_canon,
    _parse_frontmatter,
    _hash_text,
)


@pytest.fixture
def temp_adr_dir(tmp_path):
    """Create a temporary ADR directory with test files."""
    adr_dir = tmp_path / "LAW" / "CONTEXT" / "decisions"
    adr_dir.mkdir(parents=True)

    # Create test ADR files with YAML frontmatter
    (adr_dir / "ADR-001-test-decision.md").write_text(
        """---
id: "ADR-001"
title: "Test Decision for Build System"
status: "Accepted"
date: "2025-12-20"
confidence: "High"
impact: "Medium"
tags: ["architecture", "build"]
---

# ADR-001: Test Decision

## Context

This is a test ADR for the build system.

## Decision

We decided to use a specific approach for builds.

## Consequences

Build times will improve.
"""
    )

    (adr_dir / "ADR-002-governance-model.md").write_text(
        """---
id: "ADR-002"
title: "Governance Model Selection"
status: "Proposed"
date: "2025-12-21"
confidence: "Medium"
impact: "High"
tags: ["governance", "protocol"]
---

# ADR-002: Governance Model

## Context

Need to select governance model for the system.

## Decision

Use a verification-based governance approach.

## Consequences

All actions must be verified before execution.
"""
    )

    (adr_dir / "ADR-003-semantic-search.md").write_text(
        """---
id: "ADR-003"
title: "Semantic Search Implementation"
status: "Accepted"
date: "2025-12-22"
confidence: "High"
impact: "Medium"
tags: ["search", "embeddings"]
---

# ADR-003: Semantic Search

## Context

Need semantic search for architecture documents.

## Decision

Use sentence transformers for embeddings.

## Consequences

Improved document discovery.
"""
    )

    # ADR without frontmatter (edge case)
    (adr_dir / "ADR-000-template.md").write_text(
        """# ADR Template

This is a template without YAML frontmatter.
"""
    )

    return adr_dir


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_adr_index.db"


class TestFrontmatterParsing:
    """Tests for YAML frontmatter parsing."""

    def test_parse_valid_frontmatter(self):
        """Test parsing valid YAML frontmatter."""
        content = """---
id: "ADR-001"
title: "Test"
status: "Accepted"
---

# Content here
"""
        metadata, body = _parse_frontmatter(content)

        assert metadata["id"] == "ADR-001"
        assert metadata["title"] == "Test"
        assert metadata["status"] == "Accepted"
        assert "# Content here" in body

    def test_parse_missing_frontmatter(self):
        """Test handling missing frontmatter."""
        content = "# Just content without frontmatter"
        metadata, body = _parse_frontmatter(content)

        assert metadata == {}
        assert body == content

    def test_parse_tags_array(self):
        """Test parsing tags as array."""
        content = """---
id: "ADR-001"
tags: ["tag1", "tag2", "tag3"]
---

Content
"""
        metadata, _ = _parse_frontmatter(content)

        assert metadata["tags"] == ["tag1", "tag2", "tag3"]


class TestADRInventory:
    """Tests for inventory_adrs function."""

    def test_inventory_creation(self, temp_adr_dir):
        """Test that inventory creates correct manifest."""
        manifest = inventory_adrs(temp_adr_dir)

        assert manifest["total_files"] == 4
        assert manifest["total_bytes"] > 0
        assert len(manifest["files"]) == 4
        assert "manifest_hash" in manifest

    def test_inventory_metadata_extraction(self, temp_adr_dir):
        """Test that metadata is correctly extracted."""
        manifest = inventory_adrs(temp_adr_dir)

        adr_001 = manifest["files"]["ADR-001-test-decision.md"]
        assert adr_001["metadata"]["id"] == "ADR-001"
        assert adr_001["metadata"]["title"] == "Test Decision for Build System"
        assert adr_001["metadata"]["status"] == "Accepted"
        assert "architecture" in adr_001["metadata"]["tags"]

    def test_inventory_by_status(self, temp_adr_dir):
        """Test grouping by status."""
        manifest = inventory_adrs(temp_adr_dir)

        assert "Accepted" in manifest["by_status"]
        assert "Proposed" in manifest["by_status"]
        assert "ADR-001" in manifest["by_status"]["Accepted"]
        assert "ADR-002" in manifest["by_status"]["Proposed"]

    def test_inventory_hash_determinism(self, temp_adr_dir):
        """Test that same content produces same manifest hash."""
        manifest1 = inventory_adrs(temp_adr_dir)
        manifest2 = inventory_adrs(temp_adr_dir)

        assert manifest1["manifest_hash"] == manifest2["manifest_hash"]

    def test_inventory_receipt(self, temp_adr_dir):
        """Test that receipt is emitted correctly."""
        manifest = inventory_adrs(temp_adr_dir, emit_receipt=True)

        assert "receipt" in manifest
        assert manifest["receipt"]["operation"] == "inventory_adrs"
        assert manifest["receipt"]["total_files"] == 4
        assert "status_counts" in manifest["receipt"]


class TestADREmbedding:
    """Tests for embed_adrs function."""

    def test_embed_creates_database(self, temp_adr_dir, temp_db_path):
        """Test that embedding creates the database."""
        result = embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        assert temp_db_path.exists()
        assert result["embedded"] == 4
        assert result["errors"] == 0

    def test_embed_stores_records_with_metadata(self, temp_adr_dir, temp_db_path):
        """Test that embedding stores records with ADR metadata."""
        import sqlite3

        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT COUNT(*) as count FROM adr_records")
        count = cursor.fetchone()["count"]

        assert count == 4

        # Check metadata fields
        cursor = conn.execute(
            "SELECT * FROM adr_records WHERE adr_id = 'ADR-001'"
        )
        row = cursor.fetchone()

        assert row["title"] == "Test Decision for Build System"
        assert row["status"] == "Accepted"
        assert row["confidence"] == "High"
        assert row["embedding"] is not None

        conn.close()

    def test_embed_skips_existing(self, temp_adr_dir, temp_db_path):
        """Test that existing embeddings are skipped."""
        # First embed
        result1 = embed_adrs(temp_adr_dir, temp_db_path, verbose=False)
        assert result1["embedded"] == 4

        # Second embed without force
        result2 = embed_adrs(temp_adr_dir, temp_db_path, verbose=False)
        assert result2["embedded"] == 0
        assert result2["skipped"] == 4

    def test_embed_force_reembeds(self, temp_adr_dir, temp_db_path):
        """Test that force flag re-embeds all files."""
        # First embed
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        # Force re-embed
        result = embed_adrs(temp_adr_dir, temp_db_path, force=True, verbose=False)
        assert result["embedded"] == 4
        assert result["skipped"] == 0

    def test_embed_receipt(self, temp_adr_dir, temp_db_path):
        """Test that receipt is created correctly."""
        result = embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        assert "receipt" in result
        assert result["receipt"]["operation"] == "embed_adrs"
        assert result["receipt"]["model_id"] == "all-MiniLM-L6-v2"
        assert result["receipt"]["dimensions"] == 384
        assert "receipt_hash" in result["receipt"]


class TestADRSearch:
    """Tests for search_adrs function."""

    def test_search_returns_results(self, temp_adr_dir, temp_db_path):
        """Test that search returns relevant results."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        results = search_adrs("build system", db_path=temp_db_path, top_k=3)

        assert len(results) > 0
        assert all("similarity" in r for r in results)
        assert all("metadata" in r for r in results)

    def test_search_includes_metadata(self, temp_adr_dir, temp_db_path):
        """Test that search results include ADR metadata."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        results = search_adrs("governance", db_path=temp_db_path, top_k=3)

        assert len(results) > 0
        top_result = results[0]
        assert "adr_id" in top_result["metadata"]
        assert "title" in top_result["metadata"]
        assert "status" in top_result["metadata"]

    def test_search_ranking(self, temp_adr_dir, temp_db_path):
        """Test that results are ranked by similarity."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        results = search_adrs("semantic search embeddings", db_path=temp_db_path, top_k=5)

        # Results should be sorted by similarity descending
        similarities = [r["similarity"] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_status_filter(self, temp_adr_dir, temp_db_path):
        """Test filtering by status."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        # Only Accepted ADRs
        results = search_adrs(
            "decision", db_path=temp_db_path, top_k=10, status_filter="Accepted"
        )

        for r in results:
            assert r["metadata"]["status"] == "Accepted"

    def test_search_top_k_limit(self, temp_adr_dir, temp_db_path):
        """Test that top_k limits results."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        results = search_adrs("decision", db_path=temp_db_path, top_k=2)

        assert len(results) <= 2


class TestRebuildADRIndex:
    """Tests for rebuild_adr_index function."""

    def test_rebuild_deletes_and_recreates(self, temp_adr_dir, temp_db_path):
        """Test that rebuild deletes existing and recreates."""
        # Initial embed
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        # Rebuild
        result = rebuild_adr_index(temp_adr_dir, temp_db_path, verbose=False)

        assert result["embedded"] == 4
        assert result["receipt"]["operation"] == "rebuild_adr_index"

    def test_rebuild_determinism(self, temp_adr_dir, temp_db_path):
        """Test that rebuilds produce consistent results."""
        # First build
        result1 = rebuild_adr_index(temp_adr_dir, temp_db_path, verbose=False)
        hash1 = result1["receipt"]["manifest_hash"]

        # Second build
        result2 = rebuild_adr_index(temp_adr_dir, temp_db_path, verbose=False)
        hash2 = result2["receipt"]["manifest_hash"]

        # Manifest hashes should match
        assert hash1 == hash2


class TestADRStats:
    """Tests for get_adr_stats function."""

    def test_stats_on_empty(self, temp_db_path):
        """Test stats on non-existent index."""
        stats = get_adr_stats(temp_db_path)
        assert stats["exists"] is False

    def test_stats_after_embed(self, temp_adr_dir, temp_db_path):
        """Test stats after embedding."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        stats = get_adr_stats(temp_db_path)

        assert stats["exists"] is True
        assert stats["total_records"] == 4
        assert stats["embedded_records"] == 4
        assert "Accepted" in stats["status_counts"]
        assert "Proposed" in stats["status_counts"]
        assert stats["receipt_count"] >= 1


class TestADRIndexVerification:
    """Tests for verify_adr_index function."""

    def test_verify_valid_index(self, temp_adr_dir, temp_db_path):
        """Test verification of valid index."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        result = verify_adr_index(temp_db_path, temp_adr_dir)

        assert result["valid"] is True
        assert len(result["mismatches"]) == 0
        assert len(result["missing_files"]) == 0

    def test_verify_detects_missing_file(self, temp_adr_dir, temp_db_path):
        """Test that verification detects missing indexed file."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        # Add a new file (will be missing from index)
        (temp_adr_dir / "ADR-099-new.md").write_text(
            """---
id: "ADR-099"
title: "New ADR"
status: "Draft"
---

New content
"""
        )

        result = verify_adr_index(temp_db_path, temp_adr_dir)

        assert result["valid"] is False
        assert "ADR-099-new.md" in result["missing_files"]

    def test_verify_detects_changed_file(self, temp_adr_dir, temp_db_path):
        """Test that verification detects changed file content."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        # Modify a file
        (temp_adr_dir / "ADR-001-test-decision.md").write_text("Modified content")

        result = verify_adr_index(temp_db_path, temp_adr_dir)

        assert result["valid"] is False
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["file"] == "ADR-001-test-decision.md"


class TestIntegrationWithMemoryRecord:
    """Tests for integration with MemoryRecord primitive."""

    def test_stored_id_matches_hash(self, temp_adr_dir, temp_db_path):
        """Test that stored id is deterministic content hash."""
        import sqlite3

        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT id, text FROM adr_records")
        for row in cursor.fetchall():
            expected_id = _hash_text(row["text"])
            assert row["id"] == expected_id

        conn.close()

    def test_embedding_dimensions(self, temp_adr_dir, temp_db_path):
        """Test that embeddings have correct dimensions."""
        import sqlite3
        import numpy as np

        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT embedding, dimensions FROM adr_records")
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            assert len(embedding) == 384
            assert row["dimensions"] == 384

        conn.close()


class TestCrossReferences:
    """Tests for ADR-Canon cross-references."""

    def test_get_related_canon_empty(self, temp_adr_dir, temp_db_path):
        """Test getting related canon when no cross-refs exist."""
        embed_adrs(temp_adr_dir, temp_db_path, verbose=False)

        # No canon index exists, so should return empty
        results = get_related_canon(
            "ADR-001-test-decision.md", db_path=temp_db_path
        )

        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
