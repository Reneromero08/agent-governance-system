#!/usr/bin/env python3
"""
Tests for Skill Discovery - Phase 5.1.4

Tests:
1. Skill inventory creation
2. Metadata parsing from SKILL.md
3. Skill embedding with MemoryRecord
4. Semantic search by intent
5. Deterministic tie-breaking
6. Index rebuild determinism
7. Known queries return expected skills
8. Results stable across runs
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.skill_index import (
    inventory_skills,
    embed_skills,
    find_skills_by_intent,
    search_skills,
    rebuild_index,
    get_skill_by_id,
    list_all_skills,
    _parse_frontmatter,
    _parse_skill_metadata,
    _build_searchable_text,
)


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory with test skill files."""
    skills_dir = tmp_path / "CAPABILITY" / "SKILLS"
    skills_dir.mkdir(parents=True)

    # Create test skill 1: Canon governance checker
    skill1_dir = skills_dir / "governance" / "canon-check"
    skill1_dir.mkdir(parents=True)
    (skill1_dir / "SKILL.md").write_text("""---
name: canon-governance-check
version: "0.1.0"
description: Enforces changelog updates for significant changes to CANON
compatibility: all
---

# Canon Governance Check Skill

**Version:** 0.1.0

## Purpose
Enforces documentation hygiene by requiring changelog updates when significant system changes are made.

## Trigger
When changes are made to CANON, TOOLS, schemas, or ADRs.

## Inputs
- Git diff output
- List of changed files

## Outputs
- Pass/Fail status
- List of violations
""")

    # Create test skill 2: File analyzer
    skill2_dir = skills_dir / "utilities" / "file-analyzer"
    skill2_dir.mkdir(parents=True)
    (skill2_dir / "SKILL.md").write_text("""# Skill: file-analyzer

**Version:** 0.2.0
**Status:** Active

## Purpose
Analyzes repository structure and identifies critical files for understanding system architecture.

## Trigger
When user requests repository analysis or wants to understand codebase structure.

## Inputs
- Repository path
- Focus areas (optional)

## Outputs
- JSON report with file statistics
- Critical file list
""")

    # Create test skill 3: Commit manager
    skill3_dir = skills_dir / "commit-manager"
    skill3_dir.mkdir(parents=True)
    (skill3_dir / "SKILL.md").write_text("""---
name: commit-manager
version: "1.0.0"
description: Manages git commits with proper governance verification
---

# Commit Manager

## Purpose
Automates git commit workflow with integrated verification and changelog enforcement.

## Trigger
When user wants to commit changes with governance compliance.

## Inputs
- Commit message
- List of files to commit

## Outputs
- Git commit hash
- Verification receipts
""")

    # Create test skill 4: ADR generator
    skill4_dir = skills_dir / "governance" / "adr-generator"
    skill4_dir.mkdir(parents=True)
    (skill4_dir / "SKILL.md").write_text("""# Architecture Decision Record Generator

**Version:** 0.5.0

## Purpose
Generates properly formatted Architecture Decision Records (ADRs) following the canonical template.

## Trigger
When an architectural decision needs to be documented.

## Inputs
- Decision title
- Context
- Decision rationale
- Consequences

## Outputs
- Formatted ADR markdown file
- Metadata for indexing
""")

    # Create test skill 5: Verification tool
    skill5_dir = skills_dir / "utilities" / "verifier"
    skill5_dir.mkdir(parents=True)
    (skill5_dir / "SKILL.md").write_text("""---
name: verification-tool
version: "2.1.0"
description: Verifies system integrity and contract compliance
---

# Verification Tool

## Purpose
Performs comprehensive verification of system state against governance contracts.

## Trigger
Before committing changes or on demand for audit.

## Inputs
- Scope of verification (full/partial)
- Contract version to verify against

## Outputs
- Verification report
- List of violations
- Compliance score
""")

    return skills_dir


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_skill_index.db"


class TestSkillInventory:
    """Tests for inventory_skills function."""

    def test_inventory_creation(self, temp_skills_dir):
        """Test that inventory creates correct manifest."""
        manifest = inventory_skills(temp_skills_dir)

        assert manifest["total_skills"] == 5
        assert manifest["total_bytes"] > 0
        assert len(manifest["skills"]) == 5
        assert "manifest_hash" in manifest

    def test_inventory_skill_paths(self, temp_skills_dir):
        """Test that skill paths are normalized."""
        manifest = inventory_skills(temp_skills_dir)

        # Check normalized paths (forward slashes)
        assert "governance/canon-check" in manifest["skills"]
        assert "utilities/file-analyzer" in manifest["skills"]
        assert "commit-manager" in manifest["skills"]

    def test_inventory_metadata_extraction(self, temp_skills_dir):
        """Test that metadata is correctly extracted."""
        manifest = inventory_skills(temp_skills_dir)

        # Check frontmatter skill
        canon_check = manifest["skills"]["governance/canon-check"]
        assert canon_check["metadata"]["frontmatter"]["name"] == "canon-governance-check"
        assert canon_check["metadata"]["version"] == "0.1.0"
        assert "changelog" in canon_check["metadata"]["description"].lower()

        # Check non-frontmatter skill
        file_analyzer = manifest["skills"]["utilities/file-analyzer"]
        assert file_analyzer["metadata"]["name"] == "file-analyzer"
        assert file_analyzer["metadata"]["version"] == "0.2.0"

    def test_inventory_hash_determinism(self, temp_skills_dir):
        """Test that same content produces same manifest hash."""
        manifest1 = inventory_skills(temp_skills_dir)
        manifest2 = inventory_skills(temp_skills_dir)

        assert manifest1["manifest_hash"] == manifest2["manifest_hash"]

    def test_inventory_receipt(self, temp_skills_dir):
        """Test that receipt is emitted correctly."""
        manifest = inventory_skills(temp_skills_dir, emit_receipt=True)

        assert "receipt" in manifest
        assert manifest["receipt"]["operation"] == "inventory_skills"
        assert manifest["receipt"]["total_skills"] == 5


class TestMetadataParsing:
    """Tests for metadata parsing functions."""

    def test_parse_frontmatter(self):
        """Test YAML frontmatter parsing."""
        content = """---
name: test-skill
version: "1.0.0"
description: Test description
---

# Content"""
        metadata = _parse_frontmatter(content)

        assert metadata["name"] == "test-skill"
        assert metadata["version"] == "1.0.0"
        assert metadata["description"] == "Test description"

    def test_parse_frontmatter_empty(self):
        """Test parsing content without frontmatter."""
        content = "# Title\n\nContent"
        metadata = _parse_frontmatter(content)

        assert metadata == {}

    def test_parse_skill_metadata_full(self):
        """Test full metadata parsing."""
        content = """---
name: full-skill
version: "2.0.0"
---

# Full Skill

## Purpose
This skill does important things.

## Trigger
When user needs something.

## Inputs
- Input parameter 1
- Input parameter 2

## Outputs
- Output data
"""
        metadata = _parse_skill_metadata(content)

        assert metadata["name"] == "full-skill"
        assert metadata["version"] == "2.0.0"
        assert "important things" in metadata["purpose"]
        assert "user needs" in metadata["trigger"]
        assert "Input parameter" in metadata["inputs"]
        assert "Output data" in metadata["outputs"]

    def test_build_searchable_text(self):
        """Test searchable text construction."""
        metadata = {
            "name": "test-skill",
            "description": "Test description",
            "purpose": "Test purpose",
            "trigger": "Test trigger",
        }
        text = _build_searchable_text(metadata, "test/path")

        assert "test-skill" in text
        assert "Test description" in text
        assert "Test purpose" in text
        assert "Test trigger" in text
        assert "test/path" in text


class TestSkillEmbedding:
    """Tests for embed_skills function."""

    def test_embed_creates_database(self, temp_skills_dir, temp_db_path):
        """Test that embedding creates the database."""
        result = embed_skills(temp_skills_dir, temp_db_path)

        assert temp_db_path.exists()
        assert result["embedded_count"] == 5
        assert result["skipped_count"] == 0
        assert result["total_skills"] == 5

    def test_embed_stores_records(self, temp_skills_dir, temp_db_path):
        """Test that embedding stores complete skill records."""
        import sqlite3

        embed_skills(temp_skills_dir, temp_db_path)

        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute("SELECT COUNT(*) as count FROM skills")
        count = cursor.fetchone()["count"]

        assert count == 5

        # Check that records have expected fields
        cursor = conn.execute("SELECT * FROM skills LIMIT 1")
        row = cursor.fetchone()

        assert "skill_id" in row.keys()
        assert "skill_path" in row.keys()
        assert "content_hash" in row.keys()
        assert "metadata_json" in row.keys()
        assert "searchable_text" in row.keys()

        conn.close()

    def test_embed_stores_embeddings(self, temp_skills_dir, temp_db_path):
        """Test that embeddings are stored correctly."""
        import sqlite3
        import numpy as np

        embed_skills(temp_skills_dir, temp_db_path)

        conn = sqlite3.connect(str(temp_db_path))

        cursor = conn.execute("SELECT COUNT(*) as count FROM embeddings")
        count = cursor.fetchone()[0]

        assert count == 5

        # Check embedding dimensions
        cursor = conn.execute("SELECT embedding_blob FROM embeddings LIMIT 1")
        embedding_bytes = cursor.fetchone()[0]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        # all-MiniLM-L6-v2 has 384 dimensions
        assert len(embedding) == 384

        conn.close()

    def test_embed_skip_existing(self, temp_skills_dir, temp_db_path):
        """Test that re-embedding skips already embedded skills."""
        # First embedding
        result1 = embed_skills(temp_skills_dir, temp_db_path)
        assert result1["embedded_count"] == 5
        assert result1["skipped_count"] == 0

        # Second embedding (should skip all)
        result2 = embed_skills(temp_skills_dir, temp_db_path)
        assert result2["embedded_count"] == 0
        assert result2["skipped_count"] == 5

    def test_embed_force_rebuild(self, temp_skills_dir, temp_db_path):
        """Test that force_rebuild re-embeds all skills."""
        # First embedding
        embed_skills(temp_skills_dir, temp_db_path)

        # Force rebuild
        result = embed_skills(temp_skills_dir, temp_db_path, force_rebuild=True)
        assert result["embedded_count"] == 5
        assert result["skipped_count"] == 0

    def test_embed_receipt(self, temp_skills_dir, temp_db_path):
        """Test that receipt is emitted correctly."""
        result = embed_skills(temp_skills_dir, temp_db_path, emit_receipt=True)

        assert "receipt" in result
        assert result["receipt"]["operation"] == "embed_skills"
        assert result["receipt"]["embedded_count"] == 5


class TestSemanticSearch:
    """Tests for find_skills_by_intent function."""

    @pytest.fixture(autouse=True)
    def setup_index(self, temp_skills_dir, temp_db_path):
        """Embed skills before each test."""
        embed_skills(temp_skills_dir, temp_db_path)
        self.skills_dir = temp_skills_dir
        self.db_path = temp_db_path

    def test_search_returns_results(self):
        """Test that search returns results."""
        result = find_skills_by_intent(
            "check governance compliance",
            top_k=3,
            db_path=self.db_path
        )

        assert "results" in result
        assert len(result["results"]) <= 3
        assert result["total_candidates"] == 5

    def test_search_relevance_governance(self):
        """Test that governance queries return governance skills."""
        result = find_skills_by_intent(
            "verify canon changes and enforce changelog",
            top_k=3,
            db_path=self.db_path
        )

        # Top result should be canon-check or verification-tool
        top_skill = result["results"][0]
        assert "governance" in top_skill["skill_id"] or "verif" in top_skill["skill_id"].lower()
        assert top_skill["score"] > 0.3  # Reasonable similarity threshold

    def test_search_relevance_analysis(self):
        """Test that analysis queries return analysis tools."""
        result = find_skills_by_intent(
            "analyze repository structure and files",
            top_k=2,
            db_path=self.db_path
        )

        # file-analyzer should rank high
        skill_ids = [r["skill_id"] for r in result["results"]]
        assert any("file-analyzer" in sid for sid in skill_ids)

    def test_search_relevance_commit(self):
        """Test that commit-related queries return commit tools."""
        result = find_skills_by_intent(
            "create git commit with verification",
            top_k=3,
            db_path=self.db_path
        )

        # commit-manager should rank high
        skill_ids = [r["skill_id"] for r in result["results"]]
        assert any("commit" in sid for sid in skill_ids)

    def test_search_determinism(self):
        """Test that same query produces identical results."""
        query = "verify system integrity"

        result1 = find_skills_by_intent(query, top_k=3, db_path=self.db_path)
        result2 = find_skills_by_intent(query, top_k=3, db_path=self.db_path)

        # Same skill IDs in same order
        ids1 = [r["skill_id"] for r in result1["results"]]
        ids2 = [r["skill_id"] for r in result2["results"]]
        assert ids1 == ids2

        # Same scores
        scores1 = [r["score"] for r in result1["results"]]
        scores2 = [r["score"] for r in result2["results"]]
        assert scores1 == scores2

    def test_search_tie_breaking(self):
        """Test deterministic tie-breaking for equal scores."""
        # Search with a very generic query that might produce similar scores
        result = find_skills_by_intent(
            "skill",
            top_k=5,
            db_path=self.db_path
        )

        # Results should be sorted by score (desc), then skill_id (asc)
        results = result["results"]
        for i in range(len(results) - 1):
            if results[i]["score"] == results[i+1]["score"]:
                # If scores equal, IDs should be in alphabetical order
                assert results[i]["skill_id"] <= results[i+1]["skill_id"]

    def test_search_threshold_filter(self):
        """Test that threshold parameter filters results."""
        result_no_threshold = find_skills_by_intent(
            "test query",
            top_k=10,
            db_path=self.db_path
        )

        result_with_threshold = find_skills_by_intent(
            "test query",
            top_k=10,
            threshold=0.5,
            db_path=self.db_path
        )

        # With threshold, should have fewer or equal results
        assert len(result_with_threshold["results"]) <= len(result_no_threshold["results"])

        # All results should meet threshold
        for r in result_with_threshold["results"]:
            assert r["score"] >= 0.5

    def test_search_receipt(self):
        """Test that search emits receipt."""
        result = find_skills_by_intent(
            "test query",
            top_k=3,
            db_path=self.db_path,
            emit_receipt=True
        )

        assert "receipt" in result
        assert result["receipt"]["operation"] == "find_skills_by_intent"
        assert result["receipt"]["query"] == "test query"
        assert result["receipt"]["top_k"] == 3

    def test_search_alias(self):
        """Test that search_skills is an alias for find_skills_by_intent."""
        query = "governance check"

        result1 = find_skills_by_intent(query, db_path=self.db_path)
        result2 = search_skills(query, db_path=self.db_path)

        assert result1["query"] == result2["query"]
        assert len(result1["results"]) == len(result2["results"])


class TestIndexManagement:
    """Tests for index management functions."""

    def test_rebuild_index(self, temp_skills_dir, temp_db_path):
        """Test that rebuild_index recreates the database."""
        # Create initial index
        embed_skills(temp_skills_dir, temp_db_path)

        # Rebuild
        result = rebuild_index(temp_skills_dir, temp_db_path)

        assert result["embedded_count"] == 5
        assert temp_db_path.exists()

    def test_rebuild_determinism(self, temp_skills_dir, temp_db_path):
        """Test that rebuilding produces identical results."""
        import sqlite3
        import numpy as np

        # Build twice
        rebuild_index(temp_skills_dir, temp_db_path)

        # Get first set of embeddings
        conn1 = sqlite3.connect(str(temp_db_path))
        cursor1 = conn1.execute(
            "SELECT skill_id, embedding_blob FROM embeddings ORDER BY skill_id"
        )
        embeddings1 = {row[0]: np.frombuffer(row[1], dtype=np.float32)
                      for row in cursor1.fetchall()}
        conn1.close()

        # Rebuild again
        rebuild_index(temp_skills_dir, temp_db_path)

        # Get second set of embeddings
        conn2 = sqlite3.connect(str(temp_db_path))
        cursor2 = conn2.execute(
            "SELECT skill_id, embedding_blob FROM embeddings ORDER BY skill_id"
        )
        embeddings2 = {row[0]: np.frombuffer(row[1], dtype=np.float32)
                      for row in cursor2.fetchall()}
        conn2.close()

        # Compare embeddings
        assert set(embeddings1.keys()) == set(embeddings2.keys())
        for skill_id in embeddings1:
            np.testing.assert_array_almost_equal(
                embeddings1[skill_id],
                embeddings2[skill_id],
                decimal=6
            )

    def test_get_skill_by_id(self, temp_skills_dir, temp_db_path):
        """Test retrieving skill by ID."""
        embed_skills(temp_skills_dir, temp_db_path)

        skill = get_skill_by_id("governance/canon-check", temp_db_path)

        assert skill is not None
        assert skill["skill_id"] == "governance/canon-check"
        assert "metadata" in skill
        assert skill["metadata"]["name"] == "canon-governance-check"

    def test_get_skill_by_id_not_found(self, temp_skills_dir, temp_db_path):
        """Test retrieving non-existent skill."""
        embed_skills(temp_skills_dir, temp_db_path)

        skill = get_skill_by_id("nonexistent/skill", temp_db_path)

        assert skill is None

    def test_list_all_skills(self, temp_skills_dir, temp_db_path):
        """Test listing all indexed skills."""
        embed_skills(temp_skills_dir, temp_db_path)

        skills = list_all_skills(temp_db_path)

        assert len(skills) == 5
        assert all("skill_id" in s for s in skills)
        assert all("path" in s for s in skills)
        assert all("metadata" in s for s in skills)

        # Check that skills are sorted by ID
        skill_ids = [s["skill_id"] for s in skills]
        assert skill_ids == sorted(skill_ids)


class TestRealWorldSkills:
    """Tests using real skills from the repository."""

    def test_real_skills_inventory(self):
        """Test inventory with real skills."""
        # Use actual skills directory
        manifest = inventory_skills()

        # Should find multiple real skills
        assert manifest["total_skills"] > 0
        assert len(manifest["skills"]) > 0

        # Check that we're finding expected skills
        skill_ids = list(manifest["skills"].keys())
        # At minimum, we should have some governance or utility skills
        assert any("governance" in sid or "utilities" in sid for sid in skill_ids)

    def test_real_skills_embedding(self, temp_db_path):
        """Test embedding real skills."""
        result = embed_skills(db_path=temp_db_path)

        assert result["embedded_count"] > 0
        assert result["total_skills"] > 0

    def test_real_skills_search_governance(self, temp_db_path):
        """Test searching real skills for governance tasks."""
        # Embed real skills first
        embed_skills(db_path=temp_db_path)

        # Search for governance-related skills
        result = find_skills_by_intent(
            "check canon governance and changelog compliance",
            top_k=3,
            db_path=temp_db_path
        )

        assert len(result["results"]) > 0
        # Should find governance-related skills
        top_result = result["results"][0]
        assert "metadata" in top_result


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
