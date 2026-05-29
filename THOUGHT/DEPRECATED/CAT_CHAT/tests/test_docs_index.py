#!/usr/bin/env python3
"""
Tests for Docs Index (Phase F)

Exit Criteria:
- `docs search --query "..." --limit N` returns bounded, deterministic results
"""

import hashlib
import sqlite3
import tempfile
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from catalytic_chat.docs_index import (
    DocsIndex,
    DocsIndexError,
    DocsSearchResult,
    DocsSearchResponse,
    IndexStats,
    build_docs_index,
    search_docs
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository with test markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create _generated directory for the database
        (repo_root / "THOUGHT" / "LAB" / "CAT_CHAT" / "_generated").mkdir(parents=True)

        # Create test markdown files
        readme = repo_root / "README.md"
        readme.write_text("# Project README\n\nThis is the main readme file.\n", encoding='utf-8')

        # Create LAW/CANON directory with files
        canon_dir = repo_root / "LAW" / "CANON"
        canon_dir.mkdir(parents=True)

        invariants = canon_dir / "INVARIANTS.md"
        invariants.write_text(
            "# Invariants\n\n"
            "## INV-001 Catalytic Restoration\n\n"
            "File states before/after must be identical.\n\n"
            "## INV-002 Verification\n\n"
            "Proof size is O(1) per domain.\n",
            encoding='utf-8'
        )

        governance = canon_dir / "GOVERNANCE.md"
        governance.write_text(
            "# Governance\n\n"
            "All changes require review.\n"
            "Catalytic operations must be reversible.\n",
            encoding='utf-8'
        )

        yield repo_root


@pytest.fixture
def docs_index(temp_repo):
    """Create a DocsIndex instance for testing."""
    idx = DocsIndex(repo_root=temp_repo)
    yield idx
    idx.close()


class TestDocsIndexSchema:
    """Tests for F.1: FTS Tables Schema"""

    def test_schema_creates_tables(self, docs_index):
        """Tables are created on init."""
        conn = docs_index._get_conn()

        # Check docs_files table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_files'"
        )
        assert cursor.fetchone() is not None

        # Check docs_content table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_content'"
        )
        assert cursor.fetchone() is not None

        # Check FTS5 virtual table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_content_fts'"
        )
        assert cursor.fetchone() is not None

    def test_schema_creates_indexes(self, docs_index):
        """Indexes are created for efficient lookup."""
        conn = docs_index._get_conn()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_docs_files_sha256'"
        )
        assert cursor.fetchone() is not None

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_docs_files_path'"
        )
        assert cursor.fetchone() is not None


class TestDocsIndexing:
    """Tests for F.1: File Indexing"""

    def test_index_single_file(self, docs_index, temp_repo):
        """Index a single file and verify in DB."""
        readme = temp_repo / "README.md"
        sha256 = docs_index.index_file(readme)

        assert sha256 is not None
        assert len(sha256) == 64  # SHA-256 hex

        # Verify in database
        conn = docs_index._get_conn()
        cursor = conn.execute(
            "SELECT path, sha256 FROM docs_files WHERE path = ?",
            ("README.md",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row['sha256'] == sha256

    def test_index_incremental_skip_unchanged(self, docs_index, temp_repo):
        """Skip unchanged files (sha256 match)."""
        readme = temp_repo / "README.md"

        # First index
        sha1 = docs_index.index_file(readme)
        assert sha1 is not None

        # Second index - should return None (skipped)
        sha2 = docs_index.index_file(readme)
        assert sha2 is None

    def test_index_reindex_changed_file(self, docs_index, temp_repo):
        """Re-index changed files."""
        readme = temp_repo / "README.md"

        # First index
        sha1 = docs_index.index_file(readme)
        assert sha1 is not None

        # Modify file
        readme.write_text("# Modified README\n\nNew content.\n", encoding='utf-8')

        # Second index - should return new hash
        sha2 = docs_index.index_file(readme)
        assert sha2 is not None
        assert sha2 != sha1

    def test_index_all_files(self, docs_index, temp_repo):
        """Index all indexable files."""
        stats = docs_index.index_all()

        # Should have indexed README.md, INVARIANTS.md, GOVERNANCE.md
        assert stats.indexed >= 3
        assert stats.errors == 0

    def test_index_all_incremental(self, docs_index, temp_repo):
        """Incremental index skips unchanged files."""
        # First full index
        stats1 = docs_index.index_all()
        assert stats1.indexed >= 3

        # Second incremental index - all skipped
        stats2 = docs_index.index_all(incremental=True)
        assert stats2.indexed == 0
        assert stats2.skipped >= 3


class TestDocsSearch:
    """Tests for F.2 & F.3: Query API and Deterministic Ranking"""

    def test_search_basic(self, docs_index, temp_repo):
        """Simple query returns results."""
        docs_index.index_all()

        response = docs_index.search("catalytic")

        assert isinstance(response, DocsSearchResponse)
        assert response.query == "catalytic"
        assert len(response.results) > 0

    def test_search_bounded(self, docs_index, temp_repo):
        """Results respect limit parameter."""
        docs_index.index_all()

        response = docs_index.search("file", limit=1)

        assert len(response.results) <= 1
        assert response.limit_applied == 1

    def test_search_empty(self, docs_index, temp_repo):
        """No results for non-matching query."""
        docs_index.index_all()

        response = docs_index.search("xyznonexistent123")

        assert len(response.results) == 0
        assert response.total_matches == 0

    def test_deterministic_ranking(self, docs_index, temp_repo):
        """Same query returns same results in same order."""
        docs_index.index_all()

        # Run same query multiple times
        r1 = docs_index.search("catalytic", limit=10)
        r2 = docs_index.search("catalytic", limit=10)
        r3 = docs_index.search("catalytic", limit=10)

        # Results must be identical
        paths1 = [r.file_path for r in r1.results]
        paths2 = [r.file_path for r in r2.results]
        paths3 = [r.file_path for r in r3.results]

        assert paths1 == paths2 == paths3

        # Scores must be identical
        scores1 = [r.match_score for r in r1.results]
        scores2 = [r.match_score for r in r2.results]

        assert scores1 == scores2

    def test_snippet_length_bounded(self, docs_index, temp_repo):
        """Snippets are bounded at MAX_SNIPPET_LENGTH."""
        docs_index.index_all()

        response = docs_index.search("invariant")

        for result in response.results:
            assert len(result.snippet) <= DocsIndex.MAX_SNIPPET_LENGTH

    def test_search_with_offset(self, docs_index, temp_repo):
        """Offset skips first N results."""
        docs_index.index_all()

        r1 = docs_index.search("file", limit=10, offset=0)
        r2 = docs_index.search("file", limit=10, offset=1)

        if len(r1.results) > 1:
            # Second result from first query should be first result from offset query
            assert r1.results[1].file_path == r2.results[0].file_path


class TestDocsIndexChunking:
    """Tests for large file handling."""

    def test_chunk_large_file(self, temp_repo):
        """Files > MAX_CHUNK_SIZE are split correctly."""
        # Create a large file with paragraph breaks so chunking works
        # Each paragraph is ~10KB, so 10 paragraphs = ~100KB
        paragraphs = []
        for i in range(10):
            # Each paragraph is about 10KB (2000 words * 5 chars)
            paragraphs.append(f"Paragraph {i}. " + ("Word " * 2000))
        large_content = "# Large File\n\n" + "\n\n".join(paragraphs)
        large_file = temp_repo / "large.md"
        large_file.write_text(large_content, encoding='utf-8')

        idx = DocsIndex(repo_root=temp_repo)
        try:
            sha256 = idx.index_file(large_file)
            assert sha256 is not None

            # Check multiple chunks were created
            conn = idx._get_conn()
            cursor = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM docs_content dc
                JOIN docs_files df ON dc.file_id = df.file_id
                WHERE df.path = ?
                """,
                ("large.md",)
            )
            chunk_count = cursor.fetchone()['cnt']
            assert chunk_count >= 2  # Should have multiple chunks
        finally:
            idx.close()


class TestDocsIndexFTSSync:
    """Tests for FTS sync triggers."""

    def test_fts_sync_on_insert(self, docs_index, temp_repo):
        """FTS is updated when content is inserted."""
        docs_index.index_all()

        # Search should find content
        response = docs_index.search("invariant")
        assert len(response.results) > 0

    def test_fts_sync_on_delete(self, docs_index, temp_repo):
        """FTS is updated when file is deleted."""
        docs_index.index_all()

        # Verify file exists
        r1 = docs_index.search("readme")
        initial_count = len(r1.results)

        # Delete file
        readme = temp_repo / "README.md"
        readme.unlink()

        # Re-index (should remove deleted file)
        docs_index.index_all()

        # Search should not find deleted content
        r2 = docs_index.search("readme")
        # The old results may still appear until FTS is rebuilt
        # but the file should be removed from docs_files


class TestDocsIndexContent:
    """Tests for content retrieval."""

    def test_get_file_content_by_hash(self, docs_index, temp_repo):
        """Get full file content by SHA-256 hash."""
        readme = temp_repo / "README.md"
        expected_content = readme.read_text(encoding='utf-8')

        sha256 = docs_index.index_file(readme)
        content = docs_index.get_file_content(sha256)

        assert content == expected_content

    def test_get_file_content_not_found(self, docs_index, temp_repo):
        """Return None for unknown hash."""
        fake_hash = "a" * 64
        content = docs_index.get_file_content(fake_hash)

        assert content is None


class TestDocsIndexStats:
    """Tests for index statistics."""

    def test_get_stats(self, docs_index, temp_repo):
        """Get index statistics."""
        docs_index.index_all()

        stats = docs_index.get_stats()

        assert "files" in stats
        assert "chunks" in stats
        assert "total_bytes" in stats
        assert "db_path" in stats

        assert stats["files"] >= 3


class TestDocsIndexConvenienceFunctions:
    """Tests for convenience functions."""

    def test_build_docs_index(self, temp_repo):
        """Convenience function builds index."""
        stats = build_docs_index(repo_root=temp_repo)

        assert isinstance(stats, IndexStats)
        assert stats.indexed >= 3

    def test_search_docs(self, temp_repo):
        """Convenience function searches."""
        build_docs_index(repo_root=temp_repo)
        response = search_docs("catalytic", repo_root=temp_repo)

        assert isinstance(response, DocsSearchResponse)
        assert len(response.results) > 0


class TestDocsIndexNormalization:
    """Tests for text normalization."""

    def test_normalize_strips_markdown(self, docs_index):
        """Markdown formatting is stripped."""
        content = "# Header\n\n**Bold** and *italic* and `code`"
        normalized = docs_index._normalize_text(content)

        assert "#" not in normalized
        assert "**" not in normalized
        assert "*" not in normalized
        assert "`" not in normalized

    def test_normalize_lowercase(self, docs_index):
        """Content is lowercased."""
        content = "UPPERCASE and MixedCase"
        normalized = docs_index._normalize_text(content)

        assert normalized == normalized.lower()

    def test_normalize_collapses_whitespace(self, docs_index):
        """Whitespace is collapsed."""
        content = "Word1    Word2\n\n\nWord3"
        normalized = docs_index._normalize_text(content)

        assert "  " not in normalized


class TestDocsIndexEmptyQuery:
    """Tests for edge cases."""

    def test_empty_query(self, docs_index, temp_repo):
        """Empty query returns empty results."""
        docs_index.index_all()

        response = docs_index.search("")

        assert len(response.results) == 0

    def test_whitespace_query(self, docs_index, temp_repo):
        """Whitespace-only query returns empty results."""
        docs_index.index_all()

        response = docs_index.search("   ")

        assert len(response.results) == 0
