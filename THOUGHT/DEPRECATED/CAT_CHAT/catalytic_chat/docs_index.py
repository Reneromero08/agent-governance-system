#!/usr/bin/env python3
"""
Docs Index (Phase F)

Full-text search index for repository documentation.
Provides fast, bounded discovery via SQLite FTS5.

Retrieval Order Position: 3b (after symbol registry, before CAS)
1. SPC -> 2. Cassette FTS -> 3a. Symbol Registry -> 3b. DocsIndex (THIS) -> 4. CAS -> 5. Vector

Exit Criteria:
- `docs search --query "..." --limit N` returns bounded, deterministic results
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

from catalytic_chat.paths import get_cat_chat_db, get_sqlite_connection


@dataclass
class DocsSearchResult:
    """Single search result."""
    file_path: str
    file_sha256: str
    snippet: str           # Max 200 chars
    match_score: float     # BM25 score (lower is better in FTS5)
    chunk_index: int


@dataclass
class DocsSearchResponse:
    """Search response with metadata."""
    query: str
    results: List[DocsSearchResult]
    total_matches: int     # Before limit applied
    limit_applied: int
    search_time_ms: float


@dataclass
class IndexStats:
    """Statistics from index operation."""
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    files_removed: int = 0


class DocsIndexError(Exception):
    """Raised when docs index operations fail."""
    pass


class DocsIndex:
    """
    Full-text search index for repository documentation.

    Uses SQLite FTS5 with Porter stemming for fast keyword search.
    All results are deterministic (same query = same order).
    """

    # Patterns for indexable documentation files
    INDEXABLE_PATTERNS = [
        "LAW/CANON/**/*.md",
        "LAW/CONTEXT/**/*.md",
        "SKILLS/**/*.md",
        "TOOLS/**/*.md",
        "THOUGHT/LAB/CAT_CHAT/docs/*.md",
        "*.md",  # Root level markdown
    ]

    # Limits
    MAX_CHUNK_SIZE = 50_000      # 50KB per chunk for FTS efficiency
    MAX_SNIPPET_LENGTH = 200     # Max chars in snippet
    DEFAULT_LIMIT = 10

    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize DocsIndex.

        Args:
            repo_root: Repository root path. Defaults to cwd.
        """
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = repo_root
        self.db_path = get_cat_chat_db(repo_root)
        self._conn = None
        self._ensure_schema()

    def _get_conn(self):
        """Get or create database connection."""
        if self._conn is None:
            self._conn = get_sqlite_connection(self.db_path)
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()

        # File metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS docs_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_files_sha256 ON docs_files(sha256)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_files_path ON docs_files(path)")

        # Content table (normalized text, chunked)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS docs_content (
                content_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL REFERENCES docs_files(file_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                content_normalized TEXT NOT NULL,
                UNIQUE(file_id, chunk_index)
            )
        """)

        # FTS5 virtual table
        # Check if FTS table exists first (can't use IF NOT EXISTS with virtual tables)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs_content_fts'"
        )
        if cursor.fetchone() is None:
            conn.execute("""
                CREATE VIRTUAL TABLE docs_content_fts USING fts5(
                    content_normalized,
                    content='docs_content',
                    content_rowid='content_id',
                    tokenize='porter unicode61'
                )
            """)

            # Triggers to keep FTS in sync
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_content_ai AFTER INSERT ON docs_content BEGIN
                    INSERT INTO docs_content_fts(rowid, content_normalized)
                    VALUES (new.content_id, new.content_normalized);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_content_ad AFTER DELETE ON docs_content BEGIN
                    INSERT INTO docs_content_fts(docs_content_fts, rowid, content_normalized)
                    VALUES('delete', old.content_id, old.content_normalized);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS docs_content_au AFTER UPDATE ON docs_content BEGIN
                    INSERT INTO docs_content_fts(docs_content_fts, rowid, content_normalized)
                    VALUES('delete', old.content_id, old.content_normalized);
                    INSERT INTO docs_content_fts(rowid, content_normalized)
                    VALUES (new.content_id, new.content_normalized);
                END
            """)

        conn.commit()

    def _compute_sha256(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _normalize_text(self, content: str) -> str:
        """
        Normalize text for FTS indexing.

        - Lowercase
        - Strip markdown formatting (headers, links, code blocks)
        - Collapse whitespace
        """
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', ' ', content)
        content = re.sub(r'`[^`]+`', ' ', content)

        # Remove markdown links but keep text: [text](url) -> text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

        # Remove markdown headers
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)

        # Remove markdown emphasis
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\*([^*]+)\*', r'\1', content)
        content = re.sub(r'__([^_]+)__', r'\1', content)
        content = re.sub(r'_([^_]+)_', r'\1', content)

        # Collapse whitespace
        content = re.sub(r'\s+', ' ', content)

        # Lowercase
        content = content.lower().strip()

        return content

    def _chunk_content(self, content: str) -> List[str]:
        """
        Split content into chunks for FTS.

        Splits at paragraph boundaries when possible.
        """
        if len(content) <= self.MAX_CHUNK_SIZE:
            return [content]

        chunks = []
        current_chunk = ""

        # Split by paragraphs (double newline)
        paragraphs = content.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.MAX_CHUNK_SIZE:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single paragraph exceeds chunk size, split it
                if len(para) > self.MAX_CHUNK_SIZE:
                    # Split at sentence boundaries
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.MAX_CHUNK_SIZE:
                            if current_chunk:
                                current_chunk += " "
                            current_chunk += sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent[:self.MAX_CHUNK_SIZE]
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [content[:self.MAX_CHUNK_SIZE]]

    def _get_indexable_files(self) -> List[Path]:
        """Get list of all files matching indexable patterns."""
        files = []

        for pattern in self.INDEXABLE_PATTERNS:
            if "**" in pattern:
                # Recursive glob
                parts = pattern.split("**")
                base = self.repo_root / parts[0].rstrip("/")
                suffix = parts[1].lstrip("/")
                if base.exists():
                    for f in base.rglob(suffix):
                        if f.is_file():
                            files.append(f)
            else:
                # Simple glob
                for f in self.repo_root.glob(pattern):
                    if f.is_file():
                        files.append(f)

        # Deduplicate and sort for determinism
        return sorted(set(files))

    def index_file(self, file_path: Path) -> Optional[str]:
        """
        Index a single file.

        Args:
            file_path: Path to file

        Returns:
            SHA-256 hash of file, or None if skipped (unchanged)
        """
        conn = self._get_conn()

        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise DocsIndexError(f"Failed to read {file_path}: {e}")

        file_sha256 = self._compute_sha256(content)
        size_bytes = len(content.encode('utf-8'))
        rel_path = file_path.relative_to(self.repo_root).as_posix()

        # Check if file already indexed with same hash
        cursor = conn.execute(
            "SELECT file_id, sha256 FROM docs_files WHERE path = ?",
            (rel_path,)
        )
        existing = cursor.fetchone()

        if existing and existing['sha256'] == file_sha256:
            # File unchanged, skip
            return None

        # Delete old entry if exists (triggers will clean up FTS)
        if existing:
            conn.execute("DELETE FROM docs_files WHERE file_id = ?", (existing['file_id'],))

        # Insert file metadata
        cursor = conn.execute(
            """
            INSERT INTO docs_files (path, sha256, size_bytes, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            (rel_path, file_sha256, size_bytes, self._get_timestamp())
        )
        file_id = cursor.lastrowid

        # Normalize and chunk content
        normalized = self._normalize_text(content)
        chunks = self._chunk_content(normalized)

        # Insert chunks
        for i, chunk in enumerate(chunks):
            conn.execute(
                """
                INSERT INTO docs_content (file_id, chunk_index, content_normalized)
                VALUES (?, ?, ?)
                """,
                (file_id, i, chunk)
            )

        conn.commit()
        return file_sha256

    def index_all(self, incremental: bool = True) -> IndexStats:
        """
        Index all indexable files.

        Args:
            incremental: If True, skip unchanged files

        Returns:
            IndexStats with counts
        """
        stats = IndexStats()
        conn = self._get_conn()

        files = self._get_indexable_files()
        indexed_paths = set()

        for file_path in files:
            rel_path = file_path.relative_to(self.repo_root).as_posix()
            indexed_paths.add(rel_path)

            try:
                result = self.index_file(file_path)
                if result is None:
                    stats.skipped += 1
                else:
                    stats.indexed += 1
            except DocsIndexError:
                stats.errors += 1

        # Remove files that no longer exist
        cursor = conn.execute("SELECT file_id, path FROM docs_files")
        for row in cursor.fetchall():
            if row['path'] not in indexed_paths:
                conn.execute("DELETE FROM docs_files WHERE file_id = ?", (row['file_id'],))
                stats.files_removed += 1

        conn.commit()
        return stats

    def search(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> DocsSearchResponse:
        """
        Search documentation with bounded, deterministic results.

        Args:
            query: Search query (FTS5 syntax supported)
            limit: Maximum results to return
            offset: Skip first N results

        Returns:
            DocsSearchResponse with results
        """
        start_time = time.time()
        conn = self._get_conn()

        # Escape special FTS5 characters for safety
        # But preserve * for prefix matching and " for phrases
        safe_query = query.strip()
        if not safe_query:
            return DocsSearchResponse(
                query=query,
                results=[],
                total_matches=0,
                limit_applied=limit,
                search_time_ms=0.0
            )

        # Count total matches first
        try:
            count_cursor = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM docs_content_fts
                WHERE docs_content_fts MATCH ?
                """,
                (safe_query,)
            )
            total_matches = count_cursor.fetchone()['cnt']
        except Exception:
            # Query syntax error, return empty
            return DocsSearchResponse(
                query=query,
                results=[],
                total_matches=0,
                limit_applied=limit,
                search_time_ms=(time.time() - start_time) * 1000
            )

        # Execute search with deterministic ordering
        # BM25: lower is better, so we sort ascending
        # Tie-breakers: path ASC, chunk_index ASC
        try:
            cursor = conn.execute(
                """
                SELECT
                    df.path,
                    df.sha256,
                    snippet(docs_content_fts, 0, '>>>', '<<<', '...', 30) as snippet,
                    bm25(docs_content_fts) as score,
                    dc.chunk_index
                FROM docs_content_fts
                JOIN docs_content dc ON docs_content_fts.rowid = dc.content_id
                JOIN docs_files df ON dc.file_id = df.file_id
                WHERE docs_content_fts MATCH ?
                ORDER BY
                    bm25(docs_content_fts) ASC,
                    df.path ASC,
                    dc.chunk_index ASC
                LIMIT ? OFFSET ?
                """,
                (safe_query, limit, offset)
            )

            results = []
            for row in cursor.fetchall():
                # Truncate snippet if needed
                snippet = row['snippet'] or ""
                if len(snippet) > self.MAX_SNIPPET_LENGTH:
                    snippet = snippet[:self.MAX_SNIPPET_LENGTH - 3] + "..."

                results.append(DocsSearchResult(
                    file_path=row['path'],
                    file_sha256=row['sha256'],
                    snippet=snippet,
                    match_score=row['score'],
                    chunk_index=row['chunk_index']
                ))

        except Exception:
            results = []

        elapsed_ms = (time.time() - start_time) * 1000

        return DocsSearchResponse(
            query=query,
            results=results,
            total_matches=total_matches,
            limit_applied=limit,
            search_time_ms=elapsed_ms
        )

    def get_file_content(self, file_sha256: str) -> Optional[str]:
        """
        Get full file content by SHA-256 hash.

        Args:
            file_sha256: SHA-256 hash of file

        Returns:
            File content or None if not found
        """
        conn = self._get_conn()

        cursor = conn.execute(
            "SELECT path FROM docs_files WHERE sha256 = ?",
            (file_sha256,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        file_path = self.repo_root / row['path']
        if not file_path.exists():
            return None

        try:
            return file_path.read_text(encoding='utf-8')
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        conn = self._get_conn()

        files_count = conn.execute("SELECT COUNT(*) as cnt FROM docs_files").fetchone()['cnt']
        chunks_count = conn.execute("SELECT COUNT(*) as cnt FROM docs_content").fetchone()['cnt']
        total_size = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) as total FROM docs_files").fetchone()['total']

        return {
            "files": files_count,
            "chunks": chunks_count,
            "total_bytes": total_size,
            "db_path": str(self.db_path)
        }

    def _get_timestamp(self) -> str:
        """Get ISO8601 timestamp."""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Convenience functions

def build_docs_index(repo_root: Optional[Path] = None, incremental: bool = True) -> IndexStats:
    """
    Build documentation index.

    Args:
        repo_root: Repository root path
        incremental: If True, skip unchanged files

    Returns:
        IndexStats with counts
    """
    idx = DocsIndex(repo_root)
    try:
        return idx.index_all(incremental)
    finally:
        idx.close()


def search_docs(
    query: str,
    limit: int = 10,
    repo_root: Optional[Path] = None
) -> DocsSearchResponse:
    """
    Search documentation.

    Args:
        query: Search query
        limit: Max results
        repo_root: Repository root path

    Returns:
        DocsSearchResponse with results
    """
    idx = DocsIndex(repo_root)
    try:
        return idx.search(query, limit)
    finally:
        idx.close()
