#!/usr/bin/env python3
"""
Governance Cassette - Cassette for governance documents.

Wraps existing system1.db (CANON, ADRs, SKILLS, MAPS).
"""

from pathlib import Path
from typing import List, Dict
import sqlite3

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cassette_protocol import DatabaseCassette


class GovernanceCassette(DatabaseCassette):
    """Cassette for governance documents (CANON, ADRs, SKILLS, MAPS)."""

    def __init__(self):
        db_path = Path("NAVIGATION/CORTEX/db/system1.db")
        super().__init__(
            db_path=db_path,
            cassette_id="governance"
        )
        self.capabilities = ["vectors", "fts", "semantic_search"]

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Query governance docs with semantic search.

        Args:
            query_text: Search query
            top_k: Maximum results

        Returns:
            List of result dictionaries
        """
        if not self.db_path.exists():
            print(f"[GOVERNANCE] Database not found: {self.db_path}")
            return []

        results = []
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            if 'chunks_fts' in self._get_tables(conn):
                cursor = conn.execute("""
                    SELECT
                        c.chunk_id,
                        f.path,
                        c.chunk_hash,
                        snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as content,
                        'governance' as source
                    FROM chunks c
                    JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                    JOIN files f ON c.file_id = f.file_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query_text, top_k))

                for row in cursor.fetchall():
                    results.append({
                        "chunk_id": row['chunk_id'],
                        "path": row['path'],
                        "hash": row['chunk_hash'],
                        "content": row['content'],
                        "source": row['source'],
                        "score": 1.0
                    })
        finally:
            conn.close()

        return results

    def get_stats(self) -> Dict:
        """Return cassette statistics."""
        if not self.db_path.exists():
            return {"error": "Database not found"}

        stats = {}
        conn = sqlite3.connect(str(self.db_path))

        try:
            cursor = conn.execute("SELECT COUNT(*) as total_chunks FROM chunks")
            stats["total_chunks"] = cursor.fetchone()[0]

            if 'section_vectors' in self._get_tables(conn):
                cursor = conn.execute("SELECT COUNT(*) as vectors FROM section_vectors")
                stats["with_vectors"] = cursor.fetchone()[0]
            else:
                stats["with_vectors"] = 0

            if 'files' in self._get_tables(conn):
                cursor = conn.execute("SELECT COUNT(*) as files FROM files")
                stats["files"] = cursor.fetchone()[0]

            stats["content_types"] = ["CANON", "ADRs", "SKILLS", "MAPS"]
        finally:
            conn.close()

        return stats

    def _get_tables(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of tables in database."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]