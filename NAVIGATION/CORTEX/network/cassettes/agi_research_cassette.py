#!/usr/bin/env python3
"""
AGI Research Cassette - Cassette for AGI research database.

Wraps external AGI/CONTEXT/research/_generated/system1.db.
"""

from pathlib import Path
from typing import List, Dict
import sqlite3

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cassette_protocol import DatabaseCassette


class AResearchCassette(DatabaseCassette):
    """Cassette for AGI research papers and experiments."""

    def __init__(self):
        db_path = Path("D:/CCC 2.0/AI/AGI/CONTEXT/research/_generated/system1.db")
        super().__init__(
            db_path=db_path,
            cassette_id="agi-research"
        )
        self.capabilities = ["research", "fts"]

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Query research docs with FTS.

        Args:
            query_text: Search query
            top_k: Maximum results

        Returns:
            List of result dictionaries
        """
        if not self.db_path.exists():
            print(f"[AGI-RESEARCH] Database not found: {self.db_path}")
            return []

        results = []
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            if 'research_chunks' in self._get_tables(conn):
                cursor = conn.execute("""
                    SELECT
                        rc.chunk_id,
                        rc.heading,
                        rc.content,
                        'research' as source
                    FROM research_chunks rc
                    WHERE rc.content LIKE ?
                    LIMIT ?
                """, (f"%{query_text}%", top_k))

                for row in cursor.fetchall():
                    results.append({
                        "chunk_id": row['chunk_id'],
                        "heading": row['heading'],
                        "content": row['content'][:300] if row['content'] else "",
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
            if 'research_chunks' in self._get_tables(conn):
                cursor = conn.execute("SELECT COUNT(*) as total_chunks FROM research_chunks")
                stats["total_chunks"] = cursor.fetchone()[0]

            if 'research_docs' in self._get_tables(conn):
                cursor = conn.execute("SELECT COUNT(*) as total_docs FROM research_docs")
                stats["total_docs"] = cursor.fetchone()[0]

            stats["content_types"] = ["Papers", "Experiments", "Theory"]
        finally:
            conn.close()

        return stats

    def _get_tables(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of tables in database."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]