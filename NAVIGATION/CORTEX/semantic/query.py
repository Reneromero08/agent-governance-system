#!/usr/bin/env python3
"""
Cortex Query Engine (Lane C3)

Provides semantic and keyword search over the cassette network.

Note: system1.db and cortex.db are deprecated. All queries now route
through the cassette network (NAVIGATION/CORTEX/cassettes/).
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional

# Configuration - cassette network paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASSETTES_DIR = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"
CANON_CASSETTE = CASSETTES_DIR / "canon.db"


def export_to_json() -> Dict:
    """
    Export cassette network summary to JSON.

    Note: cortex.db is deprecated. Returns metadata from cassette network.
    """
    result = {
        "entities": [],
        "metadata": {"note": "cortex.db deprecated - use cassette network"}
    }

    # Count chunks across cassettes
    total_chunks = 0
    for db_file in CASSETTES_DIR.glob("*.db"):
        try:
            with sqlite3.connect(str(db_file)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
                total_chunks += cursor.fetchone()[0]
        except Exception:
            pass

    result["metadata"]["total_chunks"] = total_chunks
    return result


class CortexQuery:
    """Query interface for cassette network.

    Provides FTS5 search across cassette databases.
    """

    def __init__(self, db_path: Path = None):
        # Default to canon cassette if no path specified
        self.db_path = db_path or CANON_CASSETTE
        self.cassettes_dir = CASSETTES_DIR

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Full-text search across cassette chunks."""
        results = []

        # Search all cassettes
        for db_file in self.cassettes_dir.glob("*.db"):
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT
                            f.path,
                            c.chunk_index,
                            snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as snippet,
                            rank
                        FROM chunks_fts
                        JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                        JOIN files f ON c.file_id = f.file_id
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """, (query, limit))

                    for row in cursor.fetchall():
                        result = dict(row)
                        result["cassette"] = db_file.stem
                        results.append(result)
            except Exception:
                pass

        # Sort by rank and limit
        results.sort(key=lambda x: x.get("rank", 0))
        return results[:limit]

    def get_summary(self, path: str) -> Optional[str]:
        """Retrieve summary for a file from cassettes."""
        for db_file in self.cassettes_dir.glob("*.db"):
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT summary FROM files WHERE path = ?",
                        (path,)
                    )
                    row = cursor.fetchone()
                    if row and row['summary']:
                        return row['summary']
            except Exception:
                pass
        return None

    def find_sections(self, query: str, limit: int = 5) -> List[Dict]:
        """Find specific sections matching query."""
        return self.search(query, limit)

    def get_neighbors(self, path: str) -> List[str]:
        """Find related files (placeholder for embedding search)."""
        return []

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value.

        Note: cortex.db metadata is deprecated. Returns None for backward compat.
        """
        # Metadata was stored in deprecated cortex.db
        # Return None - callers should handle gracefully
        return None

    def find_entities_containing_path(self, path_query: str) -> List[Dict]:
        """Find files where path contains substring."""
        results = []

        for db_file in self.cassettes_dir.glob("*.db"):
            try:
                with sqlite3.connect(str(db_file)) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT path FROM files WHERE path LIKE ?",
                        (f"%{path_query}%",)
                    )
                    for row in cursor.fetchall():
                        results.append({
                            "path": row["path"],
                            "paths": {"source": row["path"]},
                            "cassette": db_file.stem
                        })
            except Exception:
                pass

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query cassette network")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--summary", action="store_true", help="Get summary for top result")

    args = parser.parse_args()
    cq = CortexQuery()

    results = cq.search(args.query, args.limit)

    print(f"Results for '{args.query}':")
    for r in results:
        print(f"  [{r.get('cassette', '?')}] {r['path']}: {r['snippet']}")

    if args.summary and results:
        top_path = results[0]['path']
        summary = cq.get_summary(top_path)
        if summary:
            print(f"\nSummary for {top_path}:\n{summary}")


def get_metadata(key: str) -> Optional[str]:
    """Module-level wrapper for get_metadata."""
    return CortexQuery().get_metadata(key)


def find_entities_containing_path(path_query: str) -> List[Dict]:
    """Module-level wrapper for find_entities_containing_path."""
    return CortexQuery().find_entities_containing_path(path_query)


if __name__ == "__main__":
    main()
