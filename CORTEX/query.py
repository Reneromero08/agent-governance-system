#!/usr/bin/env python3
"""
Cortex Query Engine (Lane C3)

Provides semantic and keyword search over the System 1 Database.
Supports:
- Full-text search (FTS5)
- Section-level retrieval
- Summary retrieval
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
DB_PATH = Path("CORTEX/system1.db")

class CortexQuery:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Full-text search across all chunks."""
        with sqlite3.connect(str(self.db_path)) as conn:
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
            
            return [dict(row) for row in cursor.fetchall()]

    def get_summary(self, path: str) -> Optional[str]:
        """Retrieve summary for a file."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT summary FROM files WHERE path = ?",
                (path,)
            )
            row = cursor.fetchone()
            return row['summary'] if row else None

    def find_sections(self, query: str, limit: int = 5) -> List[Dict]:
        """Find specific sections matching query."""
        # Note: This requires section-level indexing in DB, currently we search chunks.
        # We can approximate by grouping chunks by file and returning valid anchors.
        return self.search(query, limit)

    def get_neighbors(self, path: str) -> List[str]:
        """Find related files (placeholder for embedding search)."""
        return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query Cortex System 1")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--summary", action="store_true", help="Get summary for top result")
    
    args = parser.parse_args()
    cq = CortexQuery()
    
    results = cq.search(args.query, args.limit)
    
    print(f"Results for '{args.query}':")
    for r in results:
        print(f"  {r['path']}: {r['snippet']}")
        
    if args.summary and results:
        top_path = results[0]['path']
        summary = cq.get_summary(top_path)
        if summary:
            print(f"\nSummary for {top_path}:\n{summary}")

if __name__ == "__main__":
    main()
