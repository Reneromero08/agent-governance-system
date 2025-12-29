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
CORTEX_DB_PATH = Path(__file__).resolve().parent / "_generated" / "cortex.db"


def export_to_json() -> Dict:
    """
    Export the cortex index to a JSON-serializable dictionary.
    
    Called by cortex.build.py to create the cortex.json snapshot file.
    Returns all entities from the database in a structured format.
    """
    db_path = CORTEX_DB_PATH
    if not db_path.exists():
        return {"entities": [], "metadata": {"error": "cortex.db not found"}}
    
    result = {
        "entities": [],
        "metadata": {}
    }
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Export entities table
            cursor = conn.execute("""
                SELECT id, type, path, title, tags, summary, last_modified, content_hash
                FROM entities
                ORDER BY path
            """)
            for row in cursor.fetchall():
                entity = {
                    "id": row["id"],
                    "type": row["type"],
                    "path": row["path"],
                    "title": row["title"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "summary": row["summary"],
                    "last_modified": row["last_modified"],
                    "content_hash": row["content_hash"]
                }
                result["entities"].append(entity)
            
            # Export metadata table
            meta_cursor = conn.execute("SELECT key, value FROM metadata")
            for row in meta_cursor.fetchall():
                try:
                    result["metadata"][row["key"]] = json.loads(row["value"])
                except (json.JSONDecodeError, TypeError):
                    result["metadata"][row["key"]] = row["value"]
                    
    except sqlite3.OperationalError as e:
        result["metadata"]["error"] = str(e)
    
    return result

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

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value from cortex.db."""
        path = CORTEX_DB_PATH
        if not path.exists():
            return None
        try:
            with sqlite3.connect(str(path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    val = row['value']
                    # Some values are JSON strings, some are plain.
                    # The caller typically expects the raw value or parsed? 
                    # Looking at usage: cortex_query.get_metadata("canon_version")
                    # In schema.sql, canon_version is a string.
                    # JSON decoding might be risky if it's just a string like "2.0.0" (which is valid JSON string "2.0.0"?)
                    # Let's try to decode, fallback to raw.
                    try:
                        return json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        return val
                return None
        except Exception:
            return None

    def find_entities_containing_path(self, path_query: str) -> List[Dict]:
        """Find entities where path contains substring (cortex.db)."""
        path = CORTEX_DB_PATH
        if not path.exists():
            return []
        try:
            with sqlite3.connect(str(path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM entities WHERE path LIKE ?", (f"%{path_query}%",))
                results = []
                for row in cursor.fetchall():
                    res = dict(row)
                    res['paths'] = {'source': row['path']}
                    results.append(res)
                return results
        except Exception:
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

def get_metadata(key: str) -> Optional[str]:
    """Module-level wrapper for get_metadata."""
    return CortexQuery().get_metadata(key)

def find_entities_containing_path(path_query: str) -> List[Dict]:
    """Module-level wrapper for find_entities_containing_path."""
    return CortexQuery().find_entities_containing_path(path_query)

if __name__ == "__main__":
    main()
