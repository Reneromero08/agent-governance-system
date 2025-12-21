"""
Simple query interface for the cortex index (SQLite backend).

Functions in this module allow skills and tools to search the cortex by id or
by type, or to retrieve entities that contain a particular path. The SQLite
backend provides O(1) lookups for all primary query patterns.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = Path(__file__).resolve().parent
DB_PATH = CORTEX_DIR / "_generated" / "cortex.db"

_CONN: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Get or create a connection to the cortex database."""
    global _CONN
    if _CONN is None:
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"No cortex database found at {DB_PATH}. Run 'python CORTEX/cortex.build.py' to generate."
            )
        _CONN = sqlite3.connect(DB_PATH)
        _CONN.row_factory = sqlite3.Row  # Enable dict-like access
    return _CONN


def get_metadata(key: str) -> Optional[str]:
    """Retrieve a metadata value by key."""
    conn = _get_conn()
    cursor = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row["value"] if row else None


def get_entity_by_id(entity_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an entity by its unique ID."""
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT id, type, title, source_path FROM entities WHERE id = ?",
        (entity_id,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    
    # Fetch tags
    tag_cursor = conn.execute("SELECT tag FROM tags WHERE entity_id = ?", (entity_id,))
    tags = [r["tag"] for r in tag_cursor.fetchall()]
    
    return {
        "id": row["id"],
        "type": row["type"],
        "title": row["title"],
        "tags": tags,
        "paths": {"source": row["source_path"]}
    }


def find_entities_by_type(entity_type: str) -> List[Dict[str, Any]]:
    """Find all entities of a given type."""
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT id FROM entities WHERE type = ?",
        (entity_type,)
    )
    return [get_entity_by_id(row["id"]) for row in cursor.fetchall()]


def find_entities_containing_path(path_substring: str) -> List[Dict[str, Any]]:
    """Find all entities whose source path contains the given substring."""
    conn = _get_conn()
    cursor = conn.execute(
        "SELECT id FROM entities WHERE source_path LIKE ?",
        (f"%{path_substring}%",)
    )
    return [get_entity_by_id(row["id"]) for row in cursor.fetchall()]


def export_to_json() -> Dict[str, Any]:
    """Export the entire cortex to a JSON-compatible dictionary."""
    conn = _get_conn()
    
    # Get metadata
    meta_cursor = conn.execute("SELECT key, value FROM metadata")
    metadata = {row["key"]: row["value"] for row in meta_cursor.fetchall()}
    
    # Get all entities
    entity_cursor = conn.execute("SELECT id FROM entities")
    entities = [get_entity_by_id(row["id"]) for row in entity_cursor.fetchall()]
    
    return {
        "cortex_version": metadata.get("cortex_version", "unknown"),
        "canon_version": metadata.get("canon_version", "unknown"),
        "generated_at": metadata.get("generated_at", "unknown"),
        "entities": entities
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query the AGS Cortex (SQLite)")
    parser.add_argument("--id", help="Find entity by ID")
    parser.add_argument("--type", help="Find entities by type")
    parser.add_argument("--find", help="Find entities containing path substring")
    parser.add_argument("--list", action="store_true", help="List all entities (summary)")
    parser.add_argument("--json", action="store_true", help="Export entire cortex as JSON")
    
    args = parser.parse_args()
    
    try:
        conn = _get_conn()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.json:
        print(json.dumps(export_to_json(), indent=2))
    elif args.id:
        result = get_entity_by_id(args.id)
        print(json.dumps(result, indent=2) if result else "Not found.")
    elif args.type:
        results = find_entities_by_type(args.type)
        print(json.dumps(results, indent=2))
    elif args.find:
        results = find_entities_containing_path(args.find)
        print(json.dumps(results, indent=2))
    elif args.list:
        cursor = conn.execute("SELECT id, type, title FROM entities")
        rows = cursor.fetchall()
        print(f"Cortex contains {len(rows)} indexed entities:")
        for row in rows:
            print(f"- {row['id']} ({row['type']}) : {row['title']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


__all__ = [
    "get_entity_by_id",
    "find_entities_by_type",
    "find_entities_containing_path",
    "get_metadata",
    "export_to_json",
]
