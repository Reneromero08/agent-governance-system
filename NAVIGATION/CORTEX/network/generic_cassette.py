#!/usr/bin/env python3
"""
Generic Cassette - Creates cassettes from JSON configuration.

This allows creating cassettes without writing separate Python files.
Cassettes are defined in the cassettes.json configuration file.
"""

from pathlib import Path
from typing import List, Dict, Optional
import sqlite3
import json

from cassette_protocol import DatabaseCassette


# Current schema version for all cassettes
CASSETTE_SCHEMA_VERSION = "1.5.0"  # Phase 1.5: structure-aware chunking


class GenericCassette(DatabaseCassette):
    """Generic cassette created from JSON configuration.
    
    Supports:
    - SQLite databases with FTS5
    - Simple text search
    - Vector search (if vectors table exists)
    """
    
    def __init__(self, config: Dict, project_root: Path = None):
        """Initialize from configuration dictionary.
        
        Config should contain:
            - id: Cassette ID
            - db_path: Path to SQLite database (relative to project root)
            - name: Human-readable name
            - capabilities: List of capabilities
            - description: Optional description
            - query_template: Optional SQL query template
        """
        cassette_id = config["id"]
        
        # Convert relative path to absolute
        db_path_str = config["db_path"]
        if project_root:
            db_path = project_root / db_path_str
        else:
            # Try to find project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent  # NAVIGATION/CORTEX/network -> project root
            db_path = project_root / db_path_str
        
        super().__init__(db_path, cassette_id)
        
        self.name = config.get("name", cassette_id)
        self.description = config.get("description", "")
        self.capabilities = config.get("capabilities", [])
        self.query_template = config.get("query_template", "")
        self.expected_schema = config.get("schema_version", CASSETTE_SCHEMA_VERSION)
        self.config = config
        self.project_root = project_root
        
    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Query the database using configured query template or auto-detection."""
        if not self.db_path.exists():
            print(f"[{self.cassette_id}] Database not found: {self.db_path}")
            return []
        
        # Use custom query template if provided
        if self.query_template:
            return self._execute_custom_query(query_text, top_k)
        
        # Auto-detect query strategy
        return self._auto_query(query_text, top_k)
    
    def _execute_custom_query(self, query_text: str, top_k: int) -> List[dict]:
        """Execute custom query template."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        try:
            # Replace placeholders in template
            query = self.query_template.replace("{query}", query_text)
            query = query.replace("{limit}", str(top_k))
            
            cursor = conn.execute(query)
            results = []
            
            for row in cursor.fetchall():
                result = dict(row)
                result["source"] = self.cassette_id
                result["score"] = result.get("score", 1.0)
                results.append(result)
            
            return results
        except Exception as e:
            print(f"[{self.cassette_id}] Custom query error: {e}")
            return []
        finally:
            conn.close()
    
    def _auto_query(self, query_text: str, top_k: int) -> List[dict]:
        """Auto-detect query strategy based on database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        try:
            # Get table list
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Try FTS5 first
            fts_tables = [t for t in tables if t.endswith('_fts')]
            if fts_tables:
                return self._query_fts(conn, fts_tables[0], query_text, top_k)
            
            # Try simple text search
            text_tables = [t for t in tables if t in ['documents', 'chunks', 'content', 'indexing_info']]
            if text_tables:
                return self._query_text(conn, text_tables[0], query_text, top_k)
            
            # No suitable tables found
            return []
        finally:
            conn.close()
    
    def _query_fts(self, conn: sqlite3.Connection, fts_table: str, query_text: str, top_k: int) -> List[dict]:
        """Query FTS5 virtual table."""
        # Determine base table name (remove _fts suffix)
        base_table = fts_table[:-4] if fts_table.endswith('_fts') else fts_table

        # Check if this is the standard cassette schema (chunks + chunks_fts + files)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Standard cassette schema: chunks_fts with chunk_id, joined to chunks and files
        if fts_table == "chunks_fts" and "chunks" in tables and "files" in tables:
            query = """
                SELECT
                    c.chunk_id,
                    f.path,
                    c.chunk_hash as hash,
                    snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as content,
                    ? as source,
                    1.0 as score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
                JOIN files f ON c.file_id = f.file_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            cursor = conn.execute(query, (self.cassette_id, query_text, top_k))
        else:
            # Fallback: generic FTS query for non-standard schemas
            cursor_info = conn.execute(f"PRAGMA table_info({base_table})")
            columns = [row[1] for row in cursor_info.fetchall()]

            # Build query
            if 'content' in columns:
                content_col = 'content'
            elif 'normal_indexes_content' in columns:
                content_col = 'normal_indexes_content'
            else:
                content_col = columns[1] if len(columns) > 1 else columns[0]

            query = f"""
                SELECT
                    *,
                    snippet({fts_table}, 0, '<mark>', '</mark>', '...', 32) as snippet,
                    '{self.cassette_id}' as source,
                    1.0 as score
                FROM {fts_table}
                WHERE {fts_table} MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            cursor = conn.execute(query, (query_text, top_k))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            # Clean up the result
            if 'snippet' in result:
                result['content'] = result['snippet']
                del result['snippet']
            results.append(result)

        return results
    
    def _query_text(self, conn: sqlite3.Connection, table: str, query_text: str, top_k: int) -> List[dict]:
        """Query regular table with LIKE search."""
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Find text columns
        text_columns = [c for c in columns if c in ['content', 'text', 'description', 'normal_indexes_content', 'how_to_index_content']]
        if not text_columns:
            text_columns = [c for c in columns if c not in ['id', 'created_at', 'updated_at']]
        
        if not text_columns:
            return []
        
        # Build WHERE clause
        where_clauses = [f"{col} LIKE ?" for col in text_columns]
        where_sql = " OR ".join(where_clauses)
        params = [f"%{query_text}%"] * len(text_columns)
        
        query = f"""
            SELECT 
                *,
                '{self.cassette_id}' as source,
                0.5 as score
            FROM {table}
            WHERE {where_sql}
            LIMIT ?
        """
        params.append(top_k)
        
        cursor = conn.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(row))
        
        return results
    
    def get_stats(self) -> Dict:
        """Return cassette statistics."""
        if not self.db_path.exists():
            return {"error": "Database not found"}
        
        stats = {
            "cassette_id": self.cassette_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "db_path": str(self.db_path),
            "db_exists": True
        }
        
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            # Get table list
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            stats["tables"] = tables
            
            # Count rows in main tables
            for table in tables:
                if table.endswith('_fts') or table in ['sqlite_sequence', 'sqlite_stat1']:
                    continue
                
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[f"{table}_count"] = count
                except:
                    pass
            
            # Check for FTS
            fts_tables = [t for t in tables if t.endswith('_fts')]
            stats["has_fts"] = len(fts_tables) > 0
            
            # Check for vectors
            vector_tables = [t for t in tables if 'vector' in t.lower()]
            stats["has_vectors"] = len(vector_tables) > 0

            # Get schema version
            stats["schema_version"] = self._get_schema_version(conn)
            stats["expected_schema"] = self.expected_schema

        finally:
            conn.close()

        return stats

    # =========================================================================
    # Schema Versioning
    # =========================================================================

    def _get_schema_version(self, conn: sqlite3.Connection = None) -> Optional[str]:
        """Get the schema version from the cassette database."""
        close_conn = False
        if conn is None:
            if not self.db_path.exists():
                return None
            conn = sqlite3.connect(str(self.db_path))
            close_conn = True

        try:
            # Check if cassette_metadata table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cassette_metadata'"
            )
            if not cursor.fetchone():
                return None

            cursor = conn.execute(
                "SELECT value FROM cassette_metadata WHERE key='schema_version'"
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception:
            return None
        finally:
            if close_conn:
                conn.close()

    def _ensure_schema_version(self) -> str:
        """Ensure schema_version is set in the database. Returns current version."""
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(str(self.db_path))
        try:
            # Create cassette_metadata if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cassette_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Check current version
            cursor = conn.execute(
                "SELECT value FROM cassette_metadata WHERE key='schema_version'"
            )
            row = cursor.fetchone()

            if not row:
                # Initialize with expected version for this cassette
                conn.execute(
                    "INSERT INTO cassette_metadata (key, value) VALUES (?, ?)",
                    ("schema_version", self.expected_schema)
                )
                conn.commit()
                return self.expected_schema

            return row[0]
        finally:
            conn.close()

    def validate_schema(self) -> Dict:
        """Validate that cassette schema matches expected version.

        Returns:
            Dict with: valid (bool), current_version, expected_version, message
        """
        current = self._get_schema_version()
        expected = self.expected_schema

        if current is None:
            # Initialize schema version
            self._ensure_schema_version()
            return {
                "valid": True,
                "current_version": self.expected_schema,
                "expected_version": expected,
                "message": "Schema version initialized"
            }

        if current == expected:
            return {
                "valid": True,
                "current_version": current,
                "expected_version": expected,
                "message": "Schema up to date"
            }

        # Version mismatch - compare major.minor
        current_parts = current.split(".")
        expected_parts = expected.split(".")

        if current_parts[:2] == expected_parts[:2]:
            # Same major.minor, patch difference is OK
            return {
                "valid": True,
                "current_version": current,
                "expected_version": expected,
                "message": "Patch version difference (compatible)"
            }

        return {
            "valid": False,
            "current_version": current,
            "expected_version": expected,
            "message": f"Schema mismatch: {current} vs {expected} - migration may be needed"
        }

    # =========================================================================
    # Phase 1.5: Hierarchical Navigation Methods
    # =========================================================================

    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        """Get full chunk information by ID.

        Returns:
            Dict with: chunk_id, file_path, content, token_count,
                      header_depth, header_text, parent_chunk_id, child_count
        """
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute("""
                SELECT
                    c.chunk_id,
                    f.path as file_path,
                    c.chunk_hash,
                    c.token_count,
                    c.start_offset as start_line,
                    c.end_offset as end_line,
                    c.header_depth,
                    c.header_text,
                    c.parent_chunk_id,
                    (SELECT COUNT(*) FROM chunks ch
                     WHERE ch.parent_chunk_id = c.chunk_id) as child_count
                FROM chunks c
                JOIN files f ON c.file_id = f.file_id
                WHERE c.chunk_id = ?
            """, (chunk_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_parent(self, chunk_id: int) -> Optional[Dict]:
        """Get parent chunk (one level up in hierarchy).

        Returns:
            Parent chunk dict or None if no parent
        """
        chunk = self.get_chunk(chunk_id)
        if not chunk or not chunk.get('parent_chunk_id'):
            return None
        return self.get_chunk(chunk['parent_chunk_id'])

    def get_children(self, chunk_id: int) -> List[Dict]:
        """Get direct children of a chunk.

        Returns:
            List of child chunk dicts
        """
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute("""
                SELECT
                    c.chunk_id,
                    f.path as file_path,
                    c.chunk_hash,
                    c.token_count,
                    c.header_depth,
                    c.header_text,
                    c.parent_chunk_id
                FROM chunks c
                JOIN files f ON c.file_id = f.file_id
                WHERE c.parent_chunk_id = ?
                ORDER BY c.chunk_index
            """, (chunk_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_siblings(self, chunk_id: int) -> Dict:
        """Get previous and next siblings at same header depth.

        Returns:
            {prev: chunk_dict or None, next: chunk_dict or None}
        """
        chunk = self.get_chunk(chunk_id)
        if not chunk:
            return {"prev": None, "next": None}

        if not self.db_path.exists():
            return {"prev": None, "next": None}

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Get file_id for this chunk
            cursor = conn.execute(
                "SELECT file_id FROM chunks WHERE chunk_id = ?",
                (chunk_id,)
            )
            row = cursor.fetchone()
            if not row:
                return {"prev": None, "next": None}
            file_id = row[0]

            header_depth = chunk.get('header_depth')
            parent_id = chunk.get('parent_chunk_id')

            # Find previous sibling (same depth, same parent, lower chunk_index)
            cursor = conn.execute("""
                SELECT c.chunk_id, f.path as file_path, c.header_text, c.header_depth
                FROM chunks c
                JOIN files f ON c.file_id = f.file_id
                WHERE c.file_id = ?
                  AND c.header_depth IS NOT NULL
                  AND (c.header_depth = ? OR (? IS NULL AND c.header_depth IS NULL))
                  AND (c.parent_chunk_id = ? OR (? IS NULL AND c.parent_chunk_id IS NULL))
                  AND c.chunk_id < ?
                ORDER BY c.chunk_id DESC
                LIMIT 1
            """, (file_id, header_depth, header_depth, parent_id, parent_id, chunk_id))
            prev_row = cursor.fetchone()

            # Find next sibling
            cursor = conn.execute("""
                SELECT c.chunk_id, f.path as file_path, c.header_text, c.header_depth
                FROM chunks c
                JOIN files f ON c.file_id = f.file_id
                WHERE c.file_id = ?
                  AND c.header_depth IS NOT NULL
                  AND (c.header_depth = ? OR (? IS NULL AND c.header_depth IS NULL))
                  AND (c.parent_chunk_id = ? OR (? IS NULL AND c.parent_chunk_id IS NULL))
                  AND c.chunk_id > ?
                ORDER BY c.chunk_id ASC
                LIMIT 1
            """, (file_id, header_depth, header_depth, parent_id, parent_id, chunk_id))
            next_row = cursor.fetchone()

            return {
                "prev": dict(prev_row) if prev_row else None,
                "next": dict(next_row) if next_row else None
            }
        finally:
            conn.close()

    def get_path(self, chunk_id: int) -> List[Dict]:
        """Get hierarchical path from root to this chunk (breadcrumbs).

        Returns:
            List of ancestor chunks from root to this chunk
            Example: [{"header_text": "# Doc"}, {"header_text": "## Section"}, ...]
        """
        path = []
        current_id = chunk_id

        while current_id is not None:
            chunk = self.get_chunk(current_id)
            if not chunk:
                break
            path.insert(0, {
                "chunk_id": chunk['chunk_id'],
                "header_text": chunk.get('header_text'),
                "header_depth": chunk.get('header_depth')
            })
            current_id = chunk.get('parent_chunk_id')

        return path

    def navigate(self, chunk_id: int, direction: str) -> Optional[Dict]:
        """Navigate from a chunk in a given direction.

        Args:
            chunk_id: Starting chunk ID
            direction: One of 'parent', 'first_child', 'prev', 'next'

        Returns:
            Target chunk or None if navigation not possible
        """
        if direction == 'parent':
            return self.get_parent(chunk_id)
        elif direction == 'first_child':
            children = self.get_children(chunk_id)
            return children[0] if children else None
        elif direction == 'prev':
            return self.get_siblings(chunk_id)['prev']
        elif direction == 'next':
            return self.get_siblings(chunk_id)['next']
        return None


def create_cassette_from_config(config: Dict, project_root: Path = None) -> GenericCassette:
    """Create a GenericCassette from configuration dictionary."""
    return GenericCassette(config, project_root)


def load_cassettes_from_json(config_path: Path, project_root: Path = None) -> List[GenericCassette]:
    """Load all cassettes from JSON configuration file."""
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    cassettes = []
    for cassette_config in config.get("cassettes", []):
        if not cassette_config.get("enabled", True):
            continue
        
        try:
            cassette = create_cassette_from_config(cassette_config, project_root)
            cassettes.append(cassette)
            print(f"[INFO] Loaded cassette: {cassette.cassette_id}")
        except Exception as e:
            print(f"[ERROR] Failed to load cassette {cassette_config.get('id', 'unknown')}: {e}")
    
    return cassettes


if __name__ == "__main__":
    # Test the generic cassette
    config_path = Path(__file__).parent / "cassettes.json"
    cassettes = load_cassettes_from_json(config_path)
    
    print(f"\nLoaded {len(cassettes)} cassettes:")
    for cassette in cassettes:
        stats = cassette.get_stats()
        print(f"  - {cassette.cassette_id}: {stats.get('tables', [])}")