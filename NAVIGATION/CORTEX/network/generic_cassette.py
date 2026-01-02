#!/usr/bin/env python3
"""
Generic Cassette - Creates cassettes from JSON configuration.

This allows creating cassettes without writing separate Python files.
Cassettes are defined in the cassettes.json configuration file.
"""

from pathlib import Path
from typing import List, Dict
import sqlite3
import json

from cassette_protocol import DatabaseCassette


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
        
        # Try to get schema to determine columns
        cursor = conn.execute(f"PRAGMA table_info({base_table})")
        columns = [row[1] for row in cursor.fetchall()]
        
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
            FROM {base_table}
            JOIN {fts_table} ON rowid = {base_table}.rowid
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
            
        finally:
            conn.close()
        
        return stats


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