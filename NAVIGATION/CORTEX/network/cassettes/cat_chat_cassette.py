#!/usr/bin/env python3
"""
CAT_CHAT Cassette - Cassette for Catalytic Chat documentation and indexing information.

Wraps cat_chat_index.db with full-text search and vector capabilities.
"""

from pathlib import Path
from typing import List, Dict
import sqlite3
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cassette_protocol import DatabaseCassette


class CatChatCassette(DatabaseCassette):
    """Cassette for CAT_CHAT documentation and indexing information."""

    def __init__(self):
        db_path = Path("THOUGHT/LAB/CAT_CHAT/cat_chat_index.db")
        super().__init__(
            db_path=db_path,
            cassette_id="cat_chat"
        )
        self.capabilities = ["vectors", "fts", "semantic_search", "indexing_info"]

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Query CAT_CHAT docs with semantic search.

        Args:
            query_text: Search query
            top_k: Maximum results

        Returns:
            List of result dictionaries
        """
        if not self.db_path.exists():
            print(f"[CAT_CHAT] Database not found: {self.db_path}")
            return []

        results = []
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        try:
            # First try FTS5 if available
            tables = self._get_tables(conn)
            
            if 'chunks_fts' in tables:
                cursor = conn.execute("""
                    SELECT
                        c.chunk_id,
                        f.path,
                        c.chunk_hash,
                        snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as content,
                        'cat_chat' as source
                    FROM chunks c
                    JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
                    JOIN files f ON c.file_id = f.file_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query_text, top_k))
            elif 'documents' in tables:
                # Fallback to simple text search
                cursor = conn.execute("""
                    SELECT 
                        id,
                        path,
                        content,
                        'cat_chat' as source
                    FROM documents 
                    WHERE content LIKE ?
                    LIMIT ?
                """, (f'%{query_text}%', top_k))
            else:
                # No searchable tables
                return []

            for row in cursor.fetchall():
                results.append({
                    "chunk_id": row.get('chunk_id', row.get('id', '')),
                    "path": row.get('path', ''),
                    "hash": row.get('chunk_hash', ''),
                    "content": row.get('content', ''),
                    "source": row.get('source', 'cat_chat'),
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
            tables = self._get_tables(conn)
            
            # Count documents/chunks
            if 'chunks' in tables:
                cursor = conn.execute("SELECT COUNT(*) as total_chunks FROM chunks")
                stats["total_chunks"] = cursor.fetchone()[0]
            elif 'documents' in tables:
                cursor = conn.execute("SELECT COUNT(*) as total_documents FROM documents")
                stats["total_documents"] = cursor.fetchone()[0]
            
            # Check for vectors
            if 'section_vectors' in tables:
                cursor = conn.execute("SELECT COUNT(*) as vectors FROM section_vectors")
                stats["with_vectors"] = cursor.fetchone()[0]
            else:
                stats["with_vectors"] = 0
            
            # Check for FTS
            if 'chunks_fts' in tables:
                stats["has_fts"] = True
            else:
                stats["has_fts"] = False
            
            stats["content_types"] = ["CAT_CHAT", "indexing_info", "merge_analysis"]
            stats["capabilities"] = self.capabilities
        finally:
            conn.close()

        return stats

    def _get_tables(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of tables in database."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def add_indexing_info(self, indexing_table: str):
        """Add the indexing information table to the database.
        
        This adds the database indexing information from the user's prompt
        so it can be queried without wasting tokens.
        """
        if not self.db_path.exists():
            print(f"[CAT_CHAT] Database not found, creating: {self.db_path}")
            # Use GuardedWriter for firewall enforcement
            try:
                from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
                repo_root = Path(__file__).resolve().parents[4] # NAVIGATION/CORTEX/network/cassettes/file -> 4 levels up? No.
                # Project root:
                # file: D:\CCC 2.0\AI\agent-governance-system\NAVIGATION\CORTEX\network\cassettes\cat_chat_cassette.py
                # parents[0]: cassettes
                # parents[1]: network
                # parents[2]: CORTEX
                # parents[3]: NAVIGATION
                # parents[4]: agent-governance-system (REPO ROOT)
                repo_root = Path(__file__).resolve().parents[4]
                
                writer = GuardedWriter(
                    project_root=repo_root,
                    durable_roots=["THOUGHT/LAB/CAT_CHAT"]
                )
                # We need to open commit gate if we are writing durable.
                # However, THOUGHT might be considered ephemeral or durable? DB suggests durable.
                writer.open_commit_gate() 
                writer.mkdir_durable("THOUGHT/LAB/CAT_CHAT")
            except ImportError:
                # If we can't import GuardedWriter, we can't comply. 
                # Fail closed or fallback? strict means fail.
                raise ImportError("GuardedWriter not available for firewall enforcement")
            except Exception as e:
                 print(f"Firewall violation or error: {e}")
                 raise
        
        conn = sqlite3.connect(str(self.db_path))
        
        try:
            # Create indexing_info table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indexing_info (
                    id INTEGER PRIMARY KEY,
                    storage_type TEXT NOT NULL,
                    column_type TEXT NOT NULL,
                    normal_indexes_content TEXT NOT NULL,
                    how_to_index_content TEXT NOT NULL,
                    example TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Clear existing data
            conn.execute("DELETE FROM indexing_info")
            
            # Add the indexing information from the user's prompt
            indexing_data = [
                ("File path or URL", "VARCHAR/TEXT", "No – only indexes the path string", 
                 "Not applicable (you'd need full-text search on metadata)", 
                 "Store path in TEXT column, create B-tree index"),
                ("File binary content (BLOB)", "BLOB / BYTEA", "No – most databases don't index BLOB contents by default", 
                 "Use special full-text indexing extensions or external tools", 
                 "PostgreSQL: tsvector + GIN index, pg_trgm for trigram search"),
                ("Text you extracted from file", "TEXT", "Yes – normal / full-text indexes work", 
                 "Extract text first (e.g., with Tika, pdfplumber, pytesseract…), then store & index the text", 
                 "MySQL: Full-text index on TEXT columns"),
                ("JSON / structured metadata", "JSONB (PostgreSQL)", "Yes – can create GIN indexes on JSON", 
                 "Store metadata as JSON → index keys/paths you care about", 
                 "PostgreSQL: GIN index on JSONB column")
            ]
            
            for storage_type, column_type, normal_index, how_to_index, example in indexing_data:
                conn.execute("""
                    INSERT INTO indexing_info (storage_type, column_type, normal_indexes_content, how_to_index_content, example)
                    VALUES (?, ?, ?, ?, ?)
                """, (storage_type, column_type, normal_index, how_to_index, example))
            
            # Create FTS5 virtual table for the indexing info
            conn.execute("DROP TABLE IF EXISTS indexing_info_fts")
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS indexing_info_fts 
                USING fts5(storage_type, column_type, normal_indexes_content, how_to_index_content, example)
            """)
            
            # Populate FTS table
            conn.execute("""
                INSERT INTO indexing_info_fts (storage_type, column_type, normal_indexes_content, how_to_index_content, example)
                SELECT storage_type, column_type, normal_indexes_content, how_to_index_content, example 
                FROM indexing_info
            """)
            
            conn.commit()
            print(f"[CAT_CHAT] Added indexing information to database")
            
        except Exception as e:
            print(f"[CAT_CHAT] Error adding indexing info: {e}")
            conn.rollback()
        finally:
            conn.close()