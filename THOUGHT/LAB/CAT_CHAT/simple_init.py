#!/usr/bin/env python3
"""
Simple script to initialize CAT_CHAT database with indexing information.
"""

import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Now we can import directly
try:
    from NAVIGATION.CORTEX.network.cassettes.cat_chat_cassette import CatChatCassette
    print("[INFO] Successfully imported CatChatCassette")
except ImportError as e:
    print(f"[ERROR] Failed to import CatChatCassette: {e}")
    # Try to create the cassette directly
    print("[INFO] Creating cassette directly...")
    
    import sqlite3
    import json
    
    class SimpleCatChatCassette:
        def __init__(self):
            self.db_path = Path("cat_chat_index.db")
            self.capabilities = ["fts", "indexing_info"]
        
        def add_indexing_info(self, indexing_table: str):
            """Add the indexing information table to the database."""
            if not self.db_path.exists():
                print(f"[CAT_CHAT] Database not found, creating: {self.db_path}")
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
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
        
        def get_stats(self):
            """Return cassette statistics."""
            if not self.db_path.exists():
                return {"error": "Database not found"}
            
            stats = {}
            conn = sqlite3.connect(str(self.db_path))
            
            try:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Count indexing_info rows
                if 'indexing_info' in tables:
                    cursor = conn.execute("SELECT COUNT(*) as total FROM indexing_info")
                    stats["total_indexing_info"] = cursor.fetchone()[0]
                
                if 'indexing_info_fts' in tables:
                    stats["has_fts"] = True
                else:
                    stats["has_fts"] = False
                
                stats["tables"] = tables
                stats["capabilities"] = self.capabilities
            finally:
                conn.close()
            
            return stats

    CatChatCassette = SimpleCatChatCassette

def main():
    """Initialize CAT_CHAT database with indexing information."""
    print("Initializing CAT_CHAT database with indexing information...")
    
    # Create cassette instance
    cassette = CatChatCassette()
    
    # Check if database exists
    if not cassette.db_path.exists():
        print(f"[WARNING] Database not found at {cassette.db_path}")
        print("Creating database...")
        cassette.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add indexing information
    print("Adding indexing information to database...")
    cassette.add_indexing_info("indexing_info")
    
    # Get stats to verify
    stats = cassette.get_stats()
    if "error" in stats:
        print(f"[ERROR] Failed to get stats: {stats['error']}")
    else:
        print(f"[SUCCESS] Database initialized successfully!")
        print(f"  - Tables: {stats.get('tables', [])}")
        print(f"  - Total indexing info rows: {stats.get('total_indexing_info', 0)}")
        print(f"  - Has FTS: {stats.get('has_fts', False)}")
        print(f"  - Capabilities: {stats.get('capabilities', [])}")
    
    # Test query
    print("\nTesting query for 'indexing'...")
    if cassette.db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(cassette.db_path))
        try:
            cursor = conn.execute("""
                SELECT storage_type, column_type, normal_indexes_content 
                FROM indexing_info 
                WHERE storage_type LIKE '%index%' OR column_type LIKE '%index%' OR normal_indexes_content LIKE '%index%'
                LIMIT 3
            """)
            results = cursor.fetchall()
            print(f"Found {len(results)} results:")
            for i, row in enumerate(results, 1):
                print(f"  {i}. {row[0]} - {row[1]}")
                print(f"     {row[2][:80]}...")
        finally:
            conn.close()

if __name__ == "__main__":
    main()