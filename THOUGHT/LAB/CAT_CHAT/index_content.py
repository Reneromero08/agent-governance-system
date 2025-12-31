#!/usr/bin/env python3
"""Index the actual text content of CAT_CHAT files with FTS5."""

import sqlite3
from pathlib import Path

db_path = Path("cat_chat_index.db")
if not db_path.exists():
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))

print("Creating FTS5 index for file content...")

try:
    # Create FTS5 virtual table for content
    conn.execute("DROP TABLE IF EXISTS content_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS content_fts 
        USING fts5(file_id, content)
    """)
    
    # Populate FTS table from content table
    print("Populating FTS table from content table...")
    conn.execute("""
        INSERT INTO content_fts (file_id, content)
        SELECT file_id, content 
        FROM content
        WHERE content IS NOT NULL AND content != ''
    """)
    
    # Count rows
    cursor = conn.execute("SELECT COUNT(*) FROM content_fts")
    fts_count = cursor.fetchone()[0]
    
    cursor = conn.execute("SELECT COUNT(*) FROM content")
    content_count = cursor.fetchone()[0]
    
    print(f"Indexed {fts_count} out of {content_count} content rows")
    
    # Test the FTS index
    print("\nTesting FTS search for 'catalytic'...")
    cursor = conn.execute("""
        SELECT f.path, snippet(content_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
        FROM content_fts
        JOIN files f ON content_fts.file_id = f.id
        WHERE content_fts MATCH 'catalytic'
        LIMIT 3
    """)
    
    results = cursor.fetchall()
    print(f"Found {len(results)} results:")
    for i, (path, snippet) in enumerate(results, 1):
        print(f"  {i}. {path}")
        print(f"     {snippet}")
    
    conn.commit()
    print(f"\nSuccessfully created content_fts table with {fts_count} indexed documents")
    
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    conn.close()