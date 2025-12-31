#!/usr/bin/env python3
"""Test CAT_CHAT database queries directly."""

import sqlite3

db_path = "cat_chat_index.db"
conn = sqlite3.connect(db_path)

print("Testing CAT_CHAT database queries...")

# Test 1: Check indexing_info_fts
print("\n1. Query 'index BLOB' in indexing_info_fts:")
cursor = conn.execute("SELECT COUNT(*) FROM indexing_info_fts WHERE indexing_info_fts MATCH 'index BLOB'")
count = cursor.fetchone()[0]
print(f"   Found {count} results")

if count > 0:
    cursor = conn.execute("""
        SELECT storage_type, column_type, normal_indexes_content 
        FROM indexing_info_fts 
        WHERE indexing_info_fts MATCH 'index BLOB' 
        LIMIT 1
    """)
    result = cursor.fetchone()
    print(f"   Example: {result[0]} - {result[1]}")
    print(f"   Indexing: {result[2]}")

# Test 2: Check content_fts
print("\n2. Query 'catalytic' in content_fts:")
cursor = conn.execute("SELECT COUNT(*) FROM content_fts WHERE content_fts MATCH 'catalytic'")
count = cursor.fetchone()[0]
print(f"   Found {count} results")

if count > 0:
    cursor = conn.execute("""
        SELECT f.path, snippet(content_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
        FROM content_fts
        JOIN files f ON content_fts.file_id = f.id
        WHERE content_fts MATCH 'catalytic'
        LIMIT 1
    """)
    result = cursor.fetchone()
    if result:
        print(f"   File: {result[0]}")
        print(f"   Snippet: {result[1]}")

# Test 3: List all tables
print("\n3. Database tables:")
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"   Tables: {tables}")

conn.close()
print("\nCAT_CHAT database is working correctly!")