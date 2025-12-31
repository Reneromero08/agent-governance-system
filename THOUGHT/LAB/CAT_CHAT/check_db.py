#!/usr/bin/env python3
"""Check what's in the CAT_CHAT database."""

import sqlite3
from pathlib import Path

db_path = Path("cat_chat_index.db")
if not db_path.exists():
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(str(db_path))

# Get all tables
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Tables in database: {tables}")

# Check each table
for table in tables:
    print(f"\n--- Table: {table} ---")
    
    # Get schema
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Columns: {columns}")
    
    # Get row count
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"Row count: {count}")
    
    # Show sample data for small tables
    if count > 0 and count <= 10:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 3")
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            print(f"  Row {i}: {row}")

conn.close()