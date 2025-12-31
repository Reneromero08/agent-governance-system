#!/usr/bin/env python3
"""
Analyze CAT_CHAT database structure for duplicate detection.
"""

import sqlite3
import json
from pathlib import Path

def analyze_database():
    db_path = Path("THOUGHT/LAB/CAT_CHAT/cat_chat_index.db")
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("=== CAT_CHAT DATABASE ANALYSIS ===")
    print(f"Database: {db_path}")
    print()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"Total tables: {len(tables)}")
    print()
    
    # Analyze each table
    for table in tables:
        print(f"--- {table} ---")
        
        # Get columns
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        print(f"  Columns ({len(columns)}):")
        for col in columns:
            col_id, col_name, col_type, notnull, default, pk = col
            pk_str = " PK" if pk else ""
            print(f"    {col_name:30} {col_type:15} {pk_str}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")
        
        # Show sample for key tables
        if table in ['files', 'content'] and count > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            rows = cursor.fetchall()
            print(f"  Sample rows:")
            for i, row in enumerate(rows):
                print(f"    Row {i}: {row}")
        print()
    
    # Now analyze for duplicates
    print("=== DUPLICATE ANALYSIS ===")
    
    # Check files table for duplicate paths
    if 'files' in tables:
        print("\n1. Checking for duplicate file paths:")
        cursor.execute("""
            SELECT path, COUNT(*) as count
            FROM files
            GROUP BY path
            HAVING COUNT(*) > 1
            ORDER BY count DESC
        """)
        duplicate_paths = cursor.fetchall()
        
        if duplicate_paths:
            print(f"  Found {len(duplicate_paths)} files with duplicate paths:")
            for path, count in duplicate_paths:
                print(f"    {path} ({count} copies)")
        else:
            print("  No duplicate file paths found")
    
    # Check content table for duplicate content
    if 'content' in tables:
        print("\n2. Checking for duplicate content (by hash or text):")
        
        # First check if there's a hash column
        cursor.execute("PRAGMA table_info(content)")
        content_cols = [row[1] for row in cursor.fetchall()]
        
        if 'hash' in content_cols:
            cursor.execute("""
                SELECT hash, COUNT(*) as count
                FROM content
                GROUP BY hash
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """)
            duplicate_hashes = cursor.fetchall()
            
            if duplicate_hashes:
                print(f"  Found {len(duplicate_hashes)} duplicate content hashes:")
                for content_hash, count in duplicate_hashes:
                    print(f"    Hash: {content_hash[:16]}... ({count} copies)")
                    
                    # Get file paths for this hash
                    cursor.execute("""
                        SELECT f.path, c.content_preview
                        FROM content c
                        JOIN files f ON c.file_id = f.id
                        WHERE c.hash = ?
                        LIMIT 3
                    """, (content_hash,))
                    files = cursor.fetchall()
                    for path, preview in files:
                        preview_str = preview[:50] + "..." if preview and len(preview) > 50 else preview
                        print(f"      - {path}")
                        print(f"        Preview: {preview_str}")
            else:
                print("  No duplicate content hashes found")
        
        # Check for similar file names
        print("\n3. Checking for similar file names (potential duplicates):")
        cursor.execute("""
            SELECT path, 
                   LOWER(SUBSTR(path, INSTR(path, '/') + 1)) as filename,
                   COUNT(*) OVER (PARTITION BY LOWER(SUBSTR(path, INSTR(path, '/') + 1))) as similar_count
            FROM files
            WHERE similar_count > 1
            ORDER BY filename, path
        """)
        
        similar_files = cursor.fetchall()
        if similar_files:
            print(f"  Found {len(similar_files)} files with similar names:")
            current_filename = None
            for path, filename, count in similar_files:
                if filename != current_filename:
                    print(f"    Files named like '{filename}':")
                    current_filename = filename
                print(f"      - {path}")
        else:
            print("  No files with similar names found")
    
    conn.close()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. Run SQL queries to find exact content duplicates")
    print("2. Compare file content for semantic similarity")
    print("3. Create merge plan based on:")
    print("   - Identical content (same hash)")
    print("   - Similar file names")
    print("   - Same document in different locations")

if __name__ == "__main__":
    analyze_database()