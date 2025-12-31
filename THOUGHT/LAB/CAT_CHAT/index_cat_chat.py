#!/usr/bin/env python3
"""Index CAT_CHAT directory into SQLite for token-efficient analysis."""

import sqlite3
import hashlib
import json
from pathlib import Path
from datetime import datetime

CAT_CHAT_ROOT = Path(__file__).parent
DB_PATH = CAT_CHAT_ROOT / "cat_chat_index.db"

def compute_hash(content: bytes) -> str:
    """SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()

def init_db():
    """Create index database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            rel_path TEXT,
            size INTEGER,
            modified TEXT,
            content_hash TEXT,
            extension TEXT,
            is_duplicate BOOLEAN DEFAULT 0
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS content (
            file_id INTEGER,
            content TEXT,
            FOREIGN KEY(file_id) REFERENCES files(id)
        )
    ''')
    
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_hash ON files(content_hash)
    ''')
    
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_ext ON files(extension)
    ''')
    
    conn.commit()
    return conn

def index_files(conn):
    """Scan and index all files."""
    c = conn.cursor()
    indexed = 0
    
    # Skip these directories
    skip_dirs = {'__pycache__', '.git', 'node_modules', '_generated'}
    
    for file_path in CAT_CHAT_ROOT.rglob('*'):
        # Skip directories and excluded paths
        if file_path.is_dir():
            continue
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
        if file_path.name == 'cat_chat_index.db':
            continue
            
        try:
            # Read file
            if file_path.suffix in ['.py', '.md', '.json', '.txt', '.yaml', '.yml']:
                content = file_path.read_bytes()
                content_text = content.decode('utf-8', errors='ignore')
            else:
                content = b''
                content_text = ''
            
            # Compute metadata
            rel_path = str(file_path.relative_to(CAT_CHAT_ROOT))
            size = file_path.stat().st_size
            modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            content_hash = compute_hash(content) if content else ''
            extension = file_path.suffix
            
            # Insert file record
            c.execute('''
                INSERT OR REPLACE INTO files 
                (path, rel_path, size, modified, content_hash, extension)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (str(file_path), rel_path, size, modified, content_hash, extension))
            
            file_id = c.lastrowid
            
            # Insert content
            if content_text:
                c.execute('''
                    INSERT OR REPLACE INTO content (file_id, content)
                    VALUES (?, ?)
                ''', (file_id, content_text))
            
            indexed += 1
            
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
    
    conn.commit()
    return indexed

def mark_duplicates(conn):
    """Mark files with identical content hashes."""
    c = conn.cursor()
    
    # Find duplicate hashes
    c.execute('''
        UPDATE files
        SET is_duplicate = 1
        WHERE content_hash IN (
            SELECT content_hash
            FROM files
            WHERE content_hash != ''
            GROUP BY content_hash
            HAVING COUNT(*) > 1
        )
    ''')
    
    conn.commit()
    return c.rowcount

def print_stats(conn):
    """Print index statistics."""
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM files')
    total_files = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM files WHERE is_duplicate = 1')
    duplicates = c.fetchone()[0]
    
    c.execute('SELECT extension, COUNT(*) FROM files GROUP BY extension ORDER BY COUNT(*) DESC')
    by_ext = c.fetchall()
    
    c.execute('SELECT content_hash, COUNT(*) as cnt FROM files WHERE is_duplicate = 1 GROUP BY content_hash ORDER BY cnt DESC LIMIT 10')
    top_dupes = c.fetchall()
    
    print(f"\n📊 CAT_CHAT Index Stats")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Total files: {total_files}")
    print(f"Duplicates: {duplicates}")
    print(f"\nBy extension:")
    for ext, cnt in by_ext[:10]:
        print(f"  {ext or '(no ext)'}: {cnt}")
    
    print(f"\nTop duplicate groups:")
    for hash_val, cnt in top_dupes:
        c.execute('SELECT rel_path FROM files WHERE content_hash = ? LIMIT 3', (hash_val,))
        paths = [row[0] for row in c.fetchall()]
        print(f"  {cnt} copies: {paths[0]}")
        for p in paths[1:]:
            print(f"           ↳ {p}")

if __name__ == '__main__':
    print(f"Indexing CAT_CHAT at {CAT_CHAT_ROOT}")
    
    conn = init_db()
    indexed = index_files(conn)
    dupes = mark_duplicates(conn)
    
    print(f"✅ Indexed {indexed} files")
    print(f"🔍 Found {dupes} duplicate files")
    
    print_stats(conn)
    
    conn.close()
    print(f"\n💾 Database saved to {DB_PATH}")
