#!/usr/bin/env python3
"""
System 1 Database Builder (Fast Retrieval Layer)

Implements the "Fast Thinking" database using SQLite FTS5 for content search
and chunk indexing. Integrates with F3 CAS for content-addressed storage.

Architecture:
- system1.db: SQLite database with FTS5 tables
- Chunks: ~500 token segments with overlap
- Metadata: File paths, sections, timestamps
- CAS Integration: Content hashes for deduplication
"""

import sqlite3
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
DB_PATH = Path("CORTEX/system1.db")

class System1DB:
    """Fast retrieval database using SQLite FTS5."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        
    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            -- Files table: tracks all indexed files
            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Chunks table: stores text chunks with metadata
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(file_id),
                UNIQUE(file_id, chunk_index)
            );
            
            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                chunk_id UNINDEXED,
                tokenize='porter unicode61'
            );
            
            -- Index for fast lookups
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);
        """)
        self.conn.commit()
        
    def add_file(self, path: str, content: str) -> int:
        """Add a file and its chunks to the database."""
        # Compute content hash
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Check if already indexed
        cursor = self.conn.execute(
            "SELECT file_id FROM files WHERE path = ? AND content_hash = ?",
            (path, content_hash)
        )
        existing = cursor.fetchone()
        if existing:
            return existing['file_id']
            
        # Insert file record
        cursor = self.conn.execute(
            "INSERT INTO files (path, content_hash, size_bytes) VALUES (?, ?, ?)",
            (path, content_hash, len(content.encode('utf-8')))
        )
        file_id = cursor.lastrowid
        
        # Chunk the content
        chunks = self._chunk_text(content)
        
        # Insert chunks
        for idx, chunk_text in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            token_count = self._count_tokens(chunk_text)
            
            # Insert chunk metadata
            cursor = self.conn.execute(
                """INSERT INTO chunks 
                   (file_id, chunk_index, chunk_hash, token_count, start_offset, end_offset)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_id, idx, chunk_hash, token_count, 0, len(chunk_text))  # TODO: track actual offsets
            )
            chunk_id = cursor.lastrowid
            
            # Insert into FTS
            self.conn.execute(
                "INSERT INTO chunks_fts (rowid, content, chunk_id) VALUES (?, ?, ?)",
                (chunk_id, chunk_text, chunk_id)
            )
            
        self.conn.commit()
        return file_id
        
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Full-text search across all chunks."""
        cursor = self.conn.execute("""
            SELECT 
                f.path,
                c.chunk_index,
                c.token_count,
                snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as snippet,
                rank
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.chunk_id = c.chunk_id
            JOIN files f ON c.file_id = f.file_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks (word-based approximation)."""
        # Approximate: 1 token ≈ 0.75 words, so 500 tokens ≈ 375 words
        words = text.split()
        word_chunk_size = int(CHUNK_SIZE * 0.75)
        word_overlap = int(CHUNK_OVERLAP * 0.75)
        
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + word_chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            start += word_chunk_size - word_overlap
            
        return chunks if chunks else [text]  # Return full text if no chunks
        
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (word-based)."""
        return int(len(text.split()) / 0.75)
        
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Demo: Index a sample file."""
    db = System1DB()
    
    # Index a sample file
    sample_path = "CANON/FORMULA.md"
    if Path(sample_path).exists():
        content = Path(sample_path).read_text(encoding='utf-8')
        file_id = db.add_file(sample_path, content)
        print(f"Indexed {sample_path} as file_id={file_id}")
        
        # Test search
        results = db.search("resonance entropy")
        print(f"\nSearch results for 'resonance entropy':")
        for r in results:
            print(f"  {r['path']}#{r['chunk_index']}: {r['snippet']}")
    
    db.close()


if __name__ == "__main__":
    main()
