#!/usr/bin/env python3
"""
Export Semantic Core data for Tiny Agents

Queries system1.db and exports section data compressed with @Symbols
for consumption by small AI models (tiny agents in swarm architecture).
"""

import sqlite3
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "system1.db"

def export_for_tiny_agents():
    """
    Query semantic database and compress results using @Symbol format.
    
    Format:
    {
        "@C1": {content_hash: "...", embedding: [...]},
        "@C2": {content_hash: "...", embedding: [...]},
        ...
    }
    
    Tiny agents receive these @Symbols and expand them as needed.
    """
    if not DB_PATH.exists():
        print("❌ ERROR: system1.db not found")
        print("Please run vector_indexer.py first to build the index.")
        return False
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # Get all sections with their embeddings
    cursor = conn.execute("""
        SELECT 
            c.chunk_hash as hash,
            fts.content,
            c.chunk_id,
            sv.embedding,
            sv.model_id
        FROM chunks c
        JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
        JOIN section_vectors sv ON c.chunk_hash = sv.hash
    """)
    
    sections = cursor.fetchall()
    
    # Build compressed output
    compressed = {}
    
    for section in sections:
        section_hash = section['hash']
        content_hash = section['hash']  # Use same hash as key
        
        # Parse embedding (stored as BLOB in SQLite)
        embedding_blob = section['embedding']
        
        # Reconstruct numpy array from bytes
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        # Compress to small float16 representation for tiny agents
        embedding_compressed = embedding.astype(np.float16).tobytes().hex()
        
        compressed[content_hash] = {
            "symbol": f"@C:{section_hash[:8]}",
            "content_hash": content_hash,
            "chunk_id": section['chunk_id'],
            "embedding_compressed": embedding_compressed[:256],  # First 256 chars
            "model": section['model_id']
        }
    
    conn.close()
    
    # Write compressed data to stdout
    output = json.dumps(compressed, indent=2)
    print(output)
    
    print(f"Exported {len(compressed)} sections with compressed embeddings")
    print(f"  Embedding precision: float32 → float16 (50% size reduction)")
    
    return True

if __name__ == "__main__":
    import sys
    success = export_for_tiny_agents()
    sys.exit(0 if success else 1)
