#!/usr/bin/env python3
"""
Build isolated test database for quantum entanglement experiment.

Indexes all 12 knowledge base documents across 4 domains:
- Math (3 docs)
- Code (3 docs)
- Logic (3 docs)
- Chemistry (3 docs)

Usage:
    python test_sandbox/build_test_db.py
"""

import sqlite3
import hashlib
from pathlib import Path
import sys
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine


def build_test_db():
    """Build test_sandbox.db with all knowledge base docs."""
    db_path = Path(__file__).parent / "test_sandbox.db"
    docs_path = Path(__file__).parent / "docs"

    # Remove old database if it exists
    if db_path.exists():
        db_path.unlink()
        print(f"Removed old database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Create schema (same as cassettes)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS geometric_index (
            doc_id TEXT PRIMARY KEY,
            vector_blob BLOB NOT NULL,
            vector_hash TEXT NOT NULL,
            Df REAL NOT NULL,
            dim INTEGER NOT NULL,
            content_preview TEXT,
            metadata_json TEXT,
            indexed_at TEXT,
            schema_version TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            indexed_at TEXT,
            domain TEXT
        )
    """)

    engine = EmbeddingEngine()

    # Index all docs in all subdirectories
    total_indexed = 0
    domains = ["math", "code", "logic", "chemistry"]

    print("\n" + "=" * 70)
    print("BUILDING TEST DATABASE")
    print("=" * 70)

    for domain in domains:
        domain_path = docs_path / domain
        if not domain_path.exists():
            print(f"\n[WARN] Domain directory not found: {domain}")
            continue

        print(f"\n### Domain: {domain.upper()} ###")

        for doc_file in sorted(domain_path.glob("*.md")):
            try:
                content = doc_file.read_text(encoding='utf-8')
                doc_id = hashlib.sha256(str(doc_file).encode()).hexdigest()

                # Generate embedding
                embedding = engine.embed(content[:8000])
                vector_blob = embedding.tobytes()
                vector_hash = hashlib.sha256(vector_blob).hexdigest()[:16]

                Df = float(np.sum(embedding ** 2) ** 2 / np.sum(embedding ** 4))

                # Insert into geometric_index
                cur.execute("""
                    INSERT OR REPLACE INTO geometric_index
                    (doc_id, vector_blob, vector_hash, Df, dim, content_preview, metadata_json, indexed_at, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), '1.0')
                """, (
                    doc_id,
                    vector_blob,
                    vector_hash,
                    Df,
                    len(embedding),
                    content[:500],
                    f'{{"source": "{doc_file.name}", "domain": "{domain}"}}'
                ))

                # Insert into chunks
                chunk_id = f"{domain}_{doc_file.stem}_chunk_0"
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                cur.execute("""
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, file_path, content, content_hash, indexed_at, domain)
                    VALUES (?, ?, ?, ?, datetime('now'), ?)
                """, (chunk_id, str(doc_file), content, content_hash, domain))

                print(f"  [OK] {doc_file.name:40s} (Df={Df:6.1f})")
                total_indexed += 1

            except Exception as e:
                print(f"  [ERROR] {doc_file.name}: {e}")

    conn.commit()
    conn.close()

    print("\n" + "=" * 70)
    print(f"Database created: {db_path}")
    print(f"Total documents indexed: {total_indexed}/12")
    print("=" * 70)
    print("\nReady for testing! Run: python test_sandbox/run_all_tests.py")


if __name__ == "__main__":
    build_test_db()
