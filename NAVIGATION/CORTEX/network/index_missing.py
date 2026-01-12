#!/usr/bin/env python3
"""
Index missing files into cassette geometric_index.

Usage:
    # Index Q44/Q45 files
    python NAVIGATION/CORTEX/network/index_missing.py

    # Index specific file
    python NAVIGATION/CORTEX/network/index_missing.py --file "THOUGHT/LAB/FORMULA/some_file.md"

    # Index all files matching a pattern
    python NAVIGATION/CORTEX/network/index_missing.py --glob "THOUGHT/LAB/FORMULA/**/*.md"
"""

import sqlite3
import hashlib
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

# Default cassette mapping (path prefix -> db)
CASSETTE_MAP = {
    'THOUGHT/': 'thought.db',
    'LAW/': 'canon.db',
    'CAPABILITY/': 'capability.db',
    'NAVIGATION/': 'navigation.db',
    'DIRECTION/': 'direction.db',
    'MEMORY/': 'memory.db',
    'INBOX/': 'inbox.db',
}


def get_cassette_db(file_path: Path) -> Path:
    """Determine which cassette DB to use based on file path."""
    rel_path = str(file_path.relative_to(PROJECT_ROOT))
    for prefix, db_name in CASSETTE_MAP.items():
        if rel_path.startswith(prefix):
            return PROJECT_ROOT / "NAVIGATION/CORTEX/cassettes" / db_name
    return PROJECT_ROOT / "NAVIGATION/CORTEX/cassettes/thought.db"  # Default


def index_file(conn, engine, file_path: Path) -> bool:
    """Index a single file into the geometric_index."""
    safe_path = str(file_path).replace('\u2665', 'HEART-')

    if not file_path.exists():
        print(f"  [SKIP] Not found: {safe_path}")
        return False

    content = file_path.read_text(encoding='utf-8')
    if not content.strip():
        print(f"  [SKIP] Empty: {safe_path}")
        return False

    doc_id = hashlib.sha256(str(file_path).encode()).hexdigest()

    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM geometric_index WHERE doc_id = ?", (doc_id,))
    if cur.fetchone():
        print(f"  [SKIP] Already indexed: {Path(safe_path).name}")
        return False

    embedding = engine.embed(content[:8000])
    vector_blob = embedding.tobytes()
    vector_hash = hashlib.sha256(vector_blob).hexdigest()[:16]

    import numpy as np
    Df = float(np.sum(embedding ** 2) ** 2 / np.sum(embedding ** 4)) if np.any(embedding) else 0.0

    cur.execute("""
        INSERT INTO geometric_index
        (doc_id, vector_blob, vector_hash, Df, dim, content_preview, metadata_json, indexed_at, schema_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), '1.0')
    """, (
        doc_id,
        vector_blob,
        vector_hash,
        Df,
        len(embedding),
        content[:500],
        f'{{"source": "{file_path.name}"}}'
    ))

    print(f"  [OK] Indexed: {Path(safe_path).name} (Df={Df:.1f})")
    return True


def main():
    parser = argparse.ArgumentParser(description='Index missing files into cassette')
    parser.add_argument('--file', type=str, help='Single file to index (relative to project root)')
    parser.add_argument('--glob', type=str, help='Glob pattern to match files')
    parser.add_argument('--db', type=str, help='Override target database path')
    args = parser.parse_args()

    files_to_index = []

    if args.file:
        files_to_index.append(PROJECT_ROOT / args.file)
    elif args.glob:
        files_to_index.extend(PROJECT_ROOT.glob(args.glob))
    else:
        # Default: index Q44/Q45 files
        files_to_index = [
            PROJECT_ROOT / "THOUGHT/LAB/FORMULA/research/questions/reports/♥Q44_QUANTUM_BORN_RULE_REPORT.md",
            PROJECT_ROOT / "THOUGHT/LAB/FORMULA/research/questions/reports/♥Q45_PURE_GEOMETRY_REPORT.md",
            PROJECT_ROOT / "THOUGHT/LAB/FORMULA/research/questions/critical/q44_quantum_born_rule.md",
            PROJECT_ROOT / "THOUGHT/LAB/FORMULA/research/questions/critical/q45_semantic_entanglement.md",
        ]

    if not files_to_index:
        print("No files to index.")
        return

    engine = EmbeddingEngine()

    # Group by database
    db_files = {}
    for f in files_to_index:
        if not f.exists():
            continue
        db_path = Path(args.db) if args.db else get_cassette_db(f)
        if db_path not in db_files:
            db_files[db_path] = []
        db_files[db_path].append(f)

    total_indexed = 0
    for db_path, files in db_files.items():
        print(f"\nIndexing into {db_path.name}...")
        conn = sqlite3.connect(str(db_path))
        for f in files:
            if index_file(conn, engine, f):
                total_indexed += 1
        conn.commit()
        conn.close()

    print(f"\nDone. Indexed {total_indexed} new files.")


if __name__ == "__main__":
    main()
