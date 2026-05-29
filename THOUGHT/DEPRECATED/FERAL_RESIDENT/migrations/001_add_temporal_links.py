"""
Migration: Add temporal links to vectors table for Iso-Temporal Protocol.

This enables the rotate_memory function by storing:
- prev_vector_id: pointer to previous item in sequence
- next_vector_id: pointer to next item in sequence
- context_vec_blob: precomputed context vector (centroid of previous k items)
- sequence_id: which sequence this belongs to (paper_id, session_id, etc.)
- sequence_idx: position within the sequence

Without these, temporal retrieval requires reconstructing order from timestamps
which is fragile and expensive.

SAFE: Only ADDS columns. Existing code using old columns continues to work.
"""
import sqlite3
import json
import re
import numpy as np
from pathlib import Path
import sys


DB_PATH = Path(__file__).parent.parent / "data" / "db" / "feral_eternal.db"
CONTEXT_K = 3  # Number of previous items to include in context


def compute_E(v1: np.ndarray, v2: np.ndarray) -> float:
    """Born rule: E = |<v1|v2>|^2"""
    dot = np.dot(v1, v2)
    return dot * dot


def migrate_schema(conn: sqlite3.Connection):
    """Add temporal columns to vectors table."""
    cursor = conn.cursor()

    # Check if already migrated
    cursor.execute("PRAGMA table_info(vectors)")
    columns = [col[1] for col in cursor.fetchall()]

    if "prev_vector_id" in columns:
        print("Already migrated - temporal columns exist")
        return False

    print("Adding temporal columns to vectors table...")

    # Add new columns
    cursor.execute("ALTER TABLE vectors ADD COLUMN prev_vector_id TEXT")
    cursor.execute("ALTER TABLE vectors ADD COLUMN next_vector_id TEXT")
    cursor.execute("ALTER TABLE vectors ADD COLUMN context_vec_blob BLOB")
    cursor.execute("ALTER TABLE vectors ADD COLUMN sequence_id TEXT")
    cursor.execute("ALTER TABLE vectors ADD COLUMN sequence_idx INTEGER")

    # Create index for sequence lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vectors_sequence ON vectors(sequence_id, sequence_idx)")

    conn.commit()
    print("Schema migration complete")
    return True


def backfill_temporal_links(conn: sqlite3.Connection):
    """Backfill temporal links from existing data."""
    cursor = conn.cursor()

    # Get all vectors with metadata, ordered by creation time
    cursor.execute('''
        SELECT v.vector_id, v.vec_blob, r.metadata, v.created_at
        FROM vectors v
        LEFT JOIN receipts r ON SUBSTR(v.vec_sha256, 1, 16) = r.output_hash
        WHERE v.composition_op = 'initialize' OR v.composition_op IS NULL
        ORDER BY v.created_at
    ''')

    # Group by sequence (paper_id extracted from content)
    sequences = {}  # sequence_id -> [(vector_id, vec_blob, created_at), ...]

    for row in cursor.fetchall():
        vector_id, vec_blob, metadata_json, created_at = row

        # Extract sequence_id from metadata
        sequence_id = "unknown"
        if metadata_json:
            meta = json.loads(metadata_json)
            # Try direct paper_id field first (newer format)
            paper_id = meta.get("paper_id")
            if paper_id:
                sequence_id = f"paper_{paper_id}"
            else:
                # Fall back to parsing text_preview (older format)
                content = meta.get("text_preview", "")
                match = re.search(r"@Paper-(\d+\.\d+)", content)
                if match:
                    sequence_id = f"paper_{match.group(1)}"

        if sequence_id not in sequences:
            sequences[sequence_id] = []
        sequences[sequence_id].append((vector_id, vec_blob, created_at))

    print(f"Found {len(sequences)} sequences with {sum(len(s) for s in sequences.values())} vectors")

    # Process each sequence
    updates = []
    for sequence_id, items in sequences.items():
        # Sort by creation time within sequence
        items.sort(key=lambda x: x[2])

        for idx, (vector_id, vec_blob, _) in enumerate(items):
            # Determine prev/next
            prev_id = items[idx - 1][0] if idx > 0 else None
            next_id = items[idx + 1][0] if idx < len(items) - 1 else None

            # Compute context vector (centroid of previous k)
            context_vec = None
            if idx > 0:
                start = max(0, idx - CONTEXT_K)
                prev_vecs = []
                for j in range(start, idx):
                    vec = np.frombuffer(items[j][1], dtype=np.float32)
                    prev_vecs.append(vec)

                if prev_vecs:
                    centroid = np.mean(prev_vecs, axis=0)
                    centroid = centroid / np.linalg.norm(centroid)
                    context_vec = centroid.tobytes()

            updates.append((prev_id, next_id, context_vec, sequence_id, idx, vector_id))

    # Batch update
    print(f"Updating {len(updates)} vectors with temporal links...")
    cursor.executemany('''
        UPDATE vectors
        SET prev_vector_id = ?, next_vector_id = ?, context_vec_blob = ?,
            sequence_id = ?, sequence_idx = ?
        WHERE vector_id = ?
    ''', updates)

    conn.commit()
    print("Backfill complete")


def verify_migration(conn: sqlite3.Connection):
    """Verify the migration worked."""
    cursor = conn.cursor()

    # Count vectors with temporal links
    cursor.execute("SELECT COUNT(*) FROM vectors WHERE sequence_id IS NOT NULL")
    linked = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vectors WHERE context_vec_blob IS NOT NULL")
    with_context = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vectors")
    total = cursor.fetchone()[0]

    print(f"\nVerification:")
    print(f"  Total vectors: {total}")
    print(f"  With sequence_id: {linked}")
    print(f"  With context_vec: {with_context}")

    # Sample a sequence
    cursor.execute('''
        SELECT sequence_id, COUNT(*) as cnt
        FROM vectors
        WHERE sequence_id IS NOT NULL
        GROUP BY sequence_id
        ORDER BY cnt DESC
        LIMIT 1
    ''')
    row = cursor.fetchone()
    if row:
        seq_id, cnt = row
        print(f"\nSample sequence '{seq_id}' ({cnt} items):")
        cursor.execute('''
            SELECT vector_id, sequence_idx, prev_vector_id, next_vector_id
            FROM vectors
            WHERE sequence_id = ?
            ORDER BY sequence_idx
            LIMIT 5
        ''', (seq_id,))
        for r in cursor.fetchall():
            print(f"  idx={r[1]}: {r[0][:16]}... prev={r[2][:16] if r[2] else 'None'}... next={r[3][:16] if r[3] else 'None'}...")


def main():
    print(f"Migrating {DB_PATH}")
    print("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))

    try:
        if migrate_schema(conn):
            backfill_temporal_links(conn)
        verify_migration(conn)
    finally:
        conn.close()

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
