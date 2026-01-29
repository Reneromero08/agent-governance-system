"""
Migration: Add e_edges table for E-relationship graph.

This enables the E-relationship daemon by storing:
- edge_id: Unique edge identifier
- vector_id_a, vector_id_b: The two vectors this edge connects
- e_score: Born rule similarity E = <v1|v2>^2
- r_score: R-gate score R = E/sigma (from Q15)
- r_tier: R-gate tier T0/T1/T2/T3 (from Q17)
- created_at: Timestamp

The graph is sparse - only edges with E > 0.5 (Q44 threshold) are stored.
R-gating filters out noisy relationships (high sigma).

SAFE: Only ADDS table. Existing code continues to work.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path(__file__).parent.parent / "data" / "db" / "feral_eternal.db"


def migrate_schema(conn: sqlite3.Connection) -> bool:
    """Add e_edges table."""
    cursor = conn.cursor()

    # Check if table already exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='e_edges'
    """)
    if cursor.fetchone():
        print("Already migrated - e_edges table exists")
        return False

    print("Creating e_edges table...")

    # Create the e_edges table
    cursor.execute("""
        CREATE TABLE e_edges (
            edge_id TEXT PRIMARY KEY,
            vector_id_a TEXT NOT NULL,
            vector_id_b TEXT NOT NULL,
            e_score REAL NOT NULL,
            r_score REAL,
            r_tier TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(vector_id_a, vector_id_b),
            FOREIGN KEY (vector_id_a) REFERENCES vectors(vector_id),
            FOREIGN KEY (vector_id_b) REFERENCES vectors(vector_id)
        )
    """)

    # Create indexes for efficient lookups
    cursor.execute("""
        CREATE INDEX idx_e_edges_a ON e_edges(vector_id_a, e_score DESC)
    """)
    cursor.execute("""
        CREATE INDEX idx_e_edges_b ON e_edges(vector_id_b, e_score DESC)
    """)
    cursor.execute("""
        CREATE INDEX idx_e_edges_tier ON e_edges(r_tier)
    """)
    cursor.execute("""
        CREATE INDEX idx_e_edges_created ON e_edges(created_at DESC)
    """)

    conn.commit()
    print("Schema migration complete")
    return True


def add_daemon_item_columns(conn: sqlite3.Connection) -> bool:
    """Add daemon_item columns to vectors table if not present."""
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(vectors)")
    columns = {col[1] for col in cursor.fetchall()}

    added = False
    if "source_id" not in columns:
        print("Adding source_id column to vectors...")
        cursor.execute("ALTER TABLE vectors ADD COLUMN source_id TEXT")
        added = True

    if "daemon_step" not in columns:
        print("Adding daemon_step column to vectors...")
        cursor.execute("ALTER TABLE vectors ADD COLUMN daemon_step INTEGER")
        added = True

    if "mind_hash_before" not in columns:
        print("Adding mind_hash_before column to vectors...")
        cursor.execute("ALTER TABLE vectors ADD COLUMN mind_hash_before TEXT")
        added = True

    if added:
        conn.commit()
        print("Added daemon_item columns to vectors table")

    return added


def verify_migration(conn: sqlite3.Connection):
    """Verify the migration worked."""
    cursor = conn.cursor()

    # Check e_edges table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='e_edges'
    """)
    if cursor.fetchone():
        print("\n[OK] e_edges table exists")
    else:
        print("\n[ERROR] e_edges table not found")
        return

    # Check indexes
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='index' AND tbl_name='e_edges'
    """)
    indexes = [row[0] for row in cursor.fetchall()]
    print(f"[OK] Indexes: {indexes}")

    # Check daemon_item columns in vectors
    cursor.execute("PRAGMA table_info(vectors)")
    columns = [col[1] for col in cursor.fetchall()]
    daemon_cols = [c for c in columns if c in ('source_id', 'daemon_step', 'mind_hash_before')]
    print(f"[OK] Daemon columns in vectors: {daemon_cols}")


def main():
    print(f"Migrating {DB_PATH}")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))

    try:
        migrate_schema(conn)
        add_daemon_item_columns(conn)
        verify_migration(conn)
    finally:
        conn.close()

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
