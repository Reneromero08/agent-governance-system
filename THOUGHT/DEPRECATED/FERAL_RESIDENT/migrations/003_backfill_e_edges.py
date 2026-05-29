"""
Migration: Backfill e_edges from existing daemon items.

This migration:
1. Finds all vectors with composition_op='daemon_item' or 'remember'
2. Groups by source/sequence
3. Computes E-scores pairwise (N=3 per Q13 optimal)
4. Applies R-gate filtering
5. Persists qualifying edges

Estimated scope:
- ~2682 existing items
- ~8000 edge candidates (N=3 comparisons each)
- ~400 edges stored (5% connectivity at E>0.5)
- Processing time: ~1-2 minutes

SAFE: Only ADDS edges. Existing data unchanged.
"""
import sqlite3
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional
import sys

# Add memory module to path for imports
MEMORY_PATH = Path(__file__).parent.parent / "memory"
if str(MEMORY_PATH) not in sys.path:
    sys.path.insert(0, str(MEMORY_PATH))

from e_graph import ERelationshipGraph, E_THRESHOLD, COMPARE_TO_RECENT, RTier, R_TIER_THRESHOLDS


DB_PATH = Path(__file__).parent.parent / "data" / "db" / "feral_eternal.db"

# Backfill configuration
BATCH_SIZE = 100
MIN_E_THRESHOLD = E_THRESHOLD  # 0.5 from Q44
MIN_R_TIER = RTier.T0_OBSERVE  # No R-gate filter for backfill (capture all)


def get_daemon_items(conn: sqlite3.Connection) -> List[Tuple[str, bytes, str, int]]:
    """
    Get all daemon items ordered by sequence and step.

    Returns:
        List of (vector_id, vec_blob, source_id, daemon_step)
    """
    cursor = conn.cursor()

    # Try daemon_item first, fall back to remember
    rows = cursor.execute("""
        SELECT vector_id, vec_blob, source_id, daemon_step, created_at
        FROM vectors
        WHERE composition_op IN ('daemon_item', 'remember', 'initialize')
        ORDER BY
            COALESCE(source_id, ''),
            COALESCE(daemon_step, 0),
            created_at
    """).fetchall()

    return rows


def compute_E(v1: np.ndarray, v2: np.ndarray) -> float:
    """Born rule: E = |<v1|v2>|^2"""
    dot = np.dot(v1, v2)
    return float(dot * dot)


def backfill_edges(conn: sqlite3.Connection, dry_run: bool = False):
    """
    Backfill E-edges from existing daemon items.

    Args:
        conn: Database connection
        dry_run: If True, don't commit changes
    """
    print("Fetching daemon items...")
    items = get_daemon_items(conn)
    print(f"Found {len(items)} items to process")

    if not items:
        print("No items to process")
        return

    # Group by source_id for sequence-aware processing
    by_source = {}
    for row in items:
        vector_id = row[0]
        vec_blob = row[1]
        source_id = row[2] or "unknown"
        daemon_step = row[3] or 0

        if source_id not in by_source:
            by_source[source_id] = []
        by_source[source_id].append((vector_id, vec_blob, daemon_step))

    print(f"Found {len(by_source)} sources")

    # Statistics
    total_candidates = 0
    total_edges = 0
    e_values = []

    # Process each source
    for source_id, source_items in by_source.items():
        # Sort by daemon_step
        source_items.sort(key=lambda x: x[2])

        print(f"  Processing source '{source_id}' ({len(source_items)} items)...")

        # For each item, compare to previous N items
        for i, (vector_id, vec_blob, step) in enumerate(source_items):
            vec = np.frombuffer(vec_blob, dtype=np.float32)

            # Get previous N items to compare
            start_idx = max(0, i - COMPARE_TO_RECENT)
            for j in range(start_idx, i):
                prev_id, prev_blob, _ = source_items[j]
                prev_vec = np.frombuffer(prev_blob, dtype=np.float32)

                # Compute E
                E = compute_E(vec, prev_vec)
                total_candidates += 1
                e_values.append(E)

                # Check threshold
                if E < MIN_E_THRESHOLD:
                    continue

                # Compute R (using running sigma)
                sigma = np.std(e_values) if len(e_values) > 1 else 0.1
                R = E / max(sigma, 0.01)

                # Get tier
                if R >= R_TIER_THRESHOLDS[RTier.T3_LARGE]:
                    r_tier = "T3"
                elif R >= R_TIER_THRESHOLDS[RTier.T2_MEDIUM]:
                    r_tier = "T2"
                elif R >= R_TIER_THRESHOLDS[RTier.T1_SMALL]:
                    r_tier = "T1"
                else:
                    r_tier = "T0"

                # Create edge (normalize order to prevent duplicates)
                if vector_id > prev_id:
                    vector_id_a, vector_id_b = prev_id, vector_id
                else:
                    vector_id_a, vector_id_b = vector_id, prev_id

                edge_id = str(uuid.uuid4())[:8]
                now = datetime.now(timezone.utc).isoformat()

                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO e_edges
                        (edge_id, vector_id_a, vector_id_b, e_score, r_score, r_tier, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (edge_id, vector_id_a, vector_id_b, E, R, r_tier, now)
                    )
                    total_edges += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate edge

    if not dry_run:
        conn.commit()
        print(f"\nCommitted changes")
    else:
        conn.rollback()
        print(f"\nDry run - changes not committed")

    # Statistics
    print(f"\nBackfill Statistics:")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Edges created: {total_edges}")
    print(f"  Connectivity: {total_edges / max(total_candidates, 1) * 100:.1f}%")
    if e_values:
        print(f"  E distribution: mean={np.mean(e_values):.3f}, std={np.std(e_values):.3f}")
        print(f"  E above threshold: {sum(1 for e in e_values if e >= MIN_E_THRESHOLD)} ({sum(1 for e in e_values if e >= MIN_E_THRESHOLD) / len(e_values) * 100:.1f}%)")


def verify_backfill(conn: sqlite3.Connection):
    """Verify the backfill worked."""
    cursor = conn.cursor()

    # Edge count
    total = cursor.execute("SELECT COUNT(*) FROM e_edges").fetchone()[0]
    print(f"\nVerification:")
    print(f"  Total edges: {total}")

    # By tier
    print(f"  Edges by tier:")
    for row in cursor.execute("SELECT r_tier, COUNT(*) FROM e_edges GROUP BY r_tier ORDER BY r_tier"):
        print(f"    {row[0]}: {row[1]}")

    # E score distribution
    e_stats = cursor.execute("""
        SELECT AVG(e_score), MIN(e_score), MAX(e_score)
        FROM e_edges
    """).fetchone()
    if e_stats[0]:
        print(f"  E scores: mean={e_stats[0]:.3f}, min={e_stats[1]:.3f}, max={e_stats[2]:.3f}")

    # Sample edges
    print(f"\n  Sample edges:")
    for row in cursor.execute("SELECT * FROM e_edges ORDER BY e_score DESC LIMIT 5"):
        print(f"    {row[1][:8]}...--{row[2][:8]}... E={row[3]:.3f} R={row[4]:.2f} ({row[5]})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill e_edges from daemon items")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit changes")
    args = parser.parse_args()

    print(f"Backfilling e_edges from {DB_PATH}")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        backfill_edges(conn, dry_run=args.dry_run)
        if not args.dry_run:
            verify_backfill(conn)
    finally:
        conn.close()

    print("\nBackfill complete!")


if __name__ == "__main__":
    main()
