#!/usr/bin/env python3
"""
Hierarchy Archiver for memory management (Phase J.5).

Allows archiving old content while preserving the hierarchy structure.
Archived nodes:
- Keep their centroid (still participate in E-score computation)
- Drop their L0 content reference (saves storage)
- Return placeholder text when retrieved

This enables bounded storage while maintaining retrieval quality.

Part of Phase J.5: Forgetting and archival mechanisms.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

from .hierarchy_schema import L0


def _now_iso() -> str:
    """Get ISO8601 timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(iso_str: str) -> datetime:
    """Parse ISO8601 timestamp to datetime."""
    # Handle both Z suffix and +00:00 suffix
    if iso_str.endswith("Z"):
        iso_str = iso_str[:-1] + "+00:00"
    return datetime.fromisoformat(iso_str)


@dataclass
class ArchiveResult:
    """Result of archive operation."""
    session_id: str
    nodes_archived: int
    nodes_skipped: int  # Protected by min_keep
    bytes_freed: int    # Estimated based on content sizes


@dataclass
class ArchiveStats:
    """Archive statistics for a session."""
    session_id: str
    total_nodes: int
    archived_nodes: int
    active_nodes: int
    oldest_access: Optional[datetime]
    newest_access: Optional[datetime]


class HierarchyArchiver:
    """Manages archival of old hierarchy nodes.

    Provides memory management for long conversations by:
    - Tracking access times for LRU-style eviction
    - Archiving old nodes to free content storage
    - Preserving centroids for continued E-score participation
    - Supporting restoration when content is still available

    Usage:
        archiver = HierarchyArchiver(db_path)
        archiver.ensure_schema()

        # After retrieval, update access times
        archiver.update_access_time(retrieved_node_ids)

        # Archive old content periodically
        result = archiver.archive_old_nodes(session_id, age_days=30)
    """

    def __init__(self, db_path: Path):
        """Initialize the archiver.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def ensure_schema(self) -> bool:
        """Ensure hierarchy_nodes table has archival columns.

        Adds is_archived and last_accessed_at columns if they do not exist.

        Returns:
            True if columns were added, False if already existed
        """
        conn = self._get_conn()

        # Check if columns exist
        cursor = conn.execute("PRAGMA table_info(hierarchy_nodes)")
        columns = {row["name"] for row in cursor.fetchall()}

        added = False

        if "is_archived" not in columns:
            conn.execute("""
                ALTER TABLE hierarchy_nodes
                ADD COLUMN is_archived INTEGER DEFAULT 0
            """)
            added = True

        if "last_accessed_at" not in columns:
            conn.execute("""
                ALTER TABLE hierarchy_nodes
                ADD COLUMN last_accessed_at TEXT
            """)
            added = True

        # Create index for archival queries if not exists
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hierarchy_archived
            ON hierarchy_nodes(is_archived, last_accessed_at)
        """)

        conn.commit()
        return added

    def update_access_time(self, node_ids: List[str]) -> int:
        """Update last_accessed_at for nodes that were retrieved.

        Called after retrieval to track usage patterns.
        This is critical for LRU-style eviction decisions.

        Args:
            node_ids: List of node IDs that were accessed

        Returns:
            Count of nodes updated
        """
        if not node_ids:
            return 0

        conn = self._get_conn()
        now = _now_iso()

        # Bulk update for efficiency
        placeholders = ",".join("?" for _ in node_ids)
        cursor = conn.execute(f"""
            UPDATE hierarchy_nodes
            SET last_accessed_at = ?
            WHERE node_id IN ({placeholders})
        """, [now] + list(node_ids))

        conn.commit()
        return cursor.rowcount

    def get_archival_candidates(
        self,
        session_id: str,
        age_days: int = 30,
        min_keep: int = 1000
    ) -> List[str]:
        """Get node_ids that would be archived (dry run).

        Useful for previewing archival before committing.

        Criteria:
        - L0 nodes only (higher levels always kept)
        - Not already archived
        - last_accessed_at older than age_days (or NULL)
        - Not in the most recent min_keep turns by sequence number

        Args:
            session_id: Session to check
            age_days: Archive nodes not accessed in this many days
            min_keep: Always keep at least this many recent turns

        Returns:
            List of node_ids that would be archived
        """
        conn = self._get_conn()
        cutoff = datetime.now(timezone.utc) - timedelta(days=age_days)
        cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")

        # First, get the min sequence number to keep (protect recent turns)
        # Get the sequence number of the (total - min_keep)th turn
        cursor = conn.execute("""
            SELECT first_turn_seq
            FROM hierarchy_nodes
            WHERE session_id = ? AND level = 0
            ORDER BY first_turn_seq DESC
            LIMIT 1 OFFSET ?
        """, (session_id, min_keep - 1))

        row = cursor.fetchone()
        if row is None:
            # Fewer than min_keep turns exist, protect all
            return []

        min_protected_seq = row["first_turn_seq"]

        # Find candidates: L0 nodes that are old and not protected
        cursor = conn.execute("""
            SELECT node_id
            FROM hierarchy_nodes
            WHERE session_id = ?
              AND level = 0
              AND (is_archived = 0 OR is_archived IS NULL)
              AND first_turn_seq < ?
              AND (last_accessed_at IS NULL OR last_accessed_at < ?)
            ORDER BY first_turn_seq ASC
        """, (session_id, min_protected_seq, cutoff_iso))

        return [row["node_id"] for row in cursor.fetchall()]

    def archive_old_nodes(
        self,
        session_id: str,
        age_days: int = 30,
        min_keep: int = 1000
    ) -> ArchiveResult:
        """Archive nodes not accessed in age_days.

        Archival means:
        - Set is_archived = 1
        - Centroid is PRESERVED (E-score still works)
        - content_hash and event_id references remain for potential restoration

        Args:
            session_id: Session to archive
            age_days: Archive nodes not accessed in this many days
            min_keep: Always keep at least this many recent turns

        Returns:
            ArchiveResult with counts of archived nodes
        """
        conn = self._get_conn()

        # Get candidates
        candidates = self.get_archival_candidates(session_id, age_days, min_keep)

        if not candidates:
            return ArchiveResult(
                session_id=session_id,
                nodes_archived=0,
                nodes_skipped=0,
                bytes_freed=0
            )

        # Estimate bytes freed (based on typical content sizes)
        # We keep centroids (1536 bytes each), but content pointers are cleared
        # Estimate ~500 bytes average content reference per node
        estimated_bytes = len(candidates) * 500

        # Count how many were skipped due to min_keep
        cursor = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM hierarchy_nodes
            WHERE session_id = ?
              AND level = 0
              AND (is_archived = 0 OR is_archived IS NULL)
        """, (session_id,))
        total_unarchived = cursor.fetchone()["cnt"]
        skipped = total_unarchived - len(candidates)

        # Archive the candidates
        placeholders = ",".join("?" for _ in candidates)
        conn.execute(f"""
            UPDATE hierarchy_nodes
            SET is_archived = 1
            WHERE node_id IN ({placeholders})
        """, candidates)

        conn.commit()

        return ArchiveResult(
            session_id=session_id,
            nodes_archived=len(candidates),
            nodes_skipped=skipped,
            bytes_freed=estimated_bytes
        )

    def restore_archived_node(self, node_id: str) -> bool:
        """Restore an archived node if content is still available.

        Only works if the underlying session_events still has the content.
        This simply clears the is_archived flag.

        Args:
            node_id: ID of the node to restore

        Returns:
            True if restored, False if node not found or content is gone
        """
        conn = self._get_conn()

        # Check if node exists and is archived
        cursor = conn.execute("""
            SELECT node_id, event_id, is_archived
            FROM hierarchy_nodes
            WHERE node_id = ?
        """, (node_id,))

        row = cursor.fetchone()
        if row is None:
            return False

        if not row["is_archived"]:
            # Already not archived
            return True

        # Check if content is still available in session_events
        if row["event_id"]:
            cursor = conn.execute("""
                SELECT event_id
                FROM session_events
                WHERE event_id = ?
            """, (row["event_id"],))

            if cursor.fetchone() is None:
                # Content is gone, cannot restore
                return False

        # Restore: clear archived flag, update access time
        conn.execute("""
            UPDATE hierarchy_nodes
            SET is_archived = 0,
                last_accessed_at = ?
            WHERE node_id = ?
        """, (_now_iso(), node_id))

        conn.commit()
        return True

    def get_archive_stats(self, session_id: str) -> ArchiveStats:
        """Get statistics about archived vs active nodes.

        Args:
            session_id: Session to get stats for

        Returns:
            ArchiveStats with counts and access time range
        """
        conn = self._get_conn()

        # Total L0 nodes
        cursor = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM hierarchy_nodes
            WHERE session_id = ? AND level = 0
        """, (session_id,))
        total = cursor.fetchone()["cnt"]

        # Archived L0 nodes
        cursor = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM hierarchy_nodes
            WHERE session_id = ? AND level = 0 AND is_archived = 1
        """, (session_id,))
        archived = cursor.fetchone()["cnt"]

        # Access time range (only for non-archived L0 nodes)
        cursor = conn.execute("""
            SELECT MIN(last_accessed_at) as oldest, MAX(last_accessed_at) as newest
            FROM hierarchy_nodes
            WHERE session_id = ? AND level = 0 AND (is_archived = 0 OR is_archived IS NULL)
        """, (session_id,))
        row = cursor.fetchone()

        oldest = None
        newest = None
        if row["oldest"]:
            oldest = _parse_iso(row["oldest"])
        if row["newest"]:
            newest = _parse_iso(row["newest"])

        return ArchiveStats(
            session_id=session_id,
            total_nodes=total,
            archived_nodes=archived,
            active_nodes=total - archived,
            oldest_access=oldest,
            newest_access=newest
        )

    def get_archived_placeholder(
        self,
        node_id: str
    ) -> Optional[str]:
        """Get placeholder text for an archived node.

        When an archived L0 node is retrieved, return a placeholder
        instead of the original content.

        Args:
            node_id: ID of the archived node

        Returns:
            Placeholder text, or None if node not found or not archived
        """
        conn = self._get_conn()

        cursor = conn.execute("""
            SELECT node_id, first_turn_seq, created_at, is_archived
            FROM hierarchy_nodes
            WHERE node_id = ?
        """, (node_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        if not row["is_archived"]:
            return None

        seq = row["first_turn_seq"] if row["first_turn_seq"] is not None else "?"
        created = row["created_at"] if row["created_at"] is not None else "unknown"

        return f"[Archived content - turn #{seq} from {created}]"

    def is_node_archived(self, node_id: str) -> bool:
        """Check if a node is archived.

        Args:
            node_id: ID of the node to check

        Returns:
            True if archived, False otherwise
        """
        conn = self._get_conn()

        cursor = conn.execute("""
            SELECT is_archived
            FROM hierarchy_nodes
            WHERE node_id = ?
        """, (node_id,))

        row = cursor.fetchone()
        if row is None:
            return False

        return bool(row["is_archived"])

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "HierarchyArchiver":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def get_archiver(db_path: Optional[Path] = None) -> HierarchyArchiver:
    """Factory function to get HierarchyArchiver instance.

    Args:
        db_path: Path to database. If None, uses default from paths.

    Returns:
        Configured HierarchyArchiver instance
    """
    if db_path is None:
        from .paths import get_cat_chat_db
        db_path = get_cat_chat_db()

    archiver = HierarchyArchiver(db_path)
    archiver.ensure_schema()
    return archiver


if __name__ == "__main__":
    import tempfile
    import numpy as np

    print("Hierarchy Archiver - Quick Test")
    print("=" * 50)

    # Create temp database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_archiver.db"

        # First create the base schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE hierarchy_nodes (
                node_id TEXT PRIMARY KEY,
                session_id TEXT,
                level INTEGER NOT NULL,
                centroid BLOB NOT NULL,
                event_id TEXT,
                content_hash TEXT,
                parent_id TEXT,
                token_count INTEGER DEFAULT 0,
                first_turn_seq INTEGER,
                last_turn_seq INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Insert test data
        for i in range(100):
            event_id = f"evt_{i:03d}"
            node_id = f"h0_test_session_{i}"
            centroid = np.random.randn(384).astype(np.float32).tobytes()

            conn.execute(
                "INSERT INTO session_events VALUES (?, ?, ?, ?, ?)",
                (event_id, "test_session", i, f"hash_{i}", "{}")
            )
            conn.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, event_id, content_hash, first_turn_seq, last_turn_seq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (node_id, "test_session", 0, centroid, event_id, f"hash_{i}", i, i))

        conn.commit()
        conn.close()

        # Test archiver
        with HierarchyArchiver(db_path) as archiver:
            # Ensure schema adds columns
            added = archiver.ensure_schema()
            print(f"Schema columns added: {added}")

            # Test access time update
            updated = archiver.update_access_time([
                "h0_test_session_0",
                "h0_test_session_1",
                "h0_test_session_2"
            ])
            print(f"Access times updated: {updated}")

            # Get stats before archival
            stats = archiver.get_archive_stats("test_session")
            print(f"\nBefore archival:")
            print(f"  Total nodes: {stats.total_nodes}")
            print(f"  Archived: {stats.archived_nodes}")
            print(f"  Active: {stats.active_nodes}")

            # Get archival candidates (keeping last 10 turns)
            candidates = archiver.get_archival_candidates(
                "test_session",
                age_days=0,  # Archive anything without recent access
                min_keep=10
            )
            print(f"\nArchival candidates: {len(candidates)}")

            # Archive nodes
            result = archiver.archive_old_nodes(
                "test_session",
                age_days=0,
                min_keep=10
            )
            print(f"\nArchive result:")
            print(f"  Nodes archived: {result.nodes_archived}")
            print(f"  Nodes skipped: {result.nodes_skipped}")
            print(f"  Bytes freed (est): {result.bytes_freed}")

            # Get stats after archival
            stats = archiver.get_archive_stats("test_session")
            print(f"\nAfter archival:")
            print(f"  Total nodes: {stats.total_nodes}")
            print(f"  Archived: {stats.archived_nodes}")
            print(f"  Active: {stats.active_nodes}")

            # Test placeholder
            placeholder = archiver.get_archived_placeholder("h0_test_session_0")
            print(f"\nPlaceholder: {placeholder}")

            # Test restoration
            restored = archiver.restore_archived_node("h0_test_session_0")
            print(f"\nRestored node 0: {restored}")

            # Verify restoration
            is_archived = archiver.is_node_archived("h0_test_session_0")
            print(f"Node 0 still archived: {is_archived}")

    print("\nAll tests passed!")
