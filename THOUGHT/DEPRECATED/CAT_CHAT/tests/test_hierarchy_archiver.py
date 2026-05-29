#!/usr/bin/env python3
"""
Unit Tests for Hierarchy Archiver (Phase J.5)

Tests for:
- Access time updates
- Archival candidate detection
- Archival execution
- min_keep protection
- Archive statistics
- Restoration
- Placeholder generation
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pytest

from catalytic_chat.hierarchy_archiver import (
    HierarchyArchiver,
    ArchiveResult,
    ArchiveStats,
    _now_iso,
    _parse_iso,
)


def _setup_test_database(conn: sqlite3.Connection, num_nodes: int = 100):
    """Setup test database with hierarchy_nodes and session_events tables.

    Creates:
    - session_events table with mock events
    - hierarchy_nodes table with L0 nodes (no archival columns yet)

    Args:
        conn: SQLite connection
        num_nodes: Number of L0 nodes to create
    """
    # Create session_events table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            sequence_num INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
    """)

    # Create hierarchy_nodes table (base schema without archival columns)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hierarchy_nodes (
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
    for i in range(num_nodes):
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


class TestSchemaEnsure:
    """Test schema migration for archival columns."""

    def test_ensure_schema_adds_columns(self, tmp_path):
        """ensure_schema adds is_archived and last_accessed_at columns."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=10)
        conn.close()

        with HierarchyArchiver(db_path) as archiver:
            added = archiver.ensure_schema()
            assert added is True

            # Check columns exist
            conn = archiver._get_conn()
            cursor = conn.execute("PRAGMA table_info(hierarchy_nodes)")
            columns = {row["name"] for row in cursor.fetchall()}

            assert "is_archived" in columns
            assert "last_accessed_at" in columns

    def test_ensure_schema_idempotent(self, tmp_path):
        """Calling ensure_schema multiple times is safe."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=10)
        conn.close()

        with HierarchyArchiver(db_path) as archiver:
            first_result = archiver.ensure_schema()
            second_result = archiver.ensure_schema()
            third_result = archiver.ensure_schema()

            assert first_result is True  # Added columns
            assert second_result is False  # Already existed
            assert third_result is False  # Already existed


class TestAccessTimeUpdates:
    """Test access time tracking."""

    @pytest.fixture
    def archiver(self, tmp_path):
        """Create archiver with test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=50)
        conn.close()

        arch = HierarchyArchiver(db_path)
        arch.ensure_schema()
        return arch

    def test_update_access_time_single_node(self, archiver):
        """Update access time for a single node."""
        updated = archiver.update_access_time(["h0_test_session_0"])
        assert updated == 1

        # Verify timestamp was set
        conn = archiver._get_conn()
        cursor = conn.execute(
            "SELECT last_accessed_at FROM hierarchy_nodes WHERE node_id = ?",
            ("h0_test_session_0",)
        )
        row = cursor.fetchone()
        assert row["last_accessed_at"] is not None

    def test_update_access_time_multiple_nodes(self, archiver):
        """Update access time for multiple nodes."""
        node_ids = [f"h0_test_session_{i}" for i in range(10)]
        updated = archiver.update_access_time(node_ids)
        assert updated == 10

    def test_update_access_time_empty_list(self, archiver):
        """Update with empty list returns 0."""
        updated = archiver.update_access_time([])
        assert updated == 0

    def test_update_access_time_nonexistent_nodes(self, archiver):
        """Update with nonexistent nodes returns 0."""
        updated = archiver.update_access_time(["nonexistent_node"])
        assert updated == 0


class TestArchivalCandidates:
    """Test archival candidate detection."""

    @pytest.fixture
    def archiver(self, tmp_path):
        """Create archiver with test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=100)
        conn.close()

        arch = HierarchyArchiver(db_path)
        arch.ensure_schema()
        return arch

    def test_get_archival_candidates_respects_min_keep(self, archiver):
        """Candidates respects min_keep parameter."""
        # With min_keep=50, should protect last 50 turns
        candidates = archiver.get_archival_candidates(
            "test_session",
            age_days=0,  # Archive any without recent access
            min_keep=50
        )

        # Should have 50 candidates (nodes 0-49)
        assert len(candidates) == 50

        # Protected nodes should NOT be in candidates
        protected = [f"h0_test_session_{i}" for i in range(50, 100)]
        for node_id in protected:
            assert node_id not in candidates

    def test_get_archival_candidates_respects_age_days(self, archiver):
        """Candidates respects age_days parameter."""
        # Update access time for some nodes to now
        archiver.update_access_time([f"h0_test_session_{i}" for i in range(10)])

        # With age_days=1, recently accessed nodes should NOT be candidates
        candidates = archiver.get_archival_candidates(
            "test_session",
            age_days=1,
            min_keep=10  # Protect last 10
        )

        # Nodes 0-9 were just accessed, should not be candidates
        # (but some may still be candidates if min_keep doesn't protect them)
        recently_accessed = {f"h0_test_session_{i}" for i in range(10)}
        for candidate in candidates:
            assert candidate not in recently_accessed

    def test_get_archival_candidates_empty_when_under_min_keep(self, archiver):
        """No candidates when total nodes < min_keep."""
        # Ask for min_keep=200 but only have 100 nodes
        candidates = archiver.get_archival_candidates(
            "test_session",
            age_days=0,
            min_keep=200
        )
        assert len(candidates) == 0


class TestArchivalExecution:
    """Test actual archival operations."""

    @pytest.fixture
    def archiver(self, tmp_path):
        """Create archiver with test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=100)
        conn.close()

        arch = HierarchyArchiver(db_path)
        arch.ensure_schema()
        return arch

    def test_archive_old_nodes_returns_result(self, archiver):
        """archive_old_nodes returns ArchiveResult."""
        result = archiver.archive_old_nodes(
            "test_session",
            age_days=0,
            min_keep=50
        )

        assert isinstance(result, ArchiveResult)
        assert result.session_id == "test_session"
        assert result.nodes_archived == 50  # First 50 nodes
        assert result.nodes_skipped == 50   # Protected by min_keep
        assert result.bytes_freed > 0

    def test_archive_old_nodes_sets_flag(self, archiver):
        """archive_old_nodes sets is_archived flag."""
        archiver.archive_old_nodes(
            "test_session",
            age_days=0,
            min_keep=90
        )

        # Check that first 10 nodes are archived
        conn = archiver._get_conn()
        cursor = conn.execute("""
            SELECT COUNT(*) as cnt
            FROM hierarchy_nodes
            WHERE session_id = 'test_session' AND is_archived = 1
        """)
        assert cursor.fetchone()["cnt"] == 10

    def test_archive_old_nodes_preserves_centroid(self, archiver):
        """archive_old_nodes preserves centroid data."""
        # Get centroid before archival
        conn = archiver._get_conn()
        cursor = conn.execute(
            "SELECT centroid FROM hierarchy_nodes WHERE node_id = 'h0_test_session_0'"
        )
        centroid_before = cursor.fetchone()["centroid"]

        # Archive
        archiver.archive_old_nodes("test_session", age_days=0, min_keep=50)

        # Get centroid after archival
        cursor = conn.execute(
            "SELECT centroid FROM hierarchy_nodes WHERE node_id = 'h0_test_session_0'"
        )
        centroid_after = cursor.fetchone()["centroid"]

        # Centroid should be unchanged
        assert centroid_before == centroid_after

    def test_archive_old_nodes_idempotent(self, archiver):
        """Archiving twice does not re-archive already archived nodes."""
        result1 = archiver.archive_old_nodes("test_session", age_days=0, min_keep=50)
        result2 = archiver.archive_old_nodes("test_session", age_days=0, min_keep=50)

        assert result1.nodes_archived == 50
        assert result2.nodes_archived == 0  # Already archived


class TestArchiveStats:
    """Test archive statistics."""

    @pytest.fixture
    def archiver(self, tmp_path):
        """Create archiver with test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=100)
        conn.close()

        arch = HierarchyArchiver(db_path)
        arch.ensure_schema()
        return arch

    def test_get_archive_stats_before_archival(self, archiver):
        """Stats show all nodes active before archival."""
        stats = archiver.get_archive_stats("test_session")

        assert isinstance(stats, ArchiveStats)
        assert stats.session_id == "test_session"
        assert stats.total_nodes == 100
        assert stats.archived_nodes == 0
        assert stats.active_nodes == 100

    def test_get_archive_stats_after_archival(self, archiver):
        """Stats reflect archived nodes after archival."""
        archiver.archive_old_nodes("test_session", age_days=0, min_keep=60)

        stats = archiver.get_archive_stats("test_session")

        assert stats.total_nodes == 100
        assert stats.archived_nodes == 40
        assert stats.active_nodes == 60

    def test_get_archive_stats_tracks_access_times(self, archiver):
        """Stats include access time range."""
        # Update access time for some nodes
        archiver.update_access_time(["h0_test_session_50"])

        stats = archiver.get_archive_stats("test_session")

        assert stats.oldest_access is not None
        assert stats.newest_access is not None


class TestRestoration:
    """Test node restoration."""

    @pytest.fixture
    def archiver(self, tmp_path):
        """Create archiver with test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=100)
        conn.close()

        arch = HierarchyArchiver(db_path)
        arch.ensure_schema()
        return arch

    def test_restore_archived_node_success(self, archiver):
        """Restore an archived node when content exists."""
        # Archive first
        archiver.archive_old_nodes("test_session", age_days=0, min_keep=90)

        # Verify node is archived
        assert archiver.is_node_archived("h0_test_session_0") is True

        # Restore
        restored = archiver.restore_archived_node("h0_test_session_0")
        assert restored is True

        # Verify no longer archived
        assert archiver.is_node_archived("h0_test_session_0") is False

    def test_restore_nonexistent_node_fails(self, archiver):
        """Restore fails for nonexistent node."""
        restored = archiver.restore_archived_node("nonexistent_node")
        assert restored is False

    def test_restore_updates_access_time(self, archiver):
        """Restore updates last_accessed_at."""
        archiver.archive_old_nodes("test_session", age_days=0, min_keep=90)
        archiver.restore_archived_node("h0_test_session_0")

        conn = archiver._get_conn()
        cursor = conn.execute(
            "SELECT last_accessed_at FROM hierarchy_nodes WHERE node_id = 'h0_test_session_0'"
        )
        row = cursor.fetchone()
        assert row["last_accessed_at"] is not None


class TestPlaceholder:
    """Test placeholder generation for archived nodes."""

    @pytest.fixture
    def archiver(self, tmp_path):
        """Create archiver with test database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=100)
        conn.close()

        arch = HierarchyArchiver(db_path)
        arch.ensure_schema()
        return arch

    def test_get_archived_placeholder_for_archived_node(self, archiver):
        """Get placeholder for archived node."""
        archiver.archive_old_nodes("test_session", age_days=0, min_keep=90)

        placeholder = archiver.get_archived_placeholder("h0_test_session_0")

        assert placeholder is not None
        assert "[Archived content" in placeholder
        assert "turn #0" in placeholder

    def test_get_archived_placeholder_for_active_node(self, archiver):
        """Active nodes return None for placeholder."""
        placeholder = archiver.get_archived_placeholder("h0_test_session_99")
        assert placeholder is None

    def test_get_archived_placeholder_for_nonexistent_node(self, archiver):
        """Nonexistent nodes return None for placeholder."""
        placeholder = archiver.get_archived_placeholder("nonexistent")
        assert placeholder is None


class TestHelperFunctions:
    """Test helper functions."""

    def test_now_iso_format(self):
        """_now_iso returns valid ISO format."""
        result = _now_iso()
        assert result.endswith("Z")
        assert "T" in result

    def test_parse_iso_with_z(self):
        """_parse_iso handles Z suffix."""
        iso = "2024-01-15T10:30:00Z"
        dt = _parse_iso(iso)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_iso_with_offset(self):
        """_parse_iso handles +00:00 suffix."""
        iso = "2024-01-15T10:30:00+00:00"
        dt = _parse_iso(iso)
        assert dt.year == 2024


class TestContextManager:
    """Test context manager protocol."""

    def test_context_manager_closes_connection(self, tmp_path):
        """Context manager properly closes connection."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _setup_test_database(conn, num_nodes=10)
        conn.close()

        with HierarchyArchiver(db_path) as archiver:
            archiver.ensure_schema()
            archiver.update_access_time(["h0_test_session_0"])

        # Connection should be closed
        assert archiver._conn is None

        # Should be able to open again
        with HierarchyArchiver(db_path) as archiver2:
            stats = archiver2.get_archive_stats("test_session")
            assert stats.total_nodes == 10
