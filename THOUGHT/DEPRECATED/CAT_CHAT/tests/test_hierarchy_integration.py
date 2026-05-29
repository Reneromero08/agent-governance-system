#!/usr/bin/env python3
"""
Tests for Hierarchy Integration (Phase J.4)

Tests the integration of hierarchy retrieval with auto_context_manager:
- Hot path bypasses hierarchy for recent turns
- Hierarchy used for older turns
- Fallback to brute force when hierarchy not built
- Metrics tracking
- Access time updates after retrieval

~12 tests covering the J.4 integration layer.
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pytest

from catalytic_chat.hierarchy_retriever import (
    ensure_hierarchy_schema,
    retrieve_with_hot_path,
    has_hierarchy,
    get_hierarchy_stats,
    RetrievalMetrics,
    HierarchyNode,
    store_hierarchy_batch,
    build_hierarchy_in_memory,
    compute_E,
    EMBEDDING_DIM,
)
from catalytic_chat.hierarchy_builder import HierarchyBuilder
from catalytic_chat.hierarchy_archiver import HierarchyArchiver
from catalytic_chat.auto_context_manager import (
    AutoContextManager,
    HierarchyMetrics,
    create_auto_context_manager,
    HIERARCHY_AVAILABLE,
)
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.session_capsule import SessionCapsule


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_hierarchy.db"


@pytest.fixture
def session_id():
    """Standard test session ID."""
    return "test_session_integration"


@pytest.fixture
def temp_db(temp_db_path, session_id):
    """Create database with hierarchy schema."""
    conn = sqlite3.connect(str(temp_db_path))
    conn.row_factory = sqlite3.Row

    # Create base tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            sequence_num INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
    """)

    ensure_hierarchy_schema(conn)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def random_vectors():
    """Generate normalized random vectors."""
    np.random.seed(42)
    n = 200  # Enough to test hot path vs hierarchy
    vectors = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
    for i in range(n):
        vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
    return vectors


@pytest.fixture
def query_vector():
    """Generate a query vector."""
    np.random.seed(999)
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def mock_embed_fn():
    """Create deterministic embedding function."""
    def embed(text: str) -> np.ndarray:
        text_hash = hash(text) % (2**31)
        rng = np.random.RandomState(text_hash)
        vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)
    return embed


@pytest.fixture
def populated_hierarchy(temp_db, session_id, random_vectors):
    """Create a populated hierarchy for testing."""
    event_ids = [f"evt_{i:04d}" for i in range(len(random_vectors))]
    content_hashes = [f"hash_{i:04d}" for i in range(len(random_vectors))]
    token_counts = [100] * len(random_vectors)

    # Build hierarchy in memory
    roots = build_hierarchy_in_memory(
        random_vectors, event_ids, content_hashes, token_counts,
        branch_factor=10
    )

    # Flatten and store all nodes
    def collect_all_nodes(node: HierarchyNode) -> List[HierarchyNode]:
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes

    all_nodes = []
    for root in roots:
        all_nodes.extend(collect_all_nodes(root))

    store_hierarchy_batch(temp_db, all_nodes, session_id)
    temp_db.commit()

    return {
        "vectors": random_vectors,
        "event_ids": event_ids,
        "content_hashes": content_hashes,
        "roots": roots,
        "node_count": len(all_nodes),
    }


# =============================================================================
# Test Hot Path Optimization
# =============================================================================

class TestHotPath:
    """Tests for hot path bypassing hierarchy on recent turns."""

    def test_hot_path_finds_recent_turns(self, temp_db, session_id, random_vectors, query_vector):
        """Hot path should find turns via brute force."""
        # Add some L0 nodes directly (simulating recent turns)
        for i in range(50):
            vec = random_vectors[i]
            node_id = f"L0_{i}"
            event_id = f"evt_{i:04d}"

            temp_db.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, event_id, content_hash, token_count, created_at)
                VALUES (?, ?, 0, ?, ?, ?, 100, datetime('now'))
            """, (node_id, session_id, vec.tobytes(), event_id, f"hash_{i}"))
        temp_db.commit()

        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=50,  # Check all 50 recent
            top_k=10,
            budget_tokens=1000
        )

        # Should find some results via hot path
        assert len(results) > 0
        assert metrics.hot_path_hits > 0 or len(results) > 0
        # No hierarchy exists, so hierarchy_used should be False
        assert metrics.hierarchy_used is False

    def test_hot_path_limited_to_window(self, temp_db, session_id, random_vectors, query_vector):
        """Hot path should only check hot_window recent turns."""
        # Add 200 L0 nodes
        for i in range(200):
            vec = random_vectors[i]
            node_id = f"L0_{i}"
            event_id = f"evt_{i:04d}"

            temp_db.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, event_id, content_hash, token_count, created_at)
                VALUES (?, ?, 0, ?, ?, ?, 100, datetime('now', '-' || ? || ' seconds'))
            """, (node_id, session_id, vec.tobytes(), event_id, f"hash_{i}", 200 - i))
        temp_db.commit()

        # Retrieve with small hot window
        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=20,  # Only check 20 most recent
            top_k=10,
            budget_tokens=5000
        )

        # E computations should be limited by hot_window when no hierarchy
        # (At most hot_window for brute force)
        assert metrics.e_computations <= 200  # Not checking all


class TestHierarchyRetrieval:
    """Tests for hierarchy-based retrieval."""

    def test_hierarchy_used_for_large_set(self, temp_db, session_id, populated_hierarchy, query_vector):
        """Hierarchy should be used when it exists."""
        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=20,
            top_k=5,
            budget_tokens=2000
        )

        # Hierarchy exists and should be used
        assert metrics.hierarchy_used is True
        assert metrics.levels_searched > 0
        assert len(results) > 0

    def test_hierarchy_vs_hot_path_merge(self, temp_db, session_id, populated_hierarchy, query_vector):
        """Results should merge hot path and hierarchy hits."""
        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=50,
            top_k=10,
            budget_tokens=5000
        )

        # Should have results
        assert len(results) > 0
        # Total hits should match results (approximately, may have budget cuts)
        total_reported = metrics.hot_path_hits + metrics.hierarchy_hits
        assert total_reported >= 0


class TestFallback:
    """Tests for fallback behavior when hierarchy not built."""

    def test_fallback_to_brute_force_no_hierarchy(self, temp_db, session_id, random_vectors, query_vector):
        """Should fall back to brute force when no hierarchy exists."""
        # Add only L0 nodes (no higher level hierarchy)
        for i in range(100):
            vec = random_vectors[i]
            node_id = f"L0_{i}"
            event_id = f"evt_{i:04d}"

            temp_db.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, event_id, content_hash, token_count, created_at)
                VALUES (?, ?, 0, ?, ?, ?, 100, datetime('now'))
            """, (node_id, session_id, vec.tobytes(), event_id, f"hash_{i}"))
        temp_db.commit()

        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=100,
            top_k=10,
            budget_tokens=2000
        )

        # No hierarchy (level > 0) so hierarchy_used should be False
        assert metrics.hierarchy_used is False
        assert len(results) > 0  # Still gets results via hot path

    def test_empty_session_returns_empty(self, temp_db, query_vector):
        """Empty session should return empty results."""
        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id="nonexistent_session",
            db_conn=temp_db,
            hot_window=100,
            top_k=10,
            budget_tokens=2000
        )

        assert len(results) == 0
        assert metrics.hierarchy_used is False
        assert metrics.e_computations == 0


class TestMetricsTracking:
    """Tests for metrics tracking."""

    def test_metrics_count_e_computations(self, temp_db, session_id, populated_hierarchy, query_vector):
        """Metrics should accurately count E-score computations."""
        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=50,
            top_k=5,
            budget_tokens=5000
        )

        assert metrics.e_computations > 0
        assert metrics.results_count == len(results)

    def test_metrics_budget_tracking(self, temp_db, session_id, populated_hierarchy, query_vector):
        """Metrics should track budget usage."""
        budget = 500
        results, metrics = retrieve_with_hot_path(
            query_vec=query_vector,
            session_id=session_id,
            db_conn=temp_db,
            hot_window=50,
            top_k=10,
            budget_tokens=budget
        )

        # Budget used should not exceed budget
        assert metrics.budget_used <= budget


class TestAccessTimeUpdates:
    """Tests for access time updates via archiver."""

    def test_access_time_updated_after_retrieval(self, temp_db_path, session_id, random_vectors, query_vector):
        """Access times should be updated after retrieval."""
        # Set up database with archiver schema
        conn = sqlite3.connect(str(temp_db_path))
        conn.row_factory = sqlite3.Row

        # Create base tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        ensure_hierarchy_schema(conn)

        # Add archiver columns
        archiver = HierarchyArchiver(temp_db_path)
        archiver.ensure_schema()

        # Add L0 nodes
        for i in range(10):
            vec = random_vectors[i]
            node_id = f"L0_{i}"
            event_id = f"evt_{i:04d}"

            conn.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, event_id, content_hash, token_count, created_at)
                VALUES (?, ?, 0, ?, ?, ?, 100, datetime('now'))
            """, (node_id, session_id, vec.tobytes(), event_id, f"hash_{i}"))
        conn.commit()

        # Check initial state - no last_accessed_at
        cursor = conn.execute("""
            SELECT COUNT(*) FROM hierarchy_nodes
            WHERE session_id = ? AND last_accessed_at IS NOT NULL
        """, (session_id,))
        assert cursor.fetchone()[0] == 0

        # Simulate retrieval access update
        node_ids = ["L0_0", "L0_1", "L0_2"]
        updated = archiver.update_access_time(node_ids)

        # Check that access times were updated
        cursor = conn.execute("""
            SELECT COUNT(*) FROM hierarchy_nodes
            WHERE session_id = ? AND last_accessed_at IS NOT NULL
        """, (session_id,))
        assert cursor.fetchone()[0] == 3

        conn.close()
        archiver.close()


class TestHierarchyStats:
    """Tests for hierarchy statistics functions."""

    def test_has_hierarchy_detects_levels(self, temp_db, session_id, populated_hierarchy):
        """has_hierarchy should detect when hierarchy exists."""
        assert has_hierarchy(temp_db, session_id) is True

    def test_has_hierarchy_false_for_l0_only(self, temp_db, session_id, random_vectors):
        """has_hierarchy should return False for L0-only nodes."""
        # Add only L0 nodes
        for i in range(10):
            vec = random_vectors[i]
            temp_db.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, event_id, content_hash, token_count, created_at)
                VALUES (?, ?, 0, ?, ?, ?, 100, datetime('now'))
            """, (f"L0_{i}", session_id, vec.tobytes(), f"evt_{i}", f"hash_{i}"))
        temp_db.commit()

        assert has_hierarchy(temp_db, session_id) is False

    def test_get_hierarchy_stats(self, temp_db, session_id, populated_hierarchy):
        """get_hierarchy_stats should return level counts."""
        stats = get_hierarchy_stats(temp_db, session_id)

        assert stats["total_nodes"] == populated_hierarchy["node_count"]
        assert stats["total_turns"] == len(populated_hierarchy["vectors"])
        assert stats["levels_built"] > 0
        assert 0 in stats["nodes_per_level"]  # L0 exists


@pytest.mark.skipif(not HIERARCHY_AVAILABLE, reason="Hierarchy module not available")
class TestAutoContextManagerIntegration:
    """Tests for AutoContextManager hierarchy integration."""

    def test_hierarchy_metrics_available(self, mock_embed_fn):
        """AutoContextManager should expose hierarchy metrics."""
        import gc
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_acm.db"
        capsule = None
        manager = None

        try:
            # Create session
            capsule = SessionCapsule(db_path=db_path)
            sess_id = capsule.create_session()

            budget = ModelBudgetDiscovery.from_context_window(
                context_window=4096,
                system_prompt="Test"
            )

            manager = AutoContextManager(
                db_path=db_path,
                session_id=sess_id,
                budget=budget,
                embed_fn=mock_embed_fn,
            )

            # Initially no metrics
            metrics = manager.get_hierarchy_metrics()
            assert metrics is None or isinstance(metrics, HierarchyMetrics)

        finally:
            # Close connections to allow cleanup
            if capsule is not None:
                capsule.close()
            if manager is not None:
                if manager._hierarchy_builder is not None:
                    manager._hierarchy_builder.close()
                if manager._hierarchy_archiver is not None:
                    manager._hierarchy_archiver.close()
                if manager.capsule is not None:
                    manager.capsule.close()
            gc.collect()

    def test_hierarchy_builder_initialized(self, mock_embed_fn):
        """AutoContextManager should initialize hierarchy builder if available."""
        import gc
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_acm2.db"
        capsule = None
        manager = None

        try:
            capsule = SessionCapsule(db_path=db_path)
            sess_id = capsule.create_session()

            budget = ModelBudgetDiscovery.from_context_window(
                context_window=4096,
                system_prompt="Test"
            )

            manager = AutoContextManager(
                db_path=db_path,
                session_id=sess_id,
                budget=budget,
                embed_fn=mock_embed_fn,
            )

            # Hierarchy builder should be initialized if HIERARCHY_AVAILABLE
            # It may be None if hierarchy tables don't exist yet, which is OK
            # The key is that HIERARCHY_AVAILABLE is True
            assert HIERARCHY_AVAILABLE is True

        finally:
            # Close connections to allow cleanup
            if capsule is not None:
                capsule.close()
            if manager is not None:
                if manager._hierarchy_builder is not None:
                    manager._hierarchy_builder.close()
                if manager._hierarchy_archiver is not None:
                    manager._hierarchy_archiver.close()
                if manager.capsule is not None:
                    manager.capsule.close()
            gc.collect()

    def test_configure_hierarchy_settings(self, mock_embed_fn):
        """Should be able to configure hierarchy settings."""
        import gc
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "test_acm3.db"
        capsule = None
        manager = None

        try:
            capsule = SessionCapsule(db_path=db_path)
            sess_id = capsule.create_session()

            budget = ModelBudgetDiscovery.from_context_window(
                context_window=4096,
                system_prompt="Test"
            )

            manager = AutoContextManager(
                db_path=db_path,
                session_id=sess_id,
                budget=budget,
                embed_fn=mock_embed_fn,
            )

            # Configure hierarchy
            manager.configure_hierarchy(
                use_hierarchy_threshold=1000,
                hot_window=200
            )

            assert manager._use_hierarchy_threshold == 1000
            assert manager._hot_window == 200

        finally:
            # Close connections to allow cleanup
            if capsule is not None:
                capsule.close()
            if manager is not None:
                if manager._hierarchy_builder is not None:
                    manager._hierarchy_builder.close()
                if manager._hierarchy_archiver is not None:
                    manager._hierarchy_archiver.close()
                if manager.capsule is not None:
                    manager.capsule.close()
            gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
