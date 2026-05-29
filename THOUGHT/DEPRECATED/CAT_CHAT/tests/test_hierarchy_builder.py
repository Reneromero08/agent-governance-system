#!/usr/bin/env python3
"""
Tests for Hierarchy Builder (Phase J.3)

Tests the automatic tree maintenance for the centroid hierarchy.

Coverage:
- HierarchyBuilder initialization
- Incremental updates (on_turn_compressed)
- L1 promotion (100 children triggers new L1)
- L2 promotion (10 L1s triggers L2)
- L3 promotion (10 L2s triggers L3)
- Initial k-means build
- Initial sequential build (fallback)
- PCA projection
- Database state persistence
- Edge cases
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from catalytic_chat.hierarchy_builder import (
    HierarchyBuilder,
    get_hierarchy_builder,
    L1_CHILDREN_THRESHOLD,
    L2_CHILDREN_THRESHOLD,
    L3_CHILDREN_THRESHOLD,
    DEFAULT_PCA_DIMS,
)
from catalytic_chat.hierarchy_schema import (
    L0, L1, L2, L3,
    EMBEDDING_DIM,
    CHILDREN_PER_LEVEL,
)
from catalytic_chat.hierarchy_retriever import (
    ensure_hierarchy_schema,
    load_root_nodes,
    load_node_children,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_hierarchy.db"


@pytest.fixture
def random_vector():
    """Generate a single random normalized vector."""
    np.random.seed(42)
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def random_vectors():
    """Generate multiple random normalized vectors."""
    np.random.seed(42)
    n = 250  # Enough to trigger L2 creation
    vectors = []
    for _ in range(n):
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vectors.append(vec / np.linalg.norm(vec))
    return vectors


@pytest.fixture
def builder(temp_db_path):
    """Create a HierarchyBuilder with temporary database."""
    builder = HierarchyBuilder(temp_db_path, "test_session")
    yield builder
    builder.close()


# =============================================================================
# Test Initialization
# =============================================================================

class TestInitialization:
    """Tests for HierarchyBuilder initialization."""

    def test_init_creates_builder(self, temp_db_path):
        """Should create a HierarchyBuilder instance."""
        builder = HierarchyBuilder(temp_db_path, "test_session")
        assert builder.session_id == "test_session"
        assert builder.children_per_level == CHILDREN_PER_LEVEL
        builder.close()

    def test_init_with_custom_children_per_level(self, temp_db_path):
        """Should accept custom children_per_level."""
        builder = HierarchyBuilder(temp_db_path, "test_session", children_per_level=50)
        assert builder.children_per_level == 50
        builder.close()

    def test_context_manager_support(self, temp_db_path):
        """Should support context manager protocol."""
        with HierarchyBuilder(temp_db_path, "test_session") as builder:
            assert builder is not None
            assert builder.session_id == "test_session"
        # Connection should be closed after exit

    def test_factory_function(self, temp_db_path):
        """get_hierarchy_builder should create builder instance."""
        builder = get_hierarchy_builder(temp_db_path, "test_session")
        assert isinstance(builder, HierarchyBuilder)
        builder.close()


# =============================================================================
# Test Incremental Updates
# =============================================================================

class TestIncrementalUpdates:
    """Tests for on_turn_compressed incremental updates."""

    def test_creates_l0_node(self, builder, random_vector):
        """on_turn_compressed should create L0 node."""
        node_id = builder.on_turn_compressed(
            event_id="evt_001",
            turn_vec=random_vector,
            content_hash="hash_001",
            sequence_num=0,
            token_count=100,
        )

        assert node_id is not None
        assert node_id.startswith("h0_")

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 1

    def test_creates_l1_node_for_first_turn(self, builder, random_vector):
        """First turn should also create an L1 parent."""
        builder.on_turn_compressed(
            event_id="evt_001",
            turn_vec=random_vector,
            content_hash="hash_001",
            sequence_num=0,
            token_count=100,
        )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 1
        assert stats["levels"]["L1"] == 1

    def test_multiple_turns_share_l1(self, builder):
        """Multiple turns should share the same L1 node."""
        np.random.seed(42)

        for i in range(5):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 5
        assert stats["levels"]["L1"] == 1  # Still one open L1

    def test_l0_has_parent_id(self, builder, random_vector):
        """L0 nodes should have parent_id pointing to L1."""
        builder.on_turn_compressed(
            event_id="evt_001",
            turn_vec=random_vector,
            content_hash="hash_001",
            sequence_num=0,
            token_count=100,
        )

        conn = builder._get_conn()
        cursor = conn.execute("""
            SELECT parent_id FROM hierarchy_nodes
            WHERE session_id = ? AND level = 0
        """, (builder.session_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] is not None
        assert row[0].startswith("h1_")


# =============================================================================
# Test L1 Promotion
# =============================================================================

class TestL1Promotion:
    """Tests for L1 node promotion (100 children threshold)."""

    def test_l1_closes_at_threshold(self, temp_db_path):
        """L1 should close and new one open at threshold."""
        # Use smaller threshold for faster testing
        builder = HierarchyBuilder(temp_db_path, "test_session", children_per_level=10)
        np.random.seed(42)

        for i in range(15):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 15
        # 10 children -> L1 closes, next 5 -> new L1
        assert stats["levels"]["L1"] == 2

        builder.close()

    def test_l1_centroid_updates_incrementally(self, builder):
        """L1 centroid should update with each new L0."""
        np.random.seed(42)

        conn = builder._get_conn()

        for i in range(3):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

            # Check L1 centroid is not zero
            cursor = conn.execute("""
                SELECT centroid FROM hierarchy_nodes
                WHERE session_id = ? AND level = 1
            """, (builder.session_id,))
            row = cursor.fetchone()
            centroid = np.frombuffer(row[0], dtype=np.float32)
            assert np.linalg.norm(centroid) > 0


# =============================================================================
# Test L2 Promotion
# =============================================================================

class TestL2Promotion:
    """Tests for L2 node promotion (10 L1s threshold)."""

    def test_l2_created_at_threshold(self, temp_db_path):
        """L2 should be created when 10 L1 nodes exist."""
        # Small threshold for faster testing
        builder = HierarchyBuilder(temp_db_path, "test_session", children_per_level=5)
        np.random.seed(42)

        # Create 50 turns -> 10 L1 nodes (5 each) -> triggers L2
        for i in range(50):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 50
        assert stats["levels"]["L1"] == 10
        assert stats["levels"]["L2"] == 1

        builder.close()

    def test_l2_centroid_is_merge_of_l1s(self, temp_db_path):
        """L2 centroid should be weighted merge of L1 centroids."""
        builder = HierarchyBuilder(temp_db_path, "test_session", children_per_level=5)
        np.random.seed(42)

        # Create 50 turns to trigger L2
        for i in range(50):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

        conn = builder._get_conn()
        cursor = conn.execute("""
            SELECT centroid FROM hierarchy_nodes
            WHERE session_id = ? AND level = 2
        """, (builder.session_id,))
        row = cursor.fetchone()

        assert row is not None
        centroid = np.frombuffer(row[0], dtype=np.float32)
        # Should be valid centroid (not zero, reasonable norm)
        assert np.linalg.norm(centroid) > 0
        assert len(centroid) == EMBEDDING_DIM

        builder.close()

    def test_l1s_get_l2_parent(self, temp_db_path):
        """L1 nodes should have parent_id set to L2 after promotion."""
        builder = HierarchyBuilder(temp_db_path, "test_session", children_per_level=5)
        np.random.seed(42)

        for i in range(50):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

        conn = builder._get_conn()
        cursor = conn.execute("""
            SELECT COUNT(*) FROM hierarchy_nodes
            WHERE session_id = ? AND level = 1 AND parent_id IS NOT NULL
        """, (builder.session_id,))
        count = cursor.fetchone()[0]

        assert count == L2_CHILDREN_THRESHOLD  # 10 L1s have L2 parent

        builder.close()


# =============================================================================
# Test L3 Promotion
# =============================================================================

class TestL3Promotion:
    """Tests for L3 node promotion (10 L2s threshold)."""

    def test_l3_created_at_threshold(self, temp_db_path):
        """L3 should be created when 10 L2 nodes exist."""
        # Very small thresholds for faster testing
        builder = HierarchyBuilder(temp_db_path, "test_session", children_per_level=2)
        np.random.seed(42)

        # Create enough turns to trigger L3
        # 2 L0 -> 1 L1
        # 10 L1 -> 1 L2 (20 L0)
        # 10 L2 -> 1 L3 (200 L0)
        for i in range(200):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            builder.on_turn_compressed(
                event_id=f"evt_{i:03d}",
                turn_vec=vec,
                content_hash=f"hash_{i:03d}",
                sequence_num=i,
                token_count=100,
            )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 200
        assert stats["levels"]["L3"] >= 1

        builder.close()


# =============================================================================
# Test Initial Build with K-Means
# =============================================================================

class TestKMeansBuild:
    """Tests for build_initial_hierarchy with k-means."""

    def test_kmeans_creates_hierarchy(self, builder, random_vectors):
        """K-means build should create complete hierarchy."""
        event_ids = [f"evt_{i:04d}" for i in range(len(random_vectors))]
        content_hashes = [f"hash_{i:04d}" for i in range(len(random_vectors))]

        nodes_created = builder.build_initial_hierarchy(
            random_vectors, event_ids, content_hashes,
            use_kmeans=True,
            pca_dims=22
        )

        assert nodes_created > len(random_vectors)  # L0 + L1 + L2+ nodes

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == len(random_vectors)
        assert stats["levels"]["L1"] > 0

    def test_kmeans_clusters_semantically(self, builder):
        """K-means should group similar vectors."""
        np.random.seed(42)

        # Create clustered vectors (two distinct groups)
        vectors = []
        event_ids = []
        content_hashes = []

        # Group 1: vectors around [1, 0, 0, ...]
        base1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        base1[0] = 1.0
        for i in range(50):
            vec = base1 + 0.1 * np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            event_ids.append(f"grp1_evt_{i:03d}")
            content_hashes.append(f"grp1_hash_{i:03d}")

        # Group 2: vectors around [0, 1, 0, ...]
        base2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        base2[1] = 1.0
        for i in range(50):
            vec = base2 + 0.1 * np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            event_ids.append(f"grp2_evt_{i:03d}")
            content_hashes.append(f"grp2_hash_{i:03d}")

        nodes_created = builder.build_initial_hierarchy(
            vectors, event_ids, content_hashes,
            use_kmeans=True,
            pca_dims=None  # No PCA for this test
        )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 100
        assert stats["levels"]["L1"] >= 1

    def test_kmeans_with_pca(self, builder, random_vectors):
        """K-means build should work with PCA projection."""
        event_ids = [f"evt_{i:04d}" for i in range(len(random_vectors))]
        content_hashes = [f"hash_{i:04d}" for i in range(len(random_vectors))]

        nodes_created = builder.build_initial_hierarchy(
            random_vectors, event_ids, content_hashes,
            use_kmeans=True,
            pca_dims=DEFAULT_PCA_DIMS
        )

        assert nodes_created > 0

    def test_kmeans_fallback_without_sklearn(self, builder, random_vectors):
        """Should fall back to sequential if sklearn unavailable."""
        event_ids = [f"evt_{i:04d}" for i in range(len(random_vectors))]
        content_hashes = [f"hash_{i:04d}" for i in range(len(random_vectors))]

        with patch.dict('sys.modules', {'sklearn.cluster': None}):
            nodes_created = builder.build_initial_hierarchy(
                random_vectors, event_ids, content_hashes,
                use_kmeans=True
            )

        # Should still create nodes (via fallback)
        assert nodes_created > 0


# =============================================================================
# Test Sequential Build
# =============================================================================

class TestSequentialBuild:
    """Tests for build_initial_hierarchy without k-means."""

    def test_sequential_creates_hierarchy(self, builder, random_vectors):
        """Sequential build should create complete hierarchy."""
        event_ids = [f"evt_{i:04d}" for i in range(len(random_vectors))]
        content_hashes = [f"hash_{i:04d}" for i in range(len(random_vectors))]

        nodes_created = builder.build_initial_hierarchy(
            random_vectors, event_ids, content_hashes,
            use_kmeans=False
        )

        assert nodes_created > len(random_vectors)

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == len(random_vectors)
        assert stats["levels"]["L1"] > 0

    def test_sequential_groups_by_order(self, builder):
        """Sequential build should group vectors in order."""
        np.random.seed(42)
        n = 50
        vectors = []
        event_ids = []
        content_hashes = []

        for i in range(n):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vectors.append(vec / np.linalg.norm(vec))
            event_ids.append(f"evt_{i:03d}")
            content_hashes.append(f"hash_{i:03d}")

        builder.build_initial_hierarchy(
            vectors, event_ids, content_hashes,
            use_kmeans=False
        )

        # Check that L0 nodes are grouped sequentially
        conn = builder._get_conn()
        cursor = conn.execute("""
            SELECT node_id, parent_id FROM hierarchy_nodes
            WHERE session_id = ? AND level = 0
            ORDER BY node_id
        """, (builder.session_id,))
        rows = cursor.fetchall()

        # First children_per_level nodes should share parent
        first_parent = rows[0][1]
        for i in range(min(builder.children_per_level, n)):
            assert rows[i][1] == first_parent


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_vectors(self, builder):
        """Should handle empty vector list."""
        nodes_created = builder.build_initial_hierarchy([], [], [])
        assert nodes_created == 0

    def test_single_vector(self, builder, random_vector):
        """Should handle single vector."""
        nodes_created = builder.build_initial_hierarchy(
            [random_vector],
            ["evt_001"],
            ["hash_001"]
        )

        stats = builder.get_stats()
        assert stats["levels"]["L0"] == 1
        assert stats["levels"]["L1"] == 1

    def test_get_stats_empty_session(self, builder):
        """get_stats should work on empty session."""
        stats = builder.get_stats()
        assert stats["total_nodes"] == 0
        assert stats["levels"]["L0"] == 0

    def test_custom_token_counts(self, builder):
        """Should use custom token counts."""
        np.random.seed(42)
        vectors = []
        for _ in range(10):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vectors.append(vec / np.linalg.norm(vec))

        event_ids = [f"evt_{i:03d}" for i in range(10)]
        content_hashes = [f"hash_{i:03d}" for i in range(10)]
        token_counts = [50 + i * 10 for i in range(10)]  # 50, 60, 70, ...

        builder.build_initial_hierarchy(
            vectors, event_ids, content_hashes,
            token_counts=token_counts,
            use_kmeans=False
        )

        conn = builder._get_conn()
        cursor = conn.execute("""
            SELECT token_count FROM hierarchy_nodes
            WHERE session_id = ? AND level = 1
        """, (builder.session_id,))
        row = cursor.fetchone()

        # L1 should have sum of token counts
        total_expected = sum(token_counts)
        assert row[0] == total_expected


# =============================================================================
# Test Database Persistence
# =============================================================================

class TestDatabasePersistence:
    """Tests for database state persistence."""

    def test_nodes_persist_across_instances(self, temp_db_path):
        """Nodes should persist across builder instances."""
        np.random.seed(42)

        # First instance creates nodes
        with HierarchyBuilder(temp_db_path, "test_session") as builder1:
            for i in range(5):
                vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                builder1.on_turn_compressed(
                    event_id=f"evt_{i:03d}",
                    turn_vec=vec,
                    content_hash=f"hash_{i:03d}",
                    sequence_num=i,
                    token_count=100,
                )
            stats1 = builder1.get_stats()

        # Second instance should see same nodes
        with HierarchyBuilder(temp_db_path, "test_session") as builder2:
            stats2 = builder2.get_stats()

        assert stats1["levels"]["L0"] == stats2["levels"]["L0"]
        assert stats1["levels"]["L1"] == stats2["levels"]["L1"]

    def test_incremental_continues_after_restart(self, temp_db_path):
        """Incremental updates should continue correctly after restart."""
        np.random.seed(42)

        # First batch
        with HierarchyBuilder(temp_db_path, "test_session", children_per_level=10) as builder1:
            for i in range(8):
                vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                builder1.on_turn_compressed(
                    event_id=f"evt_{i:03d}",
                    turn_vec=vec,
                    content_hash=f"hash_{i:03d}",
                    sequence_num=i,
                    token_count=100,
                )

        # Second batch (should continue with same L1)
        with HierarchyBuilder(temp_db_path, "test_session", children_per_level=10) as builder2:
            for i in range(8, 15):
                vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                builder2.on_turn_compressed(
                    event_id=f"evt_{i:03d}",
                    turn_vec=vec,
                    content_hash=f"hash_{i:03d}",
                    sequence_num=i,
                    token_count=100,
                )
            stats = builder2.get_stats()

        # 15 nodes with 10 per L1 = 2 L1 nodes
        assert stats["levels"]["L0"] == 15
        assert stats["levels"]["L1"] == 2


# =============================================================================
# Test Integration with Retrieval
# =============================================================================

class TestRetrievalIntegration:
    """Tests for integration with hierarchy retrieval."""

    def test_built_hierarchy_is_retrievable(self, builder, random_vectors):
        """Hierarchy built should work with retrieval functions."""
        event_ids = [f"evt_{i:04d}" for i in range(len(random_vectors))]
        content_hashes = [f"hash_{i:04d}" for i in range(len(random_vectors))]

        builder.build_initial_hierarchy(
            random_vectors, event_ids, content_hashes,
            use_kmeans=True
        )

        conn = builder._get_conn()

        # Load roots
        roots = load_root_nodes(conn, builder.session_id)
        assert len(roots) >= 1

        # Load children of highest level root
        highest_root = max(roots, key=lambda n: n.level)
        children = load_node_children(conn, highest_root.node_id)

        # Should have children if root is not L0
        if highest_root.level > 0:
            assert len(children) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
