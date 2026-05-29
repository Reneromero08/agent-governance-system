#!/usr/bin/env python3
"""
Unit Tests for Hierarchy Schema (Phase J.1)

Tests for:
- HierarchyNode dataclass creation and validation
- HierarchyNode serialization/deserialization
- Centroid math functions
- Schema creation in database

Target: ~15 tests
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from catalytic_chat.hierarchy_schema import (
    HierarchyNode,
    L0, L1, L2, L3,
    CHILDREN_PER_LEVEL,
    EMBEDDING_DIM,
    EMBEDDING_BYTES,
    LEVEL_NAMES,
    generate_node_id,
    get_turns_per_level,
)
from catalytic_chat.centroid_math import (
    compute_centroid,
    update_centroid_incremental,
    compute_E,
    batch_compute_E,
    merge_centroids,
    compute_variance,
    normalize_vector,
)
from catalytic_chat.session_capsule import SessionCapsule


# ============================================================================
# HierarchyNode Tests
# ============================================================================

class TestHierarchyNodeCreation:
    """Test HierarchyNode dataclass creation."""

    def test_create_l0_node(self):
        """Create a valid L0 (turn) node."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h0_session_1",
            session_id="session_1",
            level=L0,
            centroid=centroid,
            turn_count=1,
            first_turn_seq=1,
            last_turn_seq=1,
        )

        assert node.level == L0
        assert node.level_name == "turn"
        assert node.max_turns == 1
        assert node.turn_count == 1

    def test_create_l1_node(self):
        """Create a valid L1 (century) node."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h1_session_1",
            session_id="session_1",
            level=L1,
            centroid=centroid,
            turn_count=50,
            first_turn_seq=1,
            last_turn_seq=50,
        )

        assert node.level == L1
        assert node.level_name == "century"
        assert node.max_turns == 100
        assert not node.is_full

    def test_create_l2_node(self):
        """Create a valid L2 (millennium) node."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h2_session_1",
            session_id="session_1",
            level=L2,
            centroid=centroid,
            turn_count=1000,
        )

        assert node.level == L2
        assert node.level_name == "millennium"
        assert node.max_turns == 10000
        assert not node.is_full

    def test_create_l3_node(self):
        """Create a valid L3 (epoch) node."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h3_session_1",
            session_id="session_1",
            level=L3,
            centroid=centroid,
            turn_count=10000,
        )

        assert node.level == L3
        assert node.level_name == "epoch"
        assert node.max_turns == 1000000

    def test_invalid_level_raises(self):
        """Creating node with invalid level raises ValueError."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with pytest.raises(ValueError, match="Invalid level"):
            HierarchyNode(
                node_id="h5_invalid",
                session_id="session_1",
                level=5,  # Invalid
                centroid=centroid,
            )

    def test_invalid_centroid_shape_raises(self):
        """Creating node with wrong centroid shape raises ValueError."""
        bad_centroid = np.random.randn(128).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match="centroid must have shape"):
            HierarchyNode(
                node_id="h0_bad",
                session_id="session_1",
                level=L0,
                centroid=bad_centroid,
            )

    def test_invalid_centroid_type_raises(self):
        """Creating node with non-array centroid raises TypeError."""
        with pytest.raises(TypeError, match="centroid must be numpy array"):
            HierarchyNode(
                node_id="h0_bad",
                session_id="session_1",
                level=L0,
                centroid=[1.0] * EMBEDDING_DIM,  # List, not array
            )

    def test_is_full_property(self):
        """Test is_full property at various capacities."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        # L1 node at capacity
        full_node = HierarchyNode(
            node_id="h1_full",
            session_id="session_1",
            level=L1,
            centroid=centroid,
            turn_count=100,
        )
        assert full_node.is_full

        # L1 node not at capacity
        partial_node = HierarchyNode(
            node_id="h1_partial",
            session_id="session_1",
            level=L1,
            centroid=centroid,
            turn_count=99,
        )
        assert not partial_node.is_full


class TestHierarchyNodeSerialization:
    """Test HierarchyNode serialization and deserialization."""

    def test_serialize_centroid(self):
        """Serialize centroid to bytes."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h0_test",
            session_id="session_1",
            level=L0,
            centroid=centroid,
        )

        blob = node.serialize_centroid()

        assert isinstance(blob, bytes)
        assert len(blob) == EMBEDDING_BYTES

    def test_deserialize_centroid(self):
        """Deserialize centroid from bytes."""
        original = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        blob = original.tobytes()

        recovered = HierarchyNode.deserialize_centroid(blob)

        np.testing.assert_array_equal(original, recovered)

    def test_deserialize_wrong_size_raises(self):
        """Deserializing wrong-sized blob raises ValueError."""
        bad_blob = b"too short"

        with pytest.raises(ValueError, match="Expected"):
            HierarchyNode.deserialize_centroid(bad_blob)

    def test_to_dict_from_dict_roundtrip(self):
        """Convert to dict and back preserves data."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h1_test",
            session_id="session_1",
            level=L1,
            centroid=centroid,
            parent_id="h2_parent",
            turn_count=50,
            first_turn_seq=1,
            last_turn_seq=50,
        )

        data = node.to_dict()
        recovered = HierarchyNode.from_dict(data)

        assert recovered == node

    def test_to_json_from_json_roundtrip(self):
        """Convert to JSON and back preserves data."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        node = HierarchyNode(
            node_id="h1_test",
            session_id="session_1",
            level=L1,
            centroid=centroid,
            turn_count=50,
        )

        json_str = node.to_json()
        recovered = HierarchyNode.from_json(json_str)

        assert recovered == node


class TestHierarchyHelpers:
    """Test hierarchy helper functions."""

    def test_generate_node_id(self):
        """Generate deterministic node IDs."""
        node_id = generate_node_id("session_abc", L2, 42)

        assert node_id == "h2_session_abc_42"

    def test_get_turns_per_level(self):
        """Get correct turn counts per level."""
        assert get_turns_per_level(L0) == 1
        assert get_turns_per_level(L1) == 100
        assert get_turns_per_level(L2) == 10000
        assert get_turns_per_level(L3) == 1000000

    def test_get_turns_per_level_invalid_raises(self):
        """Negative level raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            get_turns_per_level(-1)

    def test_level_constants(self):
        """Level constants have expected values."""
        assert L0 == 0
        assert L1 == 1
        assert L2 == 2
        assert L3 == 3
        assert CHILDREN_PER_LEVEL == 100


# ============================================================================
# Centroid Math Tests
# ============================================================================

class TestComputeCentroid:
    """Test compute_centroid function."""

    def test_single_vector(self):
        """Centroid of single vector is itself."""
        v = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        centroid = compute_centroid([v])

        np.testing.assert_array_almost_equal(centroid, v)

    def test_multiple_vectors(self):
        """Centroid of multiple vectors is their mean."""
        v1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        v2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        v3 = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        centroid = compute_centroid([v1, v2, v3])
        expected = (v1 + v2 + v3) / 3

        np.testing.assert_array_almost_equal(centroid, expected)

    def test_empty_list_raises(self):
        """Empty vector list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_centroid([])

    def test_wrong_shape_raises(self):
        """Wrong vector shape raises ValueError."""
        bad_v = np.random.randn(128).astype(np.float32)

        with pytest.raises(ValueError, match="shape"):
            compute_centroid([bad_v])


class TestUpdateCentroidIncremental:
    """Test update_centroid_incremental function."""

    def test_incremental_equals_batch(self):
        """Incremental update produces same result as batch computation."""
        vectors = [np.random.randn(EMBEDDING_DIM).astype(np.float32) for _ in range(10)]

        # Batch computation
        batch_centroid = compute_centroid(vectors)

        # Incremental computation
        incremental = vectors[0].copy()
        for i, v in enumerate(vectors[1:], start=1):
            incremental = update_centroid_incremental(incremental, i, v)

        np.testing.assert_array_almost_equal(incremental, batch_centroid)

    def test_first_vector(self):
        """First vector becomes the centroid."""
        old = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        new = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        result = update_centroid_incremental(old, 0, new)

        np.testing.assert_array_equal(result, new)

    def test_negative_count_raises(self):
        """Negative old_count raises ValueError."""
        old = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        new = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        with pytest.raises(ValueError, match="non-negative"):
            update_centroid_incremental(old, -1, new)


class TestComputeE:
    """Test E-score (Born rule) computation."""

    def test_identical_vectors(self):
        """Identical vectors have E = 1."""
        v = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        E = compute_E(v, v)

        assert abs(E - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have E = 0."""
        v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0

        E = compute_E(v1, v2)

        assert abs(E) < 1e-5

    def test_anti_parallel_vectors(self):
        """Anti-parallel vectors have E = 1 (squared cosine)."""
        v = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        E = compute_E(v, -v)

        assert abs(E - 1.0) < 1e-5

    def test_e_in_range(self):
        """E-score is always in [0, 1]."""
        for _ in range(100):
            v1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            v2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)

            E = compute_E(v1, v2)

            assert 0 <= E <= 1

    def test_zero_vector(self):
        """Zero vector returns E = 0."""
        v = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        E = compute_E(v, zero)

        assert E == 0.0


class TestBatchComputeE:
    """Test batch E-score computation."""

    def test_batch_matches_individual(self):
        """Batch E-scores match individual computations."""
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        items = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)

        batch_E = batch_compute_E(query, items)

        for i in range(len(items)):
            individual_E = compute_E(query, items[i])
            assert abs(batch_E[i] - individual_E) < 1e-5

    def test_empty_items(self):
        """Empty items returns empty array."""
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        items = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        result = batch_compute_E(query, items)

        assert len(result) == 0


class TestMergeCentroids:
    """Test merge_centroids function."""

    def test_merge_two_centroids(self):
        """Merge two centroids with different weights."""
        c1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        c2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        merged, total = merge_centroids([(c1, 100), (c2, 200)])

        expected = (c1 * 100 + c2 * 200) / 300
        assert total == 300
        np.testing.assert_array_almost_equal(merged, expected)

    def test_merge_single_centroid(self):
        """Merging single centroid returns itself."""
        c = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        merged, total = merge_centroids([(c, 100)])

        assert total == 100
        np.testing.assert_array_almost_equal(merged, c)

    def test_merge_empty_raises(self):
        """Merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            merge_centroids([])


class TestComputeVariance:
    """Test compute_variance function."""

    def test_variance_at_centroid(self):
        """Variance around actual centroid is meaningful."""
        vectors = [np.random.randn(EMBEDDING_DIM).astype(np.float32) for _ in range(10)]
        centroid = compute_centroid(vectors)

        variance = compute_variance(centroid, vectors)

        assert variance >= 0

    def test_variance_empty_vectors(self):
        """Variance with no vectors is zero."""
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)

        variance = compute_variance(centroid, [])

        assert variance == 0.0


class TestNormalizeVector:
    """Test normalize_vector function."""

    def test_normalizes_to_unit_length(self):
        """Normalized vector has unit length."""
        v = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 5

        normed = normalize_vector(v)

        assert abs(np.linalg.norm(normed) - 1.0) < 1e-5

    def test_zero_vector_unchanged(self):
        """Zero vector remains zero."""
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        normed = normalize_vector(zero)

        np.testing.assert_array_equal(normed, zero)


# ============================================================================
# Schema Tests
# ============================================================================

class TestHierarchySchema:
    """Test session_hierarchy_nodes table creation."""

    @pytest.fixture
    def capsule(self, tmp_path):
        """Create SessionCapsule with temp database."""
        db_path = tmp_path / "test.db"
        return SessionCapsule(db_path=db_path)

    def test_table_created(self, capsule):
        """session_hierarchy_nodes table is created."""
        conn = capsule._get_conn()
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='session_hierarchy_nodes'
        """)

        assert cursor.fetchone() is not None

    def test_indexes_created(self, capsule):
        """Hierarchy indexes are created."""
        conn = capsule._get_conn()

        # Check session index
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_hierarchy_session'
        """)
        assert cursor.fetchone() is not None

        # Check level index
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_hierarchy_level'
        """)
        assert cursor.fetchone() is not None

        # Check parent index
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_hierarchy_parent'
        """)
        assert cursor.fetchone() is not None

    def test_insert_and_retrieve_node(self, capsule):
        """Insert and retrieve a hierarchy node."""
        conn = capsule._get_conn()
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        blob = centroid.tobytes()

        # Insert
        conn.execute("""
            INSERT INTO session_hierarchy_nodes
            (node_id, session_id, level, centroid, turn_count, first_turn_seq, last_turn_seq)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("h1_test_1", "test_session", L1, blob, 50, 1, 50))
        conn.commit()

        # Retrieve
        cursor = conn.execute("""
            SELECT * FROM session_hierarchy_nodes WHERE node_id = ?
        """, ("h1_test_1",))
        row = cursor.fetchone()

        assert row["node_id"] == "h1_test_1"
        assert row["session_id"] == "test_session"
        assert row["level"] == L1
        assert row["turn_count"] == 50

        # Verify centroid
        recovered = np.frombuffer(row["centroid"], dtype=np.float32)
        np.testing.assert_array_equal(centroid, recovered)

    def test_parent_child_relationship(self, capsule):
        """Test parent-child foreign key relationship."""
        conn = capsule._get_conn()
        centroid = np.random.randn(EMBEDDING_DIM).astype(np.float32).tobytes()

        # Insert parent (L2)
        conn.execute("""
            INSERT INTO session_hierarchy_nodes
            (node_id, session_id, level, centroid, turn_count)
            VALUES (?, ?, ?, ?, ?)
        """, ("h2_parent", "test_session", L2, centroid, 1000))

        # Insert child (L1) with parent reference
        conn.execute("""
            INSERT INTO session_hierarchy_nodes
            (node_id, session_id, level, parent_node_id, centroid, turn_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("h1_child", "test_session", L1, "h2_parent", centroid, 100))
        conn.commit()

        # Query children of parent
        cursor = conn.execute("""
            SELECT node_id FROM session_hierarchy_nodes
            WHERE parent_node_id = ?
        """, ("h2_parent",))
        children = [row["node_id"] for row in cursor.fetchall()]

        assert "h1_child" in children
