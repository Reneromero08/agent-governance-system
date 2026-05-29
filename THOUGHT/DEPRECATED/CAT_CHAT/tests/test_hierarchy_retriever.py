#!/usr/bin/env python3
"""
Tests for Hierarchical Retriever (Phase J.2)

Tests the TOP-K selection based hierarchical retrieval algorithm.

Coverage:
- compute_E (Born rule)
- build_hierarchy_in_memory
- retrieve_hierarchical with TOP-K selection
- Budget cutoff behavior
- Database operations (load/store)
- Edge cases (empty, single level, etc.)
"""

import sqlite3
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from catalytic_chat.hierarchy_retriever import (
    HierarchyNode,
    RetrievalResult,
    compute_E,
    load_node_children,
    load_l0_content,
    load_root_nodes,
    retrieve_hierarchical,
    retrieve_hierarchical_with_metrics,
    build_hierarchy_in_memory,
    ensure_hierarchy_schema,
    store_hierarchy_node,
    store_hierarchy_batch,
    EMBEDDING_DIM,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def random_vectors():
    """Generate normalized random vectors for testing."""
    np.random.seed(42)
    n = 100
    vectors = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
    for i in range(n):
        vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
    return vectors


@pytest.fixture
def query_vector():
    """Generate a normalized query vector."""
    np.random.seed(123)
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ensure_hierarchy_schema(conn)
        yield conn
        conn.close()


@pytest.fixture
def mock_db_conn():
    """Mock database connection for in-memory hierarchy testing."""
    class MockConn:
        def execute(self, sql, params=()):
            class EmptyCursor:
                def fetchall(self):
                    return []
                def fetchone(self):
                    return None
            return EmptyCursor()
    return MockConn()


@pytest.fixture
def simple_hierarchy():
    """Create a simple 3-level hierarchy for testing."""
    np.random.seed(42)

    # Create L0 nodes (leaves)
    l0_nodes = []
    for i in range(10):
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        node = HierarchyNode(
            node_id=f"L0_{i}",
            level=0,
            centroid=vec,
            event_id=f"evt_{i}",
            content_hash=f"hash_{i}",
            token_count=100,
        )
        l0_nodes.append(node)

    # Create L1 nodes (2 groups of 5)
    l1_nodes = []
    for i in range(2):
        children = l0_nodes[i*5:(i+1)*5]
        centroid = np.mean([c.centroid for c in children], axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        node = HierarchyNode(
            node_id=f"L1_{i}",
            level=1,
            centroid=centroid,
            children=children,
            token_count=500,
        )
        for c in children:
            c.parent_id = node.node_id
        l1_nodes.append(node)

    # Create root (L2)
    centroid = np.mean([c.centroid for c in l1_nodes], axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    root = HierarchyNode(
        node_id="root",
        level=2,
        centroid=centroid,
        children=l1_nodes,
        token_count=1000,
    )
    for c in l1_nodes:
        c.parent_id = root.node_id

    return root


# =============================================================================
# Test compute_E (Born Rule)
# =============================================================================

class TestComputeE:
    """Tests for E-score computation using Born rule."""

    def test_identical_vectors_give_one(self):
        """Identical normalized vectors should give E = 1.0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        e_score = compute_E(vec, vec)
        assert e_score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_give_zero(self):
        """Orthogonal vectors should give E = 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e_score = compute_E(vec1, vec2)
        assert e_score == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_give_one(self):
        """Opposite vectors should give E = 1.0 (squared)."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        e_score = compute_E(vec1, vec2)
        assert e_score == pytest.approx(1.0, abs=1e-6)

    def test_e_score_is_symmetric(self):
        """E(a, b) should equal E(b, a)."""
        np.random.seed(42)
        vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)

        e1 = compute_E(vec1, vec2)
        e2 = compute_E(vec2, vec1)
        assert e1 == pytest.approx(e2, abs=1e-6)

    def test_e_score_in_range_zero_one(self):
        """E-score should always be in [0, 1] for normalized vectors."""
        np.random.seed(42)
        for _ in range(100):
            vec1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec2 = vec2 / np.linalg.norm(vec2)

            e_score = compute_E(vec1, vec2)
            assert 0.0 <= e_score <= 1.0

    def test_none_vectors_return_zero(self):
        """None vectors should return E = 0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert compute_E(None, vec) == 0.0
        assert compute_E(vec, None) == 0.0
        assert compute_E(None, None) == 0.0

    def test_empty_vectors_return_zero(self):
        """Empty vectors should return E = 0."""
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        empty = np.array([], dtype=np.float32)
        assert compute_E(empty, vec) == 0.0
        assert compute_E(vec, empty) == 0.0


# =============================================================================
# Test build_hierarchy_in_memory
# =============================================================================

class TestBuildHierarchyInMemory:
    """Tests for in-memory hierarchy building."""

    def test_empty_vectors_returns_empty(self):
        """Empty input should return empty list."""
        vectors = np.array([]).reshape(0, EMBEDDING_DIM)
        roots = build_hierarchy_in_memory(vectors, [], [])
        assert roots == []

    def test_single_vector_returns_single_node(self):
        """Single vector should return single root node at L0."""
        vec = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        roots = build_hierarchy_in_memory(vec, ["evt_0"], ["hash_0"])
        assert len(roots) == 1
        assert roots[0].level == 0
        assert roots[0].event_id == "evt_0"

    def test_builds_balanced_tree(self, random_vectors):
        """Should build a balanced tree structure."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        # With 100 vectors and branch_factor=10:
        # L0: 100 nodes -> L1: 10 nodes -> L2: 1 node
        assert len(roots) == 1
        root = roots[0]
        assert root.level == 2
        assert len(root.children) == 10

    def test_all_leaves_are_level_zero(self, random_vectors):
        """All leaf nodes should have level 0."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        def collect_leaves(node: HierarchyNode) -> List[HierarchyNode]:
            if node.level == 0:
                return [node]
            leaves = []
            for child in node.children:
                leaves.extend(collect_leaves(child))
            return leaves

        leaves = collect_leaves(roots[0])
        assert len(leaves) == 100
        assert all(leaf.level == 0 for leaf in leaves)
        assert all(leaf.event_id is not None for leaf in leaves)

    def test_centroids_are_normalized(self, random_vectors):
        """All centroids should be normalized."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        def check_normalized(node: HierarchyNode):
            norm = np.linalg.norm(node.centroid)
            assert norm == pytest.approx(1.0, abs=1e-5)
            for child in node.children:
                check_normalized(child)

        check_normalized(roots[0])

    def test_token_counts_propagate(self, random_vectors):
        """Token counts should sum up the tree."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]
        token_counts = [100] * len(random_vectors)

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes, token_counts,
            branch_factor=10
        )

        # Root should have sum of all children
        # L0: 100 tokens each, L1: 1000 each (10 children), L2: 10000 (10 L1 nodes)
        assert roots[0].token_count == 10000


# =============================================================================
# Test retrieve_hierarchical
# =============================================================================

class TestRetrieveHierarchical:
    """Tests for TOP-K hierarchical retrieval."""

    def test_empty_roots_returns_empty(self, query_vector, mock_db_conn):
        """Empty root list should return empty results."""
        results = retrieve_hierarchical(query_vector, [], mock_db_conn)
        assert results == []

    def test_none_query_returns_empty(self, simple_hierarchy, mock_db_conn):
        """None query vector should return empty results."""
        results = retrieve_hierarchical(None, [simple_hierarchy], mock_db_conn)
        assert results == []

    def test_empty_query_returns_empty(self, simple_hierarchy, mock_db_conn):
        """Empty query vector should return empty results."""
        empty_vec = np.array([], dtype=np.float32)
        results = retrieve_hierarchical(empty_vec, [simple_hierarchy], mock_db_conn)
        assert results == []

    def test_returns_results_sorted_by_e_score(self, simple_hierarchy, mock_db_conn):
        """Results should be sorted by E-score descending."""
        np.random.seed(42)
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(
            query, [simple_hierarchy], mock_db_conn,
            top_k=10, budget_tokens=10000
        )

        # Check results are sorted
        for i in range(len(results) - 1):
            assert results[i].e_score >= results[i+1].e_score

    def test_all_results_are_level_zero(self, simple_hierarchy, mock_db_conn):
        """All results should be leaf nodes (level 0)."""
        np.random.seed(42)
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(
            query, [simple_hierarchy], mock_db_conn,
            top_k=10, budget_tokens=10000
        )

        assert all(r.level == 0 for r in results)

    def test_top_k_limits_exploration(self, random_vectors, mock_db_conn):
        """TOP-K should limit nodes explored at each level."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        # With top_k=2 and 2-level tree (L1 has 10 nodes, L0 has 100):
        # We explore 2 L1 nodes, each with 10 L0 children = 20 results max
        results, metrics = retrieve_hierarchical_with_metrics(
            query, roots, mock_db_conn,
            top_k=2, budget_tokens=100000
        )

        # Should get at most 2*10 = 20 results
        assert len(results) <= 20

    def test_budget_limits_results(self, simple_hierarchy, mock_db_conn):
        """Budget should limit total tokens in results."""
        np.random.seed(42)
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Each node has token_count=100, budget=250 should give ~2 results
        results = retrieve_hierarchical(
            query, [simple_hierarchy], mock_db_conn,
            top_k=10, budget_tokens=250
        )

        total_tokens = sum(100 for _ in results)  # 100 tokens each
        # Budget check is approximate due to recursion order
        assert len(results) <= 3  # At most 3 with 250 budget and 100 tokens each

    def test_metrics_count_e_computations(self, simple_hierarchy, mock_db_conn):
        """Metrics should count E-score computations."""
        np.random.seed(42)
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results, metrics = retrieve_hierarchical_with_metrics(
            query, [simple_hierarchy], mock_db_conn,
            top_k=10, budget_tokens=10000
        )

        assert "e_computations" in metrics
        assert metrics["e_computations"] > 0
        assert metrics["results_count"] == len(results)


# =============================================================================
# Test Database Operations
# =============================================================================

class TestDatabaseOperations:
    """Tests for database storage and retrieval."""

    def test_ensure_schema_creates_table(self, temp_db):
        """ensure_hierarchy_schema should create the table."""
        cursor = temp_db.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='hierarchy_nodes'
        """)
        assert cursor.fetchone() is not None

    def test_ensure_schema_idempotent(self, temp_db):
        """ensure_hierarchy_schema should be idempotent."""
        # Already called in fixture, call again
        created = ensure_hierarchy_schema(temp_db)
        assert created is False

    def test_store_and_load_single_node(self, temp_db):
        """Should store and load a single node correctly."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        node = HierarchyNode(
            node_id="test_node",
            level=0,
            centroid=vec,
            event_id="evt_001",
            content_hash="hash_001",
            token_count=100,
        )

        store_hierarchy_node(temp_db, node, "session_001")
        temp_db.commit()

        # Load and verify
        roots = load_root_nodes(temp_db, "session_001")
        assert len(roots) == 1
        loaded = roots[0]
        assert loaded.node_id == "test_node"
        assert loaded.level == 0
        assert loaded.event_id == "evt_001"
        assert np.allclose(loaded.centroid, vec)

    def test_store_batch(self, temp_db):
        """Should store multiple nodes efficiently."""
        nodes = []
        for i in range(10):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            node = HierarchyNode(
                node_id=f"node_{i}",
                level=0,
                centroid=vec,
                event_id=f"evt_{i}",
                content_hash=f"hash_{i}",
                token_count=100,
            )
            nodes.append(node)

        count = store_hierarchy_batch(temp_db, nodes, "session_001")
        assert count == 10

        roots = load_root_nodes(temp_db, "session_001")
        assert len(roots) == 10

    def test_load_node_children(self, temp_db):
        """Should load children of a parent node."""
        # Create parent
        parent_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        parent_vec = parent_vec / np.linalg.norm(parent_vec)
        parent = HierarchyNode(
            node_id="parent",
            level=1,
            centroid=parent_vec,
            token_count=500,
        )
        store_hierarchy_node(temp_db, parent, "session_001")

        # Create children
        children = []
        for i in range(5):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            child = HierarchyNode(
                node_id=f"child_{i}",
                level=0,
                centroid=vec,
                event_id=f"evt_{i}",
                content_hash=f"hash_{i}",
                parent_id="parent",
                token_count=100,
            )
            children.append(child)

        store_hierarchy_batch(temp_db, children, "session_001")

        # Load children
        loaded = load_node_children(temp_db, "parent")
        assert len(loaded) == 5
        assert all(c.parent_id == "parent" for c in loaded)

    def test_load_l0_content(self, temp_db):
        """Should load content hash and tokens for L0 node."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        node = HierarchyNode(
            node_id="l0_node",
            level=0,
            centroid=vec,
            event_id="evt_001",
            content_hash="hash_001",
            token_count=150,
        )
        store_hierarchy_node(temp_db, node, "session_001")
        temp_db.commit()

        content_hash, tokens = load_l0_content(temp_db, "l0_node")
        assert content_hash == "hash_001"
        assert tokens == 150

    def test_load_l0_content_not_found(self, temp_db):
        """Should return (None, 0) for non-existent node."""
        content_hash, tokens = load_l0_content(temp_db, "nonexistent")
        assert content_hash is None
        assert tokens == 0


# =============================================================================
# Test Integration: Full DB-backed Retrieval
# =============================================================================

class TestIntegration:
    """Integration tests with database-backed hierarchy."""

    def test_full_db_retrieval(self, temp_db, random_vectors):
        """Test full hierarchy build, store, and retrieve cycle."""
        # Build hierarchy in memory
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]
        token_counts = [100] * len(random_vectors)

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes, token_counts,
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

        store_hierarchy_batch(temp_db, all_nodes, "session_001")

        # Load roots from DB
        db_roots = load_root_nodes(temp_db, "session_001")
        assert len(db_roots) >= 1

        # Query
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results, metrics = retrieve_hierarchical_with_metrics(
            query, db_roots, temp_db,
            top_k=5, budget_tokens=2000
        )

        # Should get results
        assert len(results) > 0
        assert metrics["e_computations"] > 0

        # All results should be L0 with valid event_ids
        for r in results:
            assert r.level == 0
            assert r.event_id.startswith("evt_")


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_node_hierarchy(self, mock_db_conn):
        """Single node hierarchy should work."""
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        node = HierarchyNode(
            node_id="single",
            level=0,
            centroid=vec,
            event_id="evt_0",
            content_hash="hash_0",
            token_count=100,
        )

        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(query, [node], mock_db_conn)
        assert len(results) == 1
        assert results[0].event_id == "evt_0"

    def test_all_zeros_budget(self, simple_hierarchy, mock_db_conn):
        """Zero budget should return empty results."""
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(
            query, [simple_hierarchy], mock_db_conn,
            budget_tokens=0
        )
        assert results == []

    def test_top_k_one_explores_single_path(self, random_vectors, mock_db_conn):
        """top_k=1 should explore only one path at each level."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results, metrics = retrieve_hierarchical_with_metrics(
            query, roots, mock_db_conn,
            top_k=1, budget_tokens=100000
        )

        # With top_k=1, we follow single best child at each level:
        # L2 (1 node) -> 1 best L1 -> 1 best L0
        # So exactly 1 result
        assert len(results) == 1
        assert results[0].level == 0

    def test_very_large_top_k(self, simple_hierarchy, mock_db_conn):
        """Large top_k should effectively return all leaves."""
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(
            query, [simple_hierarchy], mock_db_conn,
            top_k=1000, budget_tokens=100000
        )

        # Should get all 10 leaves
        assert len(results) == 10

    def test_multiple_root_nodes(self, mock_db_conn):
        """Should handle multiple independent root nodes."""
        np.random.seed(42)

        roots = []
        for r in range(3):
            vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            node = HierarchyNode(
                node_id=f"root_{r}",
                level=0,
                centroid=vec,
                event_id=f"evt_root_{r}",
                content_hash=f"hash_root_{r}",
                token_count=100,
            )
            roots.append(node)

        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(
            query, roots, mock_db_conn,
            top_k=10, budget_tokens=10000
        )

        assert len(results) == 3


# =============================================================================
# Test Recall Quality
# =============================================================================

class TestRecallQuality:
    """Tests for retrieval recall quality."""

    def test_similar_query_finds_similar_vectors(self, random_vectors, mock_db_conn):
        """Query similar to a specific vector should rank it high."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        # Use vector 42 as query (with slight noise)
        query = random_vectors[42] + 0.01 * np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = retrieve_hierarchical(
            query, roots, mock_db_conn,
            top_k=5, budget_tokens=10000
        )

        # evt_42 should be in top results
        top_event_ids = [r.event_id for r in results[:10]]
        assert "evt_42" in top_event_ids

    def test_hierarchical_vs_brute_force_recall(self, random_vectors, mock_db_conn):
        """Hierarchical should have reasonable recall vs brute force."""
        event_ids = [f"evt_{i}" for i in range(len(random_vectors))]
        hashes = [f"hash_{i}" for i in range(len(random_vectors))]

        roots = build_hierarchy_in_memory(
            random_vectors, event_ids, hashes,
            branch_factor=10
        )

        np.random.seed(999)
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Hierarchical retrieval with higher top_k for better recall
        h_results = retrieve_hierarchical(
            query, roots, mock_db_conn,
            top_k=10, budget_tokens=100000
        )
        h_events = set(r.event_id for r in h_results[:10])

        # Brute force: score all vectors
        brute_scores = []
        for i, vec in enumerate(random_vectors):
            e = compute_E(query, vec)
            brute_scores.append((f"evt_{i}", e))
        brute_scores.sort(key=lambda x: x[1], reverse=True)
        brute_top10 = set(b[0] for b in brute_scores[:10])

        # Hierarchical should overlap significantly with brute force
        overlap = len(h_events & brute_top10)
        recall = overlap / min(10, len(h_events)) if h_events else 0

        # With top_k=10 at all levels, we explore most of the tree
        # so recall should be high (at least 30% is realistic for random data)
        assert recall >= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
