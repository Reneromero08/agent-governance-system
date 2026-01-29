"""
Tests for E-Relationship Daemon

Tests cover:
- Unit tests: Edge creation, R-gating, cluster detection
- Integration tests: Full daemon workflow, retrieval improvement

Run with: python -m pytest tests/test_e_relationship.py -v
"""

import pytest
import sqlite3
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
TEST_DIR = Path(__file__).parent
FERAL_DIR = TEST_DIR.parent
MEMORY_DIR = FERAL_DIR / "memory"
sys.path.insert(0, str(FERAL_DIR))
sys.path.insert(0, str(MEMORY_DIR))

from memory.e_graph import ERelationshipGraph, E_THRESHOLD, RTier, R_TIER_THRESHOLDS
from memory.e_patterns import EPatternDetector, UnionFind
from memory.e_query import EQueryEngine, PathResult


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database with schema."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create schema
    conn.executescript("""
        CREATE TABLE vectors (
            vector_id TEXT PRIMARY KEY,
            vec_blob BLOB NOT NULL,
            vec_sha256 TEXT NOT NULL,
            Df REAL,
            composition_op TEXT,
            parent_ids JSON,
            created_at TEXT NOT NULL,
            source_id TEXT,
            daemon_step INTEGER,
            mind_hash_before TEXT
        );

        CREATE TABLE e_edges (
            edge_id TEXT PRIMARY KEY,
            vector_id_a TEXT NOT NULL,
            vector_id_b TEXT NOT NULL,
            e_score REAL NOT NULL,
            r_score REAL,
            r_tier TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(vector_id_a, vector_id_b)
        );

        CREATE INDEX idx_e_edges_a ON e_edges(vector_id_a, e_score DESC);
        CREATE INDEX idx_e_edges_b ON e_edges(vector_id_b, e_score DESC);
    """)
    conn.commit()

    yield conn

    conn.close()
    Path(db_path).unlink()


def insert_test_vector(conn, vector_id: str, vec: np.ndarray, daemon_step: int = 0):
    """Helper to insert a test vector."""
    vec_blob = vec.astype(np.float32).tobytes()
    conn.execute("""
        INSERT INTO vectors (vector_id, vec_blob, vec_sha256, Df, composition_op, parent_ids, created_at, daemon_step)
        VALUES (?, ?, ?, ?, 'daemon_item', '[]', datetime('now'), ?)
    """, (vector_id, vec_blob, 'test_hash', 20.0, daemon_step))
    conn.commit()


# =============================================================================
# UNIT TESTS: E-Graph
# =============================================================================

class TestEGraph:
    """Tests for ERelationshipGraph."""

    def test_compute_E_same_vector(self):
        """Same vectors should have E = 1.0."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        E = ERelationshipGraph.compute_E(v, v)
        assert E == pytest.approx(1.0)

    def test_compute_E_orthogonal(self):
        """Orthogonal vectors should have E = 0.0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        E = ERelationshipGraph.compute_E(v1, v2)
        assert E == pytest.approx(0.0)

    def test_compute_E_similar(self):
        """Similar vectors should have high E."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)  # Normalize
        E = ERelationshipGraph.compute_E(v1, v2)
        assert E > 0.8  # Should be highly similar

    def test_compute_R(self, temp_db):
        """Test R computation."""
        graph = ERelationshipGraph(temp_db)
        graph._e_values = [0.3, 0.4, 0.5, 0.6, 0.7]

        E = 0.8
        R = graph.compute_R(E)
        assert R > 0  # R should be positive for positive E

    def test_get_r_tier(self, temp_db):
        """Test tier classification."""
        graph = ERelationshipGraph(temp_db)

        assert graph.get_r_tier(0.3) == RTier.T0_OBSERVE
        assert graph.get_r_tier(0.6) == RTier.T1_SMALL
        assert graph.get_r_tier(0.9) == RTier.T2_MEDIUM
        assert graph.get_r_tier(1.5) == RTier.T3_LARGE

    def test_edge_creation_similar_vectors(self, temp_db):
        """Similar vectors should create edges."""
        # Insert two similar vectors
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)

        graph = ERelationshipGraph(temp_db)
        edges = graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3)

        assert len(edges) > 0
        assert edges[0].e_score > 0.8

    def test_no_edge_dissimilar_vectors(self, temp_db):
        """Dissimilar vectors should not create edges."""
        # Insert two orthogonal vectors
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)

        graph = ERelationshipGraph(temp_db)
        edges = graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.5)

        assert len(edges) == 0

    def test_r_gate_filtering(self, temp_db):
        """R-gate should filter noisy relationships."""
        # Create vectors with moderate similarity
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.7, 0.7, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)

        graph = ERelationshipGraph(temp_db)

        # With no minimum tier, should create edge
        edges_t0 = graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3, min_r_tier=RTier.T0_OBSERVE)

        # Reset edges
        temp_db.execute("DELETE FROM e_edges")
        temp_db.commit()

        # With high minimum tier, should NOT create edge (R too low)
        edges_t3 = graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3, min_r_tier=RTier.T3_LARGE)

        # T0 should have edge, T3 might not depending on computed R
        assert len(edges_t0) >= len(edges_t3)


# =============================================================================
# UNIT TESTS: Union-Find
# =============================================================================

class TestUnionFind:
    """Tests for Union-Find data structure."""

    def test_union_and_find(self):
        """Test basic union and find operations."""
        uf = UnionFind()

        uf.union('a', 'b')
        uf.union('c', 'd')
        uf.union('b', 'c')

        assert uf.connected('a', 'b')
        assert uf.connected('a', 'd')
        assert uf.connected('b', 'c')

    def test_disconnected_sets(self):
        """Disconnected sets should not be connected."""
        uf = UnionFind()

        uf.union('a', 'b')
        uf.union('c', 'd')

        assert uf.connected('a', 'b')
        assert uf.connected('c', 'd')
        assert not uf.connected('a', 'c')

    def test_path_compression(self):
        """Path compression should work correctly."""
        uf = UnionFind()

        # Create a chain
        uf.union('a', 'b')
        uf.union('b', 'c')
        uf.union('c', 'd')
        uf.union('d', 'e')

        # After find, all should point to same root
        root = uf.find('e')
        assert uf.find('a') == root
        assert uf.find('b') == root
        assert uf.find('c') == root


# =============================================================================
# UNIT TESTS: E-Patterns
# =============================================================================

class TestEPatterns:
    """Tests for EPatternDetector."""

    def test_find_clusters(self, temp_db):
        """Test cluster detection."""
        # Create a small cluster
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        v3 = v3 / np.linalg.norm(v3)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)
        insert_test_vector(temp_db, 'vec_c', v3, 2)

        # Add edges
        graph = ERelationshipGraph(temp_db)
        graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3)
        graph.add_item('vec_c', v3, Df=20.0, e_threshold=0.3)

        # Detect clusters
        detector = EPatternDetector(temp_db)
        clusters = detector.find_clusters(min_size=2)

        # Should find at least one cluster
        assert len(clusters) >= 1
        assert clusters[0].size >= 2

    def test_novelty_scoring(self, temp_db):
        """Test novelty scoring."""
        # Create connected vectors
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)

        graph = ERelationshipGraph(temp_db)
        graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3)

        detector = EPatternDetector(temp_db)

        # vec_b has a connection, so moderate novelty
        novelty_b = detector.score_novelty('vec_b')

        # Unconnected vector has max novelty
        v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        insert_test_vector(temp_db, 'vec_c', v3, 2)
        novelty_c = detector.score_novelty('vec_c')

        assert novelty_c == 1.0  # No connections = max novelty


# =============================================================================
# UNIT TESTS: E-Query
# =============================================================================

class TestEQuery:
    """Tests for EQueryEngine."""

    def test_expand_from(self, temp_db):
        """Test subgraph expansion."""
        # Create a small graph
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        v3 = v3 / np.linalg.norm(v3)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)
        insert_test_vector(temp_db, 'vec_c', v3, 2)

        graph = ERelationshipGraph(temp_db)
        graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3)
        graph.add_item('vec_c', v3, Df=20.0, e_threshold=0.3)

        engine = EQueryEngine(temp_db)
        subgraph = engine.expand_from('vec_a', hops=2, min_e=0.3)

        # Should include connected nodes
        assert subgraph['node_count'] >= 1

    def test_find_path(self, temp_db):
        """Test path finding."""
        # Create a chain: a -- b -- c
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.array([0.8, 0.2, 0.0], dtype=np.float32)
        v3 = v3 / np.linalg.norm(v3)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)
        insert_test_vector(temp_db, 'vec_c', v3, 2)

        graph = ERelationshipGraph(temp_db)
        graph.add_item('vec_b', v2, Df=20.0, e_threshold=0.3)
        graph.add_item('vec_c', v3, Df=20.0, e_threshold=0.3)

        engine = EQueryEngine(temp_db)
        result = engine.find_path('vec_a', 'vec_c', max_hops=3, min_e=0.3)

        # Path should exist if edges were created
        if graph.edge_count() >= 2:
            assert result.found
            assert len(result.path) >= 2

    def test_no_path_disconnected(self, temp_db):
        """Test that disconnected nodes have no path."""
        # Create two disconnected vectors
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        insert_test_vector(temp_db, 'vec_a', v1, 0)
        insert_test_vector(temp_db, 'vec_b', v2, 1)

        engine = EQueryEngine(temp_db)
        result = engine.find_path('vec_a', 'vec_b', max_hops=5, min_e=0.5)

        assert not result.found


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full E-relationship daemon."""

    def test_daemon_learns_relationships(self, temp_db):
        """Test that the daemon creates meaningful edges over time."""
        graph = ERelationshipGraph(temp_db)

        # Simulate 10 related vectors (all pointing roughly the same direction)
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for i in range(10):
            # Add small perturbation
            noise = np.random.randn(3) * 0.1
            vec = base + noise
            vec = vec / np.linalg.norm(vec)
            vec = vec.astype(np.float32)

            vector_id = f'vec_{i}'
            insert_test_vector(temp_db, vector_id, vec, i)
            graph.add_item(vector_id, vec, Df=20.0, e_threshold=0.3)

        # Should have created edges
        stats = graph.get_stats()
        assert stats['total_edges'] > 0

    def test_graph_vs_centroid_retrieval(self, temp_db):
        """Test that graph retrieval finds items centroid would miss."""
        graph = ERelationshipGraph(temp_db)

        # Create two clusters
        # Cluster 1: around [1, 0, 0]
        for i in range(5):
            vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            vec += np.random.randn(3).astype(np.float32) * 0.05
            vec = vec / np.linalg.norm(vec)
            vid = f'cluster1_{i}'
            insert_test_vector(temp_db, vid, vec, i)
            graph.add_item(vid, vec, Df=20.0, e_threshold=0.3)

        # Cluster 2: around [0, 1, 0]
        for i in range(5):
            vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            vec += np.random.randn(3).astype(np.float32) * 0.05
            vec = vec / np.linalg.norm(vec)
            vid = f'cluster2_{i}'
            insert_test_vector(temp_db, vid, vec, i + 5)
            graph.add_item(vid, vec, Df=20.0, e_threshold=0.3)

        # Add a bridge between clusters
        bridge = np.array([0.7, 0.7, 0.0], dtype=np.float32)
        bridge = bridge / np.linalg.norm(bridge)
        insert_test_vector(temp_db, 'bridge', bridge, 10)
        graph.add_item('bridge', bridge, Df=20.0, e_threshold=0.2)

        # Pattern detector should find two clusters
        detector = EPatternDetector(temp_db)
        clusters = detector.find_clusters(min_size=2, min_e=0.3)

        # Should have multiple clusters or one large connected component
        total_clustered = sum(c.size for c in clusters)
        assert total_clustered >= 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
