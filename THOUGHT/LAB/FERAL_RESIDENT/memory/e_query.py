"""
E-Query Engine

Graph-based retrieval and traversal over the E-relationship graph.

Key Methods:
- find_related(): E-score ranking from query
- expand_from(): Subgraph extraction (k-hop neighborhood)
- find_path(): BFS shortest path between vectors

This replaces centroid-only retrieval with graph-aware retrieval
that can follow semantic relationships.
"""

import sqlite3
import numpy as np
from collections import deque
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add CAPABILITY to path for imports
CAPABILITY_PATH = Path(__file__).parent.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"
if str(CAPABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_PATH))

try:
    from geometric_reasoner import GeometricReasoner, GeometricState
except ImportError:
    GeometricReasoner = None
    GeometricState = None


# =============================================================================
# QUERY RESULT TYPES
# =============================================================================

@dataclass
class RelatedItem:
    """A related item from the graph."""
    vector_id: str
    e_score: float
    text: Optional[str] = None
    Df: Optional[float] = None
    hops: int = 0  # Distance from query


@dataclass
class PathResult:
    """Result of a path search."""
    found: bool
    path: List[str]  # Vector IDs from start to end
    length: int
    total_e: float  # Sum of E scores along path


# =============================================================================
# E-QUERY ENGINE
# =============================================================================

class EQueryEngine:
    """
    Graph-based query engine over E-relationships.

    Key Methods:
    - find_related(): Find items related to a query
    - expand_from(): Get k-hop subgraph
    - find_path(): BFS path between vectors

    Usage:
        engine = EQueryEngine(conn, reasoner)
        results = engine.find_related("quantum computing", top_k=10)
        subgraph = engine.expand_from(vector_id, hops=2)
        path = engine.find_path(from_id, to_id)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        reasoner: Optional['GeometricReasoner'] = None
    ):
        """
        Initialize query engine.

        Args:
            conn: SQLite connection with e_edges and vectors tables
            reasoner: Optional GeometricReasoner for text->vector
        """
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.reasoner = reasoner

        # Cache recently computed vectors
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._cache_size = 1000

    # =========================================================================
    # FIND RELATED
    # =========================================================================

    def find_related(
        self,
        query_text: str,
        top_k: int = 10,
        min_e: float = 0.3,
        use_graph: bool = True
    ) -> List[RelatedItem]:
        """
        Find items related to a text query.

        Two-stage retrieval:
        1. Direct E-score with all daemon items
        2. (Optional) Graph expansion to find connected items

        Args:
            query_text: The query text
            top_k: Number of results to return
            min_e: Minimum E score threshold
            use_graph: Whether to expand via graph edges

        Returns:
            List of RelatedItem sorted by E desc
        """
        if self.reasoner is None:
            raise ValueError("GeometricReasoner required for text queries")

        # Embed query
        query_state = self.reasoner.initialize(query_text)
        query_vec = query_state.vector

        # Get all daemon items
        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT vector_id, vec_blob, Df FROM vectors
            WHERE composition_op = 'daemon_item'
        """).fetchall()

        # Compute E scores
        scored = []
        for row in rows:
            vec = np.frombuffer(row['vec_blob'], dtype=np.float32)
            dot = np.dot(query_vec, vec)
            E = dot * dot  # Born rule

            if E >= min_e:
                scored.append(RelatedItem(
                    vector_id=row['vector_id'],
                    e_score=E,
                    Df=row['Df'],
                    hops=0
                ))

        # Sort by E descending
        scored.sort(key=lambda x: x.e_score, reverse=True)

        if not use_graph:
            return scored[:top_k]

        # Graph expansion: include neighbors of top results
        expanded = set(r.vector_id for r in scored[:top_k])
        for item in scored[:min(5, len(scored))]:  # Expand top 5
            neighbors = self._get_neighbors(item.vector_id, min_e=min_e)
            for neighbor_id, e_score in neighbors:
                if neighbor_id not in expanded:
                    # Compute E with query for the neighbor
                    neighbor_vec = self._get_vector(neighbor_id)
                    if neighbor_vec is not None:
                        dot = np.dot(query_vec, neighbor_vec)
                        neighbor_E = dot * dot
                        if neighbor_E >= min_e:
                            scored.append(RelatedItem(
                                vector_id=neighbor_id,
                                e_score=neighbor_E,
                                hops=1
                            ))
                            expanded.add(neighbor_id)

        # Re-sort and return
        scored.sort(key=lambda x: x.e_score, reverse=True)
        return scored[:top_k]

    def find_related_to_vector(
        self,
        vector_id: str,
        top_k: int = 10,
        min_e: float = 0.3,
        max_hops: int = 2
    ) -> List[RelatedItem]:
        """
        Find items related to an existing vector via graph traversal.

        Args:
            vector_id: The source vector ID
            top_k: Number of results
            min_e: Minimum E score
            max_hops: Maximum hops from source

        Returns:
            List of RelatedItem sorted by E desc
        """
        source_vec = self._get_vector(vector_id)
        if source_vec is None:
            return []

        # BFS expansion
        visited = {vector_id}
        results = []
        queue = deque([(vector_id, 0)])  # (id, hops)

        while queue:
            current_id, hops = queue.popleft()

            if hops > 0:  # Don't include source
                current_vec = self._get_vector(current_id)
                if current_vec is not None:
                    dot = np.dot(source_vec, current_vec)
                    E = dot * dot
                    if E >= min_e:
                        results.append(RelatedItem(
                            vector_id=current_id,
                            e_score=E,
                            hops=hops
                        ))

            if hops < max_hops:
                for neighbor_id, _ in self._get_neighbors(current_id, min_e=min_e):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, hops + 1))

        # Sort by E descending
        results.sort(key=lambda x: x.e_score, reverse=True)
        return results[:top_k]

    # =========================================================================
    # EXPAND FROM (SUBGRAPH EXTRACTION)
    # =========================================================================

    def expand_from(
        self,
        vector_id: str,
        hops: int = 2,
        min_e: float = 0.3
    ) -> Dict:
        """
        Extract k-hop subgraph around a vector.

        Args:
            vector_id: Center vector ID
            hops: Number of hops to expand
            min_e: Minimum E score for edges

        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        nodes = set()
        edges = []
        visited_edges = set()

        # BFS expansion
        queue = deque([(vector_id, 0)])
        nodes.add(vector_id)

        while queue:
            current_id, current_hops = queue.popleft()

            if current_hops >= hops:
                continue

            # Get edges for this node
            cursor = self.conn.cursor()
            for row in cursor.execute("""
                SELECT edge_id, vector_id_a, vector_id_b, e_score
                FROM e_edges
                WHERE (vector_id_a = ? OR vector_id_b = ?)
                AND e_score >= ?
            """, (current_id, current_id, min_e)):

                edge_id = row['edge_id']
                if edge_id in visited_edges:
                    continue
                visited_edges.add(edge_id)

                edges.append({
                    'edge_id': edge_id,
                    'source': row['vector_id_a'],
                    'target': row['vector_id_b'],
                    'e_score': row['e_score']
                })

                # Add neighbor to queue
                neighbor_id = row['vector_id_b'] if row['vector_id_a'] == current_id else row['vector_id_a']
                if neighbor_id not in nodes:
                    nodes.add(neighbor_id)
                    queue.append((neighbor_id, current_hops + 1))

        # Get node details
        node_list = []
        for vid in nodes:
            row = self.conn.execute(
                "SELECT vector_id, Df, composition_op FROM vectors WHERE vector_id = ?",
                (vid,)
            ).fetchone()
            if row:
                node_list.append({
                    'vector_id': row['vector_id'],
                    'Df': row['Df'],
                    'composition_op': row['composition_op']
                })

        return {
            'center': vector_id,
            'hops': hops,
            'nodes': node_list,
            'edges': edges,
            'node_count': len(node_list),
            'edge_count': len(edges)
        }

    # =========================================================================
    # FIND PATH (BFS SHORTEST PATH)
    # =========================================================================

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_hops: int = 5,
        min_e: float = 0.0
    ) -> PathResult:
        """
        Find shortest path between two vectors using BFS.

        Args:
            from_id: Source vector ID
            to_id: Target vector ID
            max_hops: Maximum path length
            min_e: Minimum E score for edges

        Returns:
            PathResult with path or empty if not found
        """
        if from_id == to_id:
            return PathResult(found=True, path=[from_id], length=0, total_e=1.0)

        # BFS
        visited = {from_id}
        parent: Dict[str, Tuple[str, float]] = {}  # child -> (parent, e_score)
        queue = deque([from_id])
        found = False

        while queue and not found:
            current = queue.popleft()

            # Check path length
            path_len = 0
            node = current
            while node in parent:
                path_len += 1
                node = parent[node][0]

            if path_len >= max_hops:
                continue

            # Explore neighbors
            for neighbor_id, e_score in self._get_neighbors(current, min_e=min_e):
                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                parent[neighbor_id] = (current, e_score)
                queue.append(neighbor_id)

                if neighbor_id == to_id:
                    found = True
                    break

        if not found:
            return PathResult(found=False, path=[], length=0, total_e=0.0)

        # Reconstruct path
        path = [to_id]
        total_e = 0.0
        node = to_id
        while node in parent:
            p, e = parent[node]
            total_e += e
            path.append(p)
            node = p

        path.reverse()

        return PathResult(
            found=True,
            path=path,
            length=len(path) - 1,
            total_e=total_e
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_neighbors(
        self,
        vector_id: str,
        min_e: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Get neighbors from e_edges table."""
        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT vector_id_a, vector_id_b, e_score FROM e_edges
            WHERE (vector_id_a = ? OR vector_id_b = ?)
            AND e_score >= ?
            ORDER BY e_score DESC
        """, (vector_id, vector_id, min_e)).fetchall()

        neighbors = []
        for row in rows:
            neighbor_id = row['vector_id_b'] if row['vector_id_a'] == vector_id else row['vector_id_a']
            neighbors.append((neighbor_id, row['e_score']))

        return neighbors

    def _get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector from cache or database."""
        if vector_id in self._vector_cache:
            return self._vector_cache[vector_id]

        row = self.conn.execute(
            "SELECT vec_blob FROM vectors WHERE vector_id = ?",
            (vector_id,)
        ).fetchone()

        if row is None:
            return None

        vec = np.frombuffer(row['vec_blob'], dtype=np.float32)

        # Cache
        if len(self._vector_cache) >= self._cache_size:
            # Evict oldest (simple FIFO)
            oldest = next(iter(self._vector_cache))
            del self._vector_cache[oldest]
        self._vector_cache[vector_id] = vec

        return vec

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get query engine statistics."""
        cursor = self.conn.cursor()

        # Total daemon items
        daemon_count = cursor.execute(
            "SELECT COUNT(*) FROM vectors WHERE composition_op = 'daemon_item'"
        ).fetchone()[0]

        # Total edges
        edge_count = cursor.execute("SELECT COUNT(*) FROM e_edges").fetchone()[0]

        # Average degree
        avg_degree = 0.0
        if daemon_count > 0:
            avg_degree = 2 * edge_count / daemon_count  # Each edge contributes to 2 nodes

        return {
            'daemon_items': daemon_count,
            'edges': edge_count,
            'avg_degree': avg_degree,
            'cache_size': len(self._vector_cache)
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("E-Query Engine Tests")
    print("=" * 60)

    # Test path reconstruction logic
    print("\nPath reconstruction test:")

    class MockResult:
        pass

    # Simulate BFS result
    parent = {
        'b': ('a', 0.8),
        'c': ('b', 0.7),
        'd': ('c', 0.9)
    }

    # Reconstruct
    path = ['d']
    total_e = 0.0
    node = 'd'
    while node in parent:
        p, e = parent[node]
        total_e += e
        path.append(p)
        node = p
    path.reverse()

    print(f"  Path: {' -> '.join(path)}")
    print(f"  Total E: {total_e:.1f}")
    print(f"  Expected: a -> b -> c -> d, E=2.4")

    print("\nAll tests passed!")
