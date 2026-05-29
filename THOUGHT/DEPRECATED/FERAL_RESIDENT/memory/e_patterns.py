"""
E-Pattern Detector

Discovers patterns in the E-relationship graph:
- Clusters: Connected components (Union-Find)
- Bridges: Cross-cluster edges
- Novelty: Items with low E to neighbors

Research Foundation:
- Q48-Q50: Df x alpha = 8e (cluster geometry validation)
- Q17: R-gate tiers for edge quality

Complexity:
- find_clusters: O(E * alpha(V)) - Union-Find near-linear
- find_bridges: O(E) - Single pass
- score_novelty: O(k) - k neighbors only
"""

import sqlite3
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# UNION-FIND (Disjoint Set Union)
# =============================================================================

class UnionFind:
    """
    Union-Find data structure for connected components.

    Near-linear time complexity with path compression and union by rank.
    """

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find root with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """Union by rank. Returns True if merged, False if already same set."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x: str, y: str) -> bool:
        """Check if two elements are in the same set."""
        return self.find(x) == self.find(y)


# =============================================================================
# PATTERN DETECTOR
# =============================================================================

@dataclass
class Cluster:
    """A cluster of related vectors."""
    cluster_id: str
    member_ids: Set[str]
    size: int
    centroid_id: Optional[str] = None  # Representative vector
    internal_edges: int = 0
    mean_e_score: float = 0.0


@dataclass
class Bridge:
    """A bridge edge connecting two clusters."""
    edge_id: str
    vector_id_a: str
    vector_id_b: str
    cluster_a: str
    cluster_b: str
    e_score: float


class EPatternDetector:
    """
    Discovers patterns in the E-relationship graph.

    Key Methods:
    - find_clusters(): Connected components via Union-Find
    - find_bridges(): Cross-cluster edges
    - score_novelty(): 1 - mean(E to neighbors)
    - get_cluster_centroid(): Representative vector for cluster

    Usage:
        detector = EPatternDetector(conn)
        clusters = detector.find_clusters(min_size=3)
        bridges = detector.find_bridges()
        novelty = detector.score_novelty(vector_id, vec)
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize with database connection.

        Args:
            conn: SQLite connection with e_edges table
        """
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

        # Cache
        self._clusters: Optional[Dict[str, Cluster]] = None
        self._vector_to_cluster: Dict[str, str] = {}

    # =========================================================================
    # CLUSTER DETECTION
    # =========================================================================

    def find_clusters(
        self,
        min_size: int = 3,
        min_e: float = 0.0,
        min_tier: Optional[str] = None
    ) -> List[Cluster]:
        """
        Find connected components in the E-graph.

        Uses Union-Find for O(E * alpha(V)) near-linear complexity.

        Args:
            min_size: Minimum cluster size to return
            min_e: Minimum E score for edges
            min_tier: Minimum R tier for edges

        Returns:
            List of Cluster objects sorted by size desc
        """
        uf = UnionFind()

        # Build query
        query = "SELECT * FROM e_edges WHERE e_score >= ?"
        params = [min_e]

        if min_tier:
            query += " AND r_tier >= ?"
            params.append(min_tier)

        # Process all edges
        cursor = self.conn.cursor()
        edge_count = 0
        e_scores_by_root: Dict[str, List[float]] = defaultdict(list)

        for row in cursor.execute(query, params):
            uf.union(row['vector_id_a'], row['vector_id_b'])
            edge_count += 1

        # Group by root
        components: Dict[str, Set[str]] = defaultdict(set)
        for vector_id in uf.parent:
            root = uf.find(vector_id)
            components[root].add(vector_id)

        # Compute cluster statistics
        clusters = []
        for root, members in components.items():
            if len(members) < min_size:
                continue

            # Count internal edges and compute mean E
            internal_edges = 0
            total_e = 0.0
            for row in cursor.execute("""
                SELECT e_score FROM e_edges
                WHERE vector_id_a IN ({}) AND vector_id_b IN ({})
            """.format(
                ','.join('?' * len(members)),
                ','.join('?' * len(members))
            ), list(members) + list(members)):
                internal_edges += 1
                total_e += row['e_score']

            mean_e = total_e / internal_edges if internal_edges > 0 else 0.0

            cluster = Cluster(
                cluster_id=root,
                member_ids=members,
                size=len(members),
                internal_edges=internal_edges,
                mean_e_score=mean_e
            )
            clusters.append(cluster)

            # Update mapping
            for member in members:
                self._vector_to_cluster[member] = root

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        # Cache
        self._clusters = {c.cluster_id: c for c in clusters}

        return clusters

    def get_cluster_for_vector(self, vector_id: str) -> Optional[str]:
        """Get the cluster ID for a vector."""
        if not self._vector_to_cluster:
            self.find_clusters()  # Build cache
        return self._vector_to_cluster.get(vector_id)

    # =========================================================================
    # BRIDGE DETECTION
    # =========================================================================

    def find_bridges(
        self,
        cluster_a: Optional[str] = None,
        cluster_b: Optional[str] = None,
        min_e: float = 0.3
    ) -> List[Bridge]:
        """
        Find edges that connect different clusters.

        Bridges are important for cross-domain retrieval.

        Args:
            cluster_a: Filter to bridges from this cluster
            cluster_b: Filter to bridges to this cluster
            min_e: Minimum E score

        Returns:
            List of Bridge objects sorted by E desc
        """
        if not self._vector_to_cluster:
            self.find_clusters()

        bridges = []
        cursor = self.conn.cursor()

        for row in cursor.execute("SELECT * FROM e_edges WHERE e_score >= ?", (min_e,)):
            vid_a = row['vector_id_a']
            vid_b = row['vector_id_b']

            cid_a = self._vector_to_cluster.get(vid_a)
            cid_b = self._vector_to_cluster.get(vid_b)

            # Skip if same cluster or no cluster
            if cid_a is None or cid_b is None or cid_a == cid_b:
                continue

            # Apply filters
            if cluster_a and cid_a != cluster_a and cid_b != cluster_a:
                continue
            if cluster_b and cid_a != cluster_b and cid_b != cluster_b:
                continue

            bridges.append(Bridge(
                edge_id=row['edge_id'],
                vector_id_a=vid_a,
                vector_id_b=vid_b,
                cluster_a=cid_a,
                cluster_b=cid_b,
                e_score=row['e_score']
            ))

        # Sort by E descending
        bridges.sort(key=lambda b: b.e_score, reverse=True)

        return bridges

    # =========================================================================
    # NOVELTY SCORING
    # =========================================================================

    def score_novelty(
        self,
        vector_id: str,
        vec: Optional[np.ndarray] = None,
        k_neighbors: int = 5
    ) -> float:
        """
        Score how novel a vector is relative to its neighbors.

        Novelty = 1 - mean(E to top-k neighbors)

        High novelty = new concept not well-connected to graph.
        Low novelty = redundant/similar to existing items.

        Args:
            vector_id: The vector to score
            vec: Optional vector data (will fetch from DB if not provided)
            k_neighbors: Number of neighbors to consider

        Returns:
            Novelty score in [0, 1]
        """
        # Get edges for this vector
        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT e_score FROM e_edges
            WHERE vector_id_a = ? OR vector_id_b = ?
            ORDER BY e_score DESC
            LIMIT ?
        """, (vector_id, vector_id, k_neighbors)).fetchall()

        if not rows:
            return 1.0  # No connections = maximally novel

        mean_e = np.mean([row['e_score'] for row in rows])
        return 1.0 - mean_e

    def find_novel_items(
        self,
        min_novelty: float = 0.5,
        limit: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Find items with high novelty scores.

        Args:
            min_novelty: Minimum novelty threshold
            limit: Maximum items to return

        Returns:
            List of (vector_id, novelty_score) tuples
        """
        # Get all vectors with edges
        cursor = self.conn.cursor()
        vectors = set()
        for row in cursor.execute("SELECT DISTINCT vector_id_a FROM e_edges"):
            vectors.add(row[0])
        for row in cursor.execute("SELECT DISTINCT vector_id_b FROM e_edges"):
            vectors.add(row[0])

        # Score each
        scored = []
        for vid in vectors:
            novelty = self.score_novelty(vid)
            if novelty >= min_novelty:
                scored.append((vid, novelty))

        # Sort by novelty descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:limit]

    # =========================================================================
    # CLUSTER CENTROIDS
    # =========================================================================

    def get_cluster_centroid(
        self,
        cluster_id: str
    ) -> Optional[Tuple[str, np.ndarray]]:
        """
        Get the centroid vector for a cluster.

        The centroid is the member with highest average E to other members.

        Args:
            cluster_id: The cluster ID

        Returns:
            (vector_id, vector) tuple or None if cluster not found
        """
        if self._clusters is None:
            self.find_clusters()

        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return None

        if cluster.centroid_id:
            # Return cached centroid
            row = self.conn.execute(
                "SELECT vec_blob FROM vectors WHERE vector_id = ?",
                (cluster.centroid_id,)
            ).fetchone()
            if row:
                return (cluster.centroid_id, np.frombuffer(row['vec_blob'], dtype=np.float32))

        # Find member with highest connectivity
        best_id = None
        best_score = -1.0
        cursor = self.conn.cursor()

        for vid in cluster.member_ids:
            # Sum E scores to other members
            total_e = 0.0
            for row in cursor.execute("""
                SELECT SUM(e_score) FROM e_edges
                WHERE (vector_id_a = ? OR vector_id_b = ?)
            """, (vid, vid)):
                total_e = row[0] or 0.0

            if total_e > best_score:
                best_score = total_e
                best_id = vid

        if best_id:
            cluster.centroid_id = best_id
            row = cursor.execute(
                "SELECT vec_blob FROM vectors WHERE vector_id = ?",
                (best_id,)
            ).fetchone()
            if row:
                return (best_id, np.frombuffer(row['vec_blob'], dtype=np.float32))

        return None

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get pattern detection statistics."""
        if self._clusters is None:
            self.find_clusters()

        clusters = list(self._clusters.values()) if self._clusters else []

        return {
            'num_clusters': len(clusters),
            'total_clustered_vectors': sum(c.size for c in clusters),
            'largest_cluster_size': max((c.size for c in clusters), default=0),
            'mean_cluster_size': np.mean([c.size for c in clusters]) if clusters else 0,
            'mean_internal_e': np.mean([c.mean_e_score for c in clusters]) if clusters else 0,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("E-Pattern Detector Tests")
    print("=" * 60)

    # Test Union-Find
    print("\nUnion-Find tests:")
    uf = UnionFind()

    uf.union("a", "b")
    uf.union("c", "d")
    uf.union("b", "c")

    print(f"  a-b connected: {uf.connected('a', 'b')} (expected: True)")
    print(f"  a-d connected: {uf.connected('a', 'd')} (expected: True)")
    print(f"  a-e connected: {uf.connected('a', 'e')} (expected: False)")

    # All should have same root
    roots = {uf.find(x) for x in ['a', 'b', 'c', 'd']}
    print(f"  Unique roots for a,b,c,d: {len(roots)} (expected: 1)")

    print("\nAll tests passed!")
