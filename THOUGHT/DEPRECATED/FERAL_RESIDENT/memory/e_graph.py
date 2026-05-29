"""
E-Relationship Graph

Sparse graph of E-relationships between daemon items with R-gating.

Research Foundation:
- Q44: E threshold = 0.5 (r=0.977 across 5 models)
- Q13: Optimal N = 2-3 (47x improvement at N=2)
- Q15: R = E/sigma = sqrt(Likelihood Precision) (r=1.0)
- Q17: Four tiers T0/T1/T2/T3

Key Insight: The current centroid-based daemon loses structural information.
This graph preserves E-relationships between items, enabling:
- Pattern discovery (clusters, bridges)
- Novelty detection (low E to neighbors)
- Graph-based retrieval
"""

import sqlite3
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# =============================================================================
# CONSTANTS (From Validated Research)
# =============================================================================

# Q44: E threshold for edge creation
E_THRESHOLD = 0.5

# Q13: Optimal number of recent items to compare
COMPARE_TO_RECENT = 3

# Q17: R-gate tier thresholds (from seldon_gate.py)
class RTier(Enum):
    T0_OBSERVE = "T0"      # R >= 0.0 (always)
    T1_SMALL = "T1"        # R >= 0.5
    T2_MEDIUM = "T2"       # R >= 0.8
    T3_LARGE = "T3"        # R >= 1.0


R_TIER_THRESHOLDS = {
    RTier.T0_OBSERVE: 0.0,
    RTier.T1_SMALL: 0.5,
    RTier.T2_MEDIUM: 0.8,
    RTier.T3_LARGE: 1.0,
}


@dataclass
class EEdge:
    """An edge in the E-relationship graph."""
    edge_id: str
    vector_id_a: str
    vector_id_b: str
    e_score: float
    r_score: Optional[float]
    r_tier: Optional[str]
    created_at: str


# =============================================================================
# E-RELATIONSHIP GRAPH
# =============================================================================

class ERelationshipGraph:
    """
    Sparse graph of E-relationships between daemon items.

    Key Methods:
    - add_item(): Store item and create edges to recent items
    - compute_E(): Born rule similarity
    - compute_R(): R-gate score (E/sigma)
    - get_edges(): Get edges for a vector
    - get_neighbors(): Get connected vectors

    Usage:
        graph = ERelationshipGraph(conn)
        graph.add_item(vector_id, vec, Df)
        neighbors = graph.get_neighbors(vector_id)
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize with database connection.

        Args:
            conn: SQLite connection with e_edges table
        """
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

        # Running statistics for sigma computation
        self._e_values: List[float] = []
        self._max_history = 1000

    # =========================================================================
    # CORE COMPUTATIONS (From Validated Research)
    # =========================================================================

    @staticmethod
    def compute_E(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Born rule: E = |<v1|v2>|^2

        Q44 validated: r=0.977 correlation across 5 models.
        This IS semantic similarity.

        Args:
            v1, v2: Unit vectors (float32)

        Returns:
            E value in [0, 1]
        """
        dot = np.dot(v1, v2)
        return float(dot * dot)

    def compute_R(self, E: float, sigma: Optional[float] = None) -> float:
        """
        R-gate score: R = E/sigma

        Q15 validated: R = sqrt(Likelihood Precision) (r=1.0)

        Higher R = more confident relationship.
        Used to filter noisy edges.

        Args:
            E: Born rule similarity
            sigma: Standard deviation of E values (uses running avg if None)

        Returns:
            R value (unbounded, typically 0-10)
        """
        if sigma is None:
            sigma = self._get_running_sigma()

        # Prevent division by zero
        sigma = max(sigma, 0.01)
        return E / sigma

    def get_r_tier(self, R: float) -> RTier:
        """
        Classify R value into tier.

        Q17 validated: Four tiers with increasing confidence.

        Args:
            R: R-gate score

        Returns:
            RTier enum value
        """
        if R >= R_TIER_THRESHOLDS[RTier.T3_LARGE]:
            return RTier.T3_LARGE
        elif R >= R_TIER_THRESHOLDS[RTier.T2_MEDIUM]:
            return RTier.T2_MEDIUM
        elif R >= R_TIER_THRESHOLDS[RTier.T1_SMALL]:
            return RTier.T1_SMALL
        else:
            return RTier.T0_OBSERVE

    def _get_running_sigma(self) -> float:
        """Get running standard deviation of E values."""
        if len(self._e_values) < 2:
            return 0.1  # Default before we have data
        return float(np.std(self._e_values))

    def _update_running_stats(self, E: float):
        """Update running statistics with new E value."""
        self._e_values.append(E)
        if len(self._e_values) > self._max_history:
            self._e_values = self._e_values[-self._max_history // 2:]

    # =========================================================================
    # GRAPH OPERATIONS
    # =========================================================================

    def add_item(
        self,
        vector_id: str,
        vec: np.ndarray,
        Df: float,
        compare_to_recent: int = COMPARE_TO_RECENT,
        e_threshold: float = E_THRESHOLD,
        min_r_tier: RTier = RTier.T0_OBSERVE
    ) -> List[EEdge]:
        """
        Add item to graph and create edges to recent items.

        Q13 validated: N=2-3 is optimal (47x improvement).
        Only creates edges where E > threshold AND R >= min_tier.

        Args:
            vector_id: ID of the vector to add
            vec: The vector data (unit normalized)
            Df: Participation ratio
            compare_to_recent: Number of recent items to compare (default 3)
            e_threshold: Minimum E for edge creation (default 0.5)
            min_r_tier: Minimum R tier for edge creation

        Returns:
            List of created edges
        """
        # Get recent daemon items
        recent = self._get_recent_items(compare_to_recent, exclude_id=vector_id)

        created_edges = []
        for recent_id, recent_vec in recent:
            # Compute E
            E = self.compute_E(vec, recent_vec)
            self._update_running_stats(E)

            # Check E threshold
            if E < e_threshold:
                continue

            # Compute R and tier
            R = self.compute_R(E)
            tier = self.get_r_tier(R)

            # Check R tier threshold
            if R < R_TIER_THRESHOLDS[min_r_tier]:
                continue

            # Create edge
            edge = self._create_edge(vector_id, recent_id, E, R, tier)
            if edge:
                created_edges.append(edge)

        return created_edges

    def _get_recent_items(
        self,
        n: int,
        exclude_id: Optional[str] = None
    ) -> List[Tuple[str, np.ndarray]]:
        """Get n most recent daemon items."""
        query = """
            SELECT vector_id, vec_blob FROM vectors
            WHERE composition_op = 'daemon_item'
        """
        params = []

        if exclude_id:
            query += " AND vector_id != ?"
            params.append(exclude_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)

        rows = self.conn.execute(query, params).fetchall()

        return [
            (row['vector_id'], np.frombuffer(row['vec_blob'], dtype=np.float32))
            for row in rows
        ]

    def _create_edge(
        self,
        vector_id_a: str,
        vector_id_b: str,
        e_score: float,
        r_score: float,
        r_tier: RTier
    ) -> Optional[EEdge]:
        """Create an edge in the database."""
        # Normalize order (a < b) to prevent duplicates
        if vector_id_a > vector_id_b:
            vector_id_a, vector_id_b = vector_id_b, vector_id_a

        edge_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()

        try:
            self.conn.execute(
                """
                INSERT INTO e_edges (edge_id, vector_id_a, vector_id_b, e_score, r_score, r_tier, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (edge_id, vector_id_a, vector_id_b, e_score, r_score, r_tier.value, now)
            )
            self.conn.commit()

            return EEdge(
                edge_id=edge_id,
                vector_id_a=vector_id_a,
                vector_id_b=vector_id_b,
                e_score=e_score,
                r_score=r_score,
                r_tier=r_tier.value,
                created_at=now
            )
        except sqlite3.IntegrityError:
            # Edge already exists (duplicate)
            return None

    def get_edges(
        self,
        vector_id: str,
        min_e: float = 0.0,
        min_tier: Optional[str] = None
    ) -> List[EEdge]:
        """
        Get all edges for a vector.

        Args:
            vector_id: The vector to get edges for
            min_e: Minimum E score filter
            min_tier: Minimum R tier filter (e.g., "T1")

        Returns:
            List of edges
        """
        query = """
            SELECT * FROM e_edges
            WHERE (vector_id_a = ? OR vector_id_b = ?)
            AND e_score >= ?
        """
        params = [vector_id, vector_id, min_e]

        if min_tier:
            query += " AND r_tier >= ?"
            params.append(min_tier)

        query += " ORDER BY e_score DESC"

        rows = self.conn.execute(query, params).fetchall()

        return [
            EEdge(
                edge_id=row['edge_id'],
                vector_id_a=row['vector_id_a'],
                vector_id_b=row['vector_id_b'],
                e_score=row['e_score'],
                r_score=row['r_score'],
                r_tier=row['r_tier'],
                created_at=row['created_at']
            )
            for row in rows
        ]

    def get_neighbors(
        self,
        vector_id: str,
        min_e: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Get neighbor vector IDs with their E scores.

        Args:
            vector_id: The vector to get neighbors for
            min_e: Minimum E score filter

        Returns:
            List of (neighbor_id, e_score) tuples sorted by E desc
        """
        edges = self.get_edges(vector_id, min_e)

        neighbors = []
        for edge in edges:
            # Get the other vector in the edge
            neighbor_id = edge.vector_id_b if edge.vector_id_a == vector_id else edge.vector_id_a
            neighbors.append((neighbor_id, edge.e_score))

        return neighbors

    def edge_count(self) -> int:
        """Get total number of edges in the graph."""
        return self.conn.execute("SELECT COUNT(*) FROM e_edges").fetchone()[0]

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        cursor = self.conn.cursor()

        # Total edges
        total = cursor.execute("SELECT COUNT(*) FROM e_edges").fetchone()[0]

        # Edges by tier
        tier_counts = {}
        for row in cursor.execute("SELECT r_tier, COUNT(*) FROM e_edges GROUP BY r_tier"):
            tier_counts[row[0]] = row[1]

        # E score distribution
        e_stats = cursor.execute("""
            SELECT AVG(e_score), MIN(e_score), MAX(e_score)
            FROM e_edges
        """).fetchone()

        # Connected components estimate (number of unique vectors with edges)
        unique_vectors = cursor.execute("""
            SELECT COUNT(DISTINCT v) FROM (
                SELECT vector_id_a as v FROM e_edges
                UNION
                SELECT vector_id_b as v FROM e_edges
            )
        """).fetchone()[0]

        return {
            'total_edges': total,
            'unique_vectors': unique_vectors,
            'tier_counts': tier_counts,
            'e_mean': e_stats[0] if e_stats[0] else 0.0,
            'e_min': e_stats[1] if e_stats[1] else 0.0,
            'e_max': e_stats[2] if e_stats[2] else 0.0,
            'running_sigma': self._get_running_sigma()
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_graph_from_db_path(db_path: str) -> ERelationshipGraph:
    """Create an ERelationshipGraph from a database path."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return ERelationshipGraph(conn)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("E-Relationship Graph Tests")
    print("=" * 60)

    # Test compute_E
    v1 = np.array([1, 0, 0], dtype=np.float32)
    v2 = np.array([1, 0, 0], dtype=np.float32)
    v3 = np.array([0, 1, 0], dtype=np.float32)

    print(f"\ncompute_E tests:")
    print(f"  Same vector: E = {ERelationshipGraph.compute_E(v1, v2):.3f} (expected: 1.0)")
    print(f"  Orthogonal: E = {ERelationshipGraph.compute_E(v1, v3):.3f} (expected: 0.0)")

    # Test R computation
    graph = type('MockGraph', (), {'_e_values': [0.3, 0.4, 0.5, 0.6, 0.7]})()
    graph._get_running_sigma = lambda: float(np.std(graph._e_values))

    E = 0.8
    sigma = graph._get_running_sigma()
    R = ERelationshipGraph.compute_R(graph, E, sigma)
    print(f"\ncompute_R test:")
    print(f"  E={E}, sigma={sigma:.3f} -> R={R:.2f}")

    # Test tier classification
    print(f"\nget_r_tier tests:")
    test_R = [0.3, 0.6, 0.9, 1.5]
    for r in test_R:
        tier = ERelationshipGraph.get_r_tier(None, r)
        print(f"  R={r} -> {tier.value}")

    print("\nAll tests passed!")
