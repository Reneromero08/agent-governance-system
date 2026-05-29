#!/usr/bin/env python3
"""
Hierarchy Builder for automatic tree maintenance (Phase J.3).

Handles:
- Incremental updates when turns are compressed
- Initial tree building with k-means clustering
- Level promotion (L1 full -> create new L1, 10 L1s -> create L2)

Key Design Decisions:
- Semantic clustering (k-means) is REQUIRED for meaningful centroids
- Optional PCA to Df=22 dimensions removes noise in high dimensions
- Incremental updates use running mean formula: (old * n + new) / (n + 1)

Part of Phase J.3: Hierarchy Builder for automatic tree maintenance.
"""

import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

from .hierarchy_schema import (
    HierarchyNode,
    generate_node_id,
    L0, L1, L2, L3,
    CHILDREN_PER_LEVEL,
    EMBEDDING_DIM,
)
from .centroid_math import (
    compute_centroid,
    update_centroid_incremental,
    merge_centroids,
)
from .hierarchy_retriever import (
    ensure_hierarchy_schema,
    store_hierarchy_node,
    store_hierarchy_batch,
    load_root_nodes,
    load_node_children,
)

# Constants for promotion thresholds
# L1 nodes group 100 L0 nodes
# L2 nodes group 10 L1 nodes (so 1000 L0 nodes total)
L1_CHILDREN_THRESHOLD = CHILDREN_PER_LEVEL  # 100
L2_CHILDREN_THRESHOLD = 10  # 10 L1 nodes -> 1 L2 node
L3_CHILDREN_THRESHOLD = 10  # 10 L2 nodes -> 1 L3 node

# Default PCA dimensions for initial build
DEFAULT_PCA_DIMS = 22


@dataclass
class OpenNodeInfo:
    """Tracks an open (non-full) node that can accept more children."""
    node_id: str
    level: int
    centroid: np.ndarray
    child_count: int
    first_turn_seq: int
    last_turn_seq: int


class HierarchyBuilder:
    """
    Builds and maintains the centroid hierarchy.

    Supports two modes:
    1. Incremental: on_turn_compressed() for real-time updates
    2. Batch: build_initial_hierarchy() for bootstrap from existing data

    Usage:
        builder = HierarchyBuilder(db_path, session_id)

        # Incremental mode (during conversation)
        builder.on_turn_compressed(event_id, turn_vec, content_hash, seq_num)

        # Batch mode (bootstrap)
        builder.build_initial_hierarchy(vectors, event_ids, content_hashes)
    """

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        children_per_level: int = CHILDREN_PER_LEVEL,
    ):
        """
        Initialize hierarchy builder.

        Args:
            db_path: Path to SQLite database (cat_chat.db)
            session_id: Session identifier for node storage
            children_per_level: Max children per L1 node (default: 100)
        """
        self.db_path = Path(db_path)
        self.session_id = session_id
        self.children_per_level = children_per_level

        # Track open nodes at each level (nodes still accepting children)
        # Key: level, Value: OpenNodeInfo
        self._open_nodes: Dict[int, OpenNodeInfo] = {}

        # Track sequence counters for node ID generation
        self._level_counters: Dict[int, int] = {L0: 0, L1: 0, L2: 0, L3: 0}

        # Database connection (lazy initialized)
        self._conn: Optional[sqlite3.Connection] = None

        # Logger
        self._logger = logging.getLogger(__name__)

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            ensure_hierarchy_schema(self._conn)
        return self._conn

    def _generate_node_id(self, level: int) -> str:
        """Generate unique node ID for a level."""
        seq = self._level_counters[level]
        self._level_counters[level] += 1
        return generate_node_id(self.session_id, level, seq)

    def _initialize_counters_from_db(self) -> None:
        """Initialize level counters from existing database state."""
        conn = self._get_conn()

        for level in [L0, L1, L2, L3]:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM hierarchy_nodes
                WHERE session_id = ? AND level = ?
            """, (self.session_id, level))
            count = cursor.fetchone()[0]
            self._level_counters[level] = count

    def _load_open_nodes(self) -> None:
        """Load open (non-full) nodes from database."""
        conn = self._get_conn()

        # Find the most recent non-full node at each level
        for level in [L1, L2, L3]:
            max_children = self._get_max_children_for_level(level)

            # Find nodes with less than max children
            cursor = conn.execute("""
                SELECT
                    h.node_id,
                    h.level,
                    h.centroid,
                    h.token_count,
                    COUNT(c.node_id) as child_count
                FROM hierarchy_nodes h
                LEFT JOIN hierarchy_nodes c ON c.parent_id = h.node_id
                WHERE h.session_id = ? AND h.level = ? AND h.parent_id IS NULL
                GROUP BY h.node_id
                HAVING child_count < ?
                ORDER BY h.created_at DESC
                LIMIT 1
            """, (self.session_id, level, max_children))

            row = cursor.fetchone()
            if row:
                centroid = np.frombuffer(row["centroid"], dtype=np.float32).copy()
                self._open_nodes[level] = OpenNodeInfo(
                    node_id=row["node_id"],
                    level=level,
                    centroid=centroid,
                    child_count=row["child_count"],
                    first_turn_seq=0,  # Not tracked in schema
                    last_turn_seq=0,   # Not tracked in schema
                )

    def _get_max_children_for_level(self, level: int) -> int:
        """Get maximum children for a level."""
        if level == L1:
            return self.children_per_level  # 100 L0 children
        elif level in (L2, L3):
            return L2_CHILDREN_THRESHOLD  # 10 children
        return self.children_per_level

    def on_turn_compressed(
        self,
        event_id: str,
        turn_vec: np.ndarray,
        content_hash: str,
        sequence_num: int,
        token_count: int = 100,
    ) -> str:
        """
        Called after a turn is compressed. Updates hierarchy incrementally.

        Steps:
        1. Create L0 node for this turn
        2. Add to current open L1 node (or create new one)
        3. Update L1 centroid incrementally: (old * n + new) / (n + 1)
        4. Check if L1 is full (100 children) -> close it, start new L1
        5. Check if 10 L1s exist -> create L2 parent

        Args:
            event_id: Event ID from session_events
            turn_vec: Embedding vector for the turn (384 dims)
            content_hash: Content hash for the turn
            sequence_num: Sequence number in session
            token_count: Approximate token count for budget tracking

        Returns:
            Node ID of the created L0 node
        """
        conn = self._get_conn()

        # Lazy initialize counters and open nodes
        if not self._level_counters[L0]:
            self._initialize_counters_from_db()
            self._load_open_nodes()

        # 1. Create L0 node
        l0_node_id = self._generate_node_id(L0)
        l0_node = HierarchyNode(
            node_id=l0_node_id,
            session_id=self.session_id,
            level=L0,
            centroid=turn_vec,
            turn_count=1,
            first_turn_seq=sequence_num,
            last_turn_seq=sequence_num,
        )

        # Set event_id and content_hash via to_dict workaround
        # (HierarchyNode from hierarchy_schema doesn't have these fields,
        # but hierarchy_retriever.HierarchyNode does - we store directly)
        conn.execute("""
            INSERT OR REPLACE INTO hierarchy_nodes
            (node_id, session_id, level, centroid, event_id, content_hash,
             parent_id, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            l0_node_id,
            self.session_id,
            L0,
            turn_vec.astype(np.float32).tobytes(),
            event_id,
            content_hash,
            None,  # parent_id set below
            token_count,
            datetime.now(timezone.utc).isoformat(),
        ))

        # 2. Add to open L1 (or create new one)
        l1_node_id = self._add_to_open_l1(l0_node_id, turn_vec, sequence_num, token_count)

        # 3. Update L0's parent_id
        conn.execute("""
            UPDATE hierarchy_nodes SET parent_id = ?
            WHERE node_id = ?
        """, (l1_node_id, l0_node_id))

        conn.commit()

        # 4. Check for L2 promotion
        self._check_l2_promotion()

        # 5. Check for L3 promotion
        self._check_l3_promotion()

        return l0_node_id

    def _add_to_open_l1(
        self,
        l0_node_id: str,
        l0_vec: np.ndarray,
        sequence_num: int,
        token_count: int,
    ) -> str:
        """Add L0 node to current open L1, creating new L1 if needed."""
        conn = self._get_conn()

        # Get or create open L1
        if L1 not in self._open_nodes:
            # Create new L1
            l1_node_id = self._generate_node_id(L1)
            self._open_nodes[L1] = OpenNodeInfo(
                node_id=l1_node_id,
                level=L1,
                centroid=l0_vec.copy(),  # First child's centroid
                child_count=0,
                first_turn_seq=sequence_num,
                last_turn_seq=sequence_num,
            )

            # Insert new L1 node
            conn.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                l1_node_id,
                self.session_id,
                L1,
                l0_vec.astype(np.float32).tobytes(),
                token_count,
                datetime.now(timezone.utc).isoformat(),
            ))

        open_l1 = self._open_nodes[L1]

        # Update L1 centroid incrementally
        new_centroid = update_centroid_incremental(
            open_l1.centroid,
            open_l1.child_count,
            l0_vec
        )
        open_l1.centroid = new_centroid
        open_l1.child_count += 1
        open_l1.last_turn_seq = sequence_num

        # Get current token count for L1
        cursor = conn.execute(
            "SELECT token_count FROM hierarchy_nodes WHERE node_id = ?",
            (open_l1.node_id,)
        )
        current_tokens = cursor.fetchone()[0] or 0

        # Update L1 in database
        conn.execute("""
            UPDATE hierarchy_nodes
            SET centroid = ?, token_count = ?
            WHERE node_id = ?
        """, (
            new_centroid.astype(np.float32).tobytes(),
            current_tokens + token_count,
            open_l1.node_id,
        ))

        # Check if L1 is now full
        if open_l1.child_count >= self.children_per_level:
            self._close_l1_node()

        return open_l1.node_id

    def _close_l1_node(self) -> None:
        """Close current L1 node (mark it as full)."""
        if L1 in self._open_nodes:
            self._logger.debug(
                "L1 node %s is full with %d children",
                self._open_nodes[L1].node_id,
                self._open_nodes[L1].child_count
            )
            del self._open_nodes[L1]

    def _check_l2_promotion(self) -> None:
        """Check if we need to create an L2 parent for orphan L1 nodes."""
        conn = self._get_conn()

        # Count L1 nodes without L2 parent
        cursor = conn.execute("""
            SELECT COUNT(*) FROM hierarchy_nodes
            WHERE session_id = ? AND level = ? AND parent_id IS NULL
        """, (self.session_id, L1))
        orphan_l1_count = cursor.fetchone()[0]

        if orphan_l1_count >= L2_CHILDREN_THRESHOLD:
            self._create_l2_parent()

    def _create_l2_parent(self) -> None:
        """Create L2 parent for orphan L1 nodes."""
        conn = self._get_conn()

        # Get orphan L1 nodes
        cursor = conn.execute("""
            SELECT node_id, centroid, token_count
            FROM hierarchy_nodes
            WHERE session_id = ? AND level = ? AND parent_id IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (self.session_id, L1, L2_CHILDREN_THRESHOLD))

        rows = cursor.fetchall()
        if len(rows) < L2_CHILDREN_THRESHOLD:
            return

        # Compute L2 centroid from L1 children
        centroids_with_counts = []
        total_tokens = 0
        l1_node_ids = []

        for row in rows:
            centroid = np.frombuffer(row["centroid"], dtype=np.float32)
            token_count = row["token_count"] or 0
            centroids_with_counts.append((centroid, token_count))
            total_tokens += token_count
            l1_node_ids.append(row["node_id"])

        merged_centroid, _ = merge_centroids(centroids_with_counts)

        # Create L2 node
        l2_node_id = self._generate_node_id(L2)
        conn.execute("""
            INSERT INTO hierarchy_nodes
            (node_id, session_id, level, centroid, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            l2_node_id,
            self.session_id,
            L2,
            merged_centroid.astype(np.float32).tobytes(),
            total_tokens,
            datetime.now(timezone.utc).isoformat(),
        ))

        # Update L1 nodes to point to new L2 parent
        for l1_id in l1_node_ids:
            conn.execute("""
                UPDATE hierarchy_nodes SET parent_id = ?
                WHERE node_id = ?
            """, (l2_node_id, l1_id))

        conn.commit()

        self._logger.info(
            "Created L2 node %s with %d L1 children",
            l2_node_id, len(l1_node_ids)
        )

    def _check_l3_promotion(self) -> None:
        """Check if we need to create an L3 parent for orphan L2 nodes."""
        conn = self._get_conn()

        # Count L2 nodes without L3 parent
        cursor = conn.execute("""
            SELECT COUNT(*) FROM hierarchy_nodes
            WHERE session_id = ? AND level = ? AND parent_id IS NULL
        """, (self.session_id, L2))
        orphan_l2_count = cursor.fetchone()[0]

        if orphan_l2_count >= L3_CHILDREN_THRESHOLD:
            self._create_l3_parent()

    def _create_l3_parent(self) -> None:
        """Create L3 parent for orphan L2 nodes."""
        conn = self._get_conn()

        # Get orphan L2 nodes
        cursor = conn.execute("""
            SELECT node_id, centroid, token_count
            FROM hierarchy_nodes
            WHERE session_id = ? AND level = ? AND parent_id IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (self.session_id, L2, L3_CHILDREN_THRESHOLD))

        rows = cursor.fetchall()
        if len(rows) < L3_CHILDREN_THRESHOLD:
            return

        # Compute L3 centroid from L2 children
        centroids_with_counts = []
        total_tokens = 0
        l2_node_ids = []

        for row in rows:
            centroid = np.frombuffer(row["centroid"], dtype=np.float32)
            token_count = row["token_count"] or 0
            centroids_with_counts.append((centroid, token_count))
            total_tokens += token_count
            l2_node_ids.append(row["node_id"])

        merged_centroid, _ = merge_centroids(centroids_with_counts)

        # Create L3 node
        l3_node_id = self._generate_node_id(L3)
        conn.execute("""
            INSERT INTO hierarchy_nodes
            (node_id, session_id, level, centroid, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            l3_node_id,
            self.session_id,
            L3,
            merged_centroid.astype(np.float32).tobytes(),
            total_tokens,
            datetime.now(timezone.utc).isoformat(),
        ))

        # Update L2 nodes to point to new L3 parent
        for l2_id in l2_node_ids:
            conn.execute("""
                UPDATE hierarchy_nodes SET parent_id = ?
                WHERE node_id = ?
            """, (l3_node_id, l2_id))

        conn.commit()

        self._logger.info(
            "Created L3 node %s with %d L2 children",
            l3_node_id, len(l2_node_ids)
        )

    def build_initial_hierarchy(
        self,
        vectors: List[np.ndarray],
        event_ids: List[str],
        content_hashes: List[str],
        token_counts: Optional[List[int]] = None,
        use_kmeans: bool = True,
        pca_dims: Optional[int] = DEFAULT_PCA_DIMS,
    ) -> int:
        """
        Build hierarchy from existing vectors.

        If use_kmeans=True (recommended), clusters vectors semantically.
        If pca_dims is set, projects to that dimensionality first.

        Args:
            vectors: List of embedding vectors
            event_ids: List of event IDs for each vector
            content_hashes: List of content hashes for each vector
            token_counts: Optional list of token counts (default: 100 each)
            use_kmeans: Use k-means clustering (recommended)
            pca_dims: PCA dimensions (default: 22), None to disable

        Returns:
            Number of nodes created
        """
        if not vectors:
            return 0

        n = len(vectors)
        if token_counts is None:
            token_counts = [100] * n

        conn = self._get_conn()

        # Convert to numpy array
        vectors_array = np.stack(vectors, axis=0)

        if use_kmeans:
            nodes_created = self._build_with_kmeans(
                vectors_array, event_ids, content_hashes, token_counts,
                pca_dims=pca_dims
            )
        else:
            nodes_created = self._build_sequential(
                vectors_array, event_ids, content_hashes, token_counts
            )

        conn.commit()
        return nodes_created

    def _build_with_kmeans(
        self,
        vectors: np.ndarray,
        event_ids: List[str],
        content_hashes: List[str],
        token_counts: List[int],
        pca_dims: Optional[int] = None,
    ) -> int:
        """Build hierarchy using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError:
            self._logger.warning(
                "sklearn not available, falling back to sequential build"
            )
            return self._build_sequential(
                vectors, event_ids, content_hashes, token_counts
            )

        n = len(vectors)
        conn = self._get_conn()
        nodes_created = 0

        # Optional PCA projection (only if enough samples)
        if pca_dims is not None and pca_dims < vectors.shape[1] and n > pca_dims:
            self._logger.info("Applying PCA to %d dimensions", pca_dims)
            pca = PCA(n_components=pca_dims, random_state=42)
            projected = pca.fit_transform(vectors)
        else:
            projected = vectors

        # Determine number of L1 clusters
        # Aim for ~children_per_level items per cluster
        n_l1_clusters = max(1, n // self.children_per_level)
        if n_l1_clusters > n:
            n_l1_clusters = n

        self._logger.info(
            "Building hierarchy with k-means: %d vectors -> %d L1 clusters",
            n, n_l1_clusters
        )

        # Cluster into L1 groups
        kmeans = KMeans(n_clusters=n_l1_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(projected)

        # Group vectors by cluster
        clusters: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        # Create L0 and L1 nodes
        l1_nodes = []

        for cluster_id, indices in clusters.items():
            # Create L1 node
            l1_node_id = self._generate_node_id(L1)

            # Compute L1 centroid from actual vectors (not projected)
            cluster_vectors = [vectors[i] for i in indices]
            l1_centroid = compute_centroid(cluster_vectors)
            cluster_tokens = sum(token_counts[i] for i in indices)

            # Insert L1 node
            conn.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                l1_node_id,
                self.session_id,
                L1,
                l1_centroid.astype(np.float32).tobytes(),
                cluster_tokens,
                datetime.now(timezone.utc).isoformat(),
            ))
            nodes_created += 1

            # Create L0 nodes for this cluster
            for i in indices:
                l0_node_id = self._generate_node_id(L0)
                conn.execute("""
                    INSERT INTO hierarchy_nodes
                    (node_id, session_id, level, centroid, event_id, content_hash,
                     parent_id, token_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    l0_node_id,
                    self.session_id,
                    L0,
                    vectors[i].astype(np.float32).tobytes(),
                    event_ids[i],
                    content_hashes[i],
                    l1_node_id,
                    token_counts[i],
                    datetime.now(timezone.utc).isoformat(),
                ))
                nodes_created += 1

            l1_nodes.append((l1_node_id, l1_centroid, cluster_tokens))

        # Build L2+ levels if needed
        nodes_created += self._build_higher_levels(l1_nodes)

        return nodes_created

    def _build_sequential(
        self,
        vectors: np.ndarray,
        event_ids: List[str],
        content_hashes: List[str],
        token_counts: List[int],
    ) -> int:
        """Build hierarchy using sequential grouping (fallback)."""
        n = len(vectors)
        conn = self._get_conn()
        nodes_created = 0

        self._logger.info(
            "Building hierarchy sequentially: %d vectors", n
        )

        # Group into L1 nodes of children_per_level
        l1_nodes = []

        for start in range(0, n, self.children_per_level):
            end = min(start + self.children_per_level, n)
            group_indices = list(range(start, end))

            # Create L1 node
            l1_node_id = self._generate_node_id(L1)
            cluster_vectors = [vectors[i] for i in group_indices]
            l1_centroid = compute_centroid(cluster_vectors)
            cluster_tokens = sum(token_counts[i] for i in group_indices)

            # Insert L1 node
            conn.execute("""
                INSERT INTO hierarchy_nodes
                (node_id, session_id, level, centroid, token_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                l1_node_id,
                self.session_id,
                L1,
                l1_centroid.astype(np.float32).tobytes(),
                cluster_tokens,
                datetime.now(timezone.utc).isoformat(),
            ))
            nodes_created += 1

            # Create L0 nodes
            for i in group_indices:
                l0_node_id = self._generate_node_id(L0)
                conn.execute("""
                    INSERT INTO hierarchy_nodes
                    (node_id, session_id, level, centroid, event_id, content_hash,
                     parent_id, token_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    l0_node_id,
                    self.session_id,
                    L0,
                    vectors[i].astype(np.float32).tobytes(),
                    event_ids[i],
                    content_hashes[i],
                    l1_node_id,
                    token_counts[i],
                    datetime.now(timezone.utc).isoformat(),
                ))
                nodes_created += 1

            l1_nodes.append((l1_node_id, l1_centroid, cluster_tokens))

        # Build L2+ levels if needed
        nodes_created += self._build_higher_levels(l1_nodes)

        return nodes_created

    def _build_higher_levels(
        self,
        lower_nodes: List[Tuple[str, np.ndarray, int]],
    ) -> int:
        """Build L2 and L3 levels from lower level nodes."""
        conn = self._get_conn()
        nodes_created = 0
        current_level = L1
        current_nodes = lower_nodes

        while len(current_nodes) > 1:
            next_level = current_level + 1
            if next_level > L3:
                break

            threshold = L2_CHILDREN_THRESHOLD if next_level >= L2 else self.children_per_level
            next_nodes = []

            for start in range(0, len(current_nodes), threshold):
                end = min(start + threshold, len(current_nodes))
                group = current_nodes[start:end]

                if len(group) < threshold and len(next_nodes) > 0:
                    # Partial group at end, merge with previous or keep as is
                    break

                # Create parent node
                parent_node_id = self._generate_node_id(next_level)
                centroids_with_counts = [(c, t) for _, c, t in group]
                parent_centroid, _ = merge_centroids(centroids_with_counts)
                parent_tokens = sum(t for _, _, t in group)

                conn.execute("""
                    INSERT INTO hierarchy_nodes
                    (node_id, session_id, level, centroid, token_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    parent_node_id,
                    self.session_id,
                    next_level,
                    parent_centroid.astype(np.float32).tobytes(),
                    parent_tokens,
                    datetime.now(timezone.utc).isoformat(),
                ))
                nodes_created += 1

                # Update children to point to parent
                for child_id, _, _ in group:
                    conn.execute("""
                        UPDATE hierarchy_nodes SET parent_id = ?
                        WHERE node_id = ?
                    """, (parent_node_id, child_id))

                next_nodes.append((parent_node_id, parent_centroid, parent_tokens))

            if not next_nodes:
                break

            current_nodes = next_nodes
            current_level = next_level

        return nodes_created

    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchy statistics for this session."""
        conn = self._get_conn()

        stats = {
            "session_id": self.session_id,
            "levels": {},
            "total_nodes": 0,
        }

        for level in [L0, L1, L2, L3]:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM hierarchy_nodes
                WHERE session_id = ? AND level = ?
            """, (self.session_id, level))
            count = cursor.fetchone()[0]
            stats["levels"][f"L{level}"] = count
            stats["total_nodes"] += count

        # Get orphan counts (nodes without parents that should have them)
        for level in [L1, L2]:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM hierarchy_nodes
                WHERE session_id = ? AND level = ? AND parent_id IS NULL
            """, (self.session_id, level))
            stats["levels"][f"L{level}_orphans"] = cursor.fetchone()[0]

        return stats

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "HierarchyBuilder":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def get_hierarchy_builder(
    db_path: Path,
    session_id: str,
) -> HierarchyBuilder:
    """
    Factory function to create a HierarchyBuilder.

    Args:
        db_path: Path to database file
        session_id: Session identifier

    Returns:
        Configured HierarchyBuilder instance
    """
    return HierarchyBuilder(db_path, session_id)


if __name__ == "__main__":
    import tempfile

    print("Hierarchy Builder - Quick Test")
    print("=" * 50)

    # Create test data
    np.random.seed(42)
    n_vectors = 250  # Enough to trigger L2 creation

    vectors = []
    event_ids = []
    content_hashes = []

    for i in range(n_vectors):
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
        event_ids.append(f"evt_{i:04d}")
        content_hashes.append(f"hash_{i:04d}")

    # Test with temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_hierarchy.db"

        with HierarchyBuilder(db_path, "test_session") as builder:
            # Test batch build with k-means
            print("\nTesting batch build with k-means...")
            nodes_created = builder.build_initial_hierarchy(
                vectors, event_ids, content_hashes,
                use_kmeans=True,
                pca_dims=22
            )
            print(f"Nodes created: {nodes_created}")

            stats = builder.get_stats()
            print(f"Stats: {stats}")

            # Verify structure
            assert stats["levels"]["L0"] == n_vectors
            assert stats["levels"]["L1"] > 0
            print("Batch build with k-means: PASSED")

        # Test incremental build
        print("\nTesting incremental build...")
        with HierarchyBuilder(db_path, "test_session_incr") as builder:
            for i in range(150):  # Enough to trigger L1 completion
                vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                builder.on_turn_compressed(
                    event_id=f"incr_evt_{i:04d}",
                    turn_vec=vec,
                    content_hash=f"incr_hash_{i:04d}",
                    sequence_num=i,
                    token_count=100
                )

            stats = builder.get_stats()
            print(f"Incremental stats: {stats}")

            assert stats["levels"]["L0"] == 150
            assert stats["levels"]["L1"] >= 1
            print("Incremental build: PASSED")

    print("\nAll tests passed!")
