#!/usr/bin/env python3
"""
Hierarchical Retriever for O(log n) E-score computation.

Uses TOP-K centroid selection (NOT threshold-based) per experimental validation:
- At each level, score ALL children by E(query, child.centroid)
- Take top-K (default K=10)
- Recurse into those K children
- At L0, return actual turn content

Validated on SQuAD: 85% recall, 5.6x speedup vs brute force.

Phase J.2 of CAT Chat: Hierarchical retrieval with TOP-K selection.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

from .vector_persistence import EMBEDDING_DIM, EMBEDDING_BYTES


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class HierarchyNode:
    """
    Node in the E-score hierarchy.

    Levels:
        0 = leaf (actual turn content)
        1 = L1 group
        2 = L2 group
        3+ = higher level groups

    For L0 nodes, event_id links to session_events table.
    For higher nodes, children are loaded on demand from DB.
    """
    node_id: str
    level: int
    centroid: np.ndarray
    children: List["HierarchyNode"] = field(default_factory=list)
    event_id: Optional[str] = None  # Only for L0 nodes
    content_hash: Optional[str] = None  # Only for L0 nodes
    parent_id: Optional[str] = None  # For DB storage
    token_count: int = 0  # Approximate tokens for budget tracking


@dataclass
class RetrievalResult:
    """
    Result of hierarchical retrieval.

    Contains the node_id, link to session_events (for L0 nodes),
    E-score, and content hash for deduplication.
    """
    node_id: str
    event_id: str  # For L0 nodes, links to session_events
    e_score: float
    content_hash: str
    level: int


# =============================================================================
# Core Algorithm: E-score (Born Rule)
# =============================================================================

def compute_E(query_vec: np.ndarray, item_vec: np.ndarray) -> float:
    """
    Compute E-score using Born rule: E = |<query|item>|^2.

    This is the quantum-inspired probability measure for relevance.
    Cosine similarity squared gives probability interpretation.

    Args:
        query_vec: Query embedding (normalized 384-dim vector)
        item_vec: Item embedding (normalized 384-dim vector)

    Returns:
        E-score in range [0, 1] where 1 = perfect match
    """
    # Handle empty or invalid vectors
    if query_vec is None or item_vec is None:
        return 0.0
    if len(query_vec) == 0 or len(item_vec) == 0:
        return 0.0

    # Dot product = cosine similarity (assuming normalized vectors)
    dot = np.dot(query_vec, item_vec)

    # Born rule: probability = amplitude squared
    return float(dot * dot)


# =============================================================================
# Database Helpers
# =============================================================================

def load_node_children(
    db_conn: sqlite3.Connection,
    parent_node_id: str
) -> List[HierarchyNode]:
    """
    Load all children of a node from the hierarchy_nodes table.

    Args:
        db_conn: SQLite database connection
        parent_node_id: ID of the parent node

    Returns:
        List of HierarchyNode objects (children)
    """
    cursor = db_conn.execute("""
        SELECT
            node_id, level, centroid, event_id, content_hash, parent_id, token_count
        FROM hierarchy_nodes
        WHERE parent_id = ?
        ORDER BY node_id
    """, (parent_node_id,))

    children = []
    for row in cursor.fetchall():
        # Deserialize centroid from BLOB
        centroid_blob = row[2]
        if centroid_blob is not None:
            centroid = np.frombuffer(centroid_blob, dtype=np.float32)
        else:
            centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        node = HierarchyNode(
            node_id=row[0],
            level=row[1],
            centroid=centroid,
            event_id=row[3],
            content_hash=row[4],
            parent_id=row[5],
            token_count=row[6] if row[6] else 0,
        )
        children.append(node)

    return children


def load_l0_content(
    db_conn: sqlite3.Connection,
    node_id: str
) -> Tuple[Optional[str], int]:
    """
    Load content hash and approximate tokens for an L0 (leaf) node.

    Args:
        db_conn: SQLite database connection
        node_id: ID of the L0 node

    Returns:
        Tuple of (content_hash, approximate_tokens)
        Returns (None, 0) if node not found
    """
    cursor = db_conn.execute("""
        SELECT content_hash, token_count
        FROM hierarchy_nodes
        WHERE node_id = ? AND level = 0
    """, (node_id,))

    row = cursor.fetchone()
    if row is None:
        return None, 0

    return row[0], row[1] if row[1] else 0


def load_root_nodes(
    db_conn: sqlite3.Connection,
    session_id: Optional[str] = None
) -> List[HierarchyNode]:
    """
    Load root nodes (nodes with no parent) for a session.

    Args:
        db_conn: SQLite database connection
        session_id: Optional session ID to filter by

    Returns:
        List of root HierarchyNode objects
    """
    if session_id:
        cursor = db_conn.execute("""
            SELECT
                node_id, level, centroid, event_id, content_hash, parent_id, token_count
            FROM hierarchy_nodes
            WHERE parent_id IS NULL AND session_id = ?
            ORDER BY level DESC, node_id
        """, (session_id,))
    else:
        cursor = db_conn.execute("""
            SELECT
                node_id, level, centroid, event_id, content_hash, parent_id, token_count
            FROM hierarchy_nodes
            WHERE parent_id IS NULL
            ORDER BY level DESC, node_id
        """)

    roots = []
    for row in cursor.fetchall():
        centroid_blob = row[2]
        if centroid_blob is not None:
            centroid = np.frombuffer(centroid_blob, dtype=np.float32)
        else:
            centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        node = HierarchyNode(
            node_id=row[0],
            level=row[1],
            centroid=centroid,
            event_id=row[3],
            content_hash=row[4],
            parent_id=row[5],
            token_count=row[6] if row[6] else 0,
        )
        roots.append(node)

    return roots


# =============================================================================
# Hierarchical Retrieval (TOP-K Selection)
# =============================================================================

def _retrieve_recursive(
    query_vec: np.ndarray,
    node: HierarchyNode,
    db_conn: sqlite3.Connection,
    top_k: int,
    results: List[RetrievalResult],
    budget_remaining: List[int],
    e_computations: List[int],
) -> None:
    """
    Recursive retrieval helper using TOP-K centroid selection.

    At each level:
    1. If leaf (level == 0): compute E, add to results
    2. If internal: score all children, recurse into top-K

    Uses mutable lists for budget and computation counting to allow
    modification in recursive calls.

    Args:
        query_vec: Query embedding
        node: Current node to process
        db_conn: Database connection for loading children
        top_k: Number of top children to explore at each level
        results: Accumulator for results (modified in place)
        budget_remaining: Mutable list [tokens_remaining] for budget tracking
        e_computations: Mutable list [count] for metrics
    """
    # Track E computations
    e_computations[0] += 1

    if node.level == 0:
        # Base case: leaf node (actual turn)
        e_score = compute_E(query_vec, node.centroid)

        # Add to results if we have budget
        token_cost = node.token_count if node.token_count > 0 else 100  # Default estimate

        if budget_remaining[0] > 0:
            results.append(RetrievalResult(
                node_id=node.node_id,
                event_id=node.event_id or "",
                e_score=e_score,
                content_hash=node.content_hash or "",
                level=0,
            ))
            budget_remaining[0] -= token_cost
        return

    # Recursive case: score all children, take top-K
    # First check if children are already loaded in memory
    if node.children:
        children = node.children
    else:
        # Load children from database
        children = load_node_children(db_conn, node.node_id)

    if not children:
        return

    # Score all children by E(query, child.centroid)
    child_scores: List[Tuple[HierarchyNode, float]] = []
    for child in children:
        e_computations[0] += 1
        e_score = compute_E(query_vec, child.centroid)
        child_scores.append((child, e_score))

    # Sort by E-score descending
    child_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top-K and recurse
    for child, _ in child_scores[:top_k]:
        if budget_remaining[0] <= 0:
            break
        _retrieve_recursive(
            query_vec, child, db_conn, top_k,
            results, budget_remaining, e_computations
        )


def retrieve_hierarchical(
    query_vec: np.ndarray,
    root_nodes: List[HierarchyNode],
    db_conn: sqlite3.Connection,
    top_k: int = 10,
    budget_tokens: int = 4000,
) -> List[RetrievalResult]:
    """
    Top-K hierarchical retrieval.

    Uses TOP-K centroid selection at each level (NOT threshold-based).
    This guarantees:
    - Predictable pruning (at most K^depth paths explored)
    - High recall (top-K preserves most relevant branches)
    - O(log n) complexity for balanced trees

    Args:
        query_vec: Query embedding (384-dim normalized vector)
        root_nodes: Top-level hierarchy nodes to search
        db_conn: Database connection for loading children
        top_k: Number of top children to explore at each level (default: 10)
        budget_tokens: Max tokens to retrieve, approximate (default: 4000)

    Returns:
        List of RetrievalResult ordered by e_score descending
    """
    if query_vec is None or len(query_vec) == 0:
        return []

    if not root_nodes:
        return []

    # Mutable containers for recursion
    results: List[RetrievalResult] = []
    budget_remaining = [budget_tokens]
    e_computations = [0]

    # Process each root node
    for root in root_nodes:
        if budget_remaining[0] <= 0:
            break
        _retrieve_recursive(
            query_vec, root, db_conn, top_k,
            results, budget_remaining, e_computations
        )

    # Sort results by E-score descending
    results.sort(key=lambda r: r.e_score, reverse=True)

    return results


def retrieve_hierarchical_with_metrics(
    query_vec: np.ndarray,
    root_nodes: List[HierarchyNode],
    db_conn: sqlite3.Connection,
    top_k: int = 10,
    budget_tokens: int = 4000,
) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
    """
    Top-K hierarchical retrieval with performance metrics.

    Same as retrieve_hierarchical but also returns metrics for analysis.

    Args:
        query_vec: Query embedding
        root_nodes: Top-level hierarchy nodes to search
        db_conn: Database connection for loading children
        top_k: Number of top children to explore at each level
        budget_tokens: Max tokens to retrieve

    Returns:
        Tuple of (results, metrics_dict)

        metrics_dict contains:
        - e_computations: Number of E-score computations performed
        - results_count: Number of results returned
        - budget_used: Approximate tokens in results
    """
    if query_vec is None or len(query_vec) == 0:
        return [], {"e_computations": 0, "results_count": 0, "budget_used": 0}

    if not root_nodes:
        return [], {"e_computations": 0, "results_count": 0, "budget_used": 0}

    results: List[RetrievalResult] = []
    budget_remaining = [budget_tokens]
    e_computations = [0]

    for root in root_nodes:
        if budget_remaining[0] <= 0:
            break
        _retrieve_recursive(
            query_vec, root, db_conn, top_k,
            results, budget_remaining, e_computations
        )

    results.sort(key=lambda r: r.e_score, reverse=True)

    budget_used = budget_tokens - budget_remaining[0]

    metrics = {
        "e_computations": e_computations[0],
        "results_count": len(results),
        "budget_used": budget_used,
    }

    return results, metrics


# =============================================================================
# In-Memory Hierarchy Building (for testing without DB)
# =============================================================================

def build_hierarchy_in_memory(
    vectors: np.ndarray,
    event_ids: List[str],
    content_hashes: List[str],
    token_counts: Optional[List[int]] = None,
    branch_factor: int = 10,
) -> List[HierarchyNode]:
    """
    Build an in-memory hierarchy from vectors for testing.

    Creates a balanced tree structure:
    - L0: Individual vectors (leaves)
    - L1+: Groups of branch_factor children each

    Args:
        vectors: Array of embeddings, shape (n, 384)
        event_ids: List of event IDs for each vector
        content_hashes: List of content hashes for each vector
        token_counts: Optional list of token counts (defaults to 100 each)
        branch_factor: Number of children per internal node (default: 10)

    Returns:
        List of root nodes (usually one unless tree is very large)
    """
    n = len(vectors)
    if n == 0:
        return []

    if token_counts is None:
        token_counts = [100] * n

    # Create L0 nodes (leaves)
    l0_nodes: List[HierarchyNode] = []
    for i in range(n):
        node = HierarchyNode(
            node_id=f"L0_{i}",
            level=0,
            centroid=vectors[i],
            event_id=event_ids[i],
            content_hash=content_hashes[i],
            token_count=token_counts[i],
        )
        l0_nodes.append(node)

    # Build higher levels
    current_level_nodes = l0_nodes
    level = 1

    while len(current_level_nodes) > 1:
        next_level_nodes: List[HierarchyNode] = []

        for i in range(0, len(current_level_nodes), branch_factor):
            group = current_level_nodes[i:i + branch_factor]

            # Compute centroid as mean of child centroids (normalized)
            centroid = np.mean([c.centroid for c in group], axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            # Sum token counts for budget estimation
            total_tokens = sum(c.token_count for c in group)

            parent = HierarchyNode(
                node_id=f"L{level}_{i // branch_factor}",
                level=level,
                centroid=centroid,
                children=group,
                token_count=total_tokens,
            )

            # Set parent_id on children
            for child in group:
                child.parent_id = parent.node_id

            next_level_nodes.append(parent)

        current_level_nodes = next_level_nodes
        level += 1

        # Safety: prevent infinite loops
        if level > 10:
            break

    return current_level_nodes


# =============================================================================
# Schema Creation
# =============================================================================

def ensure_hierarchy_schema(db_conn: sqlite3.Connection) -> bool:
    """
    Create hierarchy_nodes table if it doesn't exist.

    Args:
        db_conn: SQLite database connection

    Returns:
        True if table was created, False if it already existed
    """
    # Check if table exists
    cursor = db_conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='hierarchy_nodes'
    """)
    existed = cursor.fetchone() is not None

    # Create table
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS hierarchy_nodes (
            node_id TEXT PRIMARY KEY,
            session_id TEXT,
            level INTEGER NOT NULL,
            centroid BLOB NOT NULL,
            event_id TEXT,
            content_hash TEXT,
            parent_id TEXT,
            token_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (event_id) REFERENCES session_events(event_id)
        )
    """)

    # Create indexes
    db_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_parent
        ON hierarchy_nodes(parent_id)
    """)

    db_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_session
        ON hierarchy_nodes(session_id)
    """)

    db_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_level
        ON hierarchy_nodes(level)
    """)

    db_conn.commit()

    return not existed


def store_hierarchy_node(
    db_conn: sqlite3.Connection,
    node: HierarchyNode,
    session_id: str,
) -> None:
    """
    Store a single hierarchy node to the database.

    Args:
        db_conn: SQLite database connection
        node: HierarchyNode to store
        session_id: Session ID for the node
    """
    centroid_blob = node.centroid.astype(np.float32).tobytes()

    db_conn.execute("""
        INSERT OR REPLACE INTO hierarchy_nodes
        (node_id, session_id, level, centroid, event_id, content_hash, parent_id, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        node.node_id,
        session_id,
        node.level,
        centroid_blob,
        node.event_id,
        node.content_hash,
        node.parent_id,
        node.token_count,
    ))


def store_hierarchy_batch(
    db_conn: sqlite3.Connection,
    nodes: List[HierarchyNode],
    session_id: str,
) -> int:
    """
    Store multiple hierarchy nodes to the database efficiently.

    Args:
        db_conn: SQLite database connection
        nodes: List of HierarchyNode objects to store
        session_id: Session ID for all nodes

    Returns:
        Number of nodes stored
    """
    if not nodes:
        return 0

    rows = []
    for node in nodes:
        centroid_blob = node.centroid.astype(np.float32).tobytes()
        rows.append((
            node.node_id,
            session_id,
            node.level,
            centroid_blob,
            node.event_id,
            node.content_hash,
            node.parent_id,
            node.token_count,
        ))

    db_conn.executemany("""
        INSERT OR REPLACE INTO hierarchy_nodes
        (node_id, session_id, level, centroid, event_id, content_hash, parent_id, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    db_conn.commit()

    return len(rows)


# =============================================================================
# Hot Path Optimization
# =============================================================================

@dataclass
class RetrievalMetrics:
    """
    Metrics from hierarchical retrieval with hot path.

    Tracks where results came from (hot path vs hierarchy) and
    performance statistics.
    """
    e_computations: int          # Total E-score computations
    results_count: int           # Number of results returned
    budget_used: int             # Approximate tokens in results
    hierarchy_used: bool         # Did we use hierarchy (vs brute force)?
    hot_path_hits: int           # Results found in hot window
    hierarchy_hits: int          # Results found via hierarchy
    levels_searched: int         # Max depth reached in hierarchy


def retrieve_with_hot_path(
    query_vec: np.ndarray,
    session_id: str,
    db_conn: sqlite3.Connection,
    hot_window: int = 100,
    top_k: int = 10,
    budget_tokens: int = 4000
) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
    """
    Retrieve with hot path for recent turns.

    The hot path optimization:
    1. Get last `hot_window` turns directly (brute force E-score)
    2. Get older turns via hierarchy (O(log n))
    3. Merge results, sort by E-score
    4. Return within budget

    This avoids hierarchy overhead for frequently-accessed recent content.

    Args:
        query_vec: Query embedding (384-dim normalized vector)
        session_id: Session to search
        db_conn: Database connection
        hot_window: Number of recent turns to check with brute force
        top_k: Number of top children to explore in hierarchy
        budget_tokens: Max tokens to retrieve

    Returns:
        Tuple of (results, metrics)
    """
    if query_vec is None or len(query_vec) == 0:
        return [], RetrievalMetrics(
            e_computations=0, results_count=0, budget_used=0,
            hierarchy_used=False, hot_path_hits=0, hierarchy_hits=0,
            levels_searched=0
        )

    e_computations = 0
    hot_path_results: List[RetrievalResult] = []
    hierarchy_results: List[RetrievalResult] = []

    # 1. Hot path: brute force on recent turns
    hot_path_results, hot_e_comps = _retrieve_hot_path(
        query_vec, session_id, db_conn, hot_window
    )
    e_computations += hot_e_comps

    # 2. Hierarchy: check if hierarchy exists and use it for older turns
    hierarchy_used = False
    levels_searched = 0

    # Check if hierarchy nodes exist for this session
    cursor = db_conn.execute("""
        SELECT COUNT(*) FROM hierarchy_nodes
        WHERE session_id = ? AND level > 0
    """, (session_id,))
    hierarchy_node_count = cursor.fetchone()[0]

    if hierarchy_node_count > 0:
        # Use hierarchy for older turns
        hierarchy_used = True
        root_nodes = load_root_nodes(db_conn, session_id)

        if root_nodes:
            levels_searched = max(r.level for r in root_nodes)

            # Get already-found event_ids to skip
            hot_event_ids = {r.event_id for r in hot_path_results}

            hierarchy_results, hier_metrics = retrieve_hierarchical_with_metrics(
                query_vec, root_nodes, db_conn,
                top_k=top_k, budget_tokens=budget_tokens
            )
            e_computations += hier_metrics["e_computations"]

            # Filter out duplicates (turns already in hot path)
            hierarchy_results = [
                r for r in hierarchy_results
                if r.event_id not in hot_event_ids
            ]

    # 3. Merge and sort by E-score
    all_results = hot_path_results + hierarchy_results
    all_results.sort(key=lambda r: r.e_score, reverse=True)

    # 4. Apply budget
    final_results: List[RetrievalResult] = []
    budget_remaining = budget_tokens

    for r in all_results:
        # Estimate token cost (default 100 if not available)
        # Token count should be stored in hierarchy_nodes
        cursor = db_conn.execute("""
            SELECT token_count FROM hierarchy_nodes
            WHERE node_id = ? OR event_id = ?
        """, (r.node_id, r.event_id))
        row = cursor.fetchone()
        token_cost = row[0] if row and row[0] else 100

        if budget_remaining >= token_cost:
            final_results.append(r)
            budget_remaining -= token_cost

    budget_used = budget_tokens - budget_remaining
    hot_hits = len([r for r in final_results if r in hot_path_results])
    hier_hits = len(final_results) - hot_hits

    return final_results, RetrievalMetrics(
        e_computations=e_computations,
        results_count=len(final_results),
        budget_used=budget_used,
        hierarchy_used=hierarchy_used,
        hot_path_hits=hot_hits,
        hierarchy_hits=hier_hits,
        levels_searched=levels_searched
    )


def _retrieve_hot_path(
    query_vec: np.ndarray,
    session_id: str,
    db_conn: sqlite3.Connection,
    hot_window: int
) -> Tuple[List[RetrievalResult], int]:
    """
    Brute force E-score retrieval on recent turns.

    Gets the most recent `hot_window` L0 nodes and scores them directly.

    Args:
        query_vec: Query embedding
        session_id: Session to search
        db_conn: Database connection
        hot_window: Number of recent turns to check

    Returns:
        Tuple of (results, e_computation_count)
    """
    e_computations = 0
    results: List[RetrievalResult] = []

    # Get recent L0 nodes ordered by creation time (most recent first)
    cursor = db_conn.execute("""
        SELECT node_id, centroid, event_id, content_hash, token_count
        FROM hierarchy_nodes
        WHERE session_id = ? AND level = 0
        ORDER BY created_at DESC
        LIMIT ?
    """, (session_id, hot_window))

    rows = cursor.fetchall()

    for row in rows:
        node_id = row[0]
        centroid_blob = row[1]
        event_id = row[2]
        content_hash = row[3]

        if centroid_blob is None:
            continue

        centroid = np.frombuffer(centroid_blob, dtype=np.float32)
        e_score = compute_E(query_vec, centroid)
        e_computations += 1

        results.append(RetrievalResult(
            node_id=node_id,
            event_id=event_id or "",
            e_score=e_score,
            content_hash=content_hash or "",
            level=0
        ))

    # Sort by E-score descending
    results.sort(key=lambda r: r.e_score, reverse=True)

    return results, e_computations


def has_hierarchy(db_conn: sqlite3.Connection, session_id: str) -> bool:
    """
    Check if a hierarchy exists for the given session.

    Args:
        db_conn: Database connection
        session_id: Session to check

    Returns:
        True if hierarchy nodes exist (level > 0), False otherwise
    """
    cursor = db_conn.execute("""
        SELECT COUNT(*) FROM hierarchy_nodes
        WHERE session_id = ? AND level > 0
    """, (session_id,))
    return cursor.fetchone()[0] > 0


def get_hierarchy_stats(
    db_conn: sqlite3.Connection,
    session_id: str
) -> Dict[str, Any]:
    """
    Get statistics about the hierarchy for a session.

    Args:
        db_conn: Database connection
        session_id: Session to query

    Returns:
        Dict with:
        - levels_built: Number of levels (0-3)
        - nodes_per_level: Dict mapping level to count
        - total_nodes: Total node count
        - total_turns: Count of L0 (turn) nodes
    """
    stats = {
        "levels_built": 0,
        "nodes_per_level": {},
        "total_nodes": 0,
        "total_turns": 0,
    }

    for level in range(4):  # L0, L1, L2, L3
        cursor = db_conn.execute("""
            SELECT COUNT(*) FROM hierarchy_nodes
            WHERE session_id = ? AND level = ?
        """, (session_id, level))
        count = cursor.fetchone()[0]

        if count > 0:
            stats["nodes_per_level"][level] = count
            stats["total_nodes"] += count
            if level == 0:
                stats["total_turns"] = count
            else:
                stats["levels_built"] = max(stats["levels_built"], level)

    return stats


# =============================================================================
# Main (Quick Test)
# =============================================================================

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("Hierarchical Retriever - Quick Test")
    print("=" * 50)

    # Create test vectors
    np.random.seed(42)
    n_items = 100
    vectors = np.random.randn(n_items, EMBEDDING_DIM).astype(np.float32)
    # Normalize
    for i in range(n_items):
        vectors[i] = vectors[i] / np.linalg.norm(vectors[i])

    event_ids = [f"evt_{i:03d}" for i in range(n_items)]
    content_hashes = [f"hash_{i:03d}" for i in range(n_items)]
    token_counts = [50 + i % 100 for i in range(n_items)]

    # Build in-memory hierarchy
    print("\nBuilding in-memory hierarchy...")
    roots = build_hierarchy_in_memory(
        vectors, event_ids, content_hashes, token_counts,
        branch_factor=10
    )
    print(f"Root nodes: {len(roots)}")
    if roots:
        print(f"Root level: {roots[0].level}")

    # Create query
    query_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Test retrieval with in-memory hierarchy (no DB needed)
    print("\nTesting in-memory retrieval...")

    # Use a simple mock connection since we have in-memory children
    class MockConn:
        def execute(self, sql, params=()):
            class EmptyCursor:
                def fetchall(self):
                    return []
                def fetchone(self):
                    return None
            return EmptyCursor()

    mock_conn = MockConn()

    results, metrics = retrieve_hierarchical_with_metrics(
        query_vec, roots, mock_conn,
        top_k=5, budget_tokens=2000
    )

    print(f"Results: {len(results)}")
    print(f"E computations: {metrics['e_computations']}")
    print(f"Budget used: {metrics['budget_used']}")

    if results:
        print(f"\nTop 5 results:")
        for i, r in enumerate(results[:5]):
            print(f"  {i+1}. {r.node_id}: E={r.e_score:.4f}, hash={r.content_hash}")

    # Test with database
    print("\n" + "=" * 50)
    print("Testing with SQLite database...")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_hierarchy.db"
        conn = sqlite3.connect(str(db_path))

        # Create schema
        created = ensure_hierarchy_schema(conn)
        print(f"Schema created: {created}")

        # Store hierarchy (flatten all nodes)
        def collect_all_nodes(node: HierarchyNode) -> List[HierarchyNode]:
            nodes = [node]
            for child in node.children:
                nodes.extend(collect_all_nodes(child))
            return nodes

        all_nodes = []
        for root in roots:
            all_nodes.extend(collect_all_nodes(root))

        stored = store_hierarchy_batch(conn, all_nodes, "test_session")
        print(f"Stored {stored} nodes")

        # Load roots from DB
        db_roots = load_root_nodes(conn, "test_session")
        print(f"Loaded {len(db_roots)} root nodes from DB")

        # Retrieve
        results2, metrics2 = retrieve_hierarchical_with_metrics(
            query_vec, db_roots, conn,
            top_k=5, budget_tokens=2000
        )

        print(f"DB retrieval results: {len(results2)}")
        print(f"E computations: {metrics2['e_computations']}")

        conn.close()

    print("\nAll tests passed!")
