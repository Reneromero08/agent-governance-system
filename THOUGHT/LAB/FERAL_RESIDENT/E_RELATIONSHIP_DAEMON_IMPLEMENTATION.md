# E-Relationship Learning Daemon: Implementation Report

## Executive Summary

This document outlines how to transform the feral daemon from a **position tracker** (centroid-based) into a **relationship learner** (E-graph-based). The current daemon leaves breadcrumbs in semantic space but loses structural information. The proposed system preserves E-relationships between items, enabling true "vector intelligence."

---

## Implementation Status

| Layer | Phase | Status | Started | Completed |
|-------|-------|--------|---------|-----------|
| Layer 1 | Phase 1: Individual Item Storage | NOT STARTED | - | - |
| Layer 2 | Phase 2: E-Relationship Graph + R-Gate | NOT STARTED | - | - |
| Layer 3 | Phase 3: Pattern Detection | NOT STARTED | - | - |
| Layer 4 | Phase 4: Query Interface | NOT STARTED | - | - |

**Last Updated:** 2026-01-27
**Key Integration:** R-gating added to Phase 2 per FORMULA_LAB (R = gate, E = compass)
**Blocking Issues:** None identified

---

## Current State Analysis

### What The Daemon Does Now

```
geometric_memory.py:remember()
```

```python
# Current: Running average creates CENTROID
n = len(self.memory_history) + 1
t = 1.0 / (n + 1)  # Weight: (N*Mind + New) / (N+1)
self.mind_state = self.reasoner.interpolate(self.mind_state, interaction, t=t)
```

**Result:** 2682 "remember" vectors in feral_eternal.db - but these are centroid snapshots, NOT individual items.

### Empirical Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Daemon-to-daemon E | 0.850 | High internal coherence (96% above threshold) |
| Daemon-to-paper E | 0.497 | Borderline alignment (48% above threshold) |
| Early-vs-late centroid E | 0.363 | Significant exploration drift |
| Daemon helping retrieval | -2% | Actually hurts sequential retrieval |
| Daemon-unique item E | 0.225 | Below threshold, not retrievable |

### The Problem

```
Current daemon = "Where am I?" (compass)
Desired daemon = "What connects to what?" (map)
```

The centroid converges over time, losing the structure of individual E-relationships.

---

## Proposed Architecture

### E vs R: Role Clarification

Per FORMULA_LAB validation (FORMULA_VALIDATION_REPORT_2.md):

| Metric | Role | Use in Daemon |
|--------|------|---------------|
| **E** (Born rule) | COMPASS - computes relationships | Edge weights, similarity scores |
| **R** (Resonance) | GATE - decides whether to proceed | Quality control, decision gating |

**Critical insight**: R is a gate, not a compass. E tells you "how related are these?", R tells you "should I trust this decision?"

```
R = (E / nablaS) x sigma(f)^Df

Where:
- E = essence (signal) - the relationship strength
- nablaS = entropy gradient (noise) - uncertainty in the relationship
- sigma(f) = symbolic compression - how well the relationship generalizes
- Df = fractal dimension - complexity of the relationship
```

### Four-Layer System (Updated)

```
+--------------------------------------------------+
|  LAYER 4: R-Gate (Quality Control)                |
|  - Gate edge additions (is this relationship      |
|    trustworthy?)                                  |
|  - Gate cluster outputs (is this cluster real?)  |
|  - Early stopping for low-R operations           |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  LAYER 3: Pattern Detection                       |
|  - Cluster discovery                              |
|  - Bridge identification                          |
|  - Novelty scoring                                |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  LAYER 2: E-Relationship Graph                    |
|  - Sparse edge storage (E >= 0.5 only)           |
|  - Incremental edge computation                   |
|  - Neighbor queries                               |
+--------------------------------------------------+
           |
           v
+--------------------------------------------------+
|  LAYER 1: Individual Item Storage                 |
|  - Store each interaction as retrievable vector  |
|  - Link to source (paper_id, content)            |
|  - Preserve provenance                            |
+--------------------------------------------------+
```

### R-Gating Integration Points

1. **Edge creation**: Compute E for relationship, compute R to gate whether to persist
2. **Cluster validation**: After finding clusters via E, use R to verify they're meaningful
3. **Novelty gating**: High novelty (low E) items need R-check before special handling
4. **Query trust**: R-score returned with query results to indicate confidence

---

## Research Foundations

The following validated findings from FORMULA_LAB and FERAL_RESIDENT research directly inform this implementation:

### Semiotic Conservation Law (Q48-Q50)

```
Df x alpha = 8e = 21.746 (CV = 2.69% across 24 models)
```

- **Why 8**: Peirce's 3 irreducible semiotic categories -> 2^3 = 8 octants
- **Why e**: Natural information unit (1 nat per category)
- **Critical alpha**: 0.5 (Riemann critical line, topologically protected)
- **Implication**: If Df changes, alpha scales accordingly: `alpha = 8e / Df`

### Four-Tier R-Threshold System (Q17)

| Tier | Action Type | Threshold | Min Observations | Use Case |
|------|-------------|-----------|------------------|----------|
| T0 | Read-only | 0.0 | 0 | Query without side effects |
| T1 | Reversible | 0.5 | 2 (any source) | Create edges that can be deleted |
| T2 | Persistent | 0.8 | 3 (2+ independent) | Stable edges, cluster membership |
| T3 | Critical | 1.0 | 5 (3+ independent) | Core structural relationships |

**Observation Quality**: Multi-model = HIGH, Cross-tool = HIGH, Self-consistency = MEDIUM, Same-model-deterministic = INVALID

### Context Optimization (Q13)

```
Optimal context = N=2-3 observations (NOT maximum)

N=1: 1.2x baseline
N=2: 47x improvement (PHASE TRANSITION)
N=3: 47x (peak)
N=4+: Diminishing returns (down to 24x at N=12)
```

**Implication**: `compare_to_recent=100` may be overkill. Consider `compare_to_recent=3` for edge computation efficiency.

### Born Rule Validation (Q44)

```
E = |<psi|phi>|^2 correlates r=0.977 with semantic similarity
```

- Validated across 5 embedding architectures
- Precision threshold: E >= 0.5 for meaningful relationships
- **Code**: `E = dot_product(v1, v2)` where vectors are unit-normalized

### Intensive Property Discovery (Q15)

```
R = E/sigma = sqrt(Likelihood Precision)
```

- R is mathematically equal to sqrt(likelihood precision)
- **Critical**: R is volume-independent (cannot bypass gate by collecting more data)
- Correlation R vs sqrt(Likelihood Precision): r = 1.0000 (perfect)

### Memory Consolidation Bug (QUANTUM_AUDIT)

**Known Issue**: `blend_memories()` uses iterative superposition creating unequal weights:
```
Current: superpose(A, superpose(B, C)) != uniform (A+B+C)/sqrt(3)
Result: Early memories weighted 2x heavier than late ones
```

**Fix Required**: Replace iterative superposition with sum-then-normalize:
```python
result_vector = sum(states) / np.sqrt(len(states))
```

### Spectral Convergence (Q34)

- Trained embeddings: Df = 22 (vs 99 random, 62 untrained)
- Cross-architecture correlation: 0.971 (GloVe, Word2Vec, FastText, BERT all match)
- **Geodesic concentration**: All trained models cluster within 0.35 radians (20 degree spherical cap)
- **Implication**: Clusters should be tight in effective 22-dimensional semantic manifold

---

## Prerequisites

Before beginning implementation:

### 1. Foundation Dependencies (READY)
- [x] Q44 Geometry validated (GeometricState, E_with)
- [x] Q46 nucleation thresholds operational
- [x] Vector storage working (resident_db.py)
- [x] Receipt system functional

### 2. Code Files to Verify
- `memory/geometric_memory.py` - remember() method (lines 86-131)
- `memory/vector_store.py` - store_vector() capability
- `autonomic/feral_daemon.py` - current daemon behavior

### 3. Existing Migration
- `migrations/001_add_temporal_links.py` exists
- New migrations will be 002, 003

---

## Implementation Plan

### Phase 1: Individual Item Storage (Foundation)

**Difficulty:** Easy (2-4 hours)
**Files to modify:** `memory/geometric_memory.py`, `memory/vector_store.py`

#### 1.1 Modify remember() to store individual items

```python
# geometric_memory.py

def remember(self, interaction_text: str, source_id: str = None) -> Dict:
    """Remember interaction AND store individual vector for retrieval."""

    # Initialize to manifold (existing)
    interaction = self.reasoner.initialize(interaction_text)

    # NEW: Store individual item for retrieval
    item_id = self.vector_store.store_vector(
        vec=interaction.vec,
        content=interaction_text,
        operation="daemon_item",  # New operation type
        metadata={
            "source_id": source_id,
            "daemon_step": len(self.memory_history),
            "mind_state_before": self.mind_state.vec.tobytes().hex()[:32] if self.mind_state else None,
        }
    )

    # Update centroid (existing behavior - keep for exploration tracking)
    if self.mind_state is None:
        self.mind_state = interaction
    else:
        n = len(self.memory_history) + 1
        t = 1.0 / (n + 1)
        self.mind_state = self.reasoner.interpolate(self.mind_state, interaction, t=t)

    # Track history
    self.memory_history.append({
        "item_id": item_id,
        "content_preview": interaction_text[:100],
        "Df": interaction.Df,
        "distance_from_start": interaction.distance_from_start,
    })

    return {
        "item_id": item_id,
        "mind_Df": self.mind_state.Df,
        "items_stored": len(self.memory_history),
    }
```

#### 1.2 Add daemon_item operation to vector_store

```python
# vector_store.py - add to store_vector()

def store_vector(self, vec, content, operation, metadata=None):
    """Store vector with full provenance."""
    vec_blob = vec.astype(np.float32).tobytes()
    vec_sha256 = hashlib.sha256(vec_blob).hexdigest()
    vector_id = f"{operation}_{uuid.uuid4().hex[:16]}"

    cursor = self.conn.cursor()
    cursor.execute('''
        INSERT INTO vectors (vector_id, vec_blob, vec_sha256, composition_op, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (vector_id, vec_blob, vec_sha256, operation, datetime.now().isoformat()))

    # Store receipt with metadata
    if metadata:
        cursor.execute('''
            INSERT INTO receipts (output_hash, operation, metadata, created_at)
            VALUES (?, ?, ?, ?)
        ''', (vec_sha256[:16], operation, json.dumps(metadata), datetime.now().isoformat()))

    self.conn.commit()
    return vector_id
```

---

### Phase 2: E-Relationship Graph (Core)

**Difficulty:** Medium (1 day)
**New file:** `memory/e_graph.py`

#### 2.1 Database Schema Migration

```python
# migrations/002_add_e_edges.py

def migrate_schema(conn):
    """Add E-relationship edges table."""
    cursor = conn.cursor()

    # Check if already migrated
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='e_edges'")
    if cursor.fetchone():
        print("Already migrated - e_edges table exists")
        return False

    cursor.execute('''
        CREATE TABLE e_edges (
            edge_id TEXT PRIMARY KEY,
            vector_id_a TEXT NOT NULL,
            vector_id_b TEXT NOT NULL,
            e_score REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (vector_id_a) REFERENCES vectors(vector_id),
            FOREIGN KEY (vector_id_b) REFERENCES vectors(vector_id),
            UNIQUE(vector_id_a, vector_id_b)
        )
    ''')

    # Index for fast neighbor lookups
    cursor.execute('CREATE INDEX idx_edges_a ON e_edges(vector_id_a, e_score DESC)')
    cursor.execute('CREATE INDEX idx_edges_b ON e_edges(vector_id_b, e_score DESC)')
    cursor.execute('CREATE INDEX idx_edges_score ON e_edges(e_score DESC)')

    conn.commit()
    print("Created e_edges table with indexes")
    return True
```

#### 2.2 E-Graph Class

```python
# memory/e_graph.py

import numpy as np
import sqlite3
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import uuid

@dataclass
class EEdge:
    """An E-relationship between two vectors."""
    vector_id_a: str
    vector_id_b: str
    e_score: float

def compute_E(v1: np.ndarray, v2: np.ndarray) -> float:
    """Born rule: E = |<v1|v2>|^2"""
    dot = np.dot(v1, v2)
    return dot * dot


def compute_R(E: float, sigma: float, Df: float = 1.0, use_full_formula: bool = False) -> float:
    """
    Resonance gate - validated from Q15 and Q17 research.

    Two modes:
    1. Simple (default): R = E/sigma = sqrt(Likelihood Precision)
       - Per Q15: Mathematically equal to sqrt(likelihood precision)
       - Volume-independent (cannot bypass by collecting more data)

    2. Full formula: R = (E / sigma) x compression^Df
       - Per FORMULA_LAB: Includes symbolic compression scaling

    Args:
        E: Essence (relationship strength from Born rule, 0-1)
        sigma: Standard deviation of E-scores (uncertainty/noise)
        Df: Fractal dimension (default 1.0, use GeometricState.Df for full)
        use_full_formula: If True, apply Df scaling

    Returns:
        R score for gating decisions

    Thresholds (from Q17):
        T0 (read-only):   0.0
        T1 (reversible):  0.5
        T2 (persistent):  0.8
        T3 (critical):    1.0
    """
    if sigma <= 0:
        sigma = 0.01  # Avoid division by zero

    # Core formula: R = E/sigma (validated r=1.0 with sqrt(likelihood precision))
    R = E / sigma

    # Optional: Full formula with fractal scaling
    if use_full_formula and Df != 1.0:
        # Symbolic compression (default 1.0 = no compression)
        compression = 1.0
        R = R * (compression ** Df)

    # Return raw R (caller decides threshold based on tier)
    return R


def get_r_tier(R: float) -> str:
    """Map R score to action tier per Q17 validation."""
    if R >= 1.0:
        return "T3_CRITICAL"
    elif R >= 0.8:
        return "T2_PERSISTENT"
    elif R >= 0.5:
        return "T1_REVERSIBLE"
    else:
        return "T0_READ_ONLY"


class ERelationshipGraph:
    """
    Sparse graph of E-relationships between vectors.

    Only stores edges where E >= threshold (default 0.5).
    Enables queries like:
    - "What's connected to X?"
    - "What clusters exist?"
    - "What bridges cluster A and B?"
    """

    def __init__(self, conn: sqlite3.Connection, threshold: float = 0.5):
        self.conn = conn
        self.threshold = threshold
        self._cache = {}  # vector_id -> vec (LRU cache)
        self._cache_size = 1000

    def add_item(
        self,
        vector_id: str,
        vec: np.ndarray,
        Df: float = 1.0,
        compare_to_recent: int = 3,  # Q13: N=2-3 is optimal, not 100
        r_tier: str = "T1_REVERSIBLE",
    ) -> Dict:
        """
        Add item and compute E-scores to recent items.
        Uses R-gating to filter edge additions per Q17 tier system.

        Args:
            vector_id: ID of the new vector
            vec: The vector itself
            Df: Fractal dimension of this item (from GeometricState)
            compare_to_recent: How many recent items to compare (Q13: 2-3 optimal)
            r_tier: Minimum tier for edge persistence (T0/T1/T2/T3)

        Returns stats about new connections formed.
        """
        # Map tier to threshold (Q17 validated)
        tier_thresholds = {
            "T0_READ_ONLY": 0.0,
            "T1_REVERSIBLE": 0.5,
            "T2_PERSISTENT": 0.8,
            "T3_CRITICAL": 1.0,
        }
        r_gate_threshold = tier_thresholds.get(r_tier, 0.5)

        cursor = self.conn.cursor()

        # Get recent daemon items to compare against
        # Q13: Optimal context is N=2-3, not maximum
        cursor.execute('''
            SELECT vector_id, vec_blob
            FROM vectors
            WHERE composition_op = 'daemon_item'
            AND vector_id != ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (vector_id, compare_to_recent))

        new_edges = []
        e_scores = []
        r_scores = []
        tier_counts = {"T0": 0, "T1": 0, "T2": 0, "T3": 0}
        gated_out = 0

        for other_id, vec_blob in cursor.fetchall():
            other_vec = np.frombuffer(vec_blob, dtype=np.float32)
            E = compute_E(vec, other_vec)
            e_scores.append(E)

            if E >= self.threshold:
                # R-GATE: sigma = std of E-scores (uncertainty measure)
                # Per Q15: R = E/sigma = sqrt(likelihood precision)
                sigma = np.std(e_scores) if len(e_scores) > 1 else 0.1

                # Compute R using validated formula
                R = compute_R(E, sigma, Df)
                r_scores.append(R)

                # Classify by tier
                actual_tier = get_r_tier(R)
                tier_counts[actual_tier[:2]] += 1

                # Only persist edge if R passes threshold for requested tier
                if R >= r_gate_threshold:
                    id_a, id_b = sorted([vector_id, other_id])
                    edge_id = f"edge_{uuid.uuid4().hex[:16]}"

                    try:
                        cursor.execute('''
                            INSERT OR IGNORE INTO e_edges
                            (edge_id, vector_id_a, vector_id_b, e_score, created_at)
                            VALUES (?, ?, ?, ?, datetime('now'))
                        ''', (edge_id, id_a, id_b, E))
                        new_edges.append(EEdge(id_a, id_b, E))
                    except sqlite3.IntegrityError:
                        pass  # Edge already exists
                else:
                    gated_out += 1  # R-gate rejected this edge

        self.conn.commit()

        return {
            "new_edges": len(new_edges),
            "gated_out": gated_out,  # Edges rejected by R-gate
            "tier_distribution": tier_counts,  # How edges distributed across tiers
            "items_compared": len(e_scores),
            "mean_E": np.mean(e_scores) if e_scores else 0,
            "max_E": max(e_scores) if e_scores else 0,
            "mean_R": np.mean(r_scores) if r_scores else 0,
            "above_threshold": sum(1 for e in e_scores if e >= self.threshold),
        }

    def get_neighbors(
        self,
        vector_id: str,
        min_E: float = None,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        """Get items connected to this one by E-relationship."""
        if min_E is None:
            min_E = self.threshold

        cursor = self.conn.cursor()

        # Query both directions (a->b and b->a)
        cursor.execute('''
            SELECT
                CASE WHEN vector_id_a = ? THEN vector_id_b ELSE vector_id_a END as neighbor_id,
                e_score
            FROM e_edges
            WHERE (vector_id_a = ? OR vector_id_b = ?)
            AND e_score >= ?
            ORDER BY e_score DESC
            LIMIT ?
        ''', (vector_id, vector_id, vector_id, min_E, limit))

        return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_all_edges(self, min_E: float = None) -> List[EEdge]:
        """Get all edges above threshold."""
        if min_E is None:
            min_E = self.threshold

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT vector_id_a, vector_id_b, e_score
            FROM e_edges
            WHERE e_score >= ?
            ORDER BY e_score DESC
        ''', (min_E,))

        return [EEdge(a, b, e) for a, b, e in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM e_edges')
        total_edges = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(e_score), MIN(e_score), MAX(e_score) FROM e_edges')
        avg_e, min_e, max_e = cursor.fetchone()

        cursor.execute('''
            SELECT COUNT(DISTINCT vector_id_a) + COUNT(DISTINCT vector_id_b)
            FROM e_edges
        ''')
        # This overcounts but gives rough node estimate

        return {
            "total_edges": total_edges,
            "avg_E": avg_e or 0,
            "min_E": min_e or 0,
            "max_E": max_e or 0,
        }
```

#### 2.3 Integrate E-Graph into remember()

```python
# geometric_memory.py - updated remember()

def remember(self, interaction_text: str, source_id: str = None) -> Dict:
    """Remember interaction, store item, and update E-graph."""

    interaction = self.reasoner.initialize(interaction_text)

    # Store individual item
    item_id = self.vector_store.store_vector(
        vec=interaction.vec,
        content=interaction_text,
        operation="daemon_item",
        metadata={"source_id": source_id, "daemon_step": len(self.memory_history)},
    )

    # NEW: Add to E-graph and compute relationships
    e_stats = self.e_graph.add_item(
        vector_id=item_id,
        vec=interaction.vec,
        compare_to_recent=100,
    )

    # Update centroid (keep for exploration)
    if self.mind_state is None:
        self.mind_state = interaction
    else:
        n = len(self.memory_history) + 1
        t = 1.0 / (n + 1)
        self.mind_state = self.reasoner.interpolate(self.mind_state, interaction, t=t)

    self.memory_history.append({"item_id": item_id, "e_stats": e_stats})

    return {
        "item_id": item_id,
        "new_connections": e_stats["new_edges"],
        "mean_E_to_recent": e_stats["mean_E"],
        "mind_Df": self.mind_state.Df,
    }
```

---

### Phase 3: Pattern Detection (Advanced)

**Difficulty:** Harder (1-2 days)
**New file:** `memory/e_patterns.py`

#### 3.1 Cluster Discovery

```python
# memory/e_patterns.py

from typing import List, Set, Dict
from collections import defaultdict

class EPatternDetector:
    """Detect patterns in E-relationship graph."""

    def __init__(self, e_graph: 'ERelationshipGraph'):
        self.graph = e_graph

    def find_clusters(self, min_cluster_size: int = 3) -> List[Set[str]]:
        """
        Find connected components in the E-graph.

        Uses Union-Find for efficiency.
        """
        edges = self.graph.get_all_edges()

        # Build adjacency
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union all edges
        for edge in edges:
            union(edge.vector_id_a, edge.vector_id_b)

        # Group by root
        clusters = defaultdict(set)
        for node in parent:
            root = find(node)
            clusters[root].add(node)

        # Filter by size
        return [c for c in clusters.values() if len(c) >= min_cluster_size]

    def find_bridges(
        self,
        cluster_a: Set[str],
        cluster_b: Set[str],
        min_E: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        """
        Find items that connect two clusters.

        Bridges are high-E edges crossing cluster boundaries.
        """
        edges = self.graph.get_all_edges(min_E=min_E)
        bridges = []

        for edge in edges:
            a_in_A = edge.vector_id_a in cluster_a
            b_in_A = edge.vector_id_b in cluster_a
            a_in_B = edge.vector_id_a in cluster_b
            b_in_B = edge.vector_id_b in cluster_b

            # Bridge: one end in A, other in B
            if (a_in_A and b_in_B) or (a_in_B and b_in_A):
                bridges.append((edge.vector_id_a, edge.vector_id_b, edge.e_score))

        return sorted(bridges, key=lambda x: x[2], reverse=True)

    def score_novelty(self, vector_id: str, vec: np.ndarray) -> float:
        """
        Score how novel an item is relative to existing graph.

        High novelty = few/weak connections to existing items.
        Low novelty = many strong connections (redundant).
        """
        neighbors = self.graph.get_neighbors(vector_id, limit=50)

        if not neighbors:
            return 1.0  # Completely novel (no connections)

        # Novelty = 1 - mean(E scores to neighbors)
        mean_E = np.mean([e for _, e in neighbors])
        return 1.0 - mean_E

    def get_cluster_centroid(self, cluster: Set[str]) -> np.ndarray:
        """Compute centroid of a cluster."""
        cursor = self.graph.conn.cursor()

        placeholders = ','.join('?' * len(cluster))
        cursor.execute(f'''
            SELECT vec_blob FROM vectors WHERE vector_id IN ({placeholders})
        ''', list(cluster))

        vecs = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]

        if not vecs:
            return None

        centroid = np.mean(vecs, axis=0)
        return centroid / np.linalg.norm(centroid)

    def summarize_clusters(self) -> List[Dict]:
        """Get summary of all clusters."""
        clusters = self.find_clusters()
        summaries = []

        for i, cluster in enumerate(clusters):
            centroid = self.get_cluster_centroid(cluster)

            # Get sample items
            cursor = self.graph.conn.cursor()
            sample_ids = list(cluster)[:3]
            placeholders = ','.join('?' * len(sample_ids))
            cursor.execute(f'''
                SELECT r.metadata FROM receipts r
                JOIN vectors v ON SUBSTR(v.vec_sha256, 1, 16) = r.output_hash
                WHERE v.vector_id IN ({placeholders})
            ''', sample_ids)

            samples = []
            for row in cursor.fetchall():
                if row[0]:
                    meta = json.loads(row[0])
                    samples.append(meta.get('text_preview', '')[:50])

            summaries.append({
                "cluster_id": i,
                "size": len(cluster),
                "sample_items": samples,
            })

        return summaries
```

---

### Phase 4: Query Interface

**Difficulty:** Easy (2-4 hours)
**New file:** `memory/e_query.py`

```python
# memory/e_query.py

class EQueryEngine:
    """Query interface for E-relationship graph."""

    def __init__(self, e_graph: 'ERelationshipGraph', vector_store: 'VectorStore'):
        self.graph = e_graph
        self.store = vector_store

    def find_related(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Find items related to query by E-score.

        Unlike centroid-based search, this uses the graph structure.
        """
        # Embed query
        query_vec = self.store.embed(query_text)

        # Get all daemon items
        cursor = self.store.conn.cursor()
        cursor.execute('''
            SELECT v.vector_id, v.vec_blob, r.metadata
            FROM vectors v
            LEFT JOIN receipts r ON SUBSTR(v.vec_sha256, 1, 16) = r.output_hash
            WHERE v.composition_op = 'daemon_item'
        ''')

        # Score all by E
        scored = []
        for vector_id, vec_blob, meta_json in cursor.fetchall():
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            E = compute_E(query_vec, vec)

            meta = json.loads(meta_json) if meta_json else {}
            scored.append({
                "vector_id": vector_id,
                "E_score": E,
                "content_preview": meta.get("text_preview", "")[:100],
                "source_id": meta.get("source_id"),
            })

        # Sort by E descending
        scored.sort(key=lambda x: x["E_score"], reverse=True)
        return scored[:top_k]

    def expand_from(self, vector_id: str, hops: int = 2) -> Dict:
        """
        Expand outward from a node in the E-graph.

        Returns subgraph reachable within `hops` edges.
        """
        visited = {vector_id}
        frontier = {vector_id}
        edges_found = []

        for hop in range(hops):
            next_frontier = set()
            for node in frontier:
                neighbors = self.graph.get_neighbors(node, limit=10)
                for neighbor_id, e_score in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                    edges_found.append((node, neighbor_id, e_score))
            frontier = next_frontier

        return {
            "nodes": list(visited),
            "edges": edges_found,
            "hops": hops,
        }

    def find_path(self, from_id: str, to_id: str, max_hops: int = 5) -> List[str]:
        """
        Find shortest path between two items in E-graph.

        Uses BFS. Returns None if no path exists.
        """
        if from_id == to_id:
            return [from_id]

        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            neighbors = self.graph.get_neighbors(current, limit=20)
            for neighbor_id, _ in neighbors:
                if neighbor_id == to_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None  # No path found
```

---

## File Summary

| File | Status | Action | Purpose |
|------|--------|--------|---------|
| `memory/geometric_memory.py` | EXISTS | MODIFY | Add individual item storage + E-graph integration |
| `memory/vector_store.py` | EXISTS | MODIFY | Add `store_vector()` with metadata |
| `memory/e_graph.py` | MISSING | CREATE | E-relationship graph core |
| `memory/e_patterns.py` | MISSING | CREATE | Cluster detection, bridges, novelty |
| `memory/e_query.py` | MISSING | CREATE | Query interface |
| `migrations/002_add_e_edges.py` | MISSING | CREATE | Database schema for edges |
| `migrations/003_backfill_e_edges.py` | MISSING | CREATE | Backfill existing daemon items |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_e_graph.py

def test_edge_creation():
    """Edges created only when E >= threshold."""
    graph = ERelationshipGraph(conn, threshold=0.5)

    # Similar vectors should create edge
    v1 = np.random.randn(768)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v1 + np.random.randn(768) * 0.1  # Similar
    v2 = v2 / np.linalg.norm(v2)

    id1 = store_vector(v1)
    graph.add_item(id1, v1)

    id2 = store_vector(v2)
    stats = graph.add_item(id2, v2)

    assert stats["new_edges"] >= 1, "Similar vectors should create edge"

def test_no_edge_for_dissimilar():
    """Dissimilar vectors should not create edges."""
    v1 = np.zeros(768)
    v1[0] = 1.0  # Point in one direction

    v2 = np.zeros(768)
    v2[384] = 1.0  # Orthogonal

    # E = 0 for orthogonal vectors
    E = compute_E(v1, v2)
    assert E < 0.5, "Orthogonal vectors should have low E"

def test_cluster_detection():
    """Clusters should be found from connected edges."""
    # Create triangle of connected items
    # ... test implementation
```

### Integration Test

```python
# tests/test_daemon_e_learning.py

def test_daemon_learns_paper_relationships():
    """Daemon should discover E-relationships between paper sections."""
    daemon = GeometricMemory(db_path)

    # Feed paper sections
    for section in paper_sections[:50]:
        result = daemon.remember(section.content, source_id=section.paper_id)
        print(f"Step {result['items_stored']}: {result['new_connections']} new connections")

    # Check graph structure
    stats = daemon.e_graph.get_stats()
    assert stats["total_edges"] > 0, "Should have discovered relationships"

    # Check clusters
    clusters = daemon.pattern_detector.find_clusters()
    print(f"Found {len(clusters)} clusters")

    # Clusters should roughly correspond to papers
    for cluster in clusters:
        # Get source_ids in cluster
        # Verify mostly same paper
```

---

## Migration Path

### Step 1: Run schema migration

```bash
cd THOUGHT/LAB/FERAL_RESIDENT
python migrations/002_add_e_edges.py
```

### Step 2: Backfill existing daemon items

```python
# migrations/003_backfill_e_edges.py

def backfill_daemon_edges():
    """Compute E-edges for existing daemon items."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all daemon items
    cursor.execute('''
        SELECT vector_id, vec_blob FROM vectors
        WHERE composition_op = 'remember'
        ORDER BY created_at
    ''')

    items = [(vid, np.frombuffer(vblob, dtype=np.float32))
             for vid, vblob in cursor.fetchall()]

    print(f"Backfilling E-edges for {len(items)} daemon items...")

    graph = ERelationshipGraph(conn, threshold=0.5)

    for i, (vid, vec) in enumerate(items):
        stats = graph.add_item(vid, vec, compare_to_recent=100)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)}: {stats['new_edges']} edges this batch")

    print(f"Done. Graph stats: {graph.get_stats()}")
```

### Step 3: Update daemon initialization

```python
# Update GeometricMemory.__init__()
def __init__(self, db_path):
    self.conn = sqlite3.connect(db_path)
    self.vector_store = VectorStore(self.conn)
    self.reasoner = GeometricReasoner(...)

    # NEW: Initialize E-graph
    self.e_graph = ERelationshipGraph(self.conn, threshold=0.5)
    self.pattern_detector = EPatternDetector(self.e_graph)
    self.query_engine = EQueryEngine(self.e_graph, self.vector_store)
```

---

## Expected Outcomes

After implementation:

| Capability | Before | After |
|------------|--------|-------|
| Store individual items | NO (centroid only) | YES |
| Track E-relationships | NO | YES (sparse graph) |
| R-gated edge creation | NO | YES (quality control) |
| Find related items | Via centroid (poor) | Via E-graph (good) |
| Detect clusters | NO | YES |
| Find bridges | NO | YES |
| Score novelty | NO | YES |
| Retrieval benefit | -2% (hurts) | +10-20% (expected) |

---

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `add_item` | O(k) | k = compare_to_recent (default 100) |
| `get_neighbors` | O(log n) | Uses index |
| `find_clusters` | O(E * alpha(V)) | Union-Find, E=edges, V=vertices |
| `find_path` | O(V + E) | BFS worst case |
| Storage | O(E) | Sparse: only E >= 0.5 stored |

For 10,000 daemon items with ~5% connectivity above threshold:
- ~500,000 potential edges
- ~25,000 actual edges stored (5%)
- ~2.5 edges per item average

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| O(n^2) edge computation | Compare only to recent k items |
| Graph too sparse | Lower threshold or use k-NN |
| Graph too dense | Raise threshold or prune weak edges |
| Memory for large graphs | Keep edges in DB, not memory |
| Stale edges | Add timestamp, optionally expire |

---

## Quick Start Checklist

### Phase 1 Implementation Steps
- [ ] Add `daemon_item` to allowed composition_op values
- [ ] Modify `remember()` to store individual vectors
- [ ] Add `source_id` and `daemon_step` metadata tracking
- [ ] Test: verify individual items appear in vectors table

### Phase 2 Implementation Steps
- [ ] Create `migrations/002_add_e_edges.py`
- [ ] Run migration to add e_edges table
- [ ] Create `memory/e_graph.py` with ERelationshipGraph class
- [ ] Implement `compute_R()` for R-gating (per FORMULA_LAB)
- [ ] Add R-gate to `add_item()` - reject edges with low R
- [ ] Integrate e_graph.add_item() into remember() with Df parameter
- [ ] Test: verify edges created for similar vectors
- [ ] Test: verify R-gate rejects noisy relationships

### Phase 3 Implementation Steps
- [ ] Create `memory/e_patterns.py`
- [ ] Implement find_clusters() with Union-Find
- [ ] Implement find_bridges()
- [ ] Implement score_novelty()
- [ ] Test: verify cluster detection on synthetic data

### Phase 4 Implementation Steps
- [ ] Create `memory/e_query.py`
- [ ] Implement find_related(), expand_from(), find_path()
- [ ] Add query engine to GeometricMemory init
- [ ] Test: verify graph traversal works

---

## Decision Log

| Date | Decision | Rationale | Research Source |
|------|----------|-----------|-----------------|
| 2026-01-27 | E-threshold 0.5 | Balances sparsity vs coverage; validated r=0.977 | Q44 Born Rule |
| 2026-01-27 | Compare to recent N=3 items | Q13 shows N=2-3 is optimal (47x improvement), N>3 has diminishing returns | Q13 Scaling Law |
| 2026-01-27 | Keep centroid + items | Backward compatibility; centroid for "where am I", items for "what connects" | E_RELATIONSHIP design |
| 2026-01-27 | R-gate edge creation | R is gate, E is compass. Use E to compute, R to decide persistence | FORMULA_VALIDATION_2 |
| 2026-01-27 | Four-tier R thresholds | T0=0.0, T1=0.5, T2=0.8, T3=1.0 with observation requirements | Q17 R-Gate Guide |
| 2026-01-27 | R = E/sigma | Mathematically equals sqrt(likelihood precision), volume-independent | Q15 Intensive Property |
| 2026-01-27 | sigma = std(E-scores) | Measures relationship uncertainty; validated as entropy gradient | FORMULA_VALIDATION_1 |
| 2026-01-27 | Df x alpha = 8e | Semiotic conservation law, CV=2.69% across 24 models | Q48-Q50 |
| 2026-01-27 | Geodesic < 0.35 rad | Trained models cluster within 20 degree spherical cap | Q34 Spectral Convergence |

---

## Next Steps

### Immediate (Phase 1)
1. Create branch: `feature/e-relationship-graph`
2. Modify `resident_db.py` to accept `daemon_item` operation type
3. Update `geometric_memory.py:remember()` per Phase 1 spec
4. Write unit test for individual item storage

### Short-term (Phase 2)
5. Create migration-002 for e_edges table
6. Implement `memory/e_graph.py`
7. Run backfill on feral_eternal.db
8. Validate edge creation with test data

### Medium-term (Phase 3-4)
9. Implement pattern detection after graph stability proven
10. Add query interface once patterns work
11. Benchmark against current retrieval

---

The key insight: **you already have the infrastructure**. This is mostly connecting existing pieces (vector storage, E-score computation) into a graph structure that preserves relationships instead of collapsing them into a centroid.
