# Phase J Implementation Plan: Recursive E-Score Hierarchy

**Created:** 2026-01-23
**Status:** Ready for parallel execution
**Prerequisite:** J.0 Vector Persistence (COMPLETE)

---

## Overview

Phase J extends CAT Chat's effective memory from ~1,000 turns to ~100,000+ turns using a hierarchical centroid structure. The key insight: E-score against centroid predicts whether contents are worth checking.

**Key Experimental Findings (from roadmap validation):**
1. **Top-K centroid selection > threshold-based pruning** - threshold is too aggressive
2. **Semantic clustering (k-means) is REQUIRED** - random grouping creates meaningless centroids
3. **PCA to Df=22 dimensions improves clustering** - removes noise in high dimensions
4. **Iso-temporal context helps (+10-17%)** - track context signatures

---

## Parallel Workstreams

### Stream A: Schema & Data Model (J.1)
**Agent:** Opus #1
**Estimated Scope:** ~4 files, ~400 LOC
**Dependencies:** J.0 complete (it is)

Creates the foundational schema and data structures.

**Tasks:**
1. **J.1.1** Create `hierarchy_schema.py`:
   - Define `HierarchyNode` dataclass with: node_id, level, centroid_vector, parent_id, turn_count, created_at
   - Define level constants: L0=0 (turns), L1=1 (100 turns), L2=2 (1000 turns), L3=3 (10000 turns)
   - Node storage in new `session_hierarchy_nodes` table

2. **J.1.2** Schema migration:
   ```sql
   CREATE TABLE session_hierarchy_nodes (
       node_id TEXT PRIMARY KEY,
       session_id TEXT NOT NULL,
       level INTEGER NOT NULL,  -- 0=turn, 1=L1, 2=L2, 3=L3
       parent_node_id TEXT,     -- NULL for root
       centroid BLOB NOT NULL,  -- 384-dim float32
       turn_count INTEGER NOT NULL,
       first_turn_seq INTEGER,  -- first turn in this subtree
       last_turn_seq INTEGER,   -- last turn in this subtree
       created_at TEXT DEFAULT (datetime('now')),
       FOREIGN KEY (parent_node_id) REFERENCES session_hierarchy_nodes(node_id)
   );
   CREATE INDEX idx_hierarchy_session ON session_hierarchy_nodes(session_id);
   CREATE INDEX idx_hierarchy_level ON session_hierarchy_nodes(session_id, level);
   CREATE INDEX idx_hierarchy_parent ON session_hierarchy_nodes(parent_node_id);
   ```

3. **J.1.3** Add `centroid_math.py`:
   - `compute_centroid(vectors: List[np.ndarray]) -> np.ndarray` - mean of vectors
   - `update_centroid_incremental(old_centroid, old_n, new_vec) -> np.ndarray` - O(1) update
   - `compute_E(query_vec, item_vec) -> float` - Born rule: |<q|i>|^2

4. **J.1.4** Tests: `tests/test_hierarchy_schema.py` (~15 tests)

**Output:** Schema ready, centroid math tested, data model defined

---

### Stream B: Recursive Retrieval Algorithm (J.2)
**Agent:** Opus #2
**Estimated Scope:** ~2 files, ~250 LOC
**Dependencies:** J.1 schema (can stub/mock for parallel dev)

Implements the core retrieval algorithm using **top-K selection** (NOT threshold).

**Tasks:**
1. **J.2.1** Create `hierarchy_retriever.py`:
   ```python
   def retrieve_hierarchical(
       query_vec: np.ndarray,
       root_nodes: List[HierarchyNode],
       top_k: int = 10,
       budget_tokens: int = 4000
   ) -> List[RetrievalResult]:
       """Top-K centroid selection - NOT threshold-based.

       At each level, scores ALL children and takes top-K.
       Recurses into top-K children until reaching L0 (turns).
       """
   ```

2. **J.2.2** Base case (L0): return turn content if within budget
3. **J.2.3** Recursive case: score all children, take top-K, recurse
4. **J.2.4** Budget tracking: stop if working_set would exceed limit
5. **J.2.5** Tests: `tests/test_hierarchy_retriever.py` (~20 tests)

**Algorithm (from roadmap validation):**
```python
def retrieve_hierarchical(query_vec, node, top_k=10):
    if node.level == 0:
        return [(node, compute_E(query_vec, node.centroid))]

    # Score ALL children, take top-K
    child_scores = [(child, compute_E(query_vec, child.centroid))
                    for child in node.children]
    child_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for child, _ in child_scores[:top_k]:
        results.extend(retrieve_hierarchical(query_vec, child, top_k))
    return results
```

**Output:** Working retrieval algorithm with top-K selection

---

### Stream C: Tree Maintenance (J.3)
**Agent:** Opus #3
**Estimated Scope:** ~3 files, ~350 LOC
**Dependencies:** J.1 schema

Handles automatic tree building and incremental updates.

**Tasks:**
1. **J.3.1** Create `hierarchy_builder.py`:
   - `on_turn_compressed(session_id, turn_vec)` - incremental update
   - Add vector to current L1 node, update centroid incrementally

2. **J.3.2** Level promotion logic:
   - When L1 has 100 children: close it, start new L1
   - When 10 L1 nodes exist: create L2 parent
   - Incremental centroid: `new_centroid = (old_centroid * n + new_vec) / (n + 1)`

3. **J.3.3** Initial tree build with k-means clustering:
   ```python
   def build_initial_hierarchy(session_id: str, vectors: List[np.ndarray]):
       """Build hierarchy from existing vectors using k-means.

       Uses semantic clustering (not random grouping) per experimental findings.
       Optional: PCA to Df=22 dimensions before clustering.
       """
   ```

4. **J.3.4** Wire into `turn_compressor.py`:
   - After storing embedding, call `hierarchy_builder.on_turn_compressed()`

5. **J.3.5** Tests: `tests/test_hierarchy_builder.py` (~18 tests)

**Output:** Tree builds automatically as turns compress

---

### Stream D: Integration & Hot Path (J.4)
**Agent:** Opus #4
**Estimated Scope:** ~2 files, ~200 LOC
**Dependencies:** J.1, J.2, J.3 (runs after Streams A-C)

Integrates hierarchy into auto_context_manager and adds hot path optimization.

**Tasks:**
1. **J.4.1** Hot path in `hierarchy_retriever.py`:
   - Last 100 turns: check directly (skip hierarchy)
   - Older turns: use recursive hierarchy
   - Configurable hot window size

2. **J.4.2** Integrate into `auto_context_manager.py`:
   - Replace brute-force E-score scan with hierarchical retrieval
   - `_hydrate_from_pointer_set()` uses hierarchy for large pointer sets
   - Fallback to brute force if hierarchy not built yet

3. **J.4.3** Add hierarchy stats to session metrics:
   - Levels built, nodes per level
   - E-computations per query (should be O(log n))
   - Hierarchy vs brute-force comparison

4. **J.4.4** Tests: `tests/test_hierarchy_integration.py` (~12 tests)

**Output:** Auto-context uses hierarchy for O(log n) retrieval

---

### Stream E: Forgetting & Archival (J.5)
**Agent:** Opus #5
**Estimated Scope:** ~2 files, ~150 LOC
**Dependencies:** J.1, J.3 (runs after Streams A, C)

Implements archival to keep storage bounded.

**Tasks:**
1. **J.5.1** Add `last_accessed_at` to hierarchy nodes
2. **J.5.2** Create `hierarchy_archiver.py`:
   - `archive_old_nodes(session_id, age_threshold)`
   - Keep centroid, drop L0 content references
   - Archived nodes still participate in E-score (centroid remains)

3. **J.5.3** Archive marker:
   - Add `is_archived BOOLEAN DEFAULT FALSE` to schema
   - Archived L0 nodes return "content archived" placeholder

4. **J.5.4** Tests: `tests/test_hierarchy_archiver.py` (~10 tests)

**Output:** Old content can be archived while maintaining hierarchy structure

---

## Execution Order

```
Phase 1 (Parallel):
  Stream A (Schema)    -----> [J.1 complete]
  Stream B (Retrieval) -----> [J.2 complete] (can mock schema)

Phase 2 (Parallel, after Schema):
  Stream C (Tree Build) -----> [J.3 complete]
  Stream E (Forgetting) -----> [J.5 complete]

Phase 3 (Sequential, after all above):
  Stream D (Integration) -----> [J.4 complete]
```

**Visualization:**
```
        A (Schema) ─────────────┬─────> C (Tree) ─────┐
                                │                      │
        B (Retrieval) ──────────┴─────> E (Forget) ───┼──> D (Integration)
                                                       │
                                                       v
                                               [Phase J Complete]
```

---

## Agent Prompts

### Opus #1: Schema & Data Model
```
You are implementing Phase J.1 of CAT Chat: the centroid hierarchy schema.

Context:
- J.0 (Vector Persistence) is complete: see catalytic_chat/vector_persistence.py
- We need a hierarchical structure for O(log n) retrieval
- Levels: L0 (turns), L1 (100 turns), L2 (1000), L3 (10000)

Your tasks:
1. Create catalytic_chat/hierarchy_schema.py with HierarchyNode dataclass
2. Create catalytic_chat/centroid_math.py with centroid operations
3. Add schema to session_capsule.py ensure_schema()
4. Create tests/test_hierarchy_schema.py

Follow existing code patterns in the catalytic_chat/ package.
Do NOT use Unicode in code (user preference).
```

### Opus #2: Recursive Retrieval
```
You are implementing Phase J.2 of CAT Chat: hierarchical retrieval.

CRITICAL: Use TOP-K selection, NOT threshold-based pruning.
Experimental validation showed threshold-based pruning is too aggressive.

Your tasks:
1. Create catalytic_chat/hierarchy_retriever.py
2. Implement retrieve_hierarchical() with top-K at each level
3. Add budget-aware cutoff
4. Create tests/test_hierarchy_retriever.py

The algorithm (validated on SQuAD, 85% recall, 5.6x speedup):
- At each level, score ALL children by E(query, child.centroid)
- Take top-K (default K=10)
- Recurse into those K children
- At L0, return the actual turn content

You can mock/stub the schema if needed - Stream A will provide the real one.
```

### Opus #3: Tree Maintenance
```
You are implementing Phase J.3 of CAT Chat: automatic tree maintenance.

Key findings from experimental validation:
- Semantic clustering (k-means) is REQUIRED - random grouping fails
- Consider PCA to Df=22 dimensions before clustering (removes noise)

Your tasks:
1. Create catalytic_chat/hierarchy_builder.py
2. Implement on_turn_compressed() for incremental updates
3. Implement build_initial_hierarchy() with k-means clustering
4. Wire into turn_compressor.py
5. Create tests/test_hierarchy_builder.py

Incremental centroid update formula:
  new_centroid = (old_centroid * n + new_vec) / (n + 1)

Level promotion:
- L1 full (100 children) -> close it, start new L1
- 10 L1 nodes -> create L2 parent with centroid of L1 centroids
```

### Opus #4: Integration & Hot Path
```
You are implementing Phase J.4 of CAT Chat: integration and hot path.

Your tasks:
1. Add hot path to hierarchy_retriever.py (last 100 turns skip hierarchy)
2. Integrate hierarchy into auto_context_manager.py
3. Add hierarchy metrics
4. Create tests/test_hierarchy_integration.py

The hot path optimization:
- Recent turns (last 100) are checked directly with brute-force E-score
- Older turns use the hierarchical retrieval
- This avoids hierarchy overhead for recent, frequently-accessed content
```

### Opus #5: Forgetting & Archival
```
You are implementing Phase J.5 of CAT Chat: forgetting mechanism.

Your tasks:
1. Create catalytic_chat/hierarchy_archiver.py
2. Add last_accessed tracking to hierarchy nodes
3. Implement archive_old_nodes() - keeps centroid, drops L0 content
4. Create tests/test_hierarchy_archiver.py

Key insight: Archived nodes STILL participate in E-score via their centroid.
Only the actual turn content is dropped. The hierarchy structure remains.
```

---

## Success Criteria

- [ ] O(log n) E-computations per query (measured)
- [ ] Recall >= 80% at 100K turns (validated)
- [ ] Zero external dependencies for retrieval
- [ ] Self-maintaining tree structure
- [ ] All tests pass
- [ ] Integration with auto_context_manager complete

---

## Also Pending: C.6.3

From Phase C, one item remains:
- [ ] C.6.3 Track E-score vs response quality correlation

This is marked "Future" in threshold_adapter.py and is separate from Phase J.
Consider adding as Stream F if desired.

---

## Files to Create/Modify

**New Files (6):**
- catalytic_chat/hierarchy_schema.py
- catalytic_chat/centroid_math.py
- catalytic_chat/hierarchy_retriever.py
- catalytic_chat/hierarchy_builder.py
- catalytic_chat/hierarchy_archiver.py
- tests/test_hierarchy_*.py (5 test files)

**Modified Files (3):**
- catalytic_chat/session_capsule.py (add schema)
- catalytic_chat/turn_compressor.py (wire builder)
- catalytic_chat/auto_context_manager.py (use hierarchy)
