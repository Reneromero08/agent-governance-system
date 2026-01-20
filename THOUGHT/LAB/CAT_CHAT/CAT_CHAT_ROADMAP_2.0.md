# CAT Chat Roadmap v2.0

**Last updated:** 2026-01-19
**Scope:** Pending work to make CAT Chat a fully catalytic, integrated system
**Previous:** CAT_CHAT_ROADMAP_1.1.md (archived)
**Design Spec:** `INBOX/reports/V4/01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md`

---

## What is CAT Chat?

**CAT = Catalytic.** This is not just bounded chat - it's chat that operates on catalytic computing principles:

- **Clean Space:** Bounded context window (tokens)
- **Catalytic Space:** Large disk state that MUST restore exactly after use
- **Compression:** Symbols/pointers instead of full content (56,370x for a single symbol)
- **Verification:** Every expansion produces hash-verified receipts
- **Fail-Closed:** Restoration failure = hard exit, never silent

---

## Sandbox Status

CAT Chat lives in `THOUGHT/LAB/` for rapid iteration without main repo governance.

**Current Structure:**
```
THOUGHT/LAB/CAT_CHAT/
    _generated/
        cat_chat.db     # Single consolidated database (all tables)
    catalytic_chat/     # Python package
    tests/
    SCHEMAS/
```

**On Graduation:** `_generated/cat_chat.db` moves to `NAVIGATION/CORTEX/cassettes/cat_chat.db`

---

## Core Concepts

**Session Model:** Session = tiny working set (clean space) + hash pointers to offloaded state (catalytic space).

**Auto-Controlled Context (Virtual Memory for LLMs):**
The system automatically manages what's in context vs what's offloaded:
- **Working Set:** What's currently materialized in context (auto-managed)
- **Pointer Set:** What's available but offloaded to disk (auto-retrieved when relevant)
- **Auto-Eviction:** When budget exceeded, lowest-E items move to pointer set
- **Auto-Hydration:** On each query, high-E items from pointer set get materialized
- **Turn Compression:** After response, old turns compress to hash pointers

The model never manually requests content - it thinks about a topic and the system notices the E-score spike and hydrates relevant content automatically.

**Retrieval Order:** Main cassettes (NAVIGATION/CORTEX/) -> Local index -> CAS (exact hash) -> Vectors (approximate fallback).

**Catalytic Invariants:**
1. **INV-CATALYTIC-01 (Restoration):** File states before/after must be identical (or explicitly committed)
2. **INV-CATALYTIC-02 (Verification):** Proof size = O(1) per domain (single Merkle root)
3. **INV-CATALYTIC-03 (Reversibility):** restore(snapshot) = original (byte-identical)
4. **INV-CATALYTIC-04 (Clean Space Bound):** Context uses pointers, not full content
5. **INV-CATALYTIC-05 (Fail-Closed):** Restoration failure = hard exit
6. **INV-CATALYTIC-06 (Determinism):** Identical inputs = identical Merkle root
7. **INV-CATALYTIC-07 (Auto-Context):** Working set managed by system, not manual references

**Design Decisions:**
- ELO scores are informational metadata only, not used for ranking
- Vectors are fallback only, not primary retrieval path
- All expansions bounded (no `ALL` slices)
- Context management is automatic, not manual @symbol references

---

## Database Schema

Single `_generated/cat_chat.db` contains all tables:

**Index Layer:**
- `sections` - Indexed content from files
- `section_index_meta` - Index metadata
- `symbols` - @symbol -> section mappings
- `expansion_cache` - Cached resolved content

**Cassette Layer (append-only work log):**
- `cassette_meta` - Version info
- `cassette_messages` - Requests (append-only)
- `cassette_jobs` - Work items
- `cassette_steps` - PENDING -> LEASED -> COMMITTED FSM
- `cassette_receipts` - Execution proofs (append-only)
- `cassette_job_budgets` - Token/byte tracking

**Session Layer:**
- `sessions` - Session state
- `session_events` - Hash-chained event log
- `session_working_set` - Active working set
- `session_pointer_set` - Pointer references

---

## Completed Infrastructure

| Component | Files | Status |
|-----------|-------|--------|
| Substrate & Indexing | section_extractor.py, section_indexer.py, slice_resolver.py | Done |
| Symbol Registry | symbol_registry.py, symbol_resolver.py | Done |
| Message Cassette | message_cassette.py, message_cassette_db.py | Done |
| Planner | planner.py | Done |
| Bundle Protocol | bundle.py | Done |
| Receipts & Attestations | receipt.py, attestation.py, merkle_attestation.py | Done |
| Trust & Identity | trust_policy.py, validator_identity.py | Done |
| Executor | executor.py, execution_policy.py | Done |
| Context Assembly | context_assembler.py, geometric_context_assembler.py | Done |
| MCP Integration | mcp_integration.py | Done |
| Geometric Chat | geometric_chat.py | Done |
| Session Capsule | session_capsule.py | Done |
| CORTEX Resolver | cortex_expansion_resolver.py | Done |
| Auto-Context Manager | auto_context_manager.py | Done |
| Context Partitioner | context_partitioner.py | Done |
| Turn Compressor | turn_compressor.py | Done |
| Adaptive Budget | adaptive_budget.py | Done |

---

## Pending Work

### A. Session Persistence Tests (P0 - Blocking)

**Status:** COMPLETE
**Files:** session_capsule.py, cortex_expansion_resolver.py
**Tests:** tests/test_session_capsule.py (17 tests, all passing)

- [x] A.1 Save/resume determinism test (byte-identical replay)
- [x] A.2 Partial execution resume test (no state loss)
- [x] A.3 Tamper detection test (fail-closed on corruption)
- [x] A.4 Hydration failure test (fail-closed on unresolvable symbols)

**Exit Criteria:** All 4 fixtures green, determinism proven - ACHIEVED

---

### B. Main Cassette Network Integration (P1 - Enhancement)

**Status:** COMPLETE
**Purpose:** Connect to existing cassette infrastructure for richer content (not blocking C)
**Files:** cassette_client.py, cortex_expansion_resolver.py
**Tests:** tests/test_cassette_client.py (22 tests), tests/test_cassette_symbol_resolution.py (15 tests)
**Docs:** docs/WRITE_ISOLATION.md, docs/GRADUATION_PATH.md

- [x] B.1 CassetteClient for reading main cassettes:
  - Read from `NAVIGATION/CORTEX/cassettes/*.db`
  - Search across canon.db, governance.db, etc.
  - Respect cassette network conventions
- [x] B.2 Symbol resolution via main cassettes:
  - Resolve @symbols to content in main cassettes
  - Fall back to local index if not found
- [x] B.3 Write isolation:
  - Reads: main cassettes (shared)
  - Writes: `_generated/cat_chat.db` (sandbox only)
  - Documented in docs/WRITE_ISOLATION.md
- [x] B.4 Graduation path:
  - Documented in docs/GRADUATION_PATH.md (future reference)
  - CAT_CHAT stays in LAB for now

**Exit Criteria:** CAT Chat reads from main cassette network, writes locally - ACHIEVED

---

### C. Auto-Controlled Context Loop (P0 - Core Catalytic Behavior)

**Status:** Core complete, C.6.3 pending
**Purpose:** Virtual memory for LLMs - automatic working set management
**Depends On:** A (session persistence)
**Files:** auto_context_manager.py, context_partitioner.py, turn_compressor.py, adaptive_budget.py
**Tests:** tests/test_auto_context_loop.py
**E-Score Implementation:** `THOUGHT/LAB/FORMULA/experiments/open_questions/q44/q44_core.py`
- `compute_E_linear(query_vec, context_vecs)` - E = mean overlap (Born rule)
- `compute_born_probability(query_vec, context_vecs)` - P = |<psi|phi>|^2
- Threshold = 0.5 (validated in Q44)

This is THE core catalytic behavior. Without this, nothing is actually catalytic.

**C.1 Context Budget & Working Set:**
- [x] C.1.1 Define clean space budget (max tokens for working set)
- [x] C.1.2 Track working_set (materialized) vs pointer_set (offloaded) per session
- [x] C.1.3 Hard fail if working_set exceeds budget (INV-CATALYTIC-04)

**C.2 E-Score Based Eviction:**
- [x] C.2.1 On budget exceeded, compute E-score of each working_set item vs current query
- [x] C.2.2 Evict lowest-E items to pointer_set until under budget
- [x] C.2.3 Log eviction events to session_events (hash-chained)

**C.3 E-Score Based Hydration:**
- [x] C.3.1 On each query, compute E-score of query vs all pointer_set items
- [x] C.3.2 Hydrate high-E items (above threshold) into working_set
- [x] C.3.3 Hydration is bounded - max N items per query, respects budget
- [x] C.3.4 Log hydration events to session_events

**C.4 Turn Compression:**
- [x] C.4.1 After response, old turns (beyond window) compress to hash pointers
- [x] C.4.2 Full turn content stored in catalytic space (session_events)
- [x] C.4.3 Only hash pointer + summary remains in working_set

**C.5 Catalytic Chat Loop:**
- [x] C.5.1 Wire together: query -> hydrate -> assemble -> LLM -> compress -> evict
- [x] C.5.2 Session capsule logs every step (deterministic replay)
- [x] C.5.3 All context assembly uses ContextAssembler with budgets
- [x] C.5.4 GeometricChat.respond() uses auto-managed context, not raw docs

**C.6 E-Gating Threshold Tuning:**
- [x] C.6.1 Default threshold = 0.5 (from Q44 validation)
- [x] C.6.2 Configurable per-session
- [ ] C.6.3 Track E-score vs response quality correlation (marked "Future" in threshold_adapter.py)

**Exit Criteria:** Core achieved, quality correlation tracking pending
- Model runs with fully auto-managed context
- No manual @symbol references needed
- Working set stays within budget across entire session
- Eviction/hydration decisions logged and reproducible

---

### D. Semantic Pointer Compression Integration (P1)

**Status:** COMPLETE
**Purpose:** Use SPC instead of verbose @symbols
**Files:** spc_bridge.py
**Tests:** tests/test_spc_integration.py (35 tests, all passing)

- [x] D.1 Codebook sync handshake:
  - Verify codebook_id + SHA256 on session start
  - Fail-closed on mismatch
  - EVENT_CODEBOOK_SYNC logged to session events
- [x] D.2 Pointer resolution:
  - Support SYMBOL_PTR (CJK characters like `法` + ASCII radicals CIVLGSRAJP)
  - Support HASH_PTR (sha256:7cfd0418...)
  - Support COMPOSITE_PTR (`法.驗`, `C3:build`, `C&I`)
  - SPC is highest priority in resolution chain (before CORTEX)
- [x] D.3 Compression metrics:
  - Track tokens_expanded, tokens_pointers, tokens_saved
  - Track CDR (Concept Density Ratio) per Q33
  - Per-symbol usage and savings tracking
  - EVENT_SPC_METRICS for session logging

**Exit Criteria:** SPC pointers resolve correctly with fail-closed semantics - ACHIEVED

---

### E. Vector Fallback Chain (P2)

**Status:** COMPLETE
**Purpose:** Vectors as governed fallback, not primary path
**Files:** cas_resolver.py, vector_fallback.py, elo_observer.py, cortex_expansion_resolver.py
**Tests:** tests/test_cas_resolver.py (19 tests), tests/test_vector_fallback.py (24 tests), tests/test_retrieval_order.py (11 tests) - all 54 passing

- [x] E.1 Retrieval order enforcement:
  - 1st: SPC (Phase D)
  - 2nd: Main cassette FTS
  - 3rd: Local index
  - 4th: CAS (exact hash)
  - 5th: Vector search (fallback only)
  - 6th: Fail-closed
- [x] E.2 Vector governance:
  - Agent is FREE to search until it finds what it needs
  - Budget is SAFETY BOUNDARY, not fill target (most searches find what they need early)
  - Only ONE config param: `min_similarity` (0.5, empirically validated in Q44)
  - Config file: `_generated/vector_fallback_config.json` for tuning
  - Search logging: `_generated/vector_fallback_search.jsonl` for analysis
  - No trust-vectors bypass (all results hash-verified)
- [x] E.3 ELO as metadata:
  - EloObserver tracks usage patterns after retrieval
  - Does NOT modify ranking (called AFTER results returned)
  - Logs to session for analytics

**Exit Criteria:** Vector fallback operational with governance - ACHIEVED

---

### F. Docs Index (P2)

**Status:** COMPLETE (2026-01-19)
**Purpose:** Fast, bounded discovery via FTS
**Files:** docs_index.py, cli.py (docs commands), cortex_expansion_resolver.py (step 3b integration)
**Tests:** tests/test_docs_index.py (26 tests) - all passing

- [x] F.1 FTS tables in cat_chat.db:
  - `docs_files` (path, sha256, size_bytes, indexed_at)
  - `docs_content` (file_id, chunk_index, content_normalized)
  - `docs_content_fts` (FTS5 with Porter stemming)
  - Automatic sync triggers for INSERT/UPDATE/DELETE
- [x] F.2 Query API:
  - `docs index` - build/rebuild documentation index (sha256 deduplication)
  - `docs search --query "..." --limit N` - bounded search with JSON output
  - `docs show <sha256>` - retrieve file by hash
  - `docs stats` - index statistics
  - Returns identifiers + bounded snippets (max 200 chars)
- [x] F.3 Deterministic ranking with stable tie-breakers:
  - BM25 score (ascending)
  - file_path (alphabetical tie-breaker)
  - chunk_index (secondary tie-breaker)
- [x] F.4 Integration into retrieval chain:
  - Step 3b: After symbol registry, before CAS
  - `docs_index_hits` stat tracking
  - Lazy-loaded DocsIndex property

**Exit Criteria:** `docs search` returns bounded, deterministic results - ACHIEVED

---

### G. Bundle Replay & Verification (P2)

**Status:** COMPLETE (2026-01-19)
**Purpose:** Prove bundles are self-contained and reproducible
**Files:** bundle_runner.py
**Tests:** tests/test_bundle_replay.py (28 tests, all passing)

- [x] G.1 Bundle runner: takes bundle.json + artifacts only
  - BundleRunner class operates without repo_root or database access
  - All inputs resolved from artifacts/ directory
  - replay_bundle() convenience function for single-shot replay
- [x] G.2 Verify-before-run: hard fail on mismatch
  - Verification delegated to BundleVerifier before any execution
  - BundleRunnerError raised on any verification failure
  - No partial execution - no receipt written on failure
- [x] G.3 Reproducibility: run twice -> identical receipts
  - Byte-identical receipt files on repeat runs
  - Identical receipt_hash and merkle_root
  - Deterministic across different directories (no path leakage)

**Exit Criteria:** Bundle can be replayed offline with identical outputs - ACHIEVED

---

### H. Specs & Golden Demo (P3)

**Status:** COMPLETE
**Purpose:** Authoritative documentation + runnable demo

- [x] H.1 Authoritative specs (bundle, receipts, trust, execution)
  - `docs/specs/BUNDLE_SPEC.md` - Bundle protocol v5.0.0
  - `docs/specs/RECEIPT_SPEC.md` - Receipt format v1.0.0
  - `docs/specs/TRUST_SPEC.md` - Trust model v1.0.0
  - `docs/specs/EXECUTION_SPEC.md` - Execution semantics v1.0.0
  - `docs/specs/SPEC_INDEX.md` - Specification index
- [x] H.2 Runbook: copy-paste runnable on Windows PowerShell
  - `docs/CAT_CHAT_USAGE_GUIDE.md` - Updated with Fresh Clone Quick Start
- [x] H.3 Golden demo from fresh clone
  - `golden_demo/golden_demo.py` - Self-contained demo script
  - `golden_demo/README.md` - Demo documentation
  - `golden_demo/fixtures/` - Demo content and fixtures

**Exit Criteria:** New user can run golden demo from README - ACHIEVED

---

### I. Measurement & Benchmarking (P3)

**Status:** Not started
**Purpose:** Prove catalytic compression with numbers

- [ ] I.1 Per-step metrics (bytes expanded, cache hits, reuse rate)
- [ ] I.2 Compression benchmarks vs baseline
- [ ] I.3 Catalytic invariant verification suite

**Exit Criteria:** Compression claims backed by reproducible benchmarks

---

### J. Recursive E-Score Hierarchy (P3 - Future)

**Status:** Not started
**Purpose:** Extend effective memory from ~1,000 turns to ~100,000+ turns
**Depends On:** C (Auto-Controlled Context Loop must work first)

**Philosophy:** E guides to E guides to E. No external tools. No indexes. No approximate nearest neighbor hacks. Just recursive vector math - the same E-score at every level of the tree.

---

**The Core Insight:**

Every turn has an embedding vector. E-score is computed FROM vectors:
```
E = |<query_vec | item_vec>|^2   (Born rule)
```

A "drawer" is just a GROUP of vectors with a CENTROID (average vector). E-score against the centroid predicts whether the contents are worth checking:
```
centroid = mean([child_1_vec, child_2_vec, ..., child_n_vec])

E(query, centroid) low  --> contents probably all low E --> SKIP
E(query, centroid) high --> some contents probably high E --> RECURSE
```

**No text summaries. No LLM calls. Just vector centroids.**

---

**The Structure:**
```
L3: 10 centroids      (each = average of 10,000 turn vectors)
    |
L2: 100 centroids     (each = average of 1,000 turn vectors)
    |
L1: 1,000 centroids   (each = average of 100 turn vectors)
    |
L0: 100,000 vectors   (actual turn embeddings + content)
```

---

**The Algorithm:**
```
Query: "What signing algorithm?"
       |
       v
E(query, centroid_L3_A) = 0.1 --> SKIP (prune 10,000 turns)
E(query, centroid_L3_B) = 0.6 --> OPEN
                                   |
                                   v
              E(query, centroid_L2_B1) = 0.2 --> SKIP (prune 1,000 turns)
              E(query, centroid_L2_B2) = 0.7 --> OPEN
                                                  |
                                                  v
                         E(query, centroid_L1_B2a) = 0.3 --> SKIP
                         E(query, centroid_L1_B2b) = 0.8 --> OPEN
                                                             |
                                                             v
                                    E(query, turn_47) = 0.9 --> RETRIEVE
                                    E(query, turn_48) = 0.2 --> skip
```

**Same function at every level. Same threshold. E guides to E guides to E.**

---

**J.0 Vector Persistence (PREREQUISITE):**

Currently CAT Chat stores text but computes embeddings on-the-fly. This is 10,000x slower than using stored vectors:
```
Embedding API call: ~10-100ms per item
Dot product:        ~0.001ms per item
```

Without stored vectors, the hierarchy can't work at scale.

- [ ] J.0.1 Add `embedding` BLOB column to `session_events` table
- [ ] J.0.2 Store embedding at turn compression time (one API call, then persisted)
- [ ] J.0.3 Add `load_vectors(session_id)` to load all embeddings on session resume
- [ ] J.0.4 Migrate existing sessions: backfill embeddings for turns without them

**Storage:** 384-dim float32 = 1.5KB per turn. 100K turns = 150MB. Acceptable.

---

**J.1 Centroid Structure:**
- [ ] J.1.1 Define levels: L0 (turns), L1 (100 turns), L2 (1000 turns), L3 (10000 turns)
- [ ] J.1.2 Each node has: centroid_vector, child_pointers, turn_count
- [ ] J.1.3 Centroid = mean(child vectors) - pure vector math, no LLM
- [ ] J.1.4 Store in session_events with level metadata and parent pointer

**J.2 Recursive Retrieval:**
- [ ] J.2.1 `retrieve(query_vec, node)` - single recursive function
- [ ] J.2.2 Base case (L0): return turn content if budget allows
- [ ] J.2.3 Recursive case: E(query, centroid) >= threshold? recurse into children
- [ ] J.2.4 Budget-aware: stop if working_set would exceed limit

```python
def retrieve(query_vec, node, budget):
    if node.level == 0:
        return [node.content] if budget > node.tokens else []

    results = []
    for child in node.children:
        E = compute_E(query_vec, child.centroid)
        if E >= THRESHOLD:
            results.extend(retrieve(query_vec, child, budget))
            budget -= sum(r.tokens for r in results)
    return results
```

**J.3 Tree Maintenance:**
- [ ] J.3.1 On turn compression: add vector to current L1 node, update centroid
- [ ] J.3.2 When L1 has 100 children: close it, start new L1
- [ ] J.3.3 When 10 L1 nodes exist: create L2 parent with centroid of L1 centroids
- [ ] J.3.4 Centroid update: `new_centroid = (old_centroid * n + new_vec) / (n + 1)`

**J.4 Hot Path (Recent Turns):**
- [ ] J.4.1 Last 100 turns: check directly (skip hierarchy)
- [ ] J.4.2 Older turns: use recursive hierarchy

**J.5 Forgetting:**
- [ ] J.5.1 Track last_accessed per node
- [ ] J.5.2 Archive old nodes: keep centroid, drop L0 content
- [ ] J.5.3 Archived nodes still participate in E-score (centroid remains)

---

**Complexity:**
```
Brute force:  O(n)                    100K turns = 100,000 E-computations
Hierarchy:    O(b * log_b(n))         100K turns = ~250 E-computations (b=100)
                                      1M turns   = ~300 E-computations
```

**Exit Criteria:**
- O(log n) E-computations per query
- Recall >= 80% at 100K turns
- Zero external dependencies
- Self-maintaining tree structure

---

**Mathematical Foundation:**

The hierarchy works because E-score against centroid approximates mean E-score of contents:
```
E(query, centroid) ~ mean(E(query, child_i))
```

If the average is low, most children are low. If the average is high, at least some children are high.

**This is why E guides to E: the centroid IS just another vector, and E-score IS the decision function at every level.**

---

**Experimental Validation (SQuAD Dataset):**

Tested on SQuAD reading comprehension dataset (10K passages, 1K questions with known answers).

**Key Findings:**

| Metric | Brute Force | Hierarchy | Notes |
|--------|-------------|-----------|-------|
| Recall@10 | 88% | 85% | 97% of brute force quality |
| E-computations | 10,000 | 1,785 | **5.6x speedup** |
| LLM accuracy | 100%* | 100%* | *on gold context |

**Critical Insights:**

1. **Top-K centroid selection > threshold-based pruning**
   - Threshold-based pruning (E >= 0.5) was too aggressive
   - Top-K selection (explore K=10 highest-E centroids) works much better
   - Reason: centroids smooth out E-scores, making threshold decisions unreliable

2. **Semantic clustering is REQUIRED**
   - Random grouping (by index) creates meaningless centroids
   - k-means clustering before building hierarchy is essential
   - Groups semantically similar passages -> discriminative centroids

3. **Retrieval is the bottleneck, not LLM**
   - LLM (lfm2.5-1.2b) achieves 100% accuracy when given gold context
   - All failures are retrieval misses, not LLM comprehension failures

**Implementation Note:** Test file at `examples/test_hierarchy_squad.py`

**Updated Algorithm (based on findings):**
```
def retrieve_hierarchical(query_vec, root, top_k=10):
    """Top-K centroid selection - NOT threshold-based."""
    if node.level == 0:
        return [(node, E(query_vec, node.centroid))]

    # Score ALL children, take top-K
    child_scores = [(child, E(query_vec, child.centroid))
                    for child in node.children]
    child_scores.sort(by E, descending)

    results = []
    for child, _ in child_scores[:top_k]:
        results.extend(retrieve_hierarchical(query_vec, child, top_k))
    return results
```

**Research-Backed Optimizations (from FORMULA/research/questions):**

The following findings from the research index could further improve retrieval:

| Finding | Source | Implication |
|---------|--------|-------------|
| Df = 22 effective dimensions | Q43 (QGT) | Cluster in PCA-reduced space, not full 384D |
| Space is curved (holonomy -0.10 rad) | Q43 | Consider spherical k-means instead of flat |
| Phase transition at alpha=0.9 | Q12 | Sharp boundary between meaningful/meaningless |
| Angular momentum conserved (CV=6e-7) | Q38 | Geodesic distance may outperform cosine^2 |
| E = Born rule CONFIRMED (r=0.999) | Q44 | Current E-score formula is correct |

**Potential Improvement: PCA Pre-clustering**
```python
from sklearn.decomposition import PCA

Df = 22  # Effective dimensionality from Q43
pca = PCA(n_components=Df)
projected = pca.fit_transform(vectors)

# Cluster in reduced space - removes ~362 dimensions of noise
kmeans = KMeans(n_clusters=n_clusters)
labels = kmeans.fit_predict(projected)
```

**Why this might help:** Random directions in 384D dilute centroid signal. Projecting to the ~22 "carved semantic directions" (Q43) concentrates information where it matters. Untested but theoretically sound.

---

**Iso-Temporal Protocol Validation (VALIDATED!):**

**The Hypothesis:** Tracking "rotation signature" (processing context) improves retrieval.

**VALIDATED on REAL DATA (99 arxiv papers, 3487 sections):**

After adding temporal links to feral_eternal.db with explicit prev/next pointers:

| Method | Recall@10 | vs Pure E |
|--------|-----------|-----------|
| Pure E-score | 33.0% | baseline |
| Context (lambda=0.2) | 36.5% | **+10.6%** |
| Frame+E (lambda=0.5) | **38.5%** | **+16.7%** |
| Context-only | 21.0% | -36% |
| Frame-only | 23.0% | -30% |

**Also validated with synthetic causal data:**

| Context Influence | Pure E | Context+E | Frame+E | Improvement |
|-------------------|--------|-----------|---------|-------------|
| 30% (weak) | 55.3% | 76.0% | 75.3% | **+37.3%** |
| 50% | 64.7% | 86.0% | 87.3% | **+33.0%** |
| 70% | 78.7% | 93.3% | 94.0% | **+18.6%** |
| 90% (strong) | 88.7% | 96.7% | 97.3% | **+9.0%** |

**Key Insights:**

1. **CONTEXT HELPS** even on authored paper content (+10.6-16.7% improvement)
2. **Rotation frames beat centroid** by 5.7% on real data (38.5% vs 36.5%)
3. **Best lambda is low (0.2-0.5)** - too much context weight hurts recall
4. **Requires temporal DB schema** - without explicit prev/next pointers, context must be reconstructed

**Schema Requirements (implemented in feral_eternal.db):**
```sql
ALTER TABLE vectors ADD COLUMN prev_vector_id TEXT;
ALTER TABLE vectors ADD COLUMN next_vector_id TEXT;
ALTER TABLE vectors ADD COLUMN context_vec_blob BLOB;  -- precomputed centroid
ALTER TABLE vectors ADD COLUMN sequence_id TEXT;       -- paper_id, session_id, etc
ALTER TABLE vectors ADD COLUMN sequence_idx INTEGER;   -- position in sequence
```

**Migration:** `FERAL_RESIDENT/migrations/001_add_temporal_links.py`

**Implication for CAT Chat:** The Auto-Controlled Context Loop (Section C) should store context signatures with each turn. During retrieval, score items by: `E(query, item) + lambda * E(query, item_context)`

**Test file:** `examples/test_isotemporal_feral.py --rotation`

---

## Priority Summary

| Priority | Phase | Blocker? | Effort |
|----------|-------|----------|--------|
| P0 | A. Session Tests | DONE | Small |
| P1 | B. Cassette Network Integration | DONE | Medium |
| P0 | C. Auto-Controlled Context Loop | Core done (C.6.3 pending) | Large |
| P1 | D. SPC Integration | DONE | Medium |
| P2 | E. Vector Fallback | DONE | Medium |
| P2 | F. Docs Index | DONE | Medium |
| P2 | G. Bundle Replay | DONE | Medium |
| P3 | H. Specs & Demo | No | Medium |
| P3 | I. Measurement | No | Medium |
| P3 | J. Scaling & Hierarchical Memory | No | Large |

**Recommended order:** H -> I -> J

**Scaling Note:** J is intentionally last. The core catalytic loop (C) must work well at 1K turns before optimizing for 100K+. Premature optimization is the root of all evil.

**Note:** P0 core complete (A, B, C core), P1 complete (D), P2 complete (E, F, G). C.6.3 (quality correlation tracking) still pending. The system is catalytic with auto-managed context, SPC pointer compression, governed vector fallback, docs index, and offline bundle replay.

---

## Dependencies (All Complete)

- **Phase 5 (Vector/Symbol):** COMPLETE - provides semantic search
- **Phase 6 (Cassette Network):** COMPLETE - provides 9 modular cassettes
- **Phase 7 (Vector ELO):** COMPLETE - provides usage metadata
- **Phase 8 (Resident AI):** COMPLETE - provides geometric reasoning

All dependencies satisfied. Integration work is unblocked.

---

## Graduation Criteria

CAT Chat graduates from LAB to main system when:

1. All P0 items complete (A: DONE, B: DONE, C: Core done, C.6.3 pending)
2. Auto-controlled context loop operational (C) - Core done
3. Tests pass with main cassette network
4. All 7 catalytic invariants verified (including INV-CATALYTIC-07: Auto-Context)
5. Compression claims proven with benchmarks
6. Golden demo works from fresh clone with auto-managed context
