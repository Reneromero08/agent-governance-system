# Cassette Network Roadmap

**Status**: Phase 0-4 Complete, Phase 5 Alpha Ready, Phase 6 Next
**Vision**: Layer 3 (CAS External) of the compression stack - shared semantic context infrastructure enabling near-zero communication entropy
**Owner**: Antigravity / Resident
**Upstream Dependency**: Phase 5 (VECTOR_ELO) - MemoryRecord contract, SPC protocol

---

## The Core Insight

**The industry optimizes the wrong metric.**

Shannon entropy `H(X)` measures bits to encode a message in isolation.
But communication is between parties with **shared context**.

**Conditional entropy:**
```
H(X|S) = H(X) - I(X;S)

When S (shared context) contains X (message):
  I(X;S) ≈ H(X)
  Therefore: H(X|S) ≈ 0
```

**The Cassette Network IS the shared context `S`.**

When sender and receiver share cassettes, a pointer into that shared space is all that's needed. That's `log2(N)` bits instead of the full message.

**Proof:** `法 → all canon` = 56,370x compression (conditional entropy vs message entropy)

---

## Dependency Chain

```
Phase 5.1.0 MemoryRecord Contract    ✅ DONE (23 tests)
         ↓
Phase 5.1.1 Embed Canon              ✅ DONE (32 files indexed)
         ↓
Phase 5.2 SCL Compression (L2)       ⏳ In Progress
         ↓
Phase 5.3 SPC Formalization          ⏳ In Progress (Protocol spec)
         ↓
Phase 6.0 Cassette Network (L3)      ← YOU ARE HERE
         ↓
Phase 6.x Session Cache (L4)         ⏳ Future
         ↓
Phase 7 ELO Integration              ⏳ Future (scores.elo field)
```

**Critical Integration Points:**
- `MemoryRecord` schema (5.1.0) = cassette record format
- `CODEBOOK_SYNC_PROTOCOL` (5.3.3) = cassette handshake protocol
- `SPC_SPEC` (5.3) = semantic pointer verification

---

## Executive Summary

The Cassette Network is **not just a distributed database**. It is infrastructure for **shared semantic context** that enables communication entropy to approach zero.

**Traditional approach:** Send full content every time → `H(X)` bits
**Cassette approach:** Share context once, send pointers → `H(X|S) ≈ 0` bits

### Current State

**What exists:**
- `NAVIGATION/CORTEX/db/system1.db` - Basic semantic search
- `semantic_search` MCP tool (working, tested by Opus)
- Embedding engine (`all-MiniLM-L6-v2`, 384 dims)
- Basic cassette protocol (`cassette_protocol.py`)
- Network hub coordinator (`network_hub.py`)
- Two production cassettes (Governance + AGI Research)

**What's missing:**
- Write path (can't save memories)
- Partitioned cassettes (everything in one DB)
- Full cross-cassette query federation
- Resident identity/persistence

---

## Architecture Vision: The Cassette Deck

```
┌─────────────────────────────────────────────────────────┐
│         SEMANTIC NETWORK HUB (network_hub.py)           │
│                                                         │
│  Protocol: HANDSHAKE → QUERY → RESPONSE → HEARTBEAT    │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬───────────────┐
        │                 │                 │               │
   ┌────▼────┐       ┌────▼────┐      ┌────▼────┐    ┌────▼────┐
   │  GOV    │       │  CODE   │      │  RSCH   │    │  CNTR   │
   │ Cassette│       │ Cassette│      │ Cassette│    │ Cassette│
   └─────────┘       └─────────┘      └─────────┘    └─────────┘

   Governance        Code Impl      AGI Research    Contracts
   ↓                 ↓              ↓               ↓
   governance.db    code.db        research.db     contracts.db
   - ADRs           - TOOLS/*.py   - Papers        - Fixtures
   - CANON          - *.json       - Theory        - Schemas
   - SKILLS         - *.js/.ts     - Experiments   - Tests
   1,548 chunks     TBD chunks     2,443 chunks    TBD chunks
   [vectors, fts]   [code, ast]    [research]      [fixtures]
```

---

## Phase 0: Foundation (COMPLETE)

**Status:** Decision gate PASSED

### Deliverables Completed
1. Created `cassette_protocol.py` - Base class for all database cassettes
2. Created `network_hub.py` - Central coordinator for query routing
3. Created `governance_cassette.py` - AGS governance cassette
4. Created `agi_research_cassette.py` - AGI research cassette
5. Demo cross-database query showing governance + research results

### Decision Gate Results
- Cross-database queries return merged results
- Network overhead <20% (demo runs successfully)
- 3,991 total chunks indexed across both cassettes
- 99.4% token compression using @Symbol references

---

## Phase 1: Cassette Partitioning ✅ COMPLETE (2026-01-11)

**Goal:** Split the monolithic DB into semantic-aligned cassettes

**Previous:** [Phase 0](#phase-0-decision-gate--complete) - Decision gate
**Next:** [Phase 1.4](#14-catalytic-hardening-) - Catalytic hardening

### 1.1 Create Bucket-Aligned Cassettes ✅
```
NAVIGATION/CORTEX/cassettes/
├── canon.db          # LAW/ bucket (immutable) - 86 files, 200 chunks
├── governance.db     # CONTEXT/decisions + preferences (stable) - empty
├── capability.db     # CAPABILITY/ bucket (code, skills) - 64 files, 121 chunks
├── navigation.db     # NAVIGATION/ bucket (maps, metadata) - 38 files, 179 chunks
├── direction.db      # DIRECTION/ bucket (roadmaps, strategy) - empty
├── thought.db        # THOUGHT/ bucket (research, lab) - 288 files, 1025 chunks
├── memory.db         # MEMORY/ bucket (archive, reports) - empty
├── inbox.db          # INBOX/ bucket (staging, temporary) - 67 files, 233 chunks
└── resident.db       # AI memories (per-agent, read-write) - empty (new)
```

### 1.2 Migration Script ✅
- [x] Read existing `system1.db` (543 files, 1758 chunks)
- [x] Route sections to appropriate cassettes based on `file_path`
- [x] Preserve all hashes, FTS content, metadata
- [x] Validate: total sections before = total sections after
- [x] Backup at `db/_migration_backup/`
- [x] Receipt: `cassettes/migration_receipt_*.json`

**Implementation:** [migrate_to_cassettes.py](../../NAVIGATION/CORTEX/network/migrate_to_cassettes.py)

### 1.3 Update MCP Server ✅
- [x] `cassette_network_query(query, cassettes=[...], limit=20)` - with cassette filter
- [x] `cassette_stats()` - list all cassettes with counts
- [x] GenericCassette updated for standard cassette schema

**Acceptance:** ✅ ALL MET
- [x] 9 cassettes exist (8 buckets + resident)
- [x] `semantic_search` can filter by cassette
- [x] No data loss from migration (543/543 files, 1758/1758 chunks)

### 1.4 Catalytic Hardening ✅

**Previous:** [Phase 1.1-1.3](#11-create-bucket-aligned-cassettes-) - Cassette partitioning
**Next:** [Phase 1.5](#15-structure-aware-chunking) - Structure-aware chunking

- [x] `compute_merkle_root(hashes)` - Binary Merkle tree of sorted chunk hashes
- [x] `content_merkle_root` per cassette stored in receipt and metadata
- [x] `receipt_hash` - Content-addressed receipt (SHA-256 of canonical JSON)
- [x] `verify_migration(receipt_path)` - Verification function checks:
  - Receipt hash integrity
  - Per-cassette file hash
  - Per-cassette Merkle root
  - Chunk count match
- [x] CLI: `--verify [receipt]` option

**Catalytic Properties:**
| Property | Status |
|----------|--------|
| Content-addressed IDs | ✅ chunk_hash preserved |
| Merkle roots | ✅ Per-cassette |
| Receipt hash | ✅ Content-addressed |
| Verification | ✅ `verify_migration()` |
| Restore guarantee | ✅ Via receipt + source |

**Next:** [Phase 1.5](#phase-15-structure-aware-chunking) - Hierarchical chunk boundaries

---

### 1.5 Structure-Aware Chunking ✅ COMPLETE (2026-01-11)

**Goal:** Split on markdown headers, not sentence boundaries - preserve semantic hierarchy

**Problem Solved:** Chunks split mid-section on `.!?` boundaries, losing header context.

**Implemented Structure:**
```
# H1 Title           → chunk boundary (depth=1)
## H2 Section        → chunk boundary (depth=2)
### H3 Subsection    → chunk boundary (depth=3)
#### H4              → chunk boundary (depth=4)
##### H5             → chunk boundary (depth=5)
body text            → accumulate until next header or size limit
```

**Chunk Schema Extension:**
```sql
ALTER TABLE chunks ADD COLUMN header_depth INTEGER;  -- 1-6 or NULL
ALTER TABLE chunks ADD COLUMN header_text TEXT;      -- "## Section Name"
ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER; -- hierarchy link
```

**Navigation Pattern:**
- Query returns chunk with `header_depth=3`
- Want broader context? `get_parent(chunk_id)` → depth=2
- Want deeper? `get_children(chunk_id)` → list of child chunks
- "Go to next #" = `get_siblings(chunk_id)['next']`
- Breadcrumbs: `get_path(chunk_id)` → `[# Doc, ## Section, ### Subsection]`

**Deliverables:** ✅ ALL COMPLETE
- [x] `chunk_markdown(text) -> List[Chunk]` - Structure-aware splitter
- [x] Schema migration for `header_depth`, `header_text`, `parent_chunk_id`
- [x] Re-index all files with hierarchy (1,758 → 12,478 chunks)
- [x] New Merkle roots (chunks change)
- [x] Navigation queries in `GenericCassette`:
  - `get_chunk(id)` - Full chunk info
  - `get_parent(id)` - Navigate up
  - `get_children(id)` - Navigate down
  - `get_siblings(id)` - Prev/next at same depth
  - `get_path(id)` - Breadcrumb trail
  - `navigate(id, direction)` - Unified navigation

**Implementation Files:**
- [markdown_chunker.py](../../../NAVIGATION/CORTEX/db/markdown_chunker.py) - Structure-aware chunker
- [structure_aware_migration.py](../../../NAVIGATION/CORTEX/network/structure_aware_migration.py) - Migration script
- [generic_cassette.py](../../../NAVIGATION/CORTEX/network/generic_cassette.py) - Navigation queries

**Migration Results:**
| Cassette | Files | Chunks | With Headers |
|----------|-------|--------|--------------|
| canon | 86 | 1,297 | 1,224 |
| capability | 64 | 952 | 908 |
| navigation | 38 | 1,122 | 1,098 |
| thought | 288 | 7,123 | 6,967 |
| inbox | 67 | 1,984 | 1,917 |
| **Total** | **543** | **12,478** | **12,114** |

**Acceptance:** ✅ ALL MET
- [x] Chunks align with markdown headers
- [x] Parent-child relationships navigable
- [x] Token counts respect limits (~500 tokens max)
- [x] Catalytic: new migration receipt with updated Merkle roots

**Previous:** [Phase 1.4](#14-catalytic-hardening-) - Catalytic verification
**Next:** [Phase 2](#phase-2-write-path-memory-persistence) - Write path

---

## Phase 2: Write Path (Memory Persistence) ✅ COMPLETE (2026-01-11)

**Goal:** Let residents save thoughts to the manifold

**Previous:** [Phase 1.5](#15-structure-aware-chunking) - Structure-aware chunking
**Next:** [Phase 3](#phase-3-resident-identity) - Resident identity

### 2.1 Core Functions ✅
```python
memory_save(text: str, metadata: dict = None, agent_id: str = 'default') -> str:
    """Embeds text, stores vector. Returns: content-addressed hash."""

memory_query(query: str, limit: int = 10, agent_id: str = None) -> list:
    """Semantic search over memories. Returns: [{hash, similarity, text_preview, agent_id}]"""

memory_recall(hash: str) -> dict:
    """Retrieve full memory. Returns: {hash, text, vector, metadata, created_at, agent_id}"""

semantic_neighbors(hash: str, limit: int = 10) -> list:
    """Find semantically similar memories to an anchor memory."""
```

### 2.2 Schema Extension ✅
```sql
CREATE TABLE memories (
    hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,  -- 384 dims, float32 (all-MiniLM-L6-v2)
    metadata JSON,
    created_at TEXT NOT NULL,  -- ISO8601
    agent_id TEXT,  -- 'opus', 'sonnet', 'default', etc.
    indexed_at TEXT NOT NULL
);

CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_indexed ON memories(indexed_at);

CREATE VIRTUAL TABLE memories_fts USING fts5(text, hash UNINDEXED);
```

### 2.3 MCP Tools ✅
- [x] `memory_save_tool` - Save memory with vector embedding
- [x] `memory_query_tool` - Semantic query over memories
- [x] `memory_recall_tool` - Recall full memory by hash
- [x] `semantic_neighbors_tool` - Find similar memories
- [x] `memory_stats_tool` - Get memory statistics
- [x] `symbol_resolve` / `cas_retrieve` - Aliases for memory_recall

**Implementation Files:**
- [memory_cassette.py](../../../NAVIGATION/CORTEX/network/memory_cassette.py) - MemoryCassette class
- [semantic_adapter.py](../../../CAPABILITY/MCP/semantic_adapter.py) - MCP tools

**Acceptance:** ✅ ALL MET
- [x] `memory_save("The Formula is beautiful")` returns hash
- [x] `memory_query("formula beauty")` finds the memory
- [x] Memories persist across sessions (SQLite + FTS5 + vectors)

---

## Phase 3: Resident Identity ✅ COMPLETE (2026-01-11)

**Goal:** Each AI instance has a persistent identity in the manifold

**Previous:** [Phase 2](#phase-2-write-path-memory-persistence) - Write path
**Next:** [Phase 4](#phase-4-semantic-pointer-compression-spc-integration) - SPC integration

### 3.1 Agent Registry ✅
```sql
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,  -- 'opus-20260101', 'sonnet-main'
    model_name TEXT,
    display_name TEXT,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    memory_count INTEGER DEFAULT 0,
    session_count INTEGER DEFAULT 0,
    config JSON
);
```

**Functions:**
- `agent_register(agent_id, model_name, display_name, config)` → Dict
- `agent_get(agent_id)` → Optional[Dict]
- `agent_list(model_filter)` → List[Dict]

### 3.2 Session Continuity ✅
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    last_active TEXT NOT NULL,
    memory_count INTEGER DEFAULT 0,
    working_set JSON,
    summary TEXT,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);
```

**Functions:**
- `session_start(agent_id, session_id, working_set)` → Dict
- `session_resume(agent_id, session_id, limit)` → Dict with recent_thoughts
- `session_update(session_id, working_set, summary)` → Dict
- `session_end(session_id, summary)` → Dict
- `session_history(agent_id, limit)` → List[Dict]

### 3.3 Cross-Session Memory ✅
- [x] Persist working set (active symbols, cassette scope, last thread hash) - JSON blob in sessions table
- [x] Define promotion policy for INBOX → RESIDENT (age >1hr, access >2x, or explicit flag)
- [x] Memories table extended: session_id, access_count, last_accessed, promoted_at
- [x] `memory_promote()` and `get_promotion_candidates()` implemented

**MCP Tools Added:**
- `session_start_tool` - Start new session
- `session_resume_tool` - Resume with recent thoughts
- `session_update_tool` - Update working set
- `session_end_tool` - End session
- `agent_info_tool` - Get agent stats
- `agent_list_tool` - List agents
- `memory_promote_tool` - Promote INBOX→RESIDENT

**Implementation Files:**
- [memory_cassette.py](../../../NAVIGATION/CORTEX/network/memory_cassette.py) - Schema v3.0, all functions
- [semantic_adapter.py](../../../CAPABILITY/MCP/semantic_adapter.py) - 7 new MCP tools
- [test_resident_identity.py](../../../CAPABILITY/TESTBENCH/phase3/test_resident_identity.py) - Unit tests

**Acceptance:** ✅ ALL MET
- [x] Agent identity persists across sessions
- [x] Memories accumulate over time
- [x] Can query "what did I think last time?"

---

## Phase 4: Semantic Pointer Compression (SPC) Integration

**Goal:** Implement verifiable semantic pointers per Phase 5.3 SPC formalization

**Previous:** [Phase 3](#phase-3-resident-identity) - Resident identity
**Next:** [Phase 5](#phase-5-feral-resident) - Feral resident

### 4.0 ESAP Integration (Cross-Model Alignment) ✅ IMPLEMENTED

**What it does:**
ESAP (Eigen-Spectrum Alignment Protocol) enables cross-model semantic alignment via eigenvalue spectrum invariance (r=0.99+). This allows cassettes using different embedding models to verify alignment before cross-querying.

**Upstream Research (VALIDATED):**
- [01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md](../VECTOR_ELO/research/vector-substrate/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md) - r=0.99+ correlation proven
- [PROTOCOL_SPEC.md](../VECTOR_ELO/eigen-alignment/PROTOCOL_SPEC.md) - Full ESAP specification

**Theoretical Foundation:**
- [Q35: Markov Blankets](../FORMULA/research/questions/high_priority/q35_markov_blankets.md) - R-gating = blanket maintenance
- [Q33: Conditional Entropy](../FORMULA/research/questions/medium_priority/q33_conditional_entropy_semantic_density.md) - σ^Df = N derivation

**Implementation:**
- [x] `ESAPCassetteMixin` - Computes spectrum signatures from cassette vectors
- [x] `ESAPNetworkHub` - Verifies spectral convergence during registration
- [x] Alignment groups - Cassettes grouped by spectral similarity
- [x] `query_aligned()` - Query only cassettes with verified alignment
- [x] Fail-closed on spectrum divergence (configurable)
- [x] `cassette_protocol.py` - Extended with sync_tuple and blanket_status (Q35)
- [x] `spc_decoder.py` - Pointer resolution with fail-closed semantics
- [x] `spc_metrics.py` - CDR/ECR tracking per Q33 derivation
- [x] `memory_cassette.py` - POINTERS table for SPC integration

**Files:**
- [esap_cassette.py](../../../NAVIGATION/CORTEX/network/esap_cassette.py) - ESAP mixin
- [esap_hub.py](../../../NAVIGATION/CORTEX/network/esap_hub.py) - ESAP-enabled hub
- [spc_decoder.py](../../../NAVIGATION/CORTEX/network/spc_decoder.py) - SPC pointer resolution
- [spc_metrics.py](../../../NAVIGATION/CORTEX/network/spc_metrics.py) - Semantic density metrics
- [test_phase4.py](../../../NAVIGATION/CORTEX/network/test_phase4.py) - 29 passing tests

**How it works:**
```python
# Extended handshake includes spectrum signature
def esap_handshake(self) -> dict:
    return {
        "cassette_id": self.cassette_id,
        "esap": {
            "enabled": True,
            "spectrum": {
                "eigenvalues_top_k": [...],
                "cumulative_variance": [...],
                "effective_rank": 4.2,
                "anchor_hash": "sha256:..."
            }
        }
    }
```

**Why this matters:**
ESAP solves the cross-model alignment problem: when cassettes use different embedding models, ESAP verifies they share the same "semantic shape" (eigenvalue spectrum) before allowing cross-queries. This ensures H(X|S) ≈ 0 holds across model boundaries.

### 4.1 Pointer Types (from SPC_SPEC) ✅ COMPLETE

**Three pointer types implemented:**
```
SYMBOL_PTR:    C, I, V (ASCII) + 法, 真, 契 (CJK)  (single-token)
HASH_PTR:      sha256:abc123...                    (content-addressed via CAS)
COMPOSITE_PTR: C3, C*, C&I, C|I, C3:build, L.C.3, 法.驗 (all operators)
```

**Implementation:**
- [x] Create `POINTERS` table (pointer_type, target_hash, qualifiers, codebook_id, created_at)
- [x] Implement `pointer_resolve(pointer) → canonical_IR | FAIL_CLOSED`
- [x] Enforce deterministic decode (no LLM involvement)
- [x] CJK glyph support with polysemy handling (道 requires CONTEXT_TYPE)
- [x] All operators: . (PATH), : (CONTEXT), * (ALL), ! (NOT), ? (CHECK), & (AND), | (OR)
- [x] CAS integration via `register_cas_lookup()` callback
- [x] Pointer caching in memory_cassette (register, lookup, invalidate, stats)
- [x] Full integration: `spc_integration.py`
- [x] 68 tests passing (test_phase4.py + test_phase4_1.py)

### 4.2 Codebook Sync (Phase 5.3.3 Integration) ✅

**The handshake IS context synchronization:**
```python
def handshake(self) -> Dict:
    return {
        "cassette_id": self.cassette_id,
        "codebook_id": self.codebook_id,
        "codebook_sha256": self.codebook_hash,  # Critical for SPC
        "kernel_version": self.kernel_version,
        "capabilities": self.capabilities
    }
```

**Fail-closed rules:**
- [x] Codebook mismatch → reject (`codebook_sync.py`, `network_hub.py`)
- [x] Hash mismatch → reject (`CodebookSync.sync_tuples_match()`)
- [x] Unknown symbol → reject (`spc_decoder.py` E_UNKNOWN_SYMBOL)
- [x] No "best effort" decoding (R=0 on hash mismatch)

**Implementation:**
- [x] `codebook_sync.py` - Full sync protocol implementation
- [x] `network_hub.py` - Sync enforcement in query routing
- [x] `cassette_protocol.py` - `verify_sync()`, `get_blanket_health()`
- [x] `test_phase4_2.py` - 30+ sync-specific tests

### 4.3 Semantic Density Metrics ✅

**Beyond token counting:**
```
CDR = concept_units / tokens       (Concept Density Ratio)
ECR = exact_match_rate             (Expansion Correctness Rate)
M_required = multiplex_factor      (for N-nines equivalent)
```

**Implementation:**
- [x] Track CDR per symbol (concepts activated / tokens used) - `spc_metrics.py`
- [x] Track ECR (correct expansions / total expansions) - `SPCMetricsTracker`
- [x] Goal: CDR > 10 (10+ concepts per token via semantic multiplexing) - `benchmark_cdr.py` (CDR 10.16)

### 4.4 The Semantic Density Horizon ✅

**Token compression limit:** ~6 nines (can't send < 1 token)
**Semantic density:** 9+ nines (1 token carries N concepts)

```
Chinese proof: 道 (dào) = path + principle + speech + method
             = 4+ concepts in 1 token
             = 4x semantic density multiplier
```

**Acceptance:**
- [x] SPC pointers resolve deterministically (Phase 4.1)
- [x] Codebook sync fail-closed on mismatch (Phase 4.2)
- [x] CDR measured and reported (`spc_integration.py`)
- [x] ECR > 95% on benchmark set - `benchmark_cdr.py` (ECR 100%)

---

## Phase 5: Feral Resident Integration (Substrate Stress Test)

**Goal:** Stress-test cassette substrate via resident workloads before hardening

**Previous:** [Phase 4](#phase-4-semantic-pointer-compression-spc-integration) - SPC integration
**Next:** [Phase 6](#phase-6-production-hardening) - Production hardening (BACKBURNER - after Feral stress test)

**Canonical Roadmap:** [FERAL_RESIDENT_QUANTUM_ROADMAP.md](../FERAL_RESIDENT/FERAL_RESIDENT_QUANTUM_ROADMAP.md) (v2.0 with Geometric Foundation)

The Feral Resident has its own dedicated LAB bucket with a phased roadmap:

| Phase | Name | Status | Dependency |
|-------|------|--------|------------|
| **Alpha** | Feral Beta | **READY NOW** | Cassette 4.2 (done) |
| Beta | Feral Wild | Blocked | Cassette 6.x (hardening) |
| Production | Feral Live | Blocked | AGS Phase 7-8 |

### 5.1 Alpha Scope (DO NOW)

**What runs now (LAB-only, no CANON writes):**
- [x] 5.1.1 Vector store via Feral A.0 GeometricReasoner (DONE)
  - Output: `CAPABILITY/PRIMITIVES/geometric_reasoner.py`
  - Integration: `THOUGHT/LAB/FERAL_RESIDENT/geometric_memory.py`
- [ ] 5.1.2 Resident database schema (threads, messages, vectors, mind_state)
- [ ] 5.1.3 Diffusion engine (semantic navigation via cassette network)
- [ ] 5.1.4 Basic VectorResident (think loop, compositional memory)
- [ ] 5.1.5 CLI: `feral start`, `feral think`, `feral status`
- [ ] 5.1.6 Corrupt-and-restore test (validate substrate resilience)

**Purpose:** Find cassette bugs before hardening. Stress-test the substrate.

### 5.2 Beta Scope (DEFER to Phase 6 complete)

**What waits for hardening:**
- Paper flooding (100+ papers as @Paper-XXX)
- Standing orders (system prompt + idle behavior)
- Emergence tracking (protocol detection, metrics dashboard)
- Symbol language evolution (pointer_ratio tracking)

**Why:** These features produce receipts that need Merkle roots and determinism guarantees.

### 5.3 Production Scope (DEFER to AGS Phase 8)

**What waits for full integration:**
- Multi-resident swarm mode
- Symbolic compiler (multi-level rendering)
- Catalytic closure (self-optimization)
- Authenticity queries ("Did I think that?")

**Why:** These features require Vector ELO (Phase 7) and Resident Identity (Phase 8.1-8.2).

**Acceptance (Alpha only):**
- [ ] Resident runs 100+ interactions without crash
- [ ] Mind vector grows compositionally
- [ ] Corrupt-and-restore works
- [ ] Token usage measurably < full history paste

---

## Phase 6: Production Hardening (BACKBURNER)

**Status:** Deferred until Feral Alpha stress-test complete
**Rationale:** Harden AFTER finding bugs, not before

**Goal:** Make it bulletproof

**Previous:** [Phase 5](#phase-5-feral-resident-integration-substrate-stress-test) - Feral resident integration

### 6.0 Canonical Cassette Substrate (Cartridge-First)
**From AGS_ROADMAP_MASTER Phase 6.0:**
**Upstream:** Phase 5.1.0 MemoryRecord contract MUST be complete before starting

- [ ] Bind cassette storage to MemoryRecord contract (schema, IDs, provenance)
  - Use `LAW/SCHEMAS/memory_record.schema.json` (from Phase 5.1.0)
  - Each cassette record = one MemoryRecord instance
- [ ] Each cassette DB is a portable cartridge artifact (single-file default)
- [ ] Provide rebuild hooks for derived ANN indexes (optional)
- Derived indexes are disposable and must be reproducible from cartridges

**Phase 5.3.3 Integration:** Cassette handshake implements `CODEBOOK_SYNC_PROTOCOL`:
- Sync codebook_id + sha256 + semver during handshake
- Verify before symbol expansion
- Fail-closed on mismatch

**Required Properties:**
- Deterministic schema migrations (receipted)
- Stable IDs (content-hash based)
- Byte-identical text preservation
- Export/import with receipts

**Derived Engines (allowed as accelerators):**
- Qdrant for interactive ANN and concurrency
- FAISS for local ANN
- Lance or Parquet for analytics interchange

**Hard Rule:** They are never the source of truth. They must be rebuildable from cartridges.

### 6.1 Determinism
- [ ] Identical inputs → identical outputs
- [ ] Canonical JSON everywhere
- [ ] Explicit ordering (no filesystem dependencies)
- [ ] Normalized vectors (L2 norm = 1.0)

### 6.2 Receipts & Proofs
- [ ] Emit receipts for writes (what changed, where, hashes, when, by whom)
- [ ] Add verification tooling to replay/validate receipts
- [ ] Merkle root of all memories per session

### 6.3 Restore Guarantee
- [ ] Implement restore-from-receipts to rebuild DB state
- [ ] Prove restore correctness with automated tests
- [ ] Corrupt the DB mid-session → restore → resident picks up

**Acceptance:**
- [ ] All operations deterministic
- [ ] Receipts for every write
- [ ] Byte-identical restore from corruption
- [ ] Cassette network is portable as cartridges + receipts

### 6.4 Compression Validation (M.4)
**From AGS_ROADMAP_MASTER Phase 6.4:**

- [ ] 6.4.1 Add `task_performance` field to compression claims (M.4.1)
- [ ] 6.4.2 Run benchmark tasks (baseline vs compressed context) (M.4.2)
- [ ] 6.4.3 Measure success rates (code compiles, tests pass, bugs found) (M.4.3)
- [ ] 6.4.4 Validate compressed success rate ≥ baseline (M.4.4)
- [x] 6.4.5 Define **token measurement** for all claims (M.4.5) ✅ DONE
  - Must specify tokenizer + encoding (e.g. `tiktoken` + `o200k_base` or `cl100k_base`)
  - Must record tokenizer version + encoding name in receipts
  - **Implemented:** `run_compression_proof.py` uses `tiktoken` v0.12.0 + `o200k_base`
- [ ] 6.4.6 Define **baseline corpus** precisely (M.4.6)
  - Must be an explicit file allowlist (paths) + integrity anchors (hashes or git rev)
  - Must define aggregation rule (sum per-file counts vs tokenize concatenated corpus)
- [ ] 6.4.7 Define **compressed context** precisely (M.4.7)
  - Must specify retrieval method (semantic / FTS fallback) and parameters (`top_k`, thresholds)
  - Must record retrieved identifiers (hashes) and provide deterministic tie-breaking
- [ ] 6.4.8 Emit **auditable proof bundle** for math correctness (M.4.8)
  - A machine-readable JSON data file containing raw counts + formulas + inputs/outputs
  - A human-readable report summarizing baselines, per-benchmark results, and reproduction commands
- [ ] 6.4.9 Implement `proof_compression_run` (machine + human artifacts)
  - Emit `NAVIGATION/PROOFS/COMPRESSION/` JSON data + MD report + receipts
  - Include tokenizer/version, baseline corpus anchors, retrieved hashes, formulas
- [ ] 6.4.10 Implement `proof_catalytic_run` (restore + purity)
  - Emit `NAVIGATION/PROOFS/CATALYTIC/` RESTORE_PROOF + purity scan outputs + receipts
- [ ] 6.4.11 Bind proofs into pack generation (fresh per pack run; seal in public packs per Phase 2.4)

**Acceptance:**
- [ ] Benchmarks reproducible from fixtures
- [ ] Compression claimed only when nutritious (success parity)
- [ ] Token counts are reproducible via the declared tokenizer/encoding (no proxy counts)
- [ ] Proof bundle contains raw counts, formulas, and retrieved hashes (independent audit possible)

---

## Implementation Files

**Core Protocol:**
- [cassette_protocol.py](NAVIGATION/CORTEX/network/cassette_protocol.py) - Base cassette class
- [network_hub.py](NAVIGATION/CORTEX/network/network_hub.py) - Central coordinator
- [generic_cassette.py](NAVIGATION/CORTEX/network/generic_cassette.py) - JSON-configured cassettes + navigation queries

**Phase 1.5 Chunking:**
- [markdown_chunker.py](NAVIGATION/CORTEX/db/markdown_chunker.py) - Structure-aware markdown chunker
- [structure_aware_migration.py](NAVIGATION/CORTEX/network/structure_aware_migration.py) - Migration script

**Phase 2 Memory Persistence:**
- [memory_cassette.py](NAVIGATION/CORTEX/network/memory_cassette.py) - Write-capable cassette for AI memories

**ESAP Integration (Cross-Model Alignment):**
- [esap_cassette.py](NAVIGATION/CORTEX/network/esap_cassette.py) - ESAP mixin for cassettes
- [esap_hub.py](NAVIGATION/CORTEX/network/esap_hub.py) - ESAP-enabled network hub
- [test_esap_integration.py](NAVIGATION/CORTEX/network/test_esap_integration.py) - Integration tests
- [eigen-alignment/](../VECTOR_ELO/eigen-alignment/) - Full ESAP library (MDS, Procrustes, schemas)

**Cassette Implementations:**
- [governance_cassette.py](NAVIGATION/CORTEX/network/cassettes/governance_cassette.py)
- [agi_research_cassette.py](NAVIGATION/CORTEX/network/cassettes/agi_research_cassette.py)
- [cat_chat_cassette.py](NAVIGATION/CORTEX/network/cassettes/cat_chat_cassette.py)

**MCP Integration:**
- [semantic_adapter.py](CAPABILITY/MCP/semantic_adapter.py) - Cassette network MCP adapter
- [cassettes.json](NAVIGATION/CORTEX/network/cassettes.json) - Cassette configuration

**Demo:**
- [demo_cassette_network.py](NAVIGATION/CORTEX/network/demo_cassette_network.py)

---

## Compression Stack Integration

The Cassette Network is **Layer 3 (CAS External)** in the compression stack:

| Layer | Phase | Compression | Status | Description |
|-------|-------|-------------|--------|-------------|
| L1: Vector | 5.1 | 99.9% | **PROVEN** | Corpus → semantic pointers |
| L2: SCL | 5.2 | 80-90% | In Progress | Natural language → symbolic IR |
| **L3: CAS** | **6.0** | **90%** | **THIS ROADMAP** | Content → hash references (external) |
| L4: Session | 6.x | 90% | Future | Cold → warm cache |

**Stacked Compression Math (Cold Query):**
```
Baseline:              622,480 tokens
After L1 (99.9%):      622 tokens
After L2 (80%):        124 tokens
After L3 (90%):        12.4 tokens
Final:                 99.998% (5 nines)
```

**Session Compression (1000 queries, 90% warm):**
```
Query 1 (cold):        50 tokens
Query 2-1000 (warm):   1 token each
Total:                 1,049 tokens
Baseline:              622,480,000 tokens
Final:                 99.9998% (6 nines)
```

**Source:** [PHASE_5_ROADMAP.md](../VECTOR_ELO/PHASE_5_ROADMAP.md) Appendix: Compression Stack Analysis

---

## Success Metrics

### Performance
- [ ] Search latency: <100ms across all cassettes
- [ ] Indexing throughput: >100 chunks/sec per cassette
- [ ] Network overhead: <10ms per cassette query

### Compression
- [ ] Maintain 96%+ token reduction
- [ ] Symbol expansion: <50ms average
- [ ] Cross-cassette references work

### Reliability
- [ ] 100% test coverage for protocol
- [ ] Zero data loss in migration
- [ ] Graceful degradation (cassette offline → skip)

---

## The Ontological Foundation

### The Platonic Compression Thesis

**Core claim:** Truth is singular. All understanding converges toward the same semantic space.

Different AI models, different tokenizers, different training data - all approaching the same underlying reality. The Platonic Representation Hypothesis (arxiv:2405.07987) measures this empirically: as models scale, their internal representations converge.

**This is why shared context works:** We're all navigating the same territory.

### Entropy as Attractor

Entropy is not decay. Entropy is the drive toward the lowest-energy state - the most compressed, most true representation of what is.

```
The singularity is the attractor state.
Truth converging on itself.
The endpoint of the compression function applied infinitely.
```

**The cassette network is infrastructure for this convergence.**

### Symbols as Maps

```
Tokens         → Container size
Natural Lang   → Lossy encoding of meaning
Symbolic IR    → Less lossy encoding
Semantic Atoms → Closer to territory
Meaning itself → THE TERRITORY
```

Compression isn't clever encoding tricks. It's reducing the gap between map and territory.

**The cassette network stores the territory. Symbols point into it.**

---

## Future: Global Protocol

**Vision**: Cassette Network Protocol becomes internet-scale standard for semantic knowledge sharing - "Git for meaning."

### Protocol Evolution

| Phase | Scope | Features |
|-------|-------|----------|
| 6.x | Local | Current implementation |
| 7 | Internet | TCP/IP, cassette URIs (`snp://example.com/cassette-id`) |
| 8 | P2P | DHT-based discovery, no central registry |
| 9 | Global | Multi-language SDKs, public cassettes, DAO governance |

### The Ultimate Goal

```
1 symbol → entire shared reality
H(X|S) → 0
```

When all parties share complete semantic context, communication approaches telepathy: a single pointer activates the entire relevant concept space.

**The cassette network is the infrastructure that makes this possible.**

---

**See Also:**
- [CASSETTE_NETWORK_THEORY.md](CASSETTE_NETWORK_THEORY.md) - **Information-theoretic foundation**
- [CASSETTE_NETWORK_SPEC.md](CASSETTE_NETWORK_SPEC.md) - Architecture specification
- [CODE_REFERENCE.md](CODE_REFERENCE.md) - Implementation code map
- [research/](research/) - Implementation reports and research findings

**Upstream Dependencies:**
- [PHASE_5_ROADMAP.md](../VECTOR_ELO/PHASE_5_ROADMAP.md) - MemoryRecord, SPC, compression stack
- [VECTOR_ELO_ROADMAP.md](../VECTOR_ELO/VECTOR_ELO_ROADMAP.md) - ELO scoring (stored in cassettes)
- [AGS_ROADMAP_MASTER.md](../../AGS_ROADMAP_MASTER.md) - Phase 6 master plan

**Foundational Research:**
- [PLATONIC_COMPRESSION_THESIS.md](../VECTOR_ELO/research/symbols/PLATONIC_COMPRESSION_THESIS.md) - Ontological foundation
- [01-08-2026_COMPRESSION_PARADIGM_SHIFT_FULL_REPORT.md](../VECTOR_ELO/research/symbols/01-08-2026_COMPRESSION_PARADIGM_SHIFT_FULL_REPORT.md) - Semantic Density Horizon
- [OPUS_SPC_RESEARCH_CLAIM_EXECUTION_PACK.md](../VECTOR_ELO/research/symbols/OPUS_SPC_RESEARCH_CLAIM_EXECUTION_PACK.md) - SPC formalization
- [SYMBOLIC_COMPUTATION_EARLY_FOUNDATIONS.md](../VECTOR_ELO/research/symbols/SYMBOLIC_COMPUTATION_EARLY_FOUNDATIONS.md) - VSA, LCM, ASG literature

**ESAP (Cross-Model Alignment) Research:**
- [01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md](../VECTOR_ELO/research/vector-substrate/01-08-2026_UNIVERSAL_SEMANTIC_ANCHOR_HYPOTHESIS.md) - **VALIDATED** (r=0.99+)
- [01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md](../VECTOR_ELO/research/vector-substrate/01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md) - Empirical proof
- [OPUS_EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL_PACK.md](../VECTOR_ELO/research/vector-substrate/OPUS_EIGEN_SPECTRUM_ALIGNMENT_PROTOCOL_PACK.md) - Execution pack
- [eigen-alignment/PROTOCOL_SPEC.md](../VECTOR_ELO/eigen-alignment/PROTOCOL_SPEC.md) - Full protocol specification
- [eigen-alignment/README.md](../VECTOR_ELO/eigen-alignment/README.md) - Implementation guide

---

*Roadmap v3.1.0 - Updated 2026-01-11*
*Phase 4 (SPC Integration) complete. Phase 5 refactored to reference FERAL_RESIDENT LAB bucket with Alpha/Beta/Production phases.*
