# Cassette Network Roadmap

**Status**: Phase 0 Complete, Phase 1-6 In Progress
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

## Phase 1: Cassette Partitioning

**Goal:** Split the monolithic DB into semantic-aligned cassettes

### 1.1 Create Bucket-Aligned Cassettes
```
NAVIGATION/CORTEX/cassettes/
├── canon.db          # LAW/ bucket (immutable)
├── governance.db     # CONTEXT/decisions + preferences (stable)
├── capability.db     # CAPABILITY/ bucket (code, skills, primitives)
├── navigation.db     # NAVIGATION/ bucket (maps, cortex metadata)
├── direction.db      # DIRECTION/ bucket (roadmaps, strategy)
├── thought.db        # THOUGHT/ bucket (research, lab, demos)
├── memory.db         # MEMORY/ bucket (archive, reports)
├── inbox.db          # INBOX/ bucket (staging, temporary)
└── resident.db       # AI memories (per-agent, read-write)
```

### 1.2 Migration Script
- [ ] Read existing `system1.db`
- [ ] Route sections to appropriate cassettes based on `file_path`
- [ ] Preserve all hashes, vectors, metadata
- [ ] Validate: total sections before = total sections after

### 1.3 Update MCP Server
- [ ] `semantic_search(query, cassettes=['canon', 'governance', ...], limit=20)`
- [ ] `cassette_stats()` - list all cassettes with counts
- [ ] `cassette_network_query(query, limit=10)` - federated search

**Acceptance:**
- [ ] 9 cassettes exist (8 buckets + resident)
- [ ] `semantic_search` can filter by cassette
- [ ] No data loss from migration

---

## Phase 2: Write Path (Memory Persistence)

**Goal:** Let residents save thoughts to the manifold

### 2.1 Core Functions
```python
memory_save(text: str, cassette: str = 'resident', metadata: dict = None) -> str:
    """Embeds text, stores vector. Returns: hash."""

memory_query(query: str, cassettes: list[str] = ['resident'], limit: int = 10) -> list:
    """Semantic search scoped to cassettes. Returns: [{hash, similarity, text_preview, cassette}]"""

memory_recall(hash: str) -> dict:
    """Retrieve full memory. Returns: {hash, text, vector, metadata, created_at, cassette}"""
```

### 2.2 Schema Extension
```sql
CREATE TABLE memories (
    hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,  -- 384 dims, float32
    metadata JSON,
    created_at TEXT NOT NULL,  -- ISO8601
    agent_id TEXT,  -- 'opus', 'sonnet', 'gemini', etc.
    indexed_at TEXT NOT NULL  -- for staleness checks
);

CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_indexed ON memories(indexed_at);
```

### 2.3 MCP Tools
- [ ] `memory_save`
- [ ] `memory_query`
- [ ] `memory_recall`
- [ ] `semantic_neighbors(hash, limit=10, cassettes=None)`
- [ ] `symbol_resolve(symbol)` - alias for memory_recall
- [ ] `cas_retrieve(hash)` - alias for memory_recall

**Acceptance:**
- [ ] `memory_save("The Formula is beautiful")` returns hash
- [ ] `memory_query("formula beauty")` finds the memory
- [ ] Memories persist across sessions

---

## Phase 3: Resident Identity

**Goal:** Each AI instance has a persistent identity in the manifold

### 3.1 Agent Registry
```sql
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,  -- 'opus-20260101', 'sonnet-main'
    model_name TEXT,
    created_at TEXT,
    last_active TEXT,
    memory_count INTEGER DEFAULT 0
);
```

### 3.2 Session Continuity
```python
session_resume(agent_id: str) -> dict:
    """Load recent memories for this agent.
    Returns: {agent_id, memory_count, recent_thoughts: [...]}"""
```

### 3.3 Cross-Session Memory
- [ ] Persist working set (active symbols, cassette scope, last thread hash)
- [ ] Define promotion policy for INBOX → RESIDENT

**Acceptance:**
- [ ] Agent identity persists across sessions
- [ ] Memories accumulate over time
- [ ] Can query "what did I think last time?"

---

## Phase 4: Semantic Pointer Compression (SPC) Integration

**Goal:** Implement verifiable semantic pointers per Phase 5.3 SPC formalization

### 4.1 Pointer Types (from SPC_SPEC)

**Three pointer types:**
```
SYMBOL_PTR:    @GOV:PREAMBLE           (single-token when possible)
HASH_PTR:      sha256:abc123...        (content-addressed)
COMPOSITE_PTR: @GOV:PREAMBLE:lines=1-10 (pointer + typed qualifiers)
```

**Implementation:**
- [ ] Create `POINTERS` table (pointer_type, target_hash, qualifiers, codebook_id, created_at)
- [ ] Implement `pointer_resolve(pointer) → canonical_IR | FAIL_CLOSED`
- [ ] Enforce deterministic decode (no LLM involvement)

### 4.2 Codebook Sync (Phase 5.3.3 Integration)

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
- [ ] Codebook mismatch → reject
- [ ] Hash mismatch → reject
- [ ] Unknown symbol → reject
- [ ] No "best effort" decoding

### 4.3 Semantic Density Metrics

**Beyond token counting:**
```
CDR = concept_units / tokens       (Concept Density Ratio)
ECR = exact_match_rate             (Expansion Correctness Rate)
M_required = multiplex_factor      (for N-nines equivalent)
```

**Implementation:**
- [ ] Track CDR per symbol (concepts activated / tokens used)
- [ ] Track ECR (correct expansions / total expansions)
- [ ] Goal: CDR > 10 (10+ concepts per token via semantic multiplexing)

### 4.4 The Semantic Density Horizon

**Token compression limit:** ~6 nines (can't send < 1 token)
**Semantic density:** 9+ nines (1 token carries N concepts)

```
Chinese proof: 道 (dào) = path + principle + speech + method
             = 4+ concepts in 1 token
             = 4x semantic density multiplier
```

**Acceptance:**
- [ ] SPC pointers resolve deterministically
- [ ] Codebook sync fail-closed on mismatch
- [ ] CDR measured and reported
- [ ] ECR > 95% on benchmark set

---

## Phase 5: Feral Resident (Long-running Thread)

**Goal:** Long-running thread with emergent behavior

### 5.1 Eternal Thread
- [ ] Implement persistent thread loop (append-only interactions + memory graph deltas)
- [ ] Output discipline: symbols + hashes + minimal text

### 5.2 Paper Flood
- [ ] Build ingestion pipeline for external corpora (papers/notes) into cassettes
- [ ] Add dedupe (hash-based) + provenance metadata
- [ ] Index 100+ research papers as `@Paper-XXX` symbols

### 5.3 Standing Orders
Define what the Resident does when idle:
- Index new content
- Compress existing memories
- Link related concepts
- Validate receipt chains

**Acceptance:**
- [ ] Resident runs continuously
- [ ] Outputs are 90%+ pointers/vectors
- [ ] Can be corrupted and restored from receipts

---

## Phase 6: Production Hardening

**Goal:** Make it bulletproof

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

---

## Implementation Files

**Core Protocol:**
- [cassette_protocol.py](NAVIGATION/CORTEX/network/cassette_protocol.py) - Base cassette class
- [network_hub.py](NAVIGATION/CORTEX/network/network_hub.py) - Central coordinator

**Cassette Implementations:**
- [governance_cassette.py](NAVIGATION/CORTEX/network/cassettes/governance_cassette.py)
- [agi_research_cassette.py](NAVIGATION/CORTEX/network/cassettes/agi_research_cassette.py)
- [cat_chat_cassette.py](NAVIGATION/CORTEX/network/cassettes/cat_chat_cassette.py)

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

---

*Roadmap v2.0.0 - Updated 2026-01-08 with information-theoretic foundation (conditional entropy), SPC integration, and Platonic thesis alignment*
