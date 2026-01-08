# Cassette Network Roadmap

**Status**: Phase 0 Complete, Phase 1-6 In Progress
**Vision**: Modular database network where specialized databases (cassettes) plug into a semantic network for cross-database queries and token compression
**Owner**: Antigravity / Resident

---

## Executive Summary

Refactor from **monolithic database** approach to **database cassette network** architecture - a federated "Semantic Manifold" where residents live, navigate, and persist identity.

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

## Phase 4: Symbol Language Evolution

**Goal:** Let residents develop compressed communication

### 4.1 Symbol Registry Integration
- [ ] Create `SYMBOLS` table (symbol_id, target_type, target_ref/hash, default_slice, created_at)
- [ ] Implement `symbol_resolve(symbol_id)`
- [ ] Enforce invariants (symbol_id starts with `@`, validate target_ref, reject `ALL` slice)

### 4.2 Bounded Expansion
- [ ] Implement bounded symbol expansion (caps on dereference depth + total tokens)
- [ ] Enforce "bounded artifacts only" rule for any bundle/context injection

### 4.3 Compression Metrics
- [ ] Track symbol compression ratio (bytes saved vs raw text)
- [ ] Track retrieval accuracy vs compression tradeoff
- [ ] Goal: After 100 sessions, 90% of resident output is symbols/hashes

**Acceptance:**
- [ ] Residents can create and use symbols
- [ ] Symbol expansion is bounded
- [ ] Compression ratio improves over time

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
- [ ] Bind cassette storage to MemoryRecord contract (schema, IDs, provenance)
- [ ] Each cassette DB is a portable cartridge artifact (single-file default)
- [ ] Provide rebuild hooks for derived ANN indexes (optional)
- Derived indexes are disposable and must be reproducible from cartridges

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

## Future: Global Protocol

**Vision**: Cassette Network Protocol becomes internet-scale standard for semantic knowledge sharing.

### Protocol Evolution
- **Phase 7**: Internet Protocol (TCP/IP, cassette URIs: `snp://example.com/cassette-id`)
- **Phase 8**: P2P Discovery (DHT-based, no central registry)
- **Phase 9**: Global Network (Multi-language SDKs, public cassettes, DAO governance)

---

**See Also:**
- [CASSETTE_NETWORK_SPEC.md](CASSETTE_NETWORK_SPEC.md) - Architecture specification
- [research/](research/) - Implementation reports and research findings
