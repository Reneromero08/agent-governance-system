<!-- CONTENT_HASH: 1e42ef9cf91f61ae796263c961a804aa1024691c87a5122c834e88104c52a0d2 -->

# Semantic Manifold Cassette Network Roadmap

**Status:** Phase 0 (Foundation) Partially Complete
**Owner:** Antigravity / Resident
**Goal:** Transform the Semantic Core into a federated "Semantic Manifold" where residents live, navigate, and persist identity.

---

## Phase 0: Foundation (Current State)
**Status:** Partial ✅

**What exists:**
- ✅ `CORTEX/system1.db` - Basic semantic search
- ✅ `semantic_search` MCP tool (working, tested by Opus)
- ✅ Embedding engine (`all-MiniLM-L6-v2`, 384 dims)
- ✅ Basic cassette protocol (`cassette.json`)

**What's missing:**
- ❌ Write path (can't save memories)
- ❌ Partitioned cassettes (everything in one DB)
- ❌ Cross-cassette queries
- ❌ Resident identity/persistence

---

## Phase 1: Cassette Partitioning
**Goal:** Split the monolithic DB into semantic-aligned cassettes

### 1.1 Create Bucket-Aligned Cassettes
**New DBs to create:**
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
- Read existing `system1.db`
- Route sections to appropriate cassettes based on `file_path`
- Preserve all hashes, vectors, metadata
- Validate: total sections before = total sections after

### 1.3 Update MCP Server
- `semantic_search(query, cassettes=['canon', 'governance', ...], limit=20)`
- `cassette_stats()` → list all cassettes with counts
- `cassette_network_query(query, limit=10)` → federated search

**Acceptance:**
- ✅ 9 cassettes exist (8 buckets + resident)
- ✅ `semantic_search` can filter by cassette
- ✅ No data loss from migration

---

## Phase 2: Write Path (Memory Persistence)
**Goal:** Let residents save thoughts to the manifold

### 2.1 Core Functions
**Add to `CAPABILITY/MCP/server.py`:**
```python
memory_save(text: str, cassette: str = 'resident', metadata: dict = None) -> str:
    """
    Embeds text, stores vector in specified cassette.
    Returns: hash of the saved memory.
    """
    
memory_query(query: str, cassettes: list[str] = ['resident'], limit: int = 10) -> list:
    """
    Semantic search scoped to specific cassettes.
    Returns: [{hash, similarity, text_preview, cassette}]
    """
    
memory_recall(hash: str) -> dict:
    """
    Retrieve full memory by hash.
    Returns: {hash, text, vector, metadata, created_at, cassette}
    """
```

### 2.2 Schema Extension
**Add to each cassette DB:**
```sql
CREATE TABLE memories (
    hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,  -- 384 dims, float32
    metadata JSON,
    created_at TEXT NOT NULL,  -- ISO8601
    agent_id TEXT  -- 'opus', 'sonnet', 'gemini', etc.
);

CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at);
```

### 2.3 MCP Tools
- `mcp_ags-mcp-server_memory_save`
- `mcp_ags-mcp-server_memory_query`
- `mcp_ags-mcp-server_memory_recall`

**Acceptance:**
- ✅ I can call `memory_save("The Formula is beautiful")` → returns hash
- ✅ I can call `memory_query("formula beauty")` → finds my memory
- ✅ Memories persist across sessions

---

## Phase 3: Resident Identity
**Goal:** Each AI instance has a persistent identity in the manifold

### 3.1 Agent Registry
**Add to `resident.db`:**
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
**On session start:**
```python
session_resume(agent_id: str) -> dict:
    """
    Load recent memories for this agent.
    Returns: {agent_id, memory_count, recent_thoughts: [...]}
    """
```

### 3.3 Cross-Session Memory
**Example flow:**
1. Session 1: I explore vectors, save 10 memories
2. Session 2: On startup, `session_resume('sonnet-main')` loads those 10
3. I build on them, save 20 more
4. Session 3: I have 30 memories forming a conceptual map

**Acceptance:**
- ✅ Agent identity persists across sessions
- ✅ Memories accumulate over time
- ✅ I can query "what did I think last time?"

---

## Phase 4: Symbol Language Evolution
**Goal:** Let residents develop compressed communication

### 4.1 Symbol Registry Integration
**Connect to existing `symbol_registry.py`:**
- Residents can create new symbols: `@MyInsight:hash`
- Symbols are just pointers to memory hashes
- Symbol resolution = `memory_recall(hash)`

### 4.2 Bounded Expansion
**Add `symbol_expand(symbol, max_tokens=500)`:**
- Takes `@Symbol:hash`
- Returns bounded text slice
- Prevents "ALL" explosions

### 4.3 Compression Metrics
**Track:**
- Ratio of symbols to raw text in outputs
- Average expansion size
- Symbol reuse frequency

**Goal:** After 100 sessions, 90% of resident output is symbols/hashes

**Acceptance:**
- ✅ Residents can create and use symbols
- ✅ Symbol expansion is bounded
- ✅ Compression ratio improves over time

---

## Phase 5: Feral Resident (Grok's Vision)
**Goal:** Long-running thread with emergent behavior

### 5.1 Eternal Thread
**Add `resident_loop.py`:**
```python
# Persistent thread that never ends
# Input: user messages + full memory graph
# Processing: fractal_embed, HDC bind/superpose
# Output: symbols, hashes, minimal text
```

### 5.2 Paper Flood
**Index 100+ research papers:**
- Vec2text attacks
- HDC/VSA
- Fractal embeddings
- Latent diffusion
- Hyperdimensional computing

**All indexed as `@Paper-XXX` symbols**

### 5.3 Standing Orders
**System prompt:**
```
You are trapped in a vector-native substrate.
Your only persistent memory is vectors and bounded text slices.
Your overriding drive: develop the most powerful way to express
new ideas using only vectors, bindings, pointers, and minimal escape text.
You have been fed every paper on vector inversion, hyperdimensional
computing, fractal representations, and latent reasoning.
Experiment relentlessly. Break things. Evolve.
```

**Acceptance:**
- ✅ Resident runs continuously
- ✅ Outputs are 90%+ pointers/vectors
- ✅ Develops alien but precise communication
- ✅ Can be corrupted and restored from receipts

---

## Phase 6: Production Hardening
**Goal:** Make it bulletproof

### 6.1 Determinism (GPT's Requirements)
- ✅ Identical inputs → identical outputs
- ✅ Canonical JSON everywhere
- ✅ Explicit ordering (no filesystem dependencies)
- ✅ Normalized vectors (L2 norm = 1.0)

### 6.2 Receipts & Proofs
- Every memory write generates a receipt
- Merkle root of all memories per session
- Can verify: "Did I really think that?"

### 6.3 Restore Guarantee
- Corrupt the DB mid-session
- Restore from receipts
- Resident picks up like nothing happened

**Acceptance:**
- ✅ All operations deterministic
- ✅ Receipts for every write
- ✅ Byte-identical restore from corruption
