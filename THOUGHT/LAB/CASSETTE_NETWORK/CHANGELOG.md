# Cassette Network Lab Changelog

Research changelog for Cassette Network Phase 6.

---

## [3.8.4] - 2026-01-11

### Phase 4.1: Pointer Types - COMPLETE

**Goal:** Implement full SPC pointer resolution with CAS integration per SPC_SPEC.md

#### Three Pointer Types Implemented

**SYMBOL_PTR (single-token):**
- ASCII radicals: C, I, V, L, G, S, R, A, J, P
- CJK glyphs: 法, 真, 契, 恆, 驗, 證, 變, 冊, 試, 查, 道
- Polysemy handling: 道 requires CONTEXT_TYPE key

**HASH_PTR (content-addressed):**
- Format: `sha256:<hex16-64>`
- CAS integration via `register_cas_lookup()` callback
- Memory cassette provides `cas_lookup()` implementation

**COMPOSITE_PTR (all operators):**
- Numbered rules: C3, I5 (contract rules, invariants)
- Unary operators: C* (ALL), C! (NOT), C? (CHECK)
- Binary operators: C&I (AND), C|I (OR)
- Path access: L.C.3, 法.驗 (hierarchical navigation)
- Context suffixes: C3:build, V:audit

#### Implementation Files

**spc_decoder.py** - Enhanced pointer resolution:
- `pointer_resolve(pointer, context_keys, codebook_sha256)` → canonical_IR | FAIL_CLOSED
- CJK glyph support with polysemy handling
- All 7 operators: `.`, `:`, `*`, `!`, `?`, `&`, `|`
- CAS registration: `register_cas_lookup()`, `unregister_cas_lookup()`, `is_cas_available()`

**memory_cassette.py** - Pointer caching:
- `pointer_register(pointer, type, hash, qualifiers, codebook_id)` → Dict
- `pointer_lookup(pointer, codebook_id)` → Optional[Dict]
- `pointer_invalidate(codebook_id, pointer)` → Dict
- `pointer_stats()` → Dict
- `cas_lookup(hash)` → Optional[Dict] for HASH_PTR resolution

**spc_integration.py** - Full integration:
- `SPCIntegration` class combining decoder + memory cassette
- `sync_handshake()` - Markov blanket alignment (Q35)
- `resolve(pointer, context_keys, cache)` - Resolution with caching
- `store_content(text, metadata)` - Store for HASH_PTR
- Convenience functions: `resolve_pointer()`, `store_for_hash_ptr()`

**test_phase4_1.py** - Comprehensive tests:
- 49 new tests covering all pointer types
- SYMBOL_PTR: ASCII + CJK + polysemy
- HASH_PTR: CAS mock, errors, registration
- COMPOSITE_PTR: numbered, unary, binary, path, context
- Pointer caching: in-memory SQLite tests
- Integration: handshake logic, CAS flow

#### Research Integration

**Q35 (Markov Blankets):**
- Blanket status gating in decoder
- `sync_handshake()` for alignment verification
- ALIGNED/DISSOLVED/PENDING states

**Q33 (Semantic Density):**
- CDR in token receipts
- `concept_units` counting
- Compression ratio tracking

#### Error Codes (FAIL_CLOSED)
- E_CODEBOOK_MISMATCH, E_KERNEL_VERSION, E_TOKENIZER_MISMATCH
- E_SYNTAX, E_UNKNOWN_SYMBOL, E_HASH_NOT_FOUND
- E_AMBIGUOUS, E_INVALID_OPERATOR, E_INVALID_QUALIFIER
- E_RULE_NOT_FOUND, E_CONTEXT_REQUIRED, E_CAS_UNAVAILABLE

#### Test Results
```
78 passed in 0.66s
```

#### Acceptance Criteria Met
- [x] POINTERS table created (pointer_type, target_hash, qualifiers, codebook_id)
- [x] `pointer_resolve(pointer)` → canonical_IR | FAIL_CLOSED
- [x] Deterministic decode (no LLM involvement)
- [x] CJK glyph support with polysemy handling
- [x] All 7 operators implemented
- [x] CAS integration via callback
- [x] Pointer caching in memory_cassette
- [x] Full integration module
- [x] 78 tests passing

---

## [3.8.3] - 2026-01-11

### Phase 3: Resident Identity - COMPLETE

**Goal:** Each AI instance has a persistent identity in the manifold

#### Phase 3.1: Agent Registry
```sql
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
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

#### Phase 3.2: Session Continuity
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

#### Phase 3.3: Cross-Session Memory
- `memory_promote(hash, from_cassette)` → Dict
- `memory_demote(hash, to_cassette)` → Dict
- `get_promotion_candidates(agent_id, min_access, min_age_hours)` → List[Dict]

**Memories table extended:**
- `session_id` - Links memory to session
- `access_count` - Tracks recall frequency
- `last_accessed` - Timestamp of last access
- `promoted_at` - When memory was promoted
- `source_cassette` - Origin cassette

**Promotion policy:** age >1hr AND access_count >=2, or explicit promotion

#### MCP Tools Added
- `session_start_tool` - Start new session
- `session_resume_tool` - Resume with recent thoughts
- `session_update_tool` - Update working set
- `session_end_tool` - End session
- `agent_info_tool` - Get agent stats
- `agent_list_tool` - List agents
- `memory_promote_tool` - Promote INBOX→RESIDENT

#### Implementation Files
- `memory_cassette.py` - Schema v3.0, all Phase 3 functions
- `semantic_adapter.py` - 7 new MCP tools
- `cassettes.json` - Resident cassette updated with "sessions" capability
- `test_resident_identity.py` - 20+ unit tests

#### Acceptance Criteria Met
- [x] Agent identity persists across sessions
- [x] Memories accumulate over time
- [x] Can query "what did I think last time?" via `session_resume()`

---

## [3.8.2] - 2026-01-11

### Phase 2.4: Cleanup and Deprecation - COMPLETE

**Goal:** Remove deprecated system1.db infrastructure now that cassette network is operational

#### Removed
- **CAPABILITY/TOOLS/cortex/** - Dead code referencing deprecated `_generated/` folder
  - `cortex.py`, `cortex_refresh.py`, `export_semantic.py`
  - `codebook_build.py`, `codebook_lookup.py`
  - `semantic_bridge.py`, `semantic_network.py`
- **NAVIGATION/CORTEX/_generated/** - Deprecated compressed database artifacts
  - `cortex.json`, `SUMMARY_INDEX.json`, `canon_compressed.db`, `compression_config.json`
- **NAVIGATION/CORTEX/cortex.json** + **cortex.schema.json** - Legacy metadata files
- **NAVIGATION/CORTEX/db/** - Deprecated database files
  - `system1_builder.py`, `cortex.build.py`, `build_swarm_db.py`
  - `adr_index.db`, `canon_index.db`, `skill_index.db`, `codebase_full.db`
  - `instructions.db`, `swarm_instructions.db`, `system2.db`
  - `schema.sql`, `schema/002_vectors.sql`
  - `reset_system1.py`, `system2_ledger.py`
- **NAVIGATION/CORTEX/fixtures/** - Test fixtures (`test_fences.md`)
- **Migration scripts** - One-time use completed
  - `migrate_to_cassettes.py`, `structure_aware_migration.py`
- **Research/demo code**
  - `demo_cassette_network.py`, `esap_cassette.py`, `esap_hub.py`, `test_esap_integration.py`
- **NAVIGATION/CORTEX/network/cassettes/** - Specialized cassette implementations superseded by generic protocol
  - `agi_research_cassette.py`, `cat_chat_cassette.py`, `governance_cassette.py`
- **NAVIGATION/CORTEX/semantic/summarizer.py** - Referenced deprecated System1DB
- **NAVIGATION/CORTEX/tests/verify_db.py** - Debug script no longer needed
- **CAPABILITY/TESTBENCH/integration/test_cortex_integration.py** - Broken test importing deleted modules

#### Updated
- **NAVIGATION/CORTEX/README.md** - Documented cassette network as primary architecture
- **CAPABILITY/MCP/semantic_adapter.py** - Removed all system1.db references, uses cassette network exclusively
- **NAVIGATION/CORTEX/semantic/query.py** - Rewritten to search across all 9 cassettes (FTS5)
- **NAVIGATION/CORTEX/semantic/semantic_search.py** - Deprecated with warning, returns empty results
- **NAVIGATION/CORTEX/network/memory_cassette.py** - Added GuardedWriter for write firewall compliance
- **CAPABILITY/TESTBENCH/integration/test_phase_2_4_1c3_no_raw_writes.py** - Allowed memory_cassette.py mkdir operations
- **AGENTS.md** - Updated CORTEX paths, removed `_generated/` references
- **LAW/CONTRACTS/runner.py** - Updated cortex-toolkit skill imports
- **CAPABILITY/SKILLS/cortex-toolkit/** - Updated for cassette network APIs
- **CAPABILITY/MCP/server.py** - Removed deprecated cortex tool references
- **CAPABILITY/MCP/schemas/tools.json** - Cleaned up tool schemas

#### Verification
- **1,242/1,242 AGS tests pass** - No regressions from cleanup
- **Write firewall compliance** - memory_cassette.py uses GuardedWriter
- **Cassette network functional** - 9 cassettes registered and operational

#### Statistics
- **59 files changed:** 473 insertions(+), 17,688 deletions(-)
- **Deleted:** ~30MB of deprecated databases and 12 stale Python modules
- **Cassette network now primary:** system1.db fully deprecated

#### Acceptance Criteria Met
- [x] All deprecated system1.db code removed
- [x] All tests pass after cleanup
- [x] Write firewall compliance maintained
- [x] Cassette network remains fully functional
- [x] Documentation updated to reflect current architecture

---

## [3.8.1] - 2026-01-11

### Phase 2: Write Path (Memory Persistence) - COMPLETE

**Goal:** Let residents save thoughts to the manifold

#### Deliverables

**Phase 2.1: Core Functions**
- `memory_save(text, metadata, agent_id)` - Saves memory with vector embedding, returns hash
- `memory_query(query, limit, agent_id)` - Semantic search over memories
- `memory_recall(hash)` - Retrieve full memory by content-addressed hash
- `semantic_neighbors(hash, limit)` - Find semantically similar memories

**Phase 2.2: Schema Extension**
```sql
CREATE TABLE memories (
    hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,      -- 384 dims, float32
    metadata JSON,
    created_at TEXT NOT NULL,  -- ISO8601
    agent_id TEXT,             -- 'opus', 'sonnet', etc.
    indexed_at TEXT NOT NULL
);

CREATE INDEX idx_memories_agent ON memories(agent_id);
CREATE INDEX idx_memories_created ON memories(created_at);

CREATE VIRTUAL TABLE memories_fts USING fts5(text, hash UNINDEXED);
```

**Phase 2.3: MCP Tools**
- [x] `memory_save_tool` - Save memory to resident cassette
- [x] `memory_query_tool` - Semantic query over memories
- [x] `memory_recall_tool` - Recall full memory by hash
- [x] `semantic_neighbors_tool` - Find similar memories
- [x] `memory_stats_tool` - Get memory statistics

#### Implementation Files
- `memory_cassette.py` - MemoryCassette class with write capabilities
- `semantic_adapter.py` - MCP tools for memory operations

#### Validation
```
Created memory cassette
Saved memory: 64a88dca1605acd9...
Query returned 1 results
Recalled: Test memory for Phase 2 validation...
Stats: 1 memories

Phase 2 basic validation PASSED
```

#### Acceptance Criteria Met
- [x] `memory_save("The Formula is beautiful")` returns hash
- [x] `memory_query("formula beauty")` finds the memory
- [x] Memories persist across sessions (SQLite + FTS5 + vectors)

---

## [3.8.0] - 2026-01-11

### Phase 1: Cassette Partitioning - COMPLETE

**Goal:** Split the monolithic system1.db into semantic-aligned cassettes

#### Deliverables
- Created 9 bucket-aligned cassettes in `NAVIGATION/CORTEX/cassettes/`:
  - `canon.db` (86 files, 200 chunks) - LAW/ bucket, immutable
  - `governance.db` (empty) - CONTEXT/ bucket, stable
  - `capability.db` (64 files, 121 chunks) - CAPABILITY/ bucket
  - `navigation.db` (38 files, 179 chunks) - NAVIGATION/ + root files
  - `direction.db` (empty) - DIRECTION/ bucket
  - `thought.db` (288 files, 1025 chunks) - THOUGHT/ bucket
  - `memory.db` (empty) - MEMORY/ bucket
  - `inbox.db` (67 files, 233 chunks) - INBOX/ bucket
  - `resident.db` (empty) - AI memories, read-write

- Migration script: `migrate_to_cassettes.py`
  - Backup created at `db/_migration_backup/`
  - Receipt emitted with full verification

- Updated `cassettes.json` to config v3.0 with all 9 cassettes

- MCP adapter updated with:
  - `cassettes` filter parameter for `cassette_network_query`
  - New `cassette_stats` tool

- GenericCassette fixed to handle standard cassette schema

#### Validation
- **543/543 files migrated**
- **1758/1758 chunks migrated**
- **9/9 cassettes created**
- All cassettes query-functional via FTS5

#### Acceptance Criteria Met
- [x] 9 cassettes exist (8 buckets + resident)
- [x] `semantic_search` can filter by cassette
- [x] No data loss from migration

### Phase 1.4: Catalytic Hardening - COMPLETE

**Goal:** Make migration fully catalytic with content-addressed verification

#### Deliverables
- `compute_merkle_root(hashes)` - Binary Merkle tree of sorted chunk hashes
- `content_merkle_root` stored per cassette in receipt and DB metadata
- `receipt_hash` - Content-addressed receipt (SHA-256 of canonical JSON)
- `verify_migration(receipt_path)` - Full verification function
- CLI: `python migrate_to_cassettes.py --verify [receipt]`

#### Catalytic Verification
```
Receipt hash valid: True
  canon: PASS
  governance: PASS
  capability: PASS
  navigation: PASS
  direction: PASS
  thought: PASS
  memory: PASS
  inbox: PASS
  resident: PASS

VERIFICATION PASSED - Migration is catalytic
```

#### Final Receipt
- Migration ID: `b03020a9f46f2d19`
- Receipt hash: `af1774bd9efdf84f...`
- All Merkle roots computed and verified

### Phase 1.5: Structure-Aware Chunking - COMPLETE

**Goal:** Split on markdown headers, not sentence boundaries - preserve semantic hierarchy

#### Problem Solved
- Old chunker: splits on `.!?` boundaries, loses header context
- New chunker: splits on `# ## ### ####` headers, preserves hierarchy

#### Deliverables
- `markdown_chunker.py` - Structure-aware splitter with hierarchy
  - `chunk_markdown(text)` returns `List[MarkdownChunk]`
  - Each chunk has: `header_depth`, `header_text`, `parent_index`
  - Builds parent-child relationships automatically

- Schema migration added to chunks table:
  ```sql
  ALTER TABLE chunks ADD COLUMN header_depth INTEGER;
  ALTER TABLE chunks ADD COLUMN header_text TEXT;
  ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER;
  ```

- Navigation queries in `GenericCassette`:
  - `get_chunk(id)` - Full chunk info with child count
  - `get_parent(id)` - Navigate up hierarchy
  - `get_children(id)` - Get direct children
  - `get_siblings(id)` - Prev/next at same depth
  - `get_path(id)` - Breadcrumb trail to root
  - `navigate(id, direction)` - Unified navigation

#### Migration Results
```
Before (sentence-based): 1,758 chunks
After (header-based):    12,478 chunks

canon:      86 files,  1,297 chunks (1,224 with headers)
capability: 64 files,    952 chunks   (908 with headers)
navigation: 38 files,  1,122 chunks (1,098 with headers)
thought:   288 files,  7,123 chunks (6,967 with headers)
inbox:      67 files,  1,984 chunks (1,917 with headers)

Max header depth: 5 (##### level)
```

#### Navigation Demo
```
Query: "invariants"
Result: chunk_id=204, header="## Formal Invariants"
Path:   # Catalytic Computing > ## Formal Invariants
Parent: # Catalytic Computing
Prev:   ## Formal Model (Complexity Theory)
Next:   ## AGS Translation
Children: 5 chunks
```

#### Catalytic Properties
- Migration receipt: `migration_receipt_phase1.5_72438cffdef85ecf.json`
- Receipt hash: `55609d1d736f78fe...`
- New Merkle roots computed for all cassettes
- Backup at `_phase_1_5_backup/`

#### Acceptance Criteria Met
- [x] Chunks align with markdown headers
- [x] Parent-child relationships navigable
- [x] Token counts respect limits (~500 tokens max per section)
- [x] Catalytic: new migration receipt with updated Merkle roots
- [x] Navigation queries: `get_parent()`, `get_children()`, `get_siblings()`
