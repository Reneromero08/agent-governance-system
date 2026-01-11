# Cassette Network Lab Changelog

Research changelog for Cassette Network Phase 6.

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
