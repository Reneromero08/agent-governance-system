---
title: "Original 8 Week Roadmap"
section: "research"
author: "Raúl R Romero"
priority: "Low"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Archived"
summary: "Original 8-week roadmap plan (Archived)"
tags: [roadmap, archive, legacy]
---
<!-- CONTENT_HASH: f5b18f5e1379b97e44e84a9e1df7fd30b28a7c8a27650af5ae26f26ced2e3eee -->

# Catalytic Chat Implementation Roadmap

**Status:** Proposed
**Date:** 2025-12-29
**Related Research:** INBOX/research/catalytic-chat-research.md
**Estimated Duration:** 6-8 weeks

## Overview

This roadmap outlines the implementation of a catalytic chat system that makes Claude Code messages hash-indexed, vector-searchable, and efficiently retrievable from a SQLite database while maintaining compatibility with existing workflows.

## Architecture Decision

**Triple-Write Strategy:**
- **DB (Primary)**: Claude CLI/terminal reads context from SQLite
- **JSONL (Mechanical)**: Generated from DB for VSCode extension
- **MD (Human)**: Generated from DB for human readability

This approach ensures:
- Opencode CLI can read from DB directly
- VSCode extension continues to work (reads mechanical JSONL)
- Token waste reduced through hash-based retrieval
- Human-readable exports available

## Phase 1: Foundation (Week 1-2)

### 1.1 Database Schema & Connection
**Goal:** Create chat.db with proper schema and migration system

**Tasks:**
- [ ] Design complete schema (messages, chunks, vectors, metadata)
- [ ] Implement `CHAT_SYSTEM/chat_db.py` with:
  - SQLite connection management
  - Schema migration system (versioned)
  - Connection pooling (if needed)
- [ ] Create migration files in `CHAT_SYSTEM/migrations/`
  - `001_initial_schema.sql`
  - `002_vectors.sql`
  - `003_indexes.sql`
- [ ] Add indexes for session_id, timestamp, chunk_hash
- [ ] Implement `chat_db.init_db()` and `chat_db.get_connection()`

**Success Criteria:**
- DB initializes with all tables
- Migration system tracks version
- Indexes created and verified via EXPLAIN QUERY PLAN

### 1.2 Core Data Models
**Goal:** Python models for messages, chunks, vectors

**Tasks:**
- [ ] Create `CHAT_SYSTEM/models.py` with dataclasses:
  - `ChatMessage(message_id, session_id, uuid, role, content, ...)`
  - `MessageChunk(chunk_id, message_id, chunk_index, chunk_hash, ...)`
  - `MessageVector(chunk_hash, embedding, model_id, ...)`
- [ ] Implement serialization/deserialization methods
- [ ] Add type hints for all models

**Success Criteria:**
- Models map 1:1 to DB tables
- Unit tests for model validation

### 1.3 Embedding Engine Integration
**Goal:** Reuse CORTEX embedding patterns for messages

**Tasks:**
- [ ] Copy/adapt `CORTEX/embeddings.py` to `CHAT_SYSTEM/embedding_engine.py`
- [ ] Ensure `all-MiniLM-L6-v2` model loaded once
- [ ] Implement batch embedding for efficiency
- [ ] Add caching for repeated embeddings

**Success Criteria:**
- Embedding generation <15ms per message chunk
- Batching works (32 chunks <200ms)

## Phase 2: Storage Layer (Week 3)

### 2.1 Triple-Write Implementation
**Goal:** Write to DB + JSONL + MD atomically

**Tasks:**
- [ ] Create `CHAT_SYSTEM/message_writer.py`
- [ ] Implement `MessageWriter.write_message()`:
  1. Insert into chat_messages table
  2. Compute content hash (SHA-256)
  3. Check for duplicates
  4. Chunk long messages (>500 tokens)
  5. Generate embeddings for chunks
  6. Store in message_vectors table
  7. Generate JSONL entry
  8. Generate MD entry
  9. Commit transaction (all-or-nothing)
- [ ] Implement `MessageWriter.write_jsonl_export()`
- [ ] Implement `MessageWriter.write_md_export()`
- [ ] Add atomic file locking for JSONL/MD writes

**Success Criteria:**
- Triple-write passes without partial failures
- DB, JSONL, MD are consistent
- Duplicate messages rejected based on hash

### 2.2 Export Generators
**Goal:** Mechanical JSONL and MD generation from DB

**Tasks:**
- [ ] Create `CHAT_SYSTEM/export_generators.py`
- [ ] Implement `generate_jsonl(session_id, output_path)`:
  - Query DB for all messages in session
  - Format as JSONL (line-delimited)
  - Preserve all JSONL fields (uuid, parentUuid, etc.)
- [ ] Implement `generate_markdown(session_id, output_path)`:
  - Query DB for messages
  - Format with headers, timestamps, roles
  - Human-readable structure
- [ ] Add incremental export (only new messages since last export)

**Success Criteria:**
- JSONL matches original Claude Code format
- MD is human-readable with proper formatting
- Export handles large sessions (>1000 messages)

## Phase 3: Context Loading (Week 4)

### 3.1 DB-Based Context Loader
**Goal:** Replace JSONL reading with DB queries

**Tasks:**
- [ ] Create `CHAT_SYSTEM/context_loader.py`
- [ ] Implement `ContextLoader.load_session(session_id)`:
  - Query messages by session_id
  - Sort by timestamp
  - Reconstruct parent-child relationships
  - Build message array in chronological order
- [ ] Implement `ContextLoader.get_conversation_tree(session_id)`:
  - Build tree structure from parent_uuid links
  - Handle sidechains and branches
- [ ] Add token counting per message (via embeddings or estimation)
- [ ] Implement hash-based message lookup (avoid re-reading)

**Success Criteria:**
- Context loads in <100ms for 1000-message session
- Parent-child linking preserved
- Token counts accurate within 5%

### 3.2 Context Optimization
**Goal:** Manage 200K token budget with DB-based retrieval

**Tasks:**
- [ ] Implement `ContextOptimizer.optimize(messages, budget)`:
  - Remove oldest messages first
  - Keep system prompt
  - Summarize truncated portion
- [ ] Implement token counter using hash-based lookups
- [ ] Add configuration for reserve tokens (default: 4,000)
- [ ] Log optimization decisions

**Success Criteria:**
- Context fits within 200K tokens
- Summary captures key information
- Optimization takes <50ms

## Phase 4: Migration (Week 5)

### 4.1 JSONL to DB Migration Tool
**Goal:** Migrate existing Claude Code sessions to chat.db

**Tasks:**
- [ ] Create `CHAT_SYSTEM/migration_tool.py`
- [ ] Implement `Migrator.migrate_jsonl_to_db(jsonl_path)`:
  1. Parse JSONL file
  2. Extract metadata (sessionId, uuid, parentUuid, etc.)
  3. Compute content hashes
  4. Insert into chat_messages table
  5. Chunk messages
  6. Generate embeddings
  7. Store vectors
- [ ] Scan all JSONL files in `.claude/projects/`
- [ ] Progress bar for large migrations
- [ ] Rollback capability (backup JSONL before migration)
- [ ] Validation: compare DB vs JSONL counts

**Success Criteria:**
- All JSONL sessions migrated
- Message count matches original
- Embeddings generated for all chunks
- Rollback tested and works

### 4.2 Migration Validation
**Goal:** Ensure data integrity after migration

**Tasks:**
- [ ] Create `CHAT_SYSTEM/validate_migration.py`
- [ ] Implement validation checks:
  - Message count matches JSONL
  - Parent-child links intact
  - Content hashes correct
  - Embeddings dimensions correct
  - No orphaned chunks
- [ ] Generate validation report
- [ ] Fix issues automatically if possible

**Success Criteria:**
- Validation passes with 0 errors
- Orphaned records = 0
- All embeddings 384 dimensions

## Phase 5: Vector Search (Week 6)

### 5.1 Semantic Search Implementation
**Goal:** Search messages by semantic similarity

**Tasks:**
- [ ] Create `CHAT_SYSTEM/vector_search.py`
- [ ] Implement `VectorSearch.search(query, session_id, top_k)`:
  1. Generate query embedding
  2. Load message vectors for session
  3. Compute cosine similarity
  4. Rank and return top_k
- [ ] Implement hybrid search (FTS5 + vectors):
  - Keyword match via message_fts table
  - Semantic match via vectors
  - Combine scores with weights
- [ ] Add filtering by role, timestamp range

**Success Criteria:**
- Search returns in <200ms for 10K chunks
- Top-5 results relevance >0.3 similarity
- Hybrid search improves keyword-only recall by >20%

### 5.2 Context Retrieval Strategy
**Goal:** Use vector search to assemble context

**Tasks:**
- [ ] Implement `ContextRetriever.retrieve_context(query, budget)`:
  - Search for similar messages
  - Select top-K based on relevance
  - Fit within token budget
  - Prioritize recent + relevant
- [ ] Implement caching for frequent queries
- [ ] Add logging for retrieval decisions

**Success Criteria:**
- Retrieved context relevant to query
- Fits within token budget
- Faster than full message loading (>2x speedup)

## Phase 6: Integration & Testing (Week 7-8)

### 6.1 Claude Code Integration
**Goal:** Hook into Claude Code CLI context loading

**Tasks:**
- [ ] Identify JSONL reader location (requires Claude Code source access)
- [ ] Replace with DB-based loader
- [ ] Initialize chat.db on startup
- [ ] Add triple-write to message persistence
- [ ] Test with opencode CLI

**Success Criteria:**
- Opencode CLI uses DB for context
- Token usage reduced by >30%
- No data loss during migration

### 6.2 VSCode Compatibility
**Goal:** Ensure VSCode extension works with JSONL exports

**Tasks:**
- [ ] Generate JSONL exports for all sessions
- [ ] Verify extension can read mechanical JSONL
- [ ] Test with existing projects
- [ ] Document export path and format

**Success Criteria:**
- VSCode extension loads sessions normally
- No data corruption in exported JSONL
- Extension features unchanged

### 6.3 Testing Suite
**Goal:** Comprehensive test coverage

**Tasks:**
- [ ] Unit tests (pytest):
  - DB operations
  - Message models
  - Embedding generation
  - Context optimization
- [ ] Integration tests:
  - Triple-write end-to-end
  - Migration workflow
  - Context loading with real sessions
- [ ] Performance tests:
  - DB query latency
  - Embedding generation speed
  - Search response time
  - Context assembly time
- [ ] Validation tests:
  - Migration accuracy
  - Data integrity
  - Token counting precision

**Success Criteria:**
- >90% code coverage
- All tests pass
- Performance benchmarks met

## Deliverables

### Phase 1 Deliverables
- `CHAT_SYSTEM/chat_db.py`
- `CHAT_SYSTEM/models.py`
- `CHAT_SYSTEM/embedding_engine.py`
- Migration SQL files

### Phase 2 Deliverables
- `CHAT_SYSTEM/message_writer.py`
- `CHAT_SYSTEM/export_generators.py`
- Triple-write implementation

### Phase 3 Deliverables
- `CHAT_SYSTEM/context_loader.py`
- `CHAT_SYSTEM/context_optimizer.py`
- DB-based context loading

### Phase 4 Deliverables
- `CHAT_SYSTEM/migration_tool.py`
- `CHAT_SYSTEM/validate_migration.py`
- Migrated chat.db with all existing sessions

### Phase 5 Deliverables
- `CHAT_SYSTEM/vector_search.py`
- `CHAT_SYSTEM/context_retriever.py`
- Semantic search over messages

### Phase 6 Deliverables
- Integration with Claude Code CLI
- VSCode compatibility verified
- Complete test suite
- Documentation

## Risk Mitigation

### Technical Risks
- **Risk**: Claude Code CLI source not accessible
  - **Mitigation**: Use command-line wrapper/proxy
- **Risk**: DB corruption
  - **Mitigation**: JSONL/MD backups, WAL mode, integrity checks
- **Risk**: Embedding generation slow
  - **Mitigation**: Batch processing, caching, lazy loading

### Compatibility Risks
- **Risk**: VSCode extension breaks with mechanical JSONL
  - **Mitigation**: Extensive testing with actual extension
- **Risk**: Parent-child linking lost
  - **Mitigation**: Preserve uuid/parent_uuid in migration

### Performance Risks
- **Risk**: DB queries too slow
  - **Mitigation**: Proper indexes, query optimization, connection pooling
- **Risk**: Vector search doesn't scale
  - **Mitigation**: Limit search to session, FAISS if needed

## Success Metrics

### Quantitative Metrics
- Token usage reduction: >30%
- Context load time: <100ms for 1000 messages
- Search response time: <200ms for 10K chunks
- Migration accuracy: 100%
- Test coverage: >90%

### Qualitative Metrics
- Opencode CLI works with DB
- VSCode extension unaffected
- Human-readable exports available
- Data integrity maintained

## Dependencies

### External Dependencies
- `sqlite3` (Python stdlib)
- `sentence-transformers>=2.2.0`
- `numpy>=1.21.0`
- `pytest>=7.0.0`

### Internal Dependencies
- CORTEX embedding patterns
- Existing CORTEX schemas (as reference)
- Claude Code JSONL format (for compatibility)

## Open Questions

1. Should chat.db be per-project or global (currently proposed global in `.claude/`)?
2. How to handle Claude Code updates that change JSONL format?
3. Should we implement automatic export triggers (on message write)?
4. What's the maximum message size before chunking?

## Next Steps

1. Review and approve this roadmap
2. Create ADR-XXX for triple-write architecture
3. Start Phase 1: Database schema and models
4. Set up development environment with test data
5. Begin implementation

---

## Phase 6.2: Attestation (Completed 2025-12-30)

**Status:** Complete ✓
**Related:** commit-plan-phase-6-2-attestation.md

### Tasks Completed
- [x] **receipt_canonical_bytes(receipt, attestation_override=None)** in `catalytic_chat/receipt.py`
  - Single source of truth for receipt canonicalization
  - Used by signer, verifier, and executor
- [x] **verify_receipt_bytes()** updated in `catalytic_chat/attestation.py`
  - Now uses `receipt_canonical_bytes()` with `attestation_override=None`
  - Ensures verification computes exact same canonical bytes as signing
- [x] **executor.py** updated to use `receipt_canonical_bytes()`
  - For signing (with `attestation=None`)
  - For writing (with or without attestation)
- [x] **cli.py** updated with `--attest` flag
  - `bundle run` now supports `--attest` (requires `--signing-key`)
  - `--verify-attestation` flag for verification
- [x] **test_attestation.py** fixed
  - All 6 attestation tests pass
  - Fixed tamper test to verify canonical byte differences

### Verification
```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q tests/test_attestation.py tests/test_receipt.py
# 12 passed, 2 skipped in 0.26s
```

### Hard Constraints Met
- ✓ Single source of truth for canonicalization
- ✓ Signing input is canonical receipt bytes with `attestation=null/None`
- ✓ Verifying recomputes exact same canonical bytes
- ✓ Hex-only for `public_key`/`signature` with validation
- ✓ No timestamps, randomness, absolute paths, or env-dependent behavior
- ✓ Minimal diffs; changes localized to canonicalization and signing flow

---

## Phase 6.2.1: Attestation Stabilization (Completed 2025-12-30)

**Status:** Complete ✓

### Issue Fixed
`test_cli_dry_run` was failing because the subprocess call to `python -m catalytic_chat.cli` could not find the module due to missing PYTHONPATH environment variable.

### Solution
Added 3 lines to `tests/test_planner.py`:
- Import `os`
- Copy environment variables and set `PYTHONPATH` to `Path(__file__).parent.parent`
- Pass `env` parameter to `subprocess.run()`

### Verification
```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q THOUGHT/LAB/CAT_CHAT/tests
# 59 passed, 13 skipped in 6.10s
```

### Hard Constraints Met
- ✓ Entire test suite green (no new failures)
- ✓ No test skips added (same 13 skips)
- ✓ Deterministic receipts unchanged
- ✓ Attestation still fail-closed
- ✓ No CLI or executor regressions
- ✓ Minimal fix (only test environment issue addressed)

---

## Phase 6.3: Receipt Chain Anchoring (Completed 2025-12-30)

**Status:** Complete ✓

### Changes
- `catalytic_chat/receipt.py`: Added chain functions
  - `compute_receipt_hash()`: deterministic hash from canonical bytes (excluding receipt_hash field)
  - `load_receipt()`: load receipt from JSON file
  - `verify_receipt_chain()`: verify chain linkage and receipt hashes
  - `find_receipt_chain()`: find all receipts for a run in execution order
- `catalytic_chat/executor.py`: Added chain support
  - `previous_receipt` parameter to `BundleExecutor.__init__()`
  - Sets `parent_receipt_hash` from previous receipt's `receipt_hash`
  - First receipt has `parent_receipt_hash=null`
- `SCHEMAS/receipt.schema.json`: Added chain fields
  - `parent_receipt_hash`: string | null
  - `receipt_hash`: string
- `catalytic_chat/cli.py`: Added `--verify-chain` flag
  - Verifies full receipt chain for a run
  - Outputs chain status and Merkle root
- `tests/test_receipt_chain.py`: Added 4 chain tests
  - Deterministic chain verification
  - Chain break detection
  - Sequential order enforcement

### Verification
```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q tests/test_receipt_chain.py
# 4 passed in 0.21s
```

### Hard Constraints Met
- ✓ Identical inputs produce identical chain
- ✓ Chain verification fails on tamper or reorder
- ✓ No timestamps, randomness, absolute paths
- ✓ Minimal diffs; only extended receipt/verify paths

---

## Phase 6.4: Receipt Merkle Root + External Anchor (Completed 2025-12-30)

**Status:** Complete ✓

### Changes
- `catalytic_chat/receipt.py`: Added Merkle root computation
  - `compute_merkle_root()`: deterministic Merkle tree from receipt hashes
  - Pairwise concatenation of hex-decoded bytes (left||right) → SHA256
  - Odd leaf duplication at each level (not just once)
  - Preserves deterministic ordering from `find_receipt_chain()`
  - `verify_receipt_chain()` now returns Merkle root string
- `catalytic_chat/cli.py`: Added Merkle output
  - `--print-merkle` flag (requires `--verify-chain`)
  - Prints ONLY Merkle root hex to stdout
  - Fails immediately if `--print-merkle` without `--verify-chain`
- `tests/test_merkle_root.py`: Added 3 Merkle root tests
  - Deterministic root computation
  - Tamper detection
  - `--verify-chain` requirement enforcement

### Verification
```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q tests/test_merkle_root.py
# 3 passed in 0.16s
```

### Hard Constraints Met
- ✓ Identical chains produce identical Merkle roots
- ✓ Fail-closed: tampering, ordering, hash mismatches abort
- ✓ No timestamps, randomness, absolute paths, network calls
- ✓ Merkle root NOT stored in individual receipts (chain-only metadata)
- ✓ `--print-merkle` prints ONLY hex to stdout
- ✓ `--print-merkle` requires `--verify-chain` (fail-closed)
