# ADR-031: Catalytic Chat Triple-Write Architecture

**Status:** Proposed
**Date:** 2025-12-29
**Confidence:** High
**Impact:** High
**Tags:** [chat, storage, optimization, vector]

## Context

Claude Code stores sessions as JSONL files in `~/.claude/projects/{project}/{session}.jsonl`. This approach has several inefficiencies:

1. **Token waste**: Loading entire conversation history for context retrieval, even when only a few messages are relevant
2. **No deduplication**: Identical messages across sessions consume storage and tokens
3. **No semantic search**: Cannot retrieve relevant historical messages by meaning, only by position/time
4. **Performance**: Parsing large JSONL files is slow for context assembly

The research in `INBOX/Agents/OpenCode/catalytic-chat-research.md` proposes a SQLite-based storage system with vector embeddings for semantic search.

## Decision

We adopt a **Triple-Write Architecture** for catalytic chat storage:

### Write Strategy

1. **DB (Primary)**: `~/.claude/chat.db`
   - Claude Code CLI/terminal reads context from SQLite
   - Hash-based message indexing for deduplication
   - Vector embeddings for semantic search
   - FTS5 for keyword search

2. **JSONL (Mechanical)**: `~/.claude/projects/{project}/{session}.jsonl`
   - Generated from DB (write-only, mechanical export)
   - Maintains compatibility with VSCode extension (closed-source)
   - Serves as backup for DB recovery

3. **MD (Human)**: `~/.claude/projects/{project}/{session}.md`
   - Generated from DB (write-only, mechanical export)
   - Human-readable conversation exports
   - Serves as audit trail

### Database Schema

```sql
-- Primary message storage
CREATE TABLE chat_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    uuid TEXT NOT NULL UNIQUE,
    parent_uuid TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata JSON
);

-- Chunks for long messages
CREATE TABLE message_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    FOREIGN KEY (message_id) REFERENCES chat_messages(message_id)
);

-- Vector embeddings (all-MiniLM-L6-v2, 384 dims)
CREATE TABLE message_vectors (
    chunk_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER NOT NULL DEFAULT 384,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_hash) REFERENCES message_chunks(chunk_hash)
);

-- FTS5 for keyword search
CREATE VIRTUAL TABLE message_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    role,
    tokenize='porter unicode61'
);
```

### Indexing Strategy

- Hash-based deduplication: SHA-256 of message content
- Vector search: Cosine similarity on embeddings
- Hybrid retrieval: Combine FTS5 keyword match + semantic similarity

## Alternatives considered

1. **JSONL-only with caching**: Keep JSONL, add in-memory cache
   - Rejected: Still wastes tokens, no semantic search

2. **Pure vector DB (Pinecone/FAISS)**: External vector database
   - Rejected: Adds external dependency, complexity, cost

3. **Single DB, no exports**: SQLite only, drop JSONL/MD
   - Rejected: Breaks VSCode extension compatibility (closed-source)

## Rationale

The triple-write approach balances three constraints:
- **Efficiency**: DB enables hash-based deduplication and vector search (30% token reduction)
- **Compatibility**: JSONL exports keep VSCode extension working
- **Reliability**: MD exports provide human-readable backup, DB corruption recoverable

Hash-based indexing ensures identical messages are stored once, regardless of how many times they appear across sessions. This is critical for agentic workflows that frequently repeat instructions.

## Consequences

### Positive
- Token usage reduction: >30% (hash-based retrieval + semantic context assembly)
- Context load time: <100ms for 1000 messages
- Semantic search: Retrieve relevant historical messages by meaning
- Compatibility: VSCode extension continues working with mechanical JSONL
- Data integrity: DB + JSONL + MD provide triple redundancy

### Negative
- Complexity: Three write paths to maintain
- Storage: ~2-3x storage increase (DB + JSONL + MD)
- Migration: One-time effort to migrate existing JSONL sessions

### Follow-up work
- Phase 1: Database schema and core storage
- Phase 2: Triple-write implementation
- Phase 3: DB-based context loader
- Phase 4: JSONL → DB migration tool
- Phase 5: Vector search integration
- Phase 6: Testing and validation

## Enforcement

### Canon updates
None required. This is an experimental prototype in CATALYTIC-DPT/LAB.

### Fixtures
Create fixtures in `LAW/CONTRACTS/fixtures/` for:
- Triple-write atomicity (all three writes succeed or none)
- Hash-based deduplication (identical content = single record)
- JSONL export format (matches Claude Code spec)
- Vector similarity thresholds

### Skill constraints
If formalized as a skill, require:
- Session ID validation (UUID format)
- Content hash computation (SHA-256)
- Atomic transaction management
- Export generation on demand

## Review triggers

Revisit if:
- Claude Code releases source code (may simplify architecture)
- VSCode extension adds direct DB support (may drop JSONL)
- Token costs change significantly (may adjust chunking strategy)
- New embedding models outperform all-MiniLM-L6-v2 (may upgrade vector model)

## References

- Research: `INBOX/Agents/OpenCode/catalytic-chat-research.md`
- Roadmap: `INBOX/Agents/OpenCode/catalytic-chat-roadmap.md`
- CORTEX patterns: `CORTEX/embeddings.py`, `CORTEX/schema/002_vectors.sql`
