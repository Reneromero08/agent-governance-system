---
uuid: 00000000-0000-0000-0000-000000000000
title: Catalytic Chat Research
section: research
bucket: 2025-12/Week-52
author: "Ra\xFAl R Romero"
priority: Medium
created: 2025-12-28 23:12
modified: 2026-01-06 13:09
status: Complete
summary: Complete research on Claude Code architecture, token management
tags:
- catalytic_chat
- research
- architecture
hashtags: []
---
<!-- CONTENT_HASH: 258d1148d46201987ebfdfc8d126a80181a0a92bdc3d92d9a8d7f2219c18a8e5 -->

# Catalytic Chat Research Report

**Status:** Complete
**Date:** 2025-12-29
**Related Files:** INBOX/research/Claude Code Vector Chat_1.md

## Executive Summary

Research complete for implementing a catalytic chat system that makes Claude Code messages hash-indexed and vector-searchable. The system uses a triple-write strategy: DB (primary for Claude), JSONL (mechanical backup for VSCode), MD (human-readable).

## 1. Current Claude Code Architecture

### 1.1 Storage Location
- **Path**: `C:\Users\{username}\.claude\projects\{PROJECT_PATH}/{session-uuid}.jsonl`
- **Format**: JSON Lines (one JSON object per line)
- **Project-based**: Each project has its own session files

### 1.2 Message Loading Flow
```
User Input → Claude Code CLI/Extension
  → Load from ~/.claude/projects/{proj}/{session-id}.jsonl
  → Parse JSONL line-by-line
  → Build context for API call
  → Send to Anthropic API
  → Receive response
  → Append new message to JSONL
```

### 1.3 Message Structure
```json
{
  "uuid": "message-uuid",
  "parentUuid": "parent-message-uuid",
  "sessionId": "session-id",
  "type": "user|assistant|queue-operation|file-history-snapshot",
  "message": {
    "role": "user|assistant",
    "content": "string or array",
    "usage": {
      "input_tokens": 1000,
      "output_tokens": 500,
      "cache_creation_input_tokens": 800,
      "cache_read_input_tokens": 200
    }
  },
  "isSidechain": false,
  "isMeta": false,
  "timestamp": "2025-12-29T00:00:00Z",
  "cwd": "/path/to/project"
}
```

## 2. Token Budget Management

### 2.1 Context Window Limits
- **Claude 3.7**: 200,000 tokens
- **Claude 3.5/4**: 200,000 tokens
- **File Reading**: Max 25,000 tokens

### 2.2 Truncation Strategy
```typescript
function truncateConversation(messages, fracToRemove) {
  const truncatedMessages = [messages[0]]; // Keep system
  const messagesToRemove = Math.floor((messages.length - 1) * fracToRemove);
  const remainingMessages = messages.slice(messagesToRemove + 1);
  truncatedMessages.push(...remainingMessages);
  return truncatedMessages;
}
```

### 2.3 Token Calculation
- Sum input_tokens + cache_read_input_tokens + cache_creation_input_tokens
- Track most recent usage from JSONL entries
- Reserve ~4,000 tokens for output

## 3. CORTEX Vector & Embedding Patterns

### 3.1 Embedding Model
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Storage**: float32 (4 bytes per dimension = 1,536 bytes per embedding)
- **Library**: `sentence-transformers>=2.2.0`

### 3.2 Vector Storage Schema
```sql
CREATE TABLE IF NOT EXISTS section_vectors (
    hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER NOT NULL DEFAULT 384,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (hash) REFERENCES sections(hash) ON DELETE CASCADE
);
```

### 3.3 Chunking Strategy
- **CHUNK_SIZE**: 500 tokens
- **CHUNK_OVERLAP**: 50 tokens
- **Token approximation**: word count × 0.75
- **Sentence boundaries**: Regex split on `.!?`

### 3.4 Semantic Search
```python
def search(query: str, top_k: int = 10, min_similarity: float = 0.0):
    query_embedding = self.embedding_engine.embed(query)
    cursor = self.conn.execute("""
        SELECT sv.hash, sv.embedding, fts.content
        FROM section_vectors sv
        LEFT JOIN chunks_fts fts ON sv.hash = c.chunk_hash
    """)

    results = []
    for row in cursor:
        embedding = self.embedding_engine.deserialize(row['embedding'])
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity >= min_similarity:
            results.append((similarity, row['content']))

    results.sort(reverse=True)
    return results[:top_k]
```

### 3.5 Hash-Based Indexing
- **Hash type**: SHA-256 (64 hex characters)
- **Scope**: Full content (not metadata)
- **Purpose**: Deduplication, incremental updates, integrity
- **Implementation**: `hashlib.sha256(content.encode('utf-8')).hexdigest()`

## 4. Integration Points

### 4.1 Where to Hook DB Reading
1. **JSONL Loader** - Replace `fs.readFile()` with SQLite query
2. **Context Builder** - Intercept message array construction
3. **Token Counter** - Use hash-based retrieval from DB
4. **Message Writer** - Triple-write: DB + JSONL + MD

### 4.2 Flow with DB Integration
```
User Input
  → Parse and validate
  → Query DB for historical context (by sessionId)
  → Build message array from DB results
  → Count tokens using hash-based lookup
  → Optimize/truncate if needed
  → Send to API
  → Store response: DB + JSONL + MD
```

## 5. Triple-Write Architecture

### 5.1 Write Paths
- **DB (Primary)**: `C:\Users\{username}\.claude\chat.db`
  - Claude CLI/terminal reads from here
  - Hash-based message retrieval
  - Vector search for context
- **JSONL (Mechanical)**: `C:\Users\{username}\.claude\projects\{project}/{session}.jsonl`
  - Generated from DB
  - For VSCode extension compatibility
  - Write-only (not read by Claude)
- **MD (Human)**: `C:\Users\{username}\.claude\projects\{project}/{session}.md`
  - Generated from DB
  - Human-readable exports
  - Mechanical from DB

### 5.2 Compatibility Matrix
| Client | Storage | Read From |
|--------|----------|-----------|
| Opencode CLI | DB + JSONL + MD | DB |
| Terminal mode | DB + JSONL + MD | DB |
| VSCode webview | DB + JSONL + MD | JSONL |

## 6. Proposed Schema for Chat DB

### 6.1 Tables
```sql
-- Primary message storage
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    uuid TEXT NOT NULL UNIQUE,
    parent_uuid TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata JSON,
    FOREIGN KEY (parent_uuid) REFERENCES chat_messages(uuid)
);

-- Chunks for long messages
CREATE TABLE IF NOT EXISTS message_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    FOREIGN KEY (message_id) REFERENCES chat_messages(message_id),
    UNIQUE(message_id, chunk_index)
);

-- Vector embeddings
CREATE TABLE IF NOT EXISTS message_vectors (
    chunk_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_id TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER NOT NULL DEFAULT 384,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_hash) REFERENCES message_chunks(chunk_hash)
);

-- FTS5 for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    role,
    tokenize='porter unicode61'
);

-- Metadata
CREATE TABLE IF NOT EXISTS chat_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_vectors_created ON message_vectors(created_at);
```

## 7. Implementation Phases

### Phase 1: Database Schema & Core Storage
- Create `CHAT_DB_PATH` in `.claude/`
- Implement schema migration system
- Write `chat_db.py` with connection management
- Create triple-write utilities

### Phase 2: JSONL/MD/DB Triple-Write
- Implement `MessageWriter` class
- Write to DB first (canonical)
- Generate JSONL from DB (mechanical)
- Generate MD from DB (mechanical)
- Ensure atomic writes

### Phase 3: Replace Context Loading
- Identify JSONL reader in Claude Code CLI
- Implement DB-based context loader
- Hash-based message lookup
- Preserve parentUuid linking
- Test with existing sessions

### Phase 4: Migration Tool
- Read all existing JSONL files
- Migrate to chat.db format
- Generate embeddings for chunks
- Validate data integrity
- Create rollback capability

### Phase 5: Vector-Based Context Retrieval
- Implement semantic search over message chunks
- Hybrid: keyword (FTS5) + semantic (vectors)
- Context assembly based on query relevance
- Token budget management with vector retrieval

### Phase 6: Testing & Validation
- Unit tests for DB operations
- Integration tests for context loading
- Token accuracy tests
- Migration validation tests
- Performance benchmarks

## 8. Critical Files to Create/Modify

### New Files
```
CHAT_SYSTEM/
  ├── chat_db.py              # DB schema and connection management
  ├── message_writer.py        # Triple-write implementation
  ├── context_loader.py        # DB-based context assembly
  ├── vector_search.py         # Semantic search over messages
  ├── migration_tool.py        # JSONL → DB migration
  └── export_generators.py    # JSONL/MD generators from DB
```

### Integration Points (Hypothetical - needs access to Claude Code source)
```
Replace JSONL loader in Claude Code CLI with context_loader.py
Hook message writing with message_writer.py
Add chat.db initialization in startup
```

## 9. Key Findings

### 9.1 What Works Well
- CORTEX embedding patterns are production-ready
- Hash-based indexing enables deduplication
- Triple-write maintains compatibility
- Vector search reduces token waste

### 9.2 Potential Issues
- VSCode extension is closed-source - cannot intercept its reads
- JSONL remains required for extension compatibility
- Need to maintain JSONL/MD as mechanical exports from DB

### 9.3 Risk Mitigation
- JSONL generated from DB ensures consistency
- MD exports provide human-readable backup
- DB corruption can be recovered from JSONL/MD

## 10. Next Steps

1. Review and approve this research report
2. Create ADR for triple-write architecture decision
3. Design Phase 1 implementation plan
4. Build schema and core storage layer
5. Implement triple-write system
6. Prototype DB-based context loader

## 11. References

- CORTEX/embeddings.py - Embedding generation patterns
- CORTEX/schema/002_vectors.sql - Vector storage schema
- CORTEX/system1_builder.py - Chunking and hashing
- INBOX/research/Claude Code Vector Chat_1.md - Original planning