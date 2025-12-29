# Catalytic Chat System (Phase 1 Complete)

**Status:** Phase 1 (Database & Core Storage) Complete
**Date:** 2025-12-29
**ADR:** ADR-031 - Catalytic Chat Triple-Write Architecture

## Overview

Triple-write storage system for Claude Code messages:
- **DB (Primary)**: SQLite with hash indexing and vector embeddings
- **JSONL (Mechanical)**: Generated from DB for VSCode compatibility
- **MD (Human)**: Generated from DB for readability

## Phase 1 Deliverables

### Core Files

- `chat_db.py` - Database schema and connection management
  - Tables: chat_messages, message_chunks, message_vectors, message_fts
  - Hash-based deduplication
  - Transaction support with context managers
  - Migration system

- `embedding_engine.py` - Vector embeddings for semantic search
  - Model: all-MiniLM-L6-v2 (384 dimensions)
  - Batch processing for efficiency
  - Cosine similarity computation
  - BLOB serialization for SQLite

- `message_writer.py` - Triple-write implementation
  - Atomic writes to DB + JSONL + MD
  - Automatic chunking of long messages
  - Embedding generation for chunks
  - JSONL export in Claude Code format
  - MD export with human-readable formatting

### Database Schema

```sql
-- Primary message storage
chat_messages (message_id, session_id, uuid, parent_uuid, role,
               content, content_hash, timestamp, metadata, ...)

-- Chunks for long messages
message_chunks (chunk_id, message_id, chunk_index, chunk_hash,
                content, token_count)

-- Vector embeddings
message_vectors (chunk_hash, embedding BLOB, model_id, dimensions, created_at)

-- Full-text search
message_fts (content, chunk_id, role)
```

## Usage Examples

### Initialize Database

```python
from chat_db import ChatDB

db = ChatDB()
db.init_db()
```

### Write Message

```python
from message_writer import MessageWriter

writer = MessageWriter()

uuid = writer.write_message(
    session_id="my-project",
    role="user",
    content="Help me debug this code."
)
```

### Query Messages

```python
# Get all messages in session
messages = db.get_session_messages("my-project")

# Get specific message
msg = db.get_message_by_uuid(uuid)
```

## Running Tests

```bash
# Test database
cd CATALYTIC-DPT/LAB/CHAT_SYSTEM
python chat_db.py

# Test embedding engine
python embedding_engine.py

# Test message writer
python message_writer.py
```

## Dependencies

- `sqlite3` (Python stdlib)
- `numpy>=1.21.0`
- `sentence-transformers>=2.2.0`

## Next Phases

**Phase 2** (Complete): Triple-write implementation
**Phase 3**: DB-based context loader
**Phase 4**: JSONL → DB migration tool
**Phase 5**: Vector search integration
**Phase 6**: Testing and validation

## Design Decisions

- **Hash-based deduplication**: SHA-256 of content enables identifying identical messages
- **Chunking at 500 tokens**: Balances embedding granularity and search performance
- **Triple-write**: Ensures compatibility with VSCode extension (JSONL) while enabling DB features
- **WAL mode**: Improves SQLite concurrency and crash recovery

## References

- Research: `INBOX/Agents/OpenCode/catalytic-chat-research.md`
- Roadmap: `INBOX/Agents/OpenCode/catalytic-chat-roadmap.md`
- ADR: `CONTEXT/decisions/ADR-031-catalytic-chat-triple-write.md`
