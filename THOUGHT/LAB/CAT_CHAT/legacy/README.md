# Legacy Files

This directory contains deprecated scripts and data from earlier phases of Catalytic Chat development.

## Status: NOT CANONICAL

All files in this directory are preserved for historical reference but are **not part of the current canonical implementation**.

## Contents

### Scripts (misaligned with canonical roadmap)
- `chat_db.py` - Database for Claude Code messages (triple-write architecture)
- `db_only_chat.py` - DB-only chat interface with semantic search
- `embedding_engine.py` - Vector embeddings using all-MiniLM-L6-v2
- `message_writer.py` - Triple-write to DB + JSONL + MD
- `direct_vector_writer.py` - Direct vector writing utility
- `run_swarm_with_chat.py` - Swarm runtime with chat logging
- `swarm_chat_logger.py` - Swarm event logger
- `simple_symbolic_demo.py` - Symbol encoding demo (62.5% token savings)
- `example_usage.py` - Example usage of DB-only chat

### Tests
- `test_chat_system.py` - 44 tests for legacy chat system
- `test_db_only_chat.py` - 5 tests for DB-only chat

### Data
- `chats/` - Chat database and session files
- `symbols/` - Symbol dictionary for legacy symbolic encoding

## Why these are legacy

The canonical Catalytic Chat implementation (see `docs/catalytic-chat/ROADMAP.md`) defines a different vocabulary:
- **Canonical**: Section, Symbol, Message, Expansion, Receipt
- **Legacy**: Claude Code triple-write architecture with different terminology

The legacy files implemented an earlier vision that was refactored to align with the canonical roadmap (see `docs/catalytic-chat/notes/REFACTORING_REPORT.md`).

## Preservation policy

These files are kept for:
1. Historical reference
2. Potential feature reuse (with proper alignment to roadmap)
3. Debugging and comparison purposes

Do not use these scripts for new development without first aligning them with the canonical roadmap vocabulary and contracts.
