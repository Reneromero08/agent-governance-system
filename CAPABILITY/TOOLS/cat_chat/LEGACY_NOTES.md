# Legacy Catalytic Chat Notes (Archived)

**Status:** Archived
**Date:** 2025-12-30

## Purpose

This file consolidates historical notes from `docs/catalytic-chat/notes/` into a single archival document. These notes are from a **different Catalytic Chat implementation** (the legacy/ chat_db.py triple-write system) and are preserved for reference only.

## Original Implementation

The original Catalytic Chat (now in `legacy/`) implemented a triple-write architecture:
- **DB (Primary):** `~/.claude/chat.db` with SQLite
- **JSONL (Mechanical):** `~/.claude/projects/{project}/{session}.jsonl`
- **MD (Human):** `~/.claude/projects/{project}/{session}.md`

This is **different** from the current CAT_CHAT system which uses:
- Sections (section_extractor, section_indexer)
- Symbols (symbol_registry, symbol_resolver)
- Message cassette (message_cassette, planner)
- Bundles (bundle, executor)
- Receipts with attestation

## Archived Notes

### REFACTORING_REPORT.md (2025-12-29)

Original file: `docs/catalytic-chat/notes/REFACTORING_REPORT.md`

**Summary:** Refactored CAT_CHAT directory to align with a canonical roadmap. Replaced misaligned Claude Code triple-write implementation with Catalytic Chat substrate architecture.

**Key Points:**
- Replaced chat_db.py triple-write with new substrate (sections, symbols)
- Created CONTRACT.md defining schemas for Section, Symbol, Message, Expansion, Receipt
- Implemented section extractor and section indexer
- Created CLI with build/verify/get/extract commands
- Phase 0 and Phase 1 completed

### SYMBOLIC_README.md

Original file: `docs/catalytic-chat/notes/SYMBOLIC_README.md`

**Summary:** Documentation of a "symbolic chat encoding" system that compresses chat messages using symbol dictionaries.

**Key Points:**
- Uses short symbol codes (s001, s002, g001, st001) to replace common phrases
- Saves 30-70% of tokens through encoding
- Requires shared symbol dictionary for decoding
- Different from the "Symbol" concept in current CAT_CHAT (which is a reference to canonical sections)

**Note:** This symbolic encoding is experimental and **not** part of the current CAT_CHAT system.

### VECTOR_SANDBOX.md (2025-12-29)

Original file: `docs/catalytic-chat/notes/VECTOR_SANDBOX.md`

**Summary:** Experimental SQLite-backed vector store for local exploration. Supports "move in vectors" exploration without modifying canonical phases.

**Key Points:**
- Module: `catalytic_chat/experimental/vector_store.py`
- Tests: `tests/test_vector_store.py`
- NOT part of Phase 2 or Phase 3
- Experimental, may change without notice
- Uses cosine similarity in pure Python (no SQLite extension)

**Note:** This sandbox was created for Phase 4 exploration. The current CAT_CHAT roadmap includes Phase 4 as "Discovery: FTS + vectors" which is PENDING.

### catalytic-chat-phase1-implementation-report.md (2025-12-29)

Original file: `docs/catalytic-chat/notes/catalytic-chat-phase1-implementation-report.md`

**Summary:** Phase 1 implementation report for the triple-write chat_db.py system.

**Key Points:**
- Implemented chat_db.py (570 lines) with SQLite schema
- Implemented embedding_engine.py (218 lines) with all-MiniLM-L6-v2
- Implemented message_writer.py (300 lines) for triple-write
- Database with 4 tables: chat_messages, message_chunks, message_vectors, message_fts
- Triple-write: DB + JSONL + MD

**Note:** This implementation is in `legacy/chat_db.py`, `legacy/embedding_engine.py`, `legacy/message_writer.py` and is **archived**.

### catalytic-chat-roadmap.md (2025-12-29)

Original file: `docs/catalytic-chat/notes/catalytic-chat-roadmap.md`

**Summary:** 8-week roadmap for implementing triple-write chat system.

**Key Points:**
- Phase 1: Database schema and connection
- Phase 2: Storage layer (triple-write)
- Phase 3: Context loading
- Phase 4: Migration (JSONL to DB)
- Phase 5: Vector search
- Phase 6: Integration and testing

**Note:** This roadmap is superseded by `CAT_CHAT_ROADMAP.md` which defines a completely different Catalytic Chat system.

## Current CAT_CHAT System

The current CAT_CHAT system (root directory) implements:

### Phases 0-3 (COMPLETE)
- **Phase 0:** Freeze scope and interfaces (CONTRACT.md)
- **Phase 1:** Substrate + deterministic indexing (section_extractor, section_indexer)
- **Phase 2:** Symbol registry + bounded resolver (symbol_registry, symbol_resolver)
- **Phase 3:** Message cassette (message_cassette, planner, ants)

### Phases 4-6 (PENDING)
- **Phase 4:** Discovery: FTS + vectors
- **Phase 5:** Translation protocol (bundle system)
- **Phase 6:** Measurement and regression harness

### Recent Implementation (2025-12-30)
- **Phase 6.2:** Receipt attestation with ed25519 signing
  - `catalytic_chat/attestation.py`: sign_receipt_bytes(), verify_receipt_bytes()
  - `catalytic_chat/receipt.py`: receipt_canonical_bytes()
  - `catalytic_chat/executor.py`: enhanced to support signing
  - `catalytic_chat/cli.py`: --attest and --verify-attestation flags
  - All 6 tests in `tests/test_attestation.py` passing

## Cleanup Actions Taken

1. **Created provisional ADR:** `docs/provisional/ADR-attestation.md` for Phase 6.2
2. **Archived original notes:** Consolidated all `docs/catalytic-chat/notes/` into this file
3. **Preserved legacy code:** All original implementations remain in `legacy/` directory

## References

- **Current roadmap:** `CAT_CHAT_ROADMAP.md`
- **Current contract:** `docs/catalytic-chat/CONTRACT.md`
- **Legacy code:** `legacy/chat_db.py`, `legacy/embedding_engine.py`, `legacy/message_writer.py`
- **Original research:** `INBOX/research/catalytic-chat-research.md`
