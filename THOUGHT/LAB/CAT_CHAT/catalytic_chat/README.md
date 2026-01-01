# Catalytic Chat Substrate

**Roadmap Phase:** Phase 2 Complete — Substrate + Symbol Registry + Bounded Resolver

## Purpose

Build persistent substrate and deterministic section index for canonical sources.

## Architecture

### Substrate Modes

**Primary: SQLite**
- Location: `CORTEX/db/system1.db` (shared with Cortex)
- Tables: `sections`, `section_index_meta`
- Advantages: Fast queries, transactions, FTS5 support

**Fallback: JSONL + Indexes**
- Location: `CAT_CORTEX/_generated/section_index.json`
- Format: JSON Lines (one Section per line)
- Advantages: Portable, no database dependency

**Implementation:** Supports both modes via substrate selector.

---

## Components

### `section_extractor.py` (Phase 1)

Extracts sections from canonical sources:
- Markdown headings → section ranges
- Code blocks → section ranges
- File-level sections

**Deterministic behavior:**
- Same file → same section boundaries
- Same content → same content_hash (SHA-256)

### `section_indexer.py` (Phase 1)

Builds and manages section index:
- Extracts sections from canonical sources
- Computes stable content_hash
- Emits SECTION_INDEX artifact
- Incremental rebuild (only changed files)

### `cli.py` (Phase 1)

Command-line interface:
- `python -m catalytic_chat.cli build` - Build index
- `python -m catalytic_chat.cli verify` - Verify determinism
- `python -m catalytic_chat.cli get <section_id>` - Fetch section

---

## Exit Criteria

- [x] Two consecutive builds on unchanged repo produce identical SECTION_INDEX (hash-stable)
- [ ] A section can be fetched by `section_id` with correct slice boundaries

---

## Next Phase

Phase 2 — Symbol registry + bounded resolver

---

## References

- Contract: `docs/catalytic-chat/CONTRACT.md`
- Roadmap: `docs/catalytic-chat/ROADMAP.md`
- Changelog: `docs/catalytic-chat/CHANGELOG.md`
- Phase Reports: `docs/catalytic-chat/phases/`
