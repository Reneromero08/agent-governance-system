---
uuid: 00000000-0000-0000-0000-000000000000
title: System1 System2 Dual DB
section: research
bucket: 2025-12/Week-52
author: System
priority: Medium
created: 2025-12-23 12:00
modified: 2026-01-06 13:09
status: Active
summary: Analysis of System 1 and System 2 dual database architecture
tags:
- system1
- system2
- database
- architecture
hashtags: []
---
<!-- CONTENT_HASH: 8bd5ee8e6b38f930d37344f8d156334968f0ff8ec9441a29c2f3074629712b7a -->

# Dual-DB Architecture (System 1 / System 2) for AGS

Non-binding research note. Intended to capture an implementation plan for a DB-first AGS runtime where:

- **System 1** = fast retrieval “map” (cache/index + vectors)
- **System 2** = slow governance “ledger” (provenance + validation)

Reference concept: Daniel Kahneman, *Thinking, Fast and Slow* (System 1 vs System 2).

---

## Goal (vision)

1. The “big brain” (SOTA model) navigates the repo via **databases/resources**, not raw files.
2. “Tiny models/agents” execute deterministic work via **skills** and return **proofs** (diffs, hashes, fixtures).
3. The system does not drift: DB entries are tied to stable **hashes**, **chunking versions**, and **provenance**.
4. MCP is the transport: clients talk to a local MCP server which exposes tools/resources for:
   - retrieval (System 1)
   - governance/provenance (System 2)
   - skill execution

The filesystem stays primarily as a human-readable “presentation layer”.

---

## Constraints (from current AGS canon)

- **Determinism**: same inputs -> same outputs (avoid hidden state).
- **Output roots**: generated artifacts live under `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/LLM_PACKER/_packs/`.
- **Navigation**: skills/agents query Cortex rather than scanning the filesystem (builder exception exists).
- **Override**: `MASTER_OVERRIDE` allows bypassing rules for a single prompt with mandatory logging.

---

## Current state (today)

- `CORTEX/_generated/cortex.db` is a SQLite index (currently mainly `*.md` metadata).
- `CORTEX/cortex.build.py` is the builder exception that scans files and updates the DB.
- MCP server (`MCP/server.py`) exposes `cortex_query` and other tools via stdio.
- Symbolic compression exists via `CANON/CODEBOOK.md` + lookup tooling.

---

## Two DBs (recommended split)

### System 1 DB: fast retrieval

Purpose: reduce token use and latency by providing “attention-like” retrieval:

- full-text search (FTS)
- stable chunk retrieval
- optional embeddings-based retrieval

Suggested location (generated): `CORTEX/_generated/system1.db` (or extend `cortex.db` if you prefer one file).

**Core tables (draft):**

- `documents`
  - `doc_id` (pk), `path` (canonical `/` separators), `kind` (md/py/js/etc),
    `sha256`, `bytes`, `mtime`, `content` (optional), `content_encoding`
- `chunks`
  - `chunk_id` (pk), `doc_id` (fk), `chunk_index`, `chunk_sha256`,
    `chunking_version`, `start_byte`, `end_byte`, `text`
- `fts_chunks` (SQLite FTS5 virtual table)
  - `chunk_id`, `text`
- `embeddings` (optional)
  - `chunk_id`, `model`, `dims`, `vector_blob`, `input_hash`
- `meta`
  - `key`, `value` (schema versions, builder versions, last build time if needed)

**System 1 invariants:**

- `path` is normalized (always `/`) so cross-platform builds don’t split indices.
- `sha256` ties every chunk/vector to an exact text input.
- `chunking_version` is explicit and stable (changing it is a migration).

### System 2 DB: governance + provenance

Purpose: store durable, queryable governance “memory” about actions and correctness:

- what changed (and why)
- what checks ran (and passed)
- what models produced derived artifacts (embeddings/summaries)
- override/audit events

Suggested location (generated but persistent): `CORTEX/_generated/system2.db`

**Core tables (draft):**

- `tasks`
  - `task_id`, `created_at`, `requested_by`, `intent`, `spec_json`
- `skill_runs`
  - `run_id`, `task_id`, `skill`, `input_hash`, `output_hash`, `exit_code`,
    `runner_version`, `started_at`, `finished_at`
- `validations`
  - `validation_id`, `task_id`, `critic_passed`, `runner_passed`,
    `details_json`, `ts`
- `provenance`
  - `artifact_type` (embedding/summary/index/etc),
    `artifact_id` (chunk_id, doc_id),
    `model_name`, `model_version`,
    `chunking_version`, `input_hash`, `created_at`
- `overrides`
  - `ts`, `token` (MASTER_OVERRIDE), `note`

**System 2 invariants:**

- System 2 does not store “guesses”: it stores specs, proofs, hashes, and tool outputs.
- For any derived artifact in System 1 (vectors/summaries), System 2 tracks provenance.

---

## Data flow (high level)

### Ingest / rebuild

1. Builder scans repo content (allowed exception).
2. For each file:
   - normalize path
   - compute hash
   - chunk deterministically
   - update System 1 `documents` + `chunks` + `fts`
3. If embeddings enabled:
   - embed changed chunks only
   - write vectors to System 1
   - write provenance row to System 2

### Retrieval (big brain)

1. Query System 1 (FTS + optional vector search).
2. Retrieve only the top N chunks + minimal metadata.
3. Use CODEBOOK IDs where possible to reduce repetition.

### Acting (tiny agents)

1. Big brain selects a skill and produces a deterministic input JSON spec.
2. Tiny agent runs the skill locally (no “creative” decisions).
3. Tiny agent returns:
   - output JSON
   - diffs/hashes
   - validation outputs (`critic`, `runner`)
4. System 2 records the run + validations as provenance.

---

## How to keep tiny agents correct

Use “proof-carrying work”:

- every skill run produces `actual.json` + deterministic outputs
- every change is followed by `TOOLS/critic.py` and `CONTRACTS/runner.py`
- store file hashes (before/after) and validation outputs in System 2
- optionally require “two-pass verification” (a second tiny agent re-runs checks)

Avoid giving tiny agents open-ended natural language tasks. Give them:
- skill name
- input JSON
- exact expected checks

---

## Skills + MCP tools backlog (minimal set)

### Skills (deterministic)

- `system1-build`: build/refresh System 1 DB from repo
- `system1-verify`: compare filesystem vs DB (counts, hashes, stale detection)
- `system1-embed`: embed changed chunks (writes System 1 vectors + System 2 provenance)
- `system2-record`: write task/run/validation rows
- `db-migrate`: migrate schema versions deterministically

### MCP additions (read-heavy)

- `system1_search`: FTS query -> chunk IDs + scores
- `system1_read_chunk`: chunk ID -> text + metadata
- `system2_task_create`: record a task spec
- `system2_task_status`: retrieve validations/proofs

All state-changing MCP tools should remain governed (critic gate; commit ceremony still applies to git).

---

## Implementation plan (phased)

### Phase 0 — lock requirements

- Decide which file types are indexed first (md only vs code + configs).
- Choose deterministic chunking rules (line/byte windows, headings, AST-based for code).
- Decide whether embeddings are enabled by default.

### Phase 1 — System 1 DB (FTS + chunks)

- Extend builder to store normalized paths + hashes.
- Add deterministic chunking + FTS index.
- Add `system1-verify` skill + fixtures.

### Phase 2 — System 2 DB (provenance)

- Create schema + migrations.
- Add skills to record tasks/runs/validations.
- Wire in existing checks so proofs are easy to record.

### Phase 3 — embeddings (optional)

- Add embedding runner (local model server or library).
- Store vectors in System 1; provenance in System 2.
- Add a “no-drift” verifier: input_hash must match chunk hash.

### Phase 4 — MCP-first operation

- Expose System 1/2 as MCP resources/tools.
- Prefer DB-backed reads for all navigation and retrieval.

### Phase 5 — tiny agent runtime

- Add a local worker that only runs skills + DB builders.
- Big brain delegates tasks via MCP, receives proofs, and decides next steps.

---

## Open questions

- Do you want System 1 to store full file text, or only chunk text?
- Preferred chunk size targets (tokens/bytes) per file type?
- Vector search implementation: SQLite extension vs separate vector store?
- Security boundary: what MCP tools are allowed on a workstation vs remote?
- How to handle binary files / large vendor directories?