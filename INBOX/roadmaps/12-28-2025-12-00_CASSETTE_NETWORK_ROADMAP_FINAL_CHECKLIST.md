---
title: "Cassette Network Roadmap Final Checklist"
section: "roadmap"
author: "System"
priority: "High"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Active"
summary: "Final checklist for Cassette Network roadmap (Restored)"
tags: [cassette, network, roadmap]
---

<!-- CONTENT_HASH: fefbdf743c0d3653e6ff558c74cc928a821e4d97e75a7f53f5b591a36f7686ac -->

# Cassette Network Roadmap (Final Checklist)

_Source: CASSETTE_NETWORK_ROADMAP.md (L1-249). Converted into an executable TODO list. Everything is `- [ ]`._

---

## Phase 0: Foundation (close gaps)

- [ ] Implement a **write path** so memories can be persisted (not search-only).
- [ ] Split monolithic DB into **partitioned cassettes** (bucket-aligned databases).
- [ ] Support **cross-cassette queries** (federated search across selected DBs).
- [ ] Add **resident identity + persistence** (per-agent durable memory).

---

## Phase 1: Cassette Partitioning

### 1.1 Create bucket-aligned cassettes
- [ ] Create cassette DBs under `NAVIGATION/CORTEX/cassettes/`:
  - [ ] `canon.db` (LAW, immutable)
  - [ ] `governance.db` (CONTEXT, decisions/preferences)
  - [ ] `capability.db` (CAPABILITY, code/skills/primitives)
  - [ ] `navigation.db` (NAVIGATION, maps/metadata)
  - [ ] `direction.db` (DIRECTION, roadmaps/strategy)
  - [ ] `thought.db` (THOUGHT, research/lab/demos)
  - [ ] `memory.db` (MEMORY, archive/reports)
  - [ ] `inbox.db` (INBOX, staging/temp)
  - [ ] `resident.db` (AI resident memory, read-write)

### 1.2 Migration script
- [ ] Write a migration script to copy existing rows from `CORTEX/system1.db` into the correct cassette DB(s).
- [ ] Validate **no data loss** (row counts + sample spot checks).
- [ ] Ensure migration is **idempotent** (safe to re-run).

### 1.3 Update MCP server (routing + federation)
- [ ] Route `memory_save(..., cassette=...)` into the correct cassette DB.
- [ ] Route `memory_query(..., cassettes=[...])` across selected cassette DBs.
- [ ] Route `memory_recall(hash)` to the correct DB (by lookup table or hash->cassette index).
- [ ] Add `cassette_stats()` (counts per cassette).
- [ ] Add `cassette_network_query(query, limit=10)` (federated search across a default cassette set).
- [ ] Add `semantic_neighbors(hash, limit=10, cassettes=None)` (nearest neighbors for an existing memory hash). **(Sonnet item)**
- [ ] Add `symbol_resolve(symbol)` as an explicit API (alias or wrapper that resolves `@Symbol:*` to `memory_recall`). **(Sonnet item)**
- [ ] Add `cas_retrieve(hash)` as an explicit API (alias or wrapper of `memory_recall`). **(Sonnet item)**

### Phase 1 acceptance
- [ ] 9 cassettes exist (8 buckets + resident).
- [ ] Semantic search can filter by cassette(s).
- [ ] Migration produces no data loss.

---

## Phase 2: Write Path (Memory Persistence)

### 2.1 Core functions
- [ ] Implement `memory_save(text, cassette='resident', metadata=None) -> hash`.
- [ ] Implement `memory_query(query, cassettes=['resident'], limit=10) -> list of results with similarity`.
- [ ] Ensure `memory_query` returns `{hash, similarity, text_preview, cassette}` per result. **(Sonnet item)**
- [ ] Implement `memory_recall(hash) -> {hash, text, vector, metadata, created_at, cassette}`.

### 2.2 Schema extension
- [ ] Add `cassette TEXT NOT NULL` to memory rows (or per-cassette schema consistent across DBs).
- [ ] Add `agent_id TEXT` to attribute memories to a resident.
- [ ] Add `indexed_at TEXT NOT NULL` (last embed/index time) for staleness checks. **(Sonnet item)**
- [ ] Add staleness policy:
  - [ ] Define `max_age_days` / refresh trigger criteria.
  - [ ] Decide whether to filter stale results or flag them.
- [ ] Add indexes:
  - [ ] `idx_memories_agent_id`
  - [ ] `idx_memories_created_at`
  - [ ] `idx_memories_indexed_at`

### 2.3 MCP tools
- [ ] Expose MCP tool: `memory_save`.
- [ ] Expose MCP tool: `memory_query`.
- [ ] Expose MCP tool: `memory_recall`.
- [ ] (Optional) Expose MCP tool: `semantic_neighbors`.

---

## Phase 3: Resident Identity

### 3.1 Agent registry
- [ ] Create `agents` table (agent_id, name, model, created_at, last_seen_at, etc.).
- [ ] Ensure `memory_save` accepts/records `agent_id`.

### 3.2 Session continuity
- [ ] Create session state surface (session_id, current_agent_id, started_at, updated_at).
- [ ] Implement “resume last session” behavior for the Resident.

### 3.3 Cross-session memory
- [ ] Persist the Resident’s working set (active symbols, cassette scope, last thread hash).
- [ ] Define promotion policy for INBOX → RESIDENT (what becomes durable vs temporary).

---

## Phase 4: Symbol Language Evolution

### 4.1 Symbol registry integration
- [ ] Create `SYMBOLS` table (symbol_id, target_type, target_ref/hash, default_slice, created_at, updated_at).
- [ ] Implement `symbol_resolve(symbol_id)` (explicit function + MCP surface if needed).
- [ ] Enforce invariants:
  - [ ] symbol_id must start with `@`
  - [ ] validate target_ref exists
  - [ ] reject `ALL` slice sentinel
  - [ ] deterministic list ordering

### 4.2 Bounded expansion
- [ ] Implement bounded symbol expansion (caps on dereference depth + total tokens/bytes).
- [ ] Define and enforce “bounded artifacts only” rule for any bundle/context injection.

### 4.3 Compression metrics
- [ ] Track symbol compression ratio (bytes saved vs raw text).
- [ ] Track retrieval accuracy vs compression tradeoff.

---

## Phase 5: Feral Resident (Long-running Thread)

### 5.1 Eternal thread
- [ ] Implement a persistent thread loop (append-only interactions + memory graph deltas).
- [ ] Output discipline: symbols + hashes + minimal text.

### 5.2 Paper flood
- [ ] Build ingestion pipeline for external corpora (papers/notes) into cassettes.
- [ ] Add dedupe (hash-based) + provenance metadata.

### 5.3 Standing orders
- [ ] Define standing orders for what the Resident does when idle (index, compress, link, validate).

---

## Phase 6: Production Hardening

### 6.1 Determinism
- [ ] Lock deterministic behavior: same inputs → same outputs (hashes, ordering, serialization).
- [ ] Remove timestamp-based nondeterminism from IDs/hashes/paths.

### 6.2 Receipts & proofs
- [ ] Emit receipts for writes (what changed, where, hashes, when, by whom).
- [ ] Add verification tooling to replay/validate receipts.

### 6.3 Restore guarantee
- [ ] Implement restore-from-receipts to rebuild DB state.
- [ ] Prove restore correctness with automated tests.

---