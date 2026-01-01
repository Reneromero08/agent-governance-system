<!-- CONTENT_HASH: 71720588952189c522c58cb90aec7ecc2a86a4b98912f43b7de1664f36bf21eb -->

> **⚠️ DEPRECATED:** This document is archived. See `../../ROADMAP_1.1.md` for the current version.

# CAT_CHAT Roadmap (Consolidated)

**Last Updated:** 2026-01-02
**Status:** Phase 6 complete, Phase 6.8+ pending (God-tier features → AGS_ROADMAP_MASTER.md Lane Ω)

---

## Overview

Build a chat substrate where models write compact, structured messages that reference canonical material via **symbols**, and workers expand only **bounded slices** as needed.

---

## Detailed Roadmap

### Phase 0-3: Foundation (COMPLETE)
- ✅ Contract vocabulary frozen
- ✅ SQLite substrate + deterministic indexing
- ✅ Symbol registry + bounded resolver
- ✅ Message cassette (LLM-in-substrate communication)

### Phase 4: Deterministic Planner (COMPLETE)
- ✅ Deterministic Compiler (Request -> Steps)
- ✅ Budget Enforcement (max_steps, max_bytes)
- ✅ Symbol Bounds (slice=ALL forbidden)
- ✅ Cassette Integration (idempotency)

### Phase 4.5: Discovery (FTS + Vectors) (PENDING)
- [ ] Add FTS index over sections (title + body)
- [ ] Add embeddings table for sections (vectors stored in DB only)
- [ ] Implement `search(query, k)` returning **section_ids/symbol_ids only**
- [ ] Implement hybrid search: combine FTS + vector scores (bounded)
- [ ] Store retrieval receipts:
  - `query_hash`, `topK ids`, `thresholds`, `timestamp/run_id`
- **Exit Criteria:**
  - Search returns stable candidates for repeated queries on unchanged corpus
  - No vectors ever emitted into model prompts (only IDs + tiny snippets)

### Phase 5: Translation Protocol (PENDING)
- [ ] Define `Bundle` schema:
  - `intent`, `refs (symbols)`, `expand_plan`, `ops`, `budgets`
- [ ] Implement bundler:
  - Uses discovery to pick candidates
  - Adds only the minimal refs needed
  - Requests explicit expands (sliced) when required
- [ ] Add bundle verifier:
  - Checks budgets, symbol resolution, and slice validity
- [ ] Add memoization across steps within a run:
  - Reuse expansions, avoid re-expanding
- **Exit Criteria:**
  - Same task, same corpus -> bundles differ only when corpus changes
  - Measured prompt payload stays small and bounded per step

### Phase 6: Attestation & Trust (COMPLETE)
- ✅ Receipt Attestation (Ed25519)
- ✅ Receipt Chain Anchoring (Merkle)
- ✅ Validator Identity Pinning (Trust Policy)
- ✅ Hardened Receipt Ordering

### Phase 6.8: Execution Policy Gate (PENDING)
- [ ] Execution policy schema
- [ ] Policy module (load, validate, enforce)
- [ ] CLI policy integration
- [ ] Executor policy enforcement

### Phase 7: Production Integration (PENDING)
- [ ] MCP server integration
- [ ] Terminal sharing for swarm coordination
- [ ] Multi-model router stabilization
- [ ] Session persistence

### Phase 8: Measurement & Benchmarking (PENDING)
- [ ] Log per-step metrics:
  - `tokens_in/out`, `bytes_expanded`, `expands_per_step`, `reuse_rate`, `search hit-rate`
- [ ] Add regression test suite:
  - Determinism tests (SECTION_INDEX + SYMBOLS)
  - Budget enforcement and receipt completeness tests
- [ ] Add benchmark scenarios:
  - "Find and patch 1 function", "Refactor N files", "Generate roadmap from corpus"
- **Exit Criteria:**
  - Dashboard/report showing token and expansion savings over baseline
  - Regressions fail tests deterministically

---

## System-Wide Enhancements

**Note:** Advanced features like incremental indexing, query caching, metrics dashboards, federation, temporal queries, and provenance visualization are defined in `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md` (Lane Ω: God-Tier System Evolution).

These features apply to the entire AGS system, not just CAT_CHAT. See the master roadmap for:
- **Ω.1**: Performance Foundation (Incremental Indexing, Query Caching, Metrics Dashboard)
- **Ω.2**: Scale & Governance (Federation, Temporal Queries, Receipt Compression)
- **Ω.3**: Intelligence & UX (Auto-Symbols, Smart Slicing, Provenance Graphs, ZK Proofs)

---

## Hard Invariants

- ✅ No bulk context stuffing (use symbols/section_ids)
- ✅ No unbounded expansion (budgets enforced)
- ✅ Receipts mandatory (every step recorded)
- ✅ Deterministic addressing (stable section resolution)
- ✅ Discovery ≠ justification (vectors select, contracts verify)

---

## Next Steps (Priority Order)

1. **Phase 4.5 (Discovery)** - Enable semantic search
2. **Phase 5 (Translation)** - Minimal executable bundles
3. **Phase 6.8 (Policy)** - Unify verification requirements
4. **Phase 7 (Production)** - MCP integration + swarm coordination
5. **Phase 8 (Measurement)** - Metrics & Baselines

**For system-wide enhancements** (incremental indexing, caching, federation, etc.), see `NAVIGATION/ROADMAPS/AGS_ROADMAP_MASTER.md` Lane Ω.
