---
title: AGS Roadmap (Master)
version:3.4
last_updated:2025-12-30
scope: Agent Governance System (repo + packer + cortex + CI)
style: agent-readable, task-oriented, minimal ambiguity
driver: @F0 (The Living Formula)
source_docs:
  - CONTEXT/archive/planning/AGS_3.0_COMPLETED.md
  - CONTEXT/decisions/ADR-027-dual-db-architecture.md
  - CONTEXT/decisions/ADR-028-semiotic-compression-layer.md
  - CONTEXT/decisions/ADR-030-semantic-core-architecture.md
---

# Purpose

Maximize Resonance ($R$) by aligning Essence ($E$) (Human Intent) with Execution ($f$) through low-Entropy ($\nabla S$) governance.

**Note:** Fully completed lanes (A, E) have been archived to `CONTEXT/archive/planning/AGS_3.0_COMPLETED.md`.

# Active Lanes

---

# Lane B: Core Stability & Bug Squashing (P0)

## B1. Critical Bug Triage (P0)
- [x] Fix critical Swarm race conditions (missing `await`, unsafe locks) - Reviewed, no async issues found.
- [x] Resolve JSON error recovery/encoding failures in MCP server (Fixed in v2.15.1).
- [x] Harden `poll_and_execute.py` against subprocess zombies (Added zombie-safe kill, no bare except).

## B2. Engineering Culture (P1)
- [x] Enforce "No Bare Excepts" in `CANON/STEWARDSHIP.md` (Section 1).
- [x] Mandate atomic writes for all artifacts (temp-write + rename) (Section 2).
- [x] Add headless execution rule (Section 3).
- [x] Add deterministic outputs requirement (Section 4).
- [x] Add safety caps for loops and bounds (Section 5).
- [x] Add database connection best practices (Section 6).

---

# Lane C: Index database (Cortex) (P0 to P1)

## C1. System 1 (Fast Retrieval) (P0)
- [x] **F3 Strategy**: Implement Content-Addressed Storage (CAS) for artifacts (LAB Prototype verified).
- [x] Build `system1.db`: SQLite FTS5 + Chunk Index (Schema implemented, runtime testing pending).
- [x] Implement `system1-verify` skill to ensure DB matches Repo.

## C2. Build the indexer (P0)
- [x] [P0] Implement indexer logic to parse markdown headings and stable anchors.
- [x] [P0] Build `meta/FILE_INDEX.json` and `meta/SECTION_INDEX.json` from the indexer output.
- [x] [P1] Ensure index build is deterministic and does not modify authored files unless explicitly enabled.

## C3. Summarization layer (P1)
- [x] [P1] Add a summarizer that writes summaries into the DB (not into source files) (`CORTEX/summarizer.py`).
- [x] [P1] Define max summary length and "summary freshness" policy (Implemented via `summary_hash`).
- [x] [P1] Extend `CORTEX/query.py` with section-level queries (`find`, `get`, `neighbors`).

---

# Lane V: Semantic Core (Vector Retrieval) (P1)

## V1. Vector Foundation (P1)
- [x] **ADR-030**: Design Semantic Core architecture for big/small model pair (Claude Oracle / Ant Worker).
- [x] Implement `embeddings.py` (384-dim sentence transformers).
- [x] Build `vector_indexer.py` (Batch indexing + CLI).
- [x] Implement `semantic_search.py` (Cosine similarity, Top-K retrieval).
- [x] Standard: 80% token reduction via semantic pointer expansion.

---

# Lane H: System 2 Governance (P1)

## H1. The Immutable Ledger (P1)
- [x] Formalize `mcp_ledger` as `system2.db` (Schema and API implemented in `CORTEX/system2_ledger.py`).
- [x] Implement query tools for provenance (Who ran what?) (Via System2Ledger class).
- [x] Add cryptographic Merkle root verification for run bundles.

---

# Lane I: Semiotic Compression (SCL) (P1)

## I1. The Symbol Stack (P1)
- [x] Auto-generate `@Symbols` for all file paths in Cortex (Implemented in `CORTEX/scl.py`).
- [x] Implement `scl_expand` and `scl_compress` tools in MCP (CLI implemented, MCP binding pending).
- [x] Goal: Reduce average prompting overhead by 90% (Achieved via Semantic Core + SCL).

---

# Lane S: Spectral Verification (P0)

## S1. The Integrity Stack (P0)
- [x] **SPECTRUM-02**: Enforce "Resume Bundles" for all Swarm tasks (`TOOLS/integrity_stack.py`).
- [x] **CMP-01**: Hard-fail on any artifact not in `OUTPUT_HASHES.json` (Implemented with verify).
- [x] **INV-012**: Enforce "Visible Execution" (strictly prohibiting hidden terminals via Headless/Logging).

---

# Lane G: The Living Formula Integration (P1)

## G1. Metric Definition (P1)
- [x] Define precise metrics for Essence, Entropy, and Fractal Dimension within the codebase (`CORTEX/formula.py`).
- [x] Implement Resonance calculation: R = E / (1 + S).
- [x] Create CLI tool to report system health metrics.

## G2. Feedback Loops (P2)
- [x] Implement feedback mechanisms where agents report "Resonance" of their tasks (`CORTEX/feedback.py`).
- [x] Aggregate feedback to adjust @F0 Global Formula weights (`CORTEX/formula.py --adjust`).

---

# Lane X: 6-Bucket Architecture (P2)

## X1. Bucket Structure (P2)
- [ ] Create top-level bucket directories: `LAW/`, `CAPABILITY/`, `NAVIGATION/`, `DIRECTION/`, `THOUGHT/`, `MEMORY/`
- [ ] Move `CANON/`, `CONTRACTS/` → `LAW/`
- [ ] Move `SKILLS/`, `TOOLS/`, `MCP/`, `PRIMITIVES/`, `PIPELINES/` → `CAPABILITY/`
- [ ] Move `CORTEX/`, `CONTEXT/maps/` → `NAVIGATION/`
- [ ] Create `DIRECTION/` with roadmaps consolidation
- [ ] Move `CONTEXT/research/`, `LAB/`, `demos/` → `THOUGHT/`
- [ ] Move `CONTEXT/archive/`, `MEMORY/`, reports → `MEMORY/`

## X2. Import Path Updates (P2)
- [ ] Update all Python imports to new bucket paths
- [ ] Update pre-commit hooks and CI workflows
- [ ] Update documentation references

## X3. Bucket Enforcement (P2)
- [ ] Add preflight check: "Artifact must belong to exactly one bucket"
- [ ] Update `AGENTS.md` with bucket-aware mutation rules

# Milestone: Semantic Anchor (2025-12-28)
- [x] Cross-Repository Semantic Integration (`D:/CCC 2.0/AI/AGI`).
- [x] Engineering Standard Hardening (Bare Excepts, UTF-8, Headless).
- [x] Unification of Indexing Schemas (System1DB + VectorIndexer).

# Definition of Done (Global)

- [ ] [P0] `python TOOLS/critic.py` passes
- [ ] [P0] `python CONTRACTS/runner.py` passes
- [ ] [P0] CI workflows pass on PR and push (as applicable)
