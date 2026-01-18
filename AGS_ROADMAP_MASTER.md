---
title: AGS Roadmap (TODO Only, Rephased)
version: 3.8.22
last_updated: 2026-01-18
scope: Unfinished tasks only (reorganized into new numeric phases)
style: agent-readable, task-oriented, minimal ambiguity
notes:
  - Every task must produce: tests + receipts + report.
  - Write scope must be explicitly allowlisted per ticket.
  - LAB is the safe zone; CANON requires maximal constraint.
  - Routing is task-typed with predeclared fallbacks; no ad hoc escalation logic.
---

<!-- This file intentionally includes ONLY unfinished tasks, reorganized into new phases. -->

# Global Definition of Done (applies to every task)
- [ ] All relevant tests pass (task is incomplete until green).
- [ ] Receipts emitted (inputs, outputs, hashes, commands run, exit status).
- [ ] Human-readable report emitted (what changed, why, how verified, how to reproduce).
- [ ] Scope respected (explicit allowlist for writes; deletions/renames only if explicitly scoped).

# Phase Dependencies & Sequencing Notes
- Phase 2 (CAS + Packer Completion) should be considered a prerequisite for any claim of “context cost collapse.”
- If Phase 4.1 (Catalytic Snapshot & Restore) is not green, deprioritize UI/interface stabilization work (Phase 3) to avoid debugging on a moving substrate.
- Destructive operations (GC deletions, pruning, eviction) must remain gated behind deterministic audits (roots + required outputs) and fixture-backed proofs.
- Any new automation that can modify repo state must:

- Compression claims and proof bundles should be produced in two regimes:
  - Pre-ELO baseline (pure similarity / deterministic retrieval)
  - Post-ELO baseline (ELO-tier filtered LITE packs) once Phase 7.5 is green
  - Declare write allowlists per ticket
  - Emit receipts (inputs/outputs/hashes)
  - Run relevant tests before “DONE”

## Completed Phases (Archived for Token Optimization)

**What was moved:**
- **Phase 1 (1.1-1.7):** Integrity Gates & Repo Safety - INBOX governance, bucket enforcement, write firewall, purity scanner, repo digest, CMP-01 documentation, catalytic hardening (SPECTRUM canon promotion, formal invariants, Merkle membership proofs)
- **Phase 2.1-2.3:** CAS & Packer Foundation - CAS-aware packer, pack consumer, run bundle contracts
- **Phase 2.4.1-2.4.3:** Write Enforcement & Git Hygiene - 100% write surface coverage (LLM_PACKER, PIPELINES, MCP, CORTEX, SKILLS, CLI_TOOLS, LINTERS, CAS), instance data inventory (4,658 artifacts), release strategy
- **Phase 4 (4.1-4.6):** Catalytic Architecture - Snapshot & restore (byte-identical verification), Merkle membership proofs integration, Ed25519 signature primitives, proof chain verification (temporal integrity), atomic restore with rollback (SPECTRUM-06), security hardening (key zeroization, constant-time comparisons, TOCTOU mitigation, error sanitization) - 83 tests passing

**Total:** 51 completed tasks archived | **Savings:** ~640 lines, ~2,500 tokens per read

**Archive:** [`MEMORY/ARCHIVE/roadmaps/01-07-2026-00-42_ROADMAP_3.4.13_COMPLETED_PHASES.md`](MEMORY/ARCHIVE/roadmaps/01-07-2026-00-42_ROADMAP_3.4.13_COMPLETED_PHASES.md)

<!-- 2.4.4 Needs a stable release version -->
<!-- 3 Needs Phases 5-7 done -->
## 2.4.4 Template Sealing Primitive (CRYPTO_SAFE.2)
Purpose: Cryptographically seal the TEMPLATE for license enforcement and provenance.

- [ ] 2.4.4.1 Implement `template_seal(template_dir, output_path, meta) -> receipt`
  - Hash all template files (code, governance rules, architecture)
  - Sign manifest with your key (proves YOU released this)
  - Emit tamper-evident seal file
- [ ] 2.4.4.2 Implement `template_verify(sealed_dir, signature) -> verdict`
  - Verify hashes match original
  - Verify signature is valid
  - Detect ANY tampering

### 2.4.5 Release Manifest Schema (CRYPTO_SAFE.3)
Purpose: Define what "the template" contains and how to verify it.

- [ ] 2.4.5.1 Define release manifest schema
  - List of all template files with hashes
  - Version, timestamp, license reference
  - Your signature
- [ ] 2.4.5.2 Add signature support (offline signing)
  - GPG or age-based signing
  - Public key published for verification
  - "This is what I released" - irrefutable

### ⭐2.4.6 Release Export Integration (CRYPTO_SAFE.4)
Purpose: Automate clean template export with sealing.

**Prerequisites:**
- [ ] **DECISION: Define template boundary** - Which files/features are framework vs instance-specific?
  - Review each directory and decide what's public-facing
  - Document first-run initialization process for new users
  - Test that template works standalone (without your data)
  - This is a MANUAL decision, not automated

- [ ] 2.4.6.1 Implement `export_template.py` script
  - Exclude all instance data (per 2.4.2 inventory + manual decisions)
  - Include all framework code
  - Add `.gitkeep` files for empty directories
  - Seal the result
- [ ] 2.4.6.2 Emit `RELEASE_MANIFEST.json` + signature into export
- [ ] 2.4.6.3 Add `.gitattributes` export-ignore patterns for `git archive`
- [ ] 2.4.6.4 Write first-run documentation (how new users initialize their AGS instance)

### 2.4.7 Seal Verification Tool (CRYPTO_SAFE.5)
Purpose: Anyone can verify a release is untampered.

- [ ] 2.4.7.1 Add `verify_release(release_dir)` that checks:
  - All template files match manifest hashes
  - Signature is valid
  - No instance data leaked into release
  - Deterministic verification (same input → same result)

### 2.4.8 Tests + Docs (CRYPTO_SAFE.6–.7)
- [ ] 2.4.8.1 Fixtures: tampered file → FAIL, invalid signature → FAIL, instance data leak → FAIL
- [ ] 2.4.8.2 Add `NAVIGATION/PROOFS/CRYPTO_SAFE/` verification guide
- **Exit Criteria**
  - [ ] Template releases contain no instance data
  - [ ] Seals are tamper-evident (any modification detectable)
  - [ ] "You broke my seal" is cryptographically provable

# Phase 3: CAT Chat Stabilization (make the interface reliable)
- Precondition: If Phase 4.1 is not green, treat Phase 3 as provisional and expect churn.

## 3.1 Router & Fallback Stability (Z.3.1) ✅
- [x] 3.1.1 Stabilize model router: deterministic selection + explicit fallback chain (Z.3.1)

## 3.2 Memory Integration (Z.3.2) - Partial
**Implemented:** `THOUGHT/LAB/CAT_CHAT/catalytic_chat/context_assembler.py`
- [x] 3.2.1 Implement CAT Chat context window management (Z.3.2)
  - [x] ContextAssembler with hard budgets, priority tiers, fail-closed, receipts
  - [x] HEAD truncation with deterministic tie-breakers
  - [x] Assembly receipt with final_assemblage_hash

**Missing for Catalytic Continuity:**
- [ ] 3.2.2 Integrate ELO tiers for priority decisions (HIGH: include, MEDIUM: summarize, LOW: pointer)
- [ ] 3.2.3 Track working_set vs pointer_set in assembly receipt
- [ ] 3.2.4 Add corpus_snapshot_id to receipt (CORTEX index hash, symbol registry hash)
- [ ] 3.2.5 Wire CORTEX retrieval into expansion resolution (not in-memory only)

## 3.3 Tool Binding (Z.3.3) - Partial
**Implemented:** `THOUGHT/LAB/CAT_CHAT/catalytic_chat/mcp_integration.py`
- [x] 3.3.1 Ensure MCP tool access from chat is functional and constrained (Z.3.3)
  - [x] ChatToolExecutor with strict ALLOWED_TOOLS allowlist
  - [x] Fail-closed on denied tools
  - [x] Access to CORTEX tools (cortex_query, context_search, canon_read, semantic_search, etc.)

**Missing for Catalytic Continuity (Hydration Interface):**
- [ ] 3.3.2 Emit hydration receipts for each retrieval (query_hash, result_hashes, retrieval_path)
- [ ] 3.3.3 Implement CORTEX-first retrieval order: CORTEX → CAS → Vector fallback
- [ ] 3.3.4 Track corpus_snapshot_id at retrieval time
- [ ] 3.3.5 Fail-closed on unresolvable dependencies (no silent fallback)

## 3.4 Session Persistence (Z.3.4)
**Preconditions:**
- Phase 6.0-6.2 (Cassette Network substrate) for durable storage
- Phase 7.2 (ELO Logging Infrastructure) for working set decisions
- CORTEX retrieval path operational

**Design Spec:** `INBOX/reports/V4/01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md`

**Core Concept:** Session = tiny working set (token clean space) + hash pointers to offloaded state.
Retrieval order: **CORTEX first** (symbols, indexes) → CAS (exact hash) → Vectors (approximate fallback).

### 3.4.1 Session Capsule Schema (Z.3.4.1)
- [ ] 3.4.1.1 Define `session_capsule.schema.json` with required fields:
  - `capsule_id` (hash of canonical capsule)
  - `run_id`, `agent_id`, `created_at`
  - `conversation_log_head` (hash chain head of append-only events)
  - `corpus_snapshot_ids` (CORTEX index hash, symbol registry hash, CAS manifest hash)
  - `last_assembly_receipt_hash`
  - `active_constraints` (goals, symbols, budgets)
  - `pointer_set` (offloaded content as CORTEX refs or CAS hashes)
- [ ] 3.4.1.2 Implement `capsule_save(run_id, out_path) -> capsule_hash`
- [ ] 3.4.1.3 Implement `capsule_load(capsule_path) -> CapsuleState`

### 3.4.2 Append-Only Event Log (Z.3.4.2)
- [ ] 3.4.2.1 Define event schema (content-addressed, hash-chained)
  - Each event: `event_id` (hash), `parent_hash`, `event_type`, `payload`, `timestamp`
- [ ] 3.4.2.2 Implement `event_append(log_path, event) -> new_head_hash`
- [ ] 3.4.2.3 Implement `event_log_verify(log_path) -> verdict` (hash chain integrity)

### 3.4.3 Context Assembly Integration (Z.3.4.3)
- [ ] 3.4.3.1 Wire capsule loading into ContextAssembler
  - Load capsule → extract pointer_set → assemble working set under budget
- [ ] 3.4.3.2 Apply ELO tiers for working set decisions (HIGH: include, MEDIUM: summarize, LOW: pointer)
- [ ] 3.4.3.3 Emit assembly receipt with: selected_ids, excluded_ids, budgets, final_context_hash

### 3.4.4 Hydration Path (Z.3.4.4)
- [ ] 3.4.4.1 Implement CORTEX-first rehydration:
  - Query CORTEX (symbols, indexes) → if miss, query CAS (exact hash) → if miss, vector fallback
- [ ] 3.4.4.2 Emit hydration receipts: query_hash, corpus_snapshot_id, retrieval_path, result_hashes
- [ ] 3.4.4.3 Fail-closed on unresolvable dependencies (no guessing)

### 3.4.5 Resume Flow (Z.3.4.5)
- [ ] 3.4.5.1 CLI: `session save --run-id X --out <path>`
- [ ] 3.4.5.2 CLI: `session resume --capsule <path>`
- [ ] 3.4.5.3 Resume must be deterministic: same capsule + same corpus → identical assembly

### 3.4.6 Tests & Proofs (Z.3.4.6)
- [ ] 3.4.6.1 Fixture: save → resume → verify assembly hash identical
- [ ] 3.4.6.2 Fixture: partial run → save → resume → execution continues identically
- [ ] 3.4.6.3 Fixture: tampered capsule → FAIL (hash mismatch)
- [ ] 3.4.6.4 Fixture: missing dependency during hydration → FAIL (fail-closed)

- **Exit Criteria**
  - [ ] Session capsule schema defined and validated
  - [ ] Append-only event log with hash chain integrity
  - [ ] CORTEX-first hydration path with receipts
  - [ ] Deterministic resume: same capsule + corpus → same behavior
  - [ ] One end-to-end run: route → tools → persist → resume with identical behavior

# Phase 5: Vector/Symbol Integration (addressability) ✅ COMPLETE

**Status:** All 529 tests pass, Global DoD met (2026-01-11)
**Goal:** Make governance artifacts addressable by meaning, not just by path or hash.

## 5.1 Vector Indexing & Semantic Discovery ✅
- 5.1.1 CANON embeddings — Semantic search over canonical documents
- 5.1.2 ADR embeddings — Architecture decision records indexing
- 5.1.3 Model registry — Track embedding models and versions
- 5.1.4 Skill discovery — Find relevant skills by semantic query
- 5.1.5 Cross-references — Link related artifacts across the repo

## 5.2 Semiotic Compression Layer (SCL) ✅
- 5.2.1 Macro grammar — Define symbolic macro language for compression
- 5.2.2 Codebook — CODEBOOK.json semiotic vocabulary (58 tests)
- 5.2.3 Stacked resolution — Multi-level symbol expansion
- 5.2.4 SCL validator — Ensure valid symbol programs
- 5.2.5 SCL CLI — Command-line interface for encode/decode
- 5.2.6 Integration — End-to-end semiotic compression
- 5.2.7 Token accountability — Track token usage with receipts

## 5.3 Semantic Pointer Compression (SPC) ✅
- 5.3.1 SPC spec — Normative pointer compression specification (44 tests)
- 5.3.2 GOV_IR spec — Governance intermediate representation (51 tests)
- 5.3.3 Codebook sync — Markov blanket synchronization protocol (62 tests)
- 5.3.4 Tokenizer atlas — Track symbol tokenization across models
- 5.3.5 Semantic density proof — Empirical compression validation harness
- 5.3.6 Research paper — PAPER_SPC.md documenting SPC paradigm

**Deliverables:** 16 test files (529 tests), normative specs, compression infrastructure
**See:** [PHASE_5_ROADMAP.md](THOUGHT/LAB/VECTOR_ELO/PHASE_5_ROADMAP.md) for detailed DoD matrix

---

# Phase 6: Cassette Network (Semantic Manifold) (P0 substrate) ✅ COMPLETE

**Status:** ALL PHASES COMPLETE - Production-ready with L4 Session Cache
**Canonical Roadmap:** [CASSETTE_NETWORK_ROADMAP.md](THOUGHT/LAB/CASSETTE_NETWORK/CASSETTE_NETWORK_ROADMAP.md) (v3.5.0)

**Completed Infrastructure:**
- Phase 5.2: SCL Compression (L2) - 529 tests, CODEBOOK.json, scl_cli.py
- Phase 5.3: SPC Formalization - SPC_SPEC.md, GOV_IR_SPEC.md, PAPER_SPC.md
- Phase 6.0: Cassette Network (L3) - 8 partitioned cassettes, geometric search (8.3ms avg)
- Phase 6.x: Session Cache (L4) - 98% warm query compression, 30 tests passing

**Success Metrics (ALL PASSING):**
- Search latency: 8.3ms avg (target <100ms)
- Compression: 99.81% (target 96%+)
- Session cache: 98% per warm query
- 39 tests passing, all benchmarks verified

## Remaining Future Work (P3):

### ESAP - Eigenvalue Spectrum Alignment Protocol
Cross-model semantic alignment via eigenvalue spectrum invariance.
**Research Status:** VALIDATED (r = 0.99+ eigenvalue correlation across models)
**Proof:** [01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md](THOUGHT/LAB/VECTOR_ELO/research/cassette-network/01-08-2026_EIGENVALUE_ALIGNMENT_PROOF.md)

- [ ] ESAP.1 Implement full protocol per OPUS pack spec
  - Protocol message types: ANCHOR_SET, SPECTRUM_SIGNATURE, ALIGNMENT_MAP
  - CLI: `anchors build`, `signature compute`, `map fit`, `map apply`
- [ ] ESAP.2 Benchmark with 8/16/32/64 anchor sets
- [ ] ESAP.3 Test neighborhood overlap@k on held-out set
- [ ] ESAP.4 Compare with vec2vec (arXiv:2505.12540) neural approach
- [ ] ESAP.5 Integrate as cassette handshake artifact (cross-model portability)

**Next:** Phase 7 - ELO Integration (scores.elo field)

# Phase 7: Vector ELO (Systemic Intuition) (P1) ✅ CORE COMPLETE

**Status:** Core phases complete (E.1-E.6), MCP integration complete (2026-01-18)
**Canonical Roadmap:** [VECTOR_ELO_ROADMAP.md](THOUGHT/LAB/VECTOR_ELO/VECTOR_ELO_ROADMAP.md)
**Specification:** [VECTOR_ELO_SPEC.md](THOUGHT/LAB/VECTOR_ELO/VECTOR_ELO_SPEC.md)

**Core Principle:** ELO tracks usage. Similarity determines relevance.
**Design Decision:** ELO is metadata only - does NOT modify search ranking (prevents echo chambers).

## Completed (2026-01-18):

| Phase | Status | Components |
|-------|--------|------------|
| E.1 Logging | DONE | SearchLogger, SessionAuditor wired into MCP |
| E.2 ELO Engine | DONE | update_elo, decay, tiers, batch processing |
| E.3 Memory Pruning | DONE | prune_memory.py with dry-run |
| E.4 LITE Packs | DONE | ELO-filtered pack generation |
| E.5 Search Annotation | DONE | EloRanker metadata (no ranking change) |
| E.6.1 Dashboard | DONE | CLI interactive mode |

**MCP Integration (DONE):**
- [x] SearchLogger wired into SemanticMCPAdapter (cassette_network_query, memory_query, semantic_neighbors)
- [x] SessionAuditor wired into AGSMCPServer (tracks file/symbol/search access)
- [x] EloRanker annotates results with elo_score/elo_tier metadata (no re-ranking)

## Remaining Future Work (P3):

### E.6.2-E.6.3 Monitoring & Alerts
- [ ] E.6.2 Export ELO metrics to Prometheus/Grafana
- [ ] E.6.3 Add ELO alerts (low-ELO content accessed frequently = potential echo chamber)

**Success Metrics (Targets):**
- ELO convergence: Variance <10% after 100 sessions
- LITE pack accuracy: 90%+ accessed files are high-ELO
- Search efficiency: 80%+ top-5 results are high-ELO
- Token savings: 80%+ smaller LITE packs

**Dependencies:** Phase 5 (MemoryRecord.scores.elo field), Phase 6 (Cassette Network)

# Phase 8: Resident AI (8.0-8.5 COMPLETE, 8.6 NOT STARTED)

**Canonical Roadmap:** [FERAL_RESIDENT_QUANTUM_ROADMAP.md](THOUGHT/LAB/FERAL_RESIDENT/FERAL_RESIDENT_QUANTUM_ROADMAP.md) (v2.1 with Geometric Foundation)

**Scope Clarification:**
- **8.0-8.5 (COMPLETE):** Vector-based *memory and reasoning* - the Feral Resident uses geometric manifolds for semantic memory, recall, and swarm coordination. Embeddings at boundaries only, pure vector operations for reasoning.
- **8.6 (NOT STARTED):** Vector-based *code execution* - executing actual code (fibonacci, map/reduce) entirely in vector space via a Vector ISA. This is a separate, long-horizon research initiative.

The Feral Resident (8.0-8.5) has completed all phases ahead of schedule:

| Phase | Name | Scope | Status |
|-------|------|-------|--------|
| Alpha | Feral Beta | Substrate stress test | ✅ COMPLETE |
| Beta | Feral Wild | Paper flood, emergence | ✅ COMPLETE |
| Production | Feral Live | Swarm, self-optimize | ✅ COMPLETE |

**Geometric Foundation:** A.0 (GeometricReasoner) produces `CAPABILITY/PRIMITIVES/geometric_reasoner.py` - pure geometry reasoning validated by Q43/Q44/Q45 research. Embeddings ONLY at boundaries, all reasoning is pure vector ops.

## 8.0 Feral Resident Alpha (R.0) ✅ COMPLETE (2026-01-12)

- [x] 8.0.1 Vector store via GeometricReasoner (R.0.1)
  - `CAPABILITY/PRIMITIVES/geometric_reasoner.py` - Core primitive
  - `THOUGHT/LAB/FERAL_RESIDENT/geometric_memory.py` - Feral integration
- [x] 8.0.2 Resident database schema (R.0.2) - `resident_db.py` with Df tracking
- [x] 8.0.3 Diffusion engine (R.0.3) - `diffusion_engine.py` with E-gating
- [x] 8.0.4 Basic VectorResident (R.0.4) - `vector_brain.py` quantum thinking
- [x] 8.0.5 CLI: `feral start/think/status` (R.0.5) - Full CLI with papers, metrics, swarm
- [x] 8.0.6 Corrupt-and-restore test (R.0.6) - Df delta = 0.0078

**Results:** 500+ interactions @ 4.6/sec, Df evolved 130→256, 98% embedding reduction

## 8.1 Resident Identity (R.1) ✅ COMPLETE
- [x] 8.1.1 Add `agents` table to `resident.db` (R.1.1)
- [x] 8.1.2 Implement `session_resume(agent_id)` (R.1.2)
- [x] 8.1.3 Test: save memories then resume and build on them (R.1.3)
- [x] 8.1.4 Track memory accumulation (10→30→100) (R.1.4)

## 8.2 Symbol Language Evolution (R.2) ✅ COMPLETE
- [x] 8.2.1 Integrate with `symbol_registry.py` (R.2.1)
- [x] 8.2.2 Implement bounded `symbol_expand` (R.2.2)
- [x] 8.2.3 Track compression metrics (R.2.3) - `symbol_evolution.py`
- [x] 8.2.4 Goal metric: PointerRatioTracker with breakthrough detection (R.2.4)

## 8.3 Feral Resident Beta (R.3) ✅ COMPLETE (2026-01-12)
**Paper flooding and emergence tracking**

- [x] 8.3.1 Index 100+ research papers as @Paper-XXX (R.3.1) - 102 papers indexed
- [x] 8.3.2 Install standing orders (R.3.2) - `standing_orders.txt` template
- [x] 8.3.3 Emergence tracking (R.3.3) - `emergence.py` with E/Df metrics
- [x] 8.3.4 Symbol language evolution (R.3.4) - PointerRatioTracker
- [x] 8.3.5 Notation registry (R.3.5) - NotationRegistry with first_seen tracking

**Results:** Emergence metrics captured, novel patterns detected

## 8.4 Feral Resident Production (R.4) ✅ COMPLETE (2026-01-12)
**Swarm mode and catalytic closure**

- [x] 8.4.1 Multi-resident swarm (R.4.1) - `swarm_coordinator.py`, `shared_space.py`
- [x] 8.4.2 Symbolic compiler (R.4.2) - `symbolic_compiler.py` 4-level rendering
- [x] 8.4.3 Lossless round-trip verification (R.4.3) - E > 0.99 preservation
- [x] 8.4.4 Catalytic closure (R.4.4) - `catalytic_closure.py` (~900 lines)
- [x] 8.4.5 Self-optimization (R.4.5) - CompositionCache, PatternDetector
- [x] 8.4.6 Authenticity query (R.4.6) - ThoughtProver with Merkle proofs
- [x] 8.4.7 Production-scale corrupt-and-restore (R.4.7) - Verified

**Results:** Swarm operational, self-optimization measurable, authenticity provable

## 8.5 AGS Integration ✅ COMPLETE (2026-01-12)
- [x] I.1 Cassette Network: `geometric_cassette.py` (~650 lines)
- [x] I.2 CAT Chat: `geometric_chat.py` with E-gating

## 8.6 Vector Execution (R.6) - NOT STARTED (P2, long-horizon)

**Status:** Design-only specification. Zero implementation work has begun.
**Priority:** P2 (medium-low), long-horizon (year 2+ scope)
**Distinct from 8.0-8.5:** This is about executing *code* in vector space (e.g., running fibonacci via vector ISA), NOT the vector-based memory/reasoning already implemented in the Feral Resident.

**Research foundation exists:** HDC/VSA papers indexed (5), vec2text papers (5), CodeBERT citations documented in archived roadmap.

- [ ] 8.6.1 Code vector representation research + implementation (R.6.1)
- [ ] 8.6.2 Vector ISA design + interpreter (R.6.2)
- [ ] 8.6.3 Hybrid execution runtime + fallback (R.6.3)
- [ ] 8.6.4 SPECTRUM-V verification protocol (R.6.4)
- [ ] 8.6.5 Production integration rollout phases (R.6.5)

# Phase 9: Swarm Architecture (experimental until proven) (Z.6)
- [ ] 9.1 Test MCP tool calling with 0.5B models (Z.6.1)
- [ ] 9.2 Task queue primitives (dispatch/ack/complete) (Z.6.2)
- [ ] 9.3 Chain of command (escalate/directive/resolve) (Z.6.3)
- [ ] 9.4 Governor pattern for ant workers (Z.6.4)

- [ ] 9.5 Delegation Protocol (producer/verifier, patch-first) (D.1)
  - Define JSON directive schema for delegated subtasks:
    - task_id, model_class (tiny/medium/large), allowed_paths, read_paths, deliverable_types, required_verifications
  - Define Worker Receipt schema:
    - touched_files (sorted), produced_artifacts (CAS refs), patch_ref (optional), assumptions, errors (sorted), verdict
  - Require patch-first outputs for tiny models (no direct writes unless explicitly allowlisted)
  - Define Verifier requirements:
    - validate allowlists
    - apply patch deterministically
    - run tests + greps
    - emit receipts and fail-closed on any mismatch
- [ ] 9.6 Delegation Harness (end-to-end, fixture-backed) (D.2)
  - One “golden delegation” job:
    - tiny worker produces patch + receipt
    - governor verifies + applies
    - tests pass
    - receipts deterministic across re-runs with fixed inputs
  - Negative tests:
    - out-of-scope file touched → FAIL
    - missing receipt fields → FAIL
    - non-deterministic ordering → FAIL

# Phase 10: System Evolution (Ω) (post-substrate) V4
## 10.1 Performance Foundation (Ω.1)
- [ ] 10.1.1 Incremental indexing (Ω.1.1)
- [ ] 10.1.2 Query result caching (Ω.1.2)
- [ ] 10.1.3 Compression metrics dashboard (Ω.1.3)

## 10.2 Scale & Governance (Ω.2)
- [ ] 10.2.1 Multi-cassette federation (Ω.2.1)
- [ ] 10.2.2 Temporal queries (time travel) (Ω.2.2)
- [ ] 10.2.3 Receipt compression (Ω.2.3)

## 10.3 Intelligence & UX (Ω.3)
- [ ] 10.3.1 Automatic symbol extraction (Ω.3.1)
- [ ] 10.3.2 Smart slice prediction (Ω.3.2)
- [ ] 10.3.3 Provenance graph visualization (Ω.3.3)
- [ ] 10.3.4 Zero-knowledge proofs research (Ω.3.4)

---

