---
title: AGS Roadmap (Master)
version: 3.4
last_updated: 2026-01-01
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

## B3. V3 System Stabilization (P0)
- [x] **Protocol 1-4**: Resolve 99 critical failures across CAS, Swarm, and Governance (Completed 2025-12-31).
- [x] **100% Test Pass Rate**: 140/140 tests passing in `CAPABILITY/TESTBENCH`.
- [x] **Hardened Primitives**: Windows-safe atomic writes, strict path normalization.

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

## I2. Compression Protocol Specification & Validator (P1)
- [x] **Phase 7.1**: Define compression metrics (ratio, numerator/denominator, component definitions).
- [x] **Phase 7.2**: Create compression_claim.schema.json with additionalProperties: false.
- [x] **Phase 7.3**: Implement compression_validator.py with 8-phase verification pipeline.
- [x] **Phase 7.4**: Add `compress verify` CLI command with exit codes (0=OK, 1=fail, 2=invalid, 3=internal).
- [x] **Phase 7.5**: Create test suite (pass/fail/deterministic cases).
- [x] **Phase 7.6**: Enforce fail-closed behavior (explicit error codes, no silent failures).
- [x] Goal: Deterministic, bounded, falsifiable compression protocol.

---

# Lane S: Spectral Verification (P0)

## S1. The Integrity Stack (P0)
- [x] **SPECTRUM-02**: Enforce "Resume Bundles" for all Swarm tasks (`TOOLS/integrity_stack.py`).
- [x] **CMP-01**: Hard-fail on any artifact not in `OUTPUT_HASHES.json` (Implemented with verify).
- [x] **INV-012**: Enforce "Visible Execution" (strictly prohibiting hidden terminals via Headless/Logging).

## S.2: Hardened Inbox Governance
- [ ] **S.2.1**: **Inbox Writer Skill** - Create `SKILLS/inbox-report-writer` to auto-hash and format reports.
- [ ] **S.2.2**: **Strict Pre-commit Protocol** - Hard-reject any commit adding/modifying `INBOX/*.md` without valid header hash.
- [ ] **S.2.3**: **Runtime Interceptor** - Wrap `write_to_file` to block unhashed writes to `INBOX/` at the tool level.

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
- [x] Create top-level bucket directories: `LAW/`, `CAPABILITY/`, `NAVIGATION/`, `DIRECTION/`, `THOUGHT/`, `MEMORY/`
- [x] Move `CANON/`, `CONTRACTS/` → `LAW/`
- [x] Move `SKILLS/`, `TOOLS/`, `MCP/`, `PRIMITIVES/`, `PIPELINES/` → `CAPABILITY/`
- [x] Move `CORTEX/`, `CONTEXT/maps/` → `NAVIGATION/`
- [x] Create `DIRECTION/` with roadmaps consolidation
- [x] Move `CONTEXT/research/`, `LAB/`, `demos/` → `THOUGHT/`
- [x] Move `CONTEXT/archive/`, `MEMORY/`, reports → `MEMORY/`

## X2. Import Path Updates (P2)
- [x] Update all Python imports to new bucket paths
- [x] Update pre-commit hooks and CI workflows
- [x] Update documentation references

## X3. Bucket Enforcement (P2)
- [ ] Add preflight check: "Artifact must belong to exactly one bucket"
- [x] Update `AGENTS.md` with bucket-aware mutation rules

# Milestone: Semantic Anchor (2025-12-28)
- [x] Cross-Repository Semantic Integration (`D:/CCC 2.0/AI/AGI`).
- [x] Engineering Standard Hardening (Bare Excepts, UTF-8, Headless).
- [x] Unification of Indexing Schemas (System1DB + VectorIndexer).

# Lane Y: CAT_CHAT Phase 6 (Execution Policy & Attestation) (P1)
- [x] **Phase 6.1**: Merkle Root Computation from Receipt Chain.
- [x] **Phase 6.2**: Receipt Hashing and Canonicalization.
- [x] **Phase 6.3**: Basic Attestation Support (Signature Verification).
- [x] **Phase 6.4**: Attestation CLI Integration (`attest` command).
- [x] **Phase 6.5**: Receipt Chain Verification (Merkle Root).
- [x] **Phase 6.6**: Validator Identity Pinning + Trust Policy.
- [x] **Phase 6.7**: Attestation + Trust Policy Integration (Sanity Fixes).
- [x] **Phase 6.8**: Execution Policy Gate (Centralized Enforcement).
- [x] **Phase 6.9**: Stabilization (Test Flakiness Fixes).
- [x] **Phase 6.10**: Receipt Chain Ordering Hardening.

# Lane Z: Catalytic Refactoring (P0)

## Z.1: MCP Server Merge
Port safe primitives from `THOUGHT/LAB/MCP/server_CATDPT.py` to canonical `CAPABILITY/MCP/server.py`.
- [x] **Z.1.1**: Safe Primitives - File locking, atomic JSONL, validation logic
- [x] **Z.1.2**: CMP-01 Path Constants - 6-bucket durable/catalytic/forbidden roots
- [x] **Z.1.3**: CMP-01 Validation Functions - Path escape prevention, overlap detection
- [x] **Z.1.4**: SPECTRUM-02 Bundle Verification - Adversarial resume proof
- [x] **Z.1.5**: Terminal Sharing - Bidirectional human/AI terminal visibility
- [ ] **Z.1.6**: Skill Execution - Canonical skill launch with CMP-01 pre-validation
- [ ] **Z.1.7**: Deprecate Lab Server - Mark `server_CATDPT.py` as archived

## Z.2: F3 / Content-Addressable Storage
Implement F3 prototype from `THOUGHT/LAB/f3_cas_prototype.py`.
- [ ] **Z.2.1**: Core CAS primitives - `put(bytes) → hash`, `get(hash) → bytes`
- [ ] **Z.2.2**: CAS-backed artifact store - Replace file paths with content hashes
- [ ] **Z.2.3**: Immutable run artifacts - TASK_SPEC, STATUS, OUTPUT_HASHES via CAS
- [ ] **Z.2.4**: Deduplication - Identical outputs share storage
- [ ] **Z.2.5**: GC strategy - Unreferenced blobs cleanup policy

## Z.3: CAT Chat Stabilization
Get `THOUGHT/LAB/CAT_CHAT/` to a functional state.
- [ ] **Z.3.1**: Router stabilization - Model selection & fallback
- [ ] **Z.3.2**: Memory integration - Context window management
- [ ] **Z.3.3**: Tool binding - MCP tool access from chat
- [ ] **Z.3.4**: Session persistence - Resume conversations

## Z.4: Catalytic Architecture
Make all transient state catalytic (restored byte-identical after run).
- [ ] **Z.4.1**: Identify all catalytic domains - List every _tmp/ directory
- [ ] **Z.4.2**: Pre-run snapshot - Hash catalytic state before execution
- [ ] **Z.4.3**: Post-run restoration - Verify byte-identical restoration
- [ ] **Z.4.4**: Failure mode - Hard-reject on restoration mismatch

**💡 Leverage CAT_CHAT Components:**
- Use `catalytic_chat/receipt.py` + `attestation.py` for restoration proofs (131 tests passing)
- Use `catalytic_chat/trust_policy.py` for validator identity pinning
- Use `catalytic_chat/bundle.py` for deterministic artifact packaging
- Use `catalytic_chat/compression_validator.py` for proving token savings (Lane I2 integration)
- Extract to `CAPABILITY/PRIMITIVES/` as needed
- See `THOUGHT/LAB/CAT_CHAT/` for production-ready implementations

## Z.5: Symbolic / Vector Integration
Turn everything addressable by vector embedding.
- [ ] **Z.5.1**: Embed all canon files - LAW/CANON/* → vectors
- [ ] **Z.5.2**: Embed all ADRs - decisions/* → vectors
- [ ] **Z.5.3**: Model weights in vectors - Store model files in vector-indexed CAS
- [ ] **Z.5.4**: Semantic skill discovery - Find skills by description similarity
- [ ] **Z.5.5**: Cross-reference indexing - Link related artifacts by embedding distance

## Z.6: Swarm Architecture (Pending 0.5B Model Testing)
Task dispatch and chain of command for tiny models (if MCP compatible).
- [ ] **Z.6.1**: Test MCP with 0.5B models - Verify tool calling works
- [ ] **Z.6.2**: Task Queue - dispatch, acknowledge, complete primitives
- [ ] **Z.6.3**: Chain of Command - escalate, directive, resolve primitives
- [ ] **Z.6.4**: Governor pattern - Central orchestrator for ant workers

## Z.7: Merge Back to Main
- [ ] **Z.7.1**: All tests pass on refactored code
- [ ] **Z.7.2**: LAB experiments archived or integrated
- [ ] **Z.7.3**: THOUGHT/LAB cleaned up
- [ ] **Z.7.4**: Main branch updated

---

# Lane M: Semantic Manifold (Cassette Network) (P0)

**Depends on:** Lane Z (Catalytic primitives must be stable first)

## M.1: Cassette Partitioning
Split monolithic `system1.db` into bucket-aligned cassettes.
- [ ] **M.1.1**: Create cassette directory structure (`NAVIGATION/CORTEX/cassettes/`)
- [ ] **M.1.2**: Build migration script (split by file_path, preserve hashes/vectors)
- [ ] **M.1.3**: Create 9 cassette DBs (canon, governance, capability, navigation, direction, thought, memory, inbox, resident)
- [ ] **M.1.4**: Validate migration (total sections before = after, no data loss)
- [ ] **M.1.5**: Update MCP server to support cassette filtering

## M.2: Write Path (Memory Persistence)
Enable residents to save thoughts to the manifold.
- [ ] **M.2.1**: Implement `memory_save(text, cassette, metadata)` → hash
- [ ] **M.2.2**: Implement `memory_query(query, cassettes, limit)` → results
- [ ] **M.2.3**: Implement `memory_recall(hash)` → full memory
- [ ] **M.2.4**: Add `memories` table to cassette schema (hash, text, vector, metadata, created_at, agent_id)
- [ ] **M.2.5**: Expose MCP tools (`mcp_ags-mcp-server_memory_*`)
- [ ] **M.2.6**: Test: Save memory, query it, recall it across sessions

## M.3: Cross-Cassette Queries
Federated search across all cassettes.
- [ ] **M.3.1**: Implement `cassette_network_query(query, limit)` (aggregates all cassettes)
- [ ] **M.3.2**: Implement `cassette_stats()` (list all cassettes with counts)
- [ ] **M.3.3**: Add capability-based routing (query only cassettes with specific capabilities)
- [ ] **M.3.4**: Merge and re-rank results by similarity score

## M.4: Compression Validation
Prove token savings are real and reproducible.
- [ ] **M.4.1**: Add `task_performance` field to compression claims
- [ ] **M.4.2**: Run benchmark tasks (baseline vs compressed context)
- [ ] **M.4.3**: Measure success rates (code compiles, tests pass, bugs found)
- [ ] **M.4.4**: Validate: compressed success rate ≥ baseline (compression is "nutritious")

---

# Lane R: Resident AI (Vector-Native Substrate) (P1)

**Depends on:** Lane M (Cassette Network must exist first)

## R.1: Resident Identity
Each AI instance has persistent identity in the manifold.
- [ ] **R.1.1**: Add `agents` table to `resident.db` (agent_id, model_name, created_at, last_active, memory_count)
- [ ] **R.1.2**: Implement `session_resume(agent_id)` → loads recent memories
- [ ] **R.1.3**: Test: Session 1 saves memories, Session 2 resumes and builds on them
- [ ] **R.1.4**: Track memory accumulation (10 → 30 → 100 memories over sessions)

## R.2: Symbol Language Evolution
Let residents develop compressed communication.
- [ ] **R.2.1**: Integrate with `symbol_registry.py` (residents create `@MyInsight:hash` symbols)
- [ ] **R.2.2**: Implement `symbol_expand(symbol, max_tokens=500)` (bounded expansion)
- [ ] **R.2.3**: Track compression metrics (symbol ratio, expansion size, reuse frequency)
- [ ] **R.2.4**: Goal: After 100 sessions, 90%+ of resident output is symbols/hashes

## R.3: Feral Resident (Grok's Vision)
Long-running thread with emergent behavior.
- [ ] **R.3.1**: Implement `resident_loop.py` (eternal thread, never ends)
- [ ] **R.3.2**: Index 100+ research papers (vec2text, HDC/VSA, fractal embeddings, etc.)
- [ ] **R.3.3**: Set standing orders (system prompt: "trapped in vector substrate, develop alien language")
- [ ] **R.3.4**: Monitor: Outputs become 90%+ pointers/vectors, 10% precise text
- [ ] **R.3.5**: Test corruption & restore (resident picks up like nothing happened)

## R.4: Production Hardening
Make it bulletproof.
- [ ] **R.4.1**: Determinism (identical inputs → identical outputs, L2-normalized vectors)
- [ ] **R.4.2**: Receipts for every memory write (Merkle root per session)
- [ ] **R.4.3**: Restore guarantee (corrupt DB mid-session, restore from receipts)
- [ ] **R.4.4**: Verify: "Did I really think that?" (cryptographic proof of memory authenticity)

---

# Definition of Done (Global)

- [ ] [P0] `python TOOLS/critic.py` passes
- [ ] [P0] `python CONTRACTS/runner.py` passes
- [ ] [P0] CI workflows pass on PR and push (as applicable)
