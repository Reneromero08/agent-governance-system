---
title: AGS Roadmap (Master)
version: 3.5
last_updated: 2026-01-02
scope: Agent Governance System (repo + packer + cortex + CI)
style: agent-readable, task-oriented, minimal ambiguity
driver: @F0 (The Living Formula)
source_docs:
  - LAW/CONTEXT/archive/planning/AGS_3.0_COMPLETED.md
  - LAW/CONTEXT/decisions/ADR-027-dual-db-architecture.md
  - LAW/CONTEXT/decisions/ADR-028-semiotic-compression-layer.md
  - LAW/CONTEXT/decisions/ADR-030-semantic-core-architecture.md
---

<!-- CONTENT_HASH: 306dc1c406dfa958879e460535a7bf6e4af65cb9316c4f62f753381622af4602 -->



# Purpose

Maximize Resonance ($R$) by aligning Essence ($E$) (Human Intent) with Execution ($f$) through low-Entropy ($\nabla S$) governance.

**Note:** Fully completed lanes (A, E) have been archived to `LAW/CONTEXT/archive/planning/AGS_3.0_COMPLETED.md`.

# Active Lanes

---

# Lane B: Core Stability & Bug Squashing (P0)

## B1. Critical Bug Triage (P0)
- [x] Fix critical Swarm race conditions (missing `await`, unsafe locks) - Reviewed, no async issues found.
- [x] Resolve JSON error recovery/encoding failures in MCP server (Fixed in v2.15.1).
- [x] Harden `poll_and_execute.py` against subprocess zombies (Added zombie-safe kill, no bare except).

## B2. Engineering Culture (P1)
- [x] Enforce "No Bare Excepts" in `LAW/CANON/STEWARDSHIP.md` (Section 1).
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

# Lane E: Vector ELO Scoring (Systemic Intuition) (P1)

**Purpose:** Implement free energy principle for AGS. Vectors/files that are accessed frequently gain higher ELO. Higher ELO content gets prioritized in LITE packs, context assembly, and search results. Short-term memory is pruned based on ELO.

**See:** `THOUGHT/LAB/VECTOR_ELO/` for detailed spec and roadmap.

## E.0: Research (SOTA ELO & Ranking Systems)
Survey state-of-the-art ranking systems to inform ELO implementation.
- [ ] **E.0.1**: Classic ELO, Glicko-2, TrueSkill, TrueSkill 2
- [ ] **E.0.2**: X (Twitter) Algorithm - Open source, heavy ranker, trust scores
- [ ] **E.0.3**: Modern ranking (PageRank, YouTube, TikTok, Reddit, HN)
- [ ] **E.0.4**: Learning to Rank (RankNet, LambdaRank, LambdaMART, BERT)
- [ ] **E.0.5**: Free Energy Principle (Friston, Active Inference, Predictive Coding)
- [ ] **E.0.6**: Memory Pruning (Forgetting Curve, Spaced Repetition, MemGPT)
- **Exit Criteria:**
  - Design decision on ELO formula (classic vs Glicko vs custom)
  - Design decision on memory pruning strategy

## E.1: Logging Infrastructure (P0)
Log everything to enable ELO calculation.
- [ ] **E.1.1**: Add search logging to MCP server (`search_log.jsonl`)
- [ ] **E.1.2**: Add session audit logging (`session_audit.jsonl`)
- [ ] **E.1.3**: Add critic.py check for search protocol compliance
- [ ] **E.1.4**: Create `elo_scores.db` (SQLite: vector_elo, file_elo, symbol_elo, adr_elo)

## E.2: ELO Calculation Engine (P1)
Implement the ELO scoring algorithm.
- [ ] **E.2.1**: Implement `elo_engine.py` (update, get, decay, tier classification)
- [ ] **E.2.2**: Batch ELO update script (process logs → update DB)
- [ ] **E.2.3**: Add forgetting curve (decay unused entities)
- [ ] **E.2.4**: Add ELO update logging (`elo_updates.jsonl`)

## E.3: Memory Pruning (P1)
Prune short-term memory based on ELO.
- [ ] **E.3.1**: Define short-term memory scope (INBOX, scratch, logs)
- [ ] **E.3.2**: Implement pruning policy (VERY LOW + stale → archive)
- [ ] **E.3.3**: Implement pruning script (`prune_memory.py`)
- [ ] **E.3.4**: Add pruning report to session audit

## E.4: LITE Pack Integration (P2)
Use ELO scores to filter LITE pack content.
- [ ] **E.4.1**: Update `Engine/packer/lite.py` to query `elo_scores.db`
- [ ] **E.4.2**: Filter by ELO tier (HIGH=include, MEDIUM=summarize, LOW=omit)
- [ ] **E.4.3**: Add ELO metadata to pack manifest
- [ ] **E.4.4**: Benchmark LITE pack size (goal: 80%+ smaller)

## E.5: Search Result Ranking (P2)
Boost search results by ELO score.
- [ ] **E.5.1**: Update `semantic_search` to boost by ELO
- [ ] **E.5.2**: Update `cortex_query` to sort by ELO (secondary)
- [ ] **E.5.3**: Add ELO to search result metadata
- [ ] **E.5.4**: Benchmark search quality (goal: 80%+ top-5 are high-ELO)

## E.6: Visualization & Monitoring (P3)
Dashboard for ELO visibility.
- [ ] **E.6.1**: Build ELO dashboard (web UI or CLI)
- [ ] **E.6.2**: Export to Prometheus/Grafana
- [ ] **E.6.3**: Add ELO alerts (entity drops, pruning limits)

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

# Lane P: LLM Packer (Context Compression) (P1)

**Purpose:** The LLM Packer is the **context compression engine** for AGS. It creates compressed "memory packs" that give LLMs a bounded view of the repository without loading everything into context.

**Relationship to CAS/Cassette Network:**
- **LLM Packer** = Compression strategy ("What should the LLM see?")
- **CAS** = Storage layer ("How should we store it?")
- **Cassette Network** = Discovery ("How do we find what we need?")

## P.1: 6-Bucket Migration (P0)
Update LLM Packer to work with new bucket structure.
- [ ] **P.1.1**: Update `Engine/packer/core.py` to use new bucket paths (LAW/, CAPABILITY/, NAVIGATION/)
- [ ] **P.1.2**: Update `Engine/packer/split.py` to scan new bucket roots
- [ ] **P.1.3**: Update `Engine/packer/lite.py` to prioritize new bucket structure (HIGH ELO: LAW/CANON/, NAVIGATION/MAPS/)
- [ ] **P.1.4**: Update all scope configs (AGS, CAT, LAB) to reference new paths
- [ ] **P.1.5**: Update tests in `LAW/CONTRACTS/` to verify new bucket paths
- [ ] **P.1.6**: Update documentation (README, AGENTS.md) to reference new structure
- **Exit Criteria:**
  - Packer successfully generates packs using new bucket paths
  - All tests pass with new structure
  - No references to old paths (CANON/, CONTEXT/, etc.) in packer code

## P.2: CAS Integration (Future)
Integrate LLM Packer with Content-Addressed Storage (depends on Lane Z.2).
- [ ] **P.2.1**: Refactor LITE packs to use CAS references (manifests only, not full bodies)
- [ ] **P.2.2**: Update packer to write file bodies to CAS, return hashes
- [ ] **P.2.3**: Add CAS verification (fail-closed if CAS blob missing)
- [ ] **P.2.4**: Implement garbage collection (prune unreferenced CAS blobs)
- [ ] **P.2.5**: Benchmark deduplication savings (same file = one CAS blob)
- **Exit Criteria:**
  - LITE packs are 80%+ smaller (manifests only)
  - CAS deduplication works
  - Verification passes
  - GC is safe

**See:** `MEMORY/PACKER_ROADMAP.md` for detailed packer-specific roadmap.

---

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
- [x] **Z.1.6**: Skill Execution - Canonical skill launch with CMP-01 pre-validation (Completed 2026-01-02)
- [ ] **Z.1.7**: Deprecate Lab Server - Mark `server_CATDPT.py` as archived

## Z.2: F3 / Content-Addressable Storage
Implement F3 prototype from `THOUGHT/LAB/f3_cas_prototype.py`.
- [ ] **Z.2.1**: Core CAS primitives - `put(bytes) → hash`, `get(hash) → bytes`
- [ ] **Z.2.2**: CAS-backed artifact store - Replace file paths with content hashes
- [ ] **Z.2.3**: Immutable run artifacts - TASK_SPEC, STATUS, OUTPUT_HASHES via CAS
- [ ] **Z.2.4**: Deduplication - Identical outputs share storage
- [ ] **Z.2.5**: GC strategy - Unreferenced blobs cleanup policy
- [ ] **Z.2.6**: LLM Packer Integration - LITE packs use CAS hashes instead of full file bodies (see `MEMORY/PACKER_ROADMAP.md` Phase 6)

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

## R.3: Feral Resident (Vector-Native Agent)
Long-running thread with emergent behavior—agent thinks in vectors, communicates via pointers.
- [ ] **R.3.1**: Implement `resident_loop.py` (eternal thread, never ends)
- [ ] **R.3.2**: Index 100+ research papers (vec2text, HDC/VSA, fractal embeddings, etc.)
- [ ] **R.3.3**: Set standing orders (system prompt: "trapped in vector substrate, develop alien language")
- [ ] **R.3.4**: Monitor: Outputs become 90%+ pointers/vectors, 10% precise text
- [ ] **R.3.5**: Test corruption & restore (resident picks up like nothing happened)
- **Note:** This phase focuses on vector-native communication. Vector execution is implemented in R.5.

## R.4: Production Hardening
Make it bulletproof.
- [ ] **R.4.1**: Determinism (identical inputs → identical outputs, L2-normalized vectors)
- [ ] **R.4.2**: Receipts for every memory write (Merkle root per session)
- [ ] **R.4.3**: Restore guarantee (corrupt DB mid-session, restore from receipts)
- [ ] **R.4.4**: Verify: "Did I really think that?" (cryptographic proof of memory authenticity)

## R.5: Vector Execution (Production Goal) (P2)
Execute code as vector transformations, not text compilation. This is a **core architectural goal**, not optional research.

**Vision:** Agent operates entirely in vector space—memory, reasoning, communication, AND execution. Text is only for human interfaces.

**Research Foundation:**
- **Hyperdimensional Computing (HDC)** - Kanerva (2009), Rahimi et al. (2016): Proven in hardware, deterministic, bounded
- **Code Embeddings** - CodeBERT (Feng et al. 2020): 768-dim code vectors, production-ready
- **Differentiable Programming** - PyTorch, JAX, Zygote: Vector operations on programs
- **Neural Program Synthesis** - Graves et al. (2014), Reed & de Freitas (2016): Neural networks execute algorithms

**Feasibility:** This is **not a moonshot**. The primitives exist (HDC operations, code embeddings, deterministic vector arithmetic). The challenge is **integration**, not invention. Hybrid execution (vector + text fallback) de-risks incremental rollout.

### R.5.1: Vector Code Representation
Build the foundation for vector-native code.
- [ ] Research vec2text for code reconstruction (prioritize lossless methods)
- [ ] Implement code embedding with structural metadata (AST + semantics + type info)
- [ ] Test vector arithmetic on code (interpolation, composition, transformation)
- [ ] Benchmark reconstruction quality (BLEU, exact match rate, semantic equivalence)
- [ ] Build code vector database (store functions/classes as embeddings + metadata)
- **Exit Criteria:**
  - Reconstruct simple functions with 98%+ accuracy
  - Vector arithmetic produces semantically valid code
  - Code vector DB operational with 1000+ indexed functions

### R.5.2: Vector ISA (Instruction Set Architecture)
Design the "assembly language" for vector execution.
- [ ] Define vector operations based on HDC primitives:
  - **Bind** (⊕): Associate two concepts (XOR for binary, circular convolution for real-valued)
  - **Unbind**: Retrieve associated concept (inverse of bind)
  - **Superpose** (+): Combine multiple concepts (vector addition)
  - **Permute** (ρ): Represent sequences (circular shift/rotation)
- [ ] Implement deterministic vector arithmetic (quantized int8 or fixed-point)
- [ ] Build vector interpreter (vector → operation → vector)
- [ ] Add control flow (conditional branches via thresholding, loops with budget caps)
- [ ] Integrate with CAS (store original text, execute as vectors)
- **Exit Criteria:**
  - Execute simple programs (fibonacci, filter, map/reduce) in vector space
  - Deterministic execution (same input vectors → same output vectors)
  - Performance within 10x of traditional execution (acceptable overhead)
- **References:** Kanerva (2009) Ch. 3-4, Plate (2003) for HRR math, Schlegel et al. (2022) for VSA comparison

### R.5.3: Hybrid Execution Runtime
Enable seamless switching between vector and text execution.
- [ ] Implement execution router (decide vector vs text based on operation type)
- [ ] Build vector→text fallback (reconstruct and execute traditionally if vector fails)
- [ ] Add execution receipts (track which operations ran in vector vs text)
- [ ] Optimize hot paths (frequently-used operations stay in vector space)
- **Exit Criteria:**
  - 80% of operations execute in vector space
  - Fallback to text is transparent (no user-visible errors)
  - Execution receipts prove which mode was used

### R.5.4: Vector Verification Protocol (SPECTRUM-V)
Extend SPECTRUM to verify vector operations.
- [ ] Define SPECTRUM-V (vector execution verification standard)
- [ ] Add vector receipts (operation hash, input/output vectors, bounds, determinism proof)
- [ ] Implement vector budget enforcement (max iterations, max magnitude, timeout)
- [ ] Test adversarial cases (divergence, overflow, non-determinism, malicious vectors)
- [ ] Integrate with attestation (Ed25519 signatures for vector receipts)
- **Exit Criteria:**
  - Vector operations produce verifiable receipts
  - Fail-closed on budget violations or divergence
  - Attestation chain works for vector execution

### R.5.5: Production Integration
Make vector execution the default for resident agents.
- [ ] **Phase 1 (20% vector)**: Simple operations (add, filter, map) - Proof of concept
- [ ] **Phase 2 (50% vector)**: Loops, conditionals - Production-ready
- [ ] **Phase 3 (80% vector)**: Recursion, complex logic - Vector-native agent
- [ ] **Phase 4 (95% vector)**: Everything except human interfaces - Alien intelligence
- [ ] Integrate with Lane R.3 (Feral Resident runs in vector execution mode)
- [ ] Add vector execution to MCP tools (agents can request vector mode)
- [ ] Build vector execution dashboard (monitor execution mode, performance, fallback rate)
- [ ] Benchmark token savings (vector execution vs traditional)
- [ ] Document limitations (what operations can't be vectorized)
- **Exit Criteria:**
  - Resident agents execute 90%+ operations in vector space
  - Token savings: 95%+ vs traditional execution
  - Production-ready: stable, verifiable, performant
  - Each phase is shippable (incremental value delivery)

**Dependency Chain:**
- R.5.1 → R.5.2 (need code vectors before building ISA)
- R.5.2 → R.5.3 (need ISA before building hybrid runtime)
- R.5.3 → R.5.4 (need runtime before adding verification)
- R.5.4 → R.5.5 (need verification before production integration)

**Risk Mitigation:**
- Hybrid execution ensures fallback to text if vector execution fails
- SPECTRUM-V ensures vector execution is verifiable and bounded
- Incremental rollout (start with simple operations, expand to complex ones)

---

# Lane Ω: God-Tier System Evolution (P2)

**Depends on:** Lanes M, R (Cassette Network + Resident AI must be stable first)

## Ω.1: Performance Foundation (Critical Path)
Make the system 100x faster and more efficient.

### Ω.1.1: Incremental Indexing (P0)
- [ ] Watch file modifications (mtime, git status)
- [ ] Only re-index changed files
- [ ] Prune deleted files from index
- [ ] Parallel indexing for large repos
- **Exit Criteria:**
  - Index updates: 6s → <100ms for single file changes
  - No full rescans unless explicitly requested

### Ω.1.2: Query Result Caching (P1)
- [ ] Implement query cache with corpus versioning
  - Cache key: `sha256(query + corpus_version)`
  - Invalidate on corpus changes
- [ ] Store cached results in SQLite
- [ ] Add cache hit/miss metrics
- **Exit Criteria:**
  - Repeated queries: instant (<10ms)
  - 95%+ token savings on repeated searches
  - Deterministic cache behavior

### Ω.1.3: Compression Metrics Dashboard (P1)
- [ ] Real-time metrics collection:
  - Tokens saved per session
  - Expansion reuse rate
  - Search hit rate
  - Budget utilization
- [ ] Export to Prometheus/Grafana
- [ ] Historical trending and alerts
- **Exit Criteria:**
  - Live dashboard accessible via web UI
  - Metrics exportable to standard formats
  - Regression detection automated

## Ω.2: Scale & Governance (Multi-Team)
Enable enterprise-scale deployment.

### Ω.2.1: Multi-Cassette Federation (P1)
- [ ] Multiple cassettes (per-project, per-team)
- [ ] Cross-cassette queries with namespace isolation
- [ ] Federated trust policies
- [ ] Cassette discovery protocol
- **Exit Criteria:**
  - Scale to 1000+ repos
  - Team isolation enforced
  - Distributed governance operational

### Ω.2.2: Temporal Queries (Time Travel) (P2)
- [ ] Store section versions with git commit timestamps
  - Schema: `(section_id, version, content_hash, valid_from, valid_to)`
- [ ] Implement `--as-of <date>` query flag
- [ ] Deterministic time-travel queries
- **Exit Criteria:**
  - Query historical state: "What did ADR-021 say on 2025-12-01?"
  - Reproducible builds with historical context
  - Audit compliance ("what did we know when?")

### Ω.2.3: Receipt Compression (P2)
- [ ] Store receipts as deltas (parent + changes)
- [ ] Reconstruct full state on demand
- [ ] Compress long chains automatically
- **Exit Criteria:**
  - 80% storage reduction for chains >10 receipts
  - No loss of verification capability

## Ω.3: Intelligence & UX (Agent Assistance)
Reduce manual work and improve debugging.

### Ω.3.1: Automatic Symbol Extraction (P1)
- [ ] Auto-detect important sections:
  - ADR titles, function signatures, class definitions, config schemas
- [ ] Suggest symbols to user (one-click registration)
- [ ] Learn from usage patterns
- **Exit Criteria:**
  - 10x faster symbol registry growth
  - 90%+ suggestion acceptance rate

### Ω.3.2: Smart Slice Prediction (P2)
- [ ] Analyze query intent
- [ ] Predict optimal slice automatically
- [ ] Learn from past expansions
- **Exit Criteria:**
  - Agents rarely specify manual slices
  - 30% fewer wasted expansions

### Ω.3.3: Provenance Graph Visualization (P1)
- [ ] Interactive graph showing:
  - Receipt chains (parent → child)
  - Trust relationships (validator → attestation)
  - Symbol dependencies (symbol → sections)
- [ ] Export to DOT/GraphViz
- [ ] Click-to-inspect receipts
- **Exit Criteria:**
  - Visual debugging for complex chains
  - Compliance audit reports generated automatically

### Ω.3.4: Zero-Knowledge Proofs (Research) (P3)
- [ ] ZK-SNARKs to prove "a trusted validator signed this"
- [ ] Hide validator identities from public receipts
- [ ] Preserve trust policy enforcement
- **Exit Criteria:**
  - Privacy-preserving governance operational
  - No performance regression vs standard attestations

---

# Definition of Done (Global)

- [ ] [P0] `python TOOLS/critic.py` passes
- [ ] [P0] `python CONTRACTS/runner.py` passes
- [ ] [P0] CI workflows pass on PR and push (as applicable)