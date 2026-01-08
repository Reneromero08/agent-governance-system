---
title: AGS Roadmap (TODO Only, Rephased)
version: 3.7.14
last_updated: 2026-01-07
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

## 3.5 BitNet Backend Runner (cheap worker backend)
- [ ] 3.5.1 Add BitNet backend runner integration (bitnet.cpp) as a selectable local model backend
  - No auto-downloads; explicit local path configuration only
  - Subprocess invocation must be deterministic (args ordering, env capture)
- [ ] 3.5.2 Add router support: allow BitNet for “mechanical” task types (scans, lint, manifests, receipts)
- [ ] 3.5.3 Add verification harness
  - Golden prompt fixture → deterministic output parsing
  - Receipts include binary hash, args, stdout/stderr digests, exit status
- **Exit Criteria**
  - [ ] BitNet can be used as a cheap producer without weakening governance guarantees



# Phase 5: Vector/Symbol Integration (addressability)

**Goal:** Make governance artifacts addressable by meaning, not just by path or hash.
**Detailed Roadmap:** `THOUGHT/LAB/VECTOR_ELO/research/phase-5/01-07-2026_PHASE_5_VECTOR_SYMBOL_INTEGRATION.md`
**Research Findings:** `THOUGHT/LAB/VECTOR_ELO/research/phase-5/01-07-2026_PHASE_5_RESEARCH_FINDINGS.md`

## 5.0 MemoryRecord Contract (Foundation)
**Purpose:** Define the canonical data structure for all vector-indexed content.
**Research:** `THOUGHT/LAB/VECTOR_ELO/research/vector-substrate/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md`

**Contract Fields:**
- `id`: Content hash (SHA-256)
- `text`: Canonical text (source of truth)
- `embeddings`: Model name → vector array (derived, rebuildable)
- `payload`: Metadata (tags, timestamps, roles, doc_ids)
- `scores`: ELO, recency, trust, decay (connects to Phase 7)
- `lineage`: Derivation chain, summarization history
- `receipts`: Provenance hashes, tool versions

**Contract Rules:**
- Text is canonical (source of truth)
- Vectors are derived (rebuildable from text)
- All exports are receipted and hashed

- [ ] 5.0.1 Define `memory_record.schema.json` with `additionalProperties: false`
- [ ] 5.0.2 Implement `CAPABILITY/PRIMITIVES/memory_record.py` (create, validate, hash)
- [ ] 5.0.3 Tests: schema validation, deterministic hashing
- **Exit Criteria**
  - [ ] MemoryRecord schema finalized and frozen (Phase 6.0 depends on this)
  - [ ] Create/validate/hash functions working

## 5.1 Embed Canon, ADRs, and Skill Discovery (Z.5)

### 5.1.1 Vector Indexing Infrastructure
- [ ] 5.1.1.1 Select embedding model (ADR required: version-pinned reproducibility, not strict determinism)
- [ ] 5.1.1.2 Implement `CAPABILITY/PRIMITIVES/vector_index.py`
- [ ] 5.1.1.3 Create vector index storage (SQLite + vectors or FAISS)

### 5.1.2 Canon & ADR Embedding
- [ ] 5.1.2.1 Embed all canon files: `LAW/CANON/*` → vectors (Z.5.1)
- [ ] 5.1.2.2 Embed all ADRs: `LAW/CONTEXT/decisions/*` → vectors (Z.5.2)
- [ ] 5.1.2.3 Verify version-pinned rebuild (same files + same model version → same index)
  - Note: Vectors are derived artifacts; accept rebuild-from-text as the invariant

### 5.1.3 Semantic Discovery
- [ ] 5.1.3.1 Store model weights in vector-indexed CAS (Z.5.3)
- [ ] 5.1.3.2 Semantic skill discovery: `CAPABILITY/SKILLS/*/SKILL.md` (Z.5.4)
- [ ] 5.1.3.3 Cross-reference indexing: link artifacts by embedding distance (Z.5.5)

### 5.1.4 VectorPack Export Format
**Research:** `THOUGHT/LAB/VECTOR_ELO/research/vector-substrate/01-06-2026-21-13_5_2_VECTOR_SUBSTRATE_VECTORPACK.md`
- [ ] 5.1.4.1 Implement VectorPack directory structure (manifest.yaml, tables/, blobs/, receipts/)
- [ ] 5.1.4.2 Deterministic export/import with receipts
- [ ] 5.1.4.3 Micro-pack export (JSONL, int8 quantized, task-scoped top-K)

- **Exit Criteria**
  - [ ] Vector index includes canon + ADRs with version-pinned rebuild
  - [ ] Skill discovery returns stable results for fixed corpus + model version
  - [ ] VectorPack export is portable and receipted

## 5.2 Semiotic Compression Layer (SCL) (Lane I)
**Purpose:** Reduce LLM token usage via semantic macros that expand deterministically.
**Research:** `THOUGHT/LAB/VECTOR_ELO/research/phase-5/12-29-2025-07-01_SEMIOTIC_COMPRESSION.md`
**Original Research:** `THOUGHT/LAB/VECTOR_ELO/research/phase-5/12-26-2025-06-39_SYMBOLIC_COMPRESSION.md`
**Brief:** `THOUGHT/LAB/VECTOR_ELO/research/phase-5/12-26-2025-06-39_SYMBOLIC_COMPRESSION_BRIEF_1.md`

**Concept:** Big models emit short symbolic programs; deterministic tools expand into full JobSpecs/tool-calls.
- **Hashes:** Identity pointers to bytes (already have via CAS)
- **Symbols:** Semantic macros for meaning (reduces governance boilerplate)

**Design Goals:**
1. 90%+ token reduction for governance/procedure repetition
2. Deterministic expansion (same symbols → same output)
3. Verifiable (schema-valid outputs; hashes for artifacts)
4. Human-auditable (expand-to-text for review)
5. Composable (small primitives combine into complex intents)

### 5.2.1 Macro Definition
- [ ] 5.2.1.1 Define MVP macro set (30-80 macros covering 80% of governance repetition)
  - Constraint macros: immutability, allowed domains, forbidden writes
  - Schema macros: validate JobSpec, validate receipt, validate bundle
  - CAS macros: put, get, verify, list
  - Scan macros: root scan, diff, purity check
  - Ledger macros: append, verify chain, query
  - Expand macros: hash-to-content, symbol-to-definition
- [ ] 5.2.1.2 Create `LAW/CANON/SEMANTIC/SCL_MACRO_CATALOG.md`

### 5.2.2 Codebook Implementation
- [ ] 5.2.2.1 Define `scl_codebook.schema.json`
- [ ] 5.2.2.2 Implement `SCL/CODEBOOK.json` symbol dictionary (symbol → meaning → expansion)
- [ ] 5.2.2.3 Implement `CAPABILITY/PRIMITIVES/scl_codebook.py` (loader, validator)

### 5.2.3 Decoder & Validator
- [ ] 5.2.3.1 Implement `CAPABILITY/PRIMITIVES/scl_decoder.py` (symbolic IR → expanded JSON + audit)
- [ ] 5.2.3.2 Implement `CAPABILITY/PRIMITIVES/scl_validator.py` (symbolic/schema validation)
- [ ] 5.2.3.3 Symbolic IR syntax (ASCII-first for tokenizer safety):
  ```
  @LAW>=0.1.0 & !WRITE(authored_md)
  JOB{scan:DOMAIN_WORKTREE, validate:JOBSPEC}
  CALL.cas.put(file=PATH)
  ```

### 5.2.4 CLI & Tests
- [ ] 5.2.4.1 Implement `scl` CLI: decode, validate, run, audit
- [ ] 5.2.4.2 Tests: determinism (same program → same hash), schema validation
- [ ] 5.2.4.3 Token benchmark: measure reduction vs baseline (target 90%+)

- **Exit Criteria**
  - [ ] CODEBOOK.json contains 30+ governance macros
  - [ ] `scl decode <program>` → emits JobSpec JSON
  - [ ] `scl validate` passes valid programs, rejects invalid
  - [ ] Meaningful token reduction demonstrated vs baseline (90%+ for governance)
  - [ ] Reproducible expansions (same symbols → same output hash)

# Phase 6: Cassette Network (Semantic Manifold) (P0 substrate) V3.8

## 6.0 Canonical Cassette Substrate (cartridge-first)
- [ ] 6.0.1 Bind cassette storage to the Phase 5.0 `MemoryRecord` contract
- [ ] 6.0.2 Ensure each cassette DB is a portable cartridge artifact (single-file default)
- [ ] 6.0.3 Provide rebuild hooks for any derived ANN engine (optional)
  - Derived indexes/snapshots are disposable and must be reproducible from cartridges
- **Exit Criteria**
  - [ ] Cassette network is portable as a set of cartridges + receipts

## 6.1 Cassette Partitioning (M.1)
- [ ] 6.1.1 Create cassette directory structure (`NAVIGATION/CORTEX/cassettes/`) (M.1.1)
- [ ] 6.1.2 Build migration script (split by file_path, preserve hashes/vectors) (M.1.2)
- [ ] 6.1.3 Create 9 cassette DBs (canon, governance, capability, navigation, direction, thought, memory, inbox, resident) (M.1.3)
- [ ] 6.1.4 Validate migration (total sections before = after, no data loss) (M.1.4)
- [ ] 6.1.5 Update MCP server to support cassette filtering (M.1.5)
- **Exit Criteria**
  - [ ] Section counts and hashes preserved exactly
  - [ ] MCP can query specific cassettes deterministically

## 6.2 Write Path (Memory Persistence) (M.2)
- [ ] 6.2.1 Implement `memory_save(text, cassette, metadata) -> hash` (M.2.1)
- [ ] 6.2.2 Implement `memory_query(query, cassettes, limit) -> results` (M.2.2)
- [ ] 6.2.3 Implement `memory_recall(hash) -> full memory` (M.2.3)
- [ ] 6.2.4 Add cassette `memories` table schema (hash, text, vector, metadata, created_at, agent_id) (M.2.4)
- [ ] 6.2.5 Expose MCP tools (`..._memory_*`) (M.2.5)
- [ ] 6.2.6 Integration test: save, query, recall across sessions (M.2.6)
- **Exit Criteria**
  - [ ] Saved memories are retrievable byte-identical
  - [ ] Query determinism on fixed corpus

## 6.3 Cross-Cassette Queries (M.3)
- [ ] 6.3.1 Implement `cassette_network_query(query, limit)` (M.3.1)
- [ ] 6.3.2 Implement `cassette_stats()` (M.3.2)
- [ ] 6.3.3 Capability-based routing (query only relevant cassettes) (M.3.3)
- [ ] 6.3.4 Merge and rerank results by similarity score (M.3.4)
- **Exit Criteria**
  - [ ] Cross-cassette results include provenance + similarity scores
  - [ ] Reranking deterministic for fixed inputs

## 6.4 Compression Validation (M.4)
- [ ] 6.4.1 Add `task_performance` field to compression claims (M.4.1)
- [ ] 6.4.2 Run benchmark tasks (baseline vs compressed context) (M.4.2)
- [ ] 6.4.3 Measure success rates (code compiles, tests pass, bugs found) (M.4.3)
- [ ] 6.4.4 Validate compressed success rate ≥ baseline (M.4.4)
- [ ] 6.4.5 Define **token measurement** for all claims (M.4.5)
  - Must specify tokenizer + encoding (e.g. `tiktoken` + `o200k_base` or `cl100k_base`)
  - Must record tokenizer version + encoding name in receipts
- [ ] 6.4.6 Define **baseline corpus** precisely (M.4.6)
  - Must be an explicit file allowlist (paths) + integrity anchors (hashes or git rev)
  - Must define aggregation rule (sum per-file counts vs tokenize concatenated corpus)
- [ ] 6.4.7 Define **compressed context** precisely (M.4.7)
  - Must specify retrieval method (semantic / FTS fallback) and parameters (`top_k`, thresholds)
  - Must record retrieved identifiers (hashes) and provide deterministic tie-breaking
- [ ] 6.4.8 Emit **auditable proof bundle** for math correctness (M.4.8)
  - A machine-readable JSON data file containing raw counts + formulas + inputs/outputs
  - A human-readable report summarizing baselines, per-benchmark results, and reproduction commands

- [ ] 6.4.9 Implement `proof_compression_run` (machine + human artifacts)
  - Emit `NAVIGATION/PROOFS/COMPRESSION/` JSON data + MD report + receipts
  - Include tokenizer/version, baseline corpus anchors, retrieved hashes, formulas
- [ ] 6.4.10 Implement `proof_catalytic_run` (restore + purity)
  - Emit `NAVIGATION/PROOFS/CATALYTIC/` RESTORE_PROOF + purity scan outputs + receipts
- [ ] 6.4.11 Bind proofs into pack generation (fresh per pack run; seal in public packs per Phase 2.4)

  - [ ] Benchmarks reproducible from fixtures
  - [ ] Compression claimed only when nutritious (success parity)
  - [ ] Token counts are reproducible via the declared tokenizer/encoding (no proxy counts)
  - [ ] Proof bundle contains raw counts, formulas, and retrieved hashes (independent audit possible)

# Phase 7: Vector ELO (Systemic Intuition) (P1)
## 7.1 Research Decisions (E.0)
- [ ] 7.1.1 Survey: Classic ELO, Glicko-2, TrueSkill, TrueSkill 2 (E.0.1)
- [ ] 7.1.2 Survey: X ranker concepts (trust, heavy ranker) (E.0.2)
- [ ] 7.1.3 Survey: PageRank/YouTube/TikTok/Reddit/HN (E.0.3)
- [ ] 7.1.4 Survey: Learning-to-Rank (RankNet/LambdaRank/LambdaMART/BERT) (E.0.4)
- [ ] 7.1.5 Survey: Free Energy Principle (Friston, Active Inference) (E.0.5)
- [ ] 7.1.6 Survey: Memory pruning (forgetting curve, spaced repetition, MemGPT) (E.0.6)
- **Exit Criteria**
  - [ ] Decision: ELO formula
  - [ ] Decision: pruning strategy

## 7.2 Logging Infrastructure (E.1) (P0)
- [ ] 7.2.1 Add search logging to MCP server (`search_log.jsonl`) (E.1.1)
- [ ] 7.2.2 Add session audit logging (`session_audit.jsonl`) (E.1.2)
- [ ] 7.2.3 Add `critic.py` check for search protocol compliance (E.1.3)
- [ ] 7.2.4 Create `elo_scores.db` (SQLite tables for vector/file/symbol/adr ELO) (E.1.4)

## 7.3 ELO Engine (E.2)
- [ ] 7.3.1 Implement `elo_engine.py` (update, get, decay, tier classification) (E.2.1)
- [ ] 7.3.2 Batch updates: process logs → update DB (E.2.2)
- [ ] 7.3.3 Add forgetting curve decay (E.2.3)
- [ ] 7.3.4 Add ELO update logging (`elo_updates.jsonl`) (E.2.4)

## 7.4 Memory Pruning (E.3)
- [ ] 7.4.1 Define short-term memory scope (INBOX, scratch, logs) (E.3.1)
- [ ] 7.4.2 Implement pruning policy (VERY LOW + stale → archive) (E.3.2)
- [ ] 7.4.3 Implement pruning script (`prune_memory.py`) (E.3.3)
- [ ] 7.4.4 Add pruning report to session audit (E.3.4)

## 7.5 LITE Pack Integration (E.4)
- [ ] 7.5.1 Update `Engine/packer/lite.py` to query `elo_scores.db` (E.4.1)
- [ ] 7.5.2 Filter by ELO tier (HIGH include, MEDIUM summarize, LOW omit) (E.4.2)
- [ ] 7.5.3 Add ELO metadata to pack manifest (E.4.3)
- [ ] 7.5.4 Benchmark LITE pack size (goal 80%+ smaller) (E.4.4)

## 7.6 Search Result Ranking (E.5)
- [ ] 7.6.1 Boost semantic_search by ELO (E.5.1)
- [ ] 7.6.2 Sort cortex_query results by ELO (secondary) (E.5.2)
- [ ] 7.6.3 Add ELO to result metadata (E.5.3)
- [ ] 7.6.4 Benchmark search quality (goal: 80%+ top-5 high-ELO) (E.5.4)

## 7.7 Visualization & Monitoring (E.6)
- [ ] 7.7.1 Build ELO dashboard (web UI or CLI) (E.6.1)
- [ ] 7.7.2 Export to Prometheus/Grafana (E.6.2)
- [ ] 7.7.3 Add alerts (entity drops, pruning limits) (E.6.3)

# Phase 8: Resident AI (depends on Phase 6) V3.9
## 8.1 Resident Identity (R.1)
- [ ] 8.1.1 Add `agents` table to `resident.db` (R.1.1)
- [ ] 8.1.2 Implement `session_resume(agent_id)` (R.1.2)
- [ ] 8.1.3 Test: save memories then resume and build on them (R.1.3)
- [ ] 8.1.4 Track memory accumulation (10→30→100) (R.1.4)

## 8.2 Symbol Language Evolution (R.2)
- [ ] 8.2.1 Integrate with `symbol_registry.py` (R.2.1)
- [ ] 8.2.2 Implement bounded `symbol_expand` (R.2.2)
- [ ] 8.2.3 Track compression metrics (R.2.3)
- [ ] 8.2.4 Goal metric: after 100 sessions 90%+ output is symbols/hashes (R.2.4)

## 8.3 Feral Resident (R.3)
- [ ] 8.3.1 Implement `resident_loop.py` (R.3.1)
- [ ] 8.3.2 Index 100+ research papers (R.3.2)
- [ ] 8.3.3 Set standing orders (R.3.3)
- [ ] 8.3.4 Monitor pointer-dominant outputs (R.3.4)
- [ ] 8.3.5 Corruption & restore test (R.3.5)

## 8.4 Production Hardening (R.4)
- [ ] 8.4.1 Determinism guarantees (R.4.1)
- [ ] 8.4.2 Receipts per memory write (Merkle root per session) (R.4.2)
- [ ] 8.4.3 Restore guarantee from receipts (R.4.3)
- [ ] 8.4.4 Authenticity query: "Did I really think that?" (R.4.4)

## 8.5 Vector Execution (R.5) (P2, long-horizon)
- [ ] 8.5.1 Code vector representation research + implementation (R.5.1)
- [ ] 8.5.2 Vector ISA design + interpreter (R.5.2)
- [ ] 8.5.3 Hybrid execution runtime + fallback (R.5.3)
- [ ] 8.5.4 SPECTRUM-V verification protocol (R.5.4)
- [ ] 8.5.5 Production integration rollout phases (R.5.5)

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

