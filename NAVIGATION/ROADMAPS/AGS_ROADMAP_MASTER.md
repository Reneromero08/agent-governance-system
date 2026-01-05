---
title: AGS Roadmap (TODO Only, Rephased)
version: 3.6.1
last_updated: 2026-01-03
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
  - Declare write allowlists per ticket
  - Emit receipts (inputs/outputs/hashes)
  - Run relevant tests before “DONE”

# Phase 1: Integrity Gates & Repo Safety (highest leverage)
## 1.1 Hardened Inbox Governance (S.2)
- [x] 1.1.1 Create `SKILLS/inbox-report-writer` to auto-hash and format reports (S.2.1)
- [x] 1.1.2 Add strict pre-commit protocol: reject commits touching `INBOX/*.md` without valid header hash (S.2.2)
- [x] 1.1.3 Add runtime interceptor: block unhashed writes to `INBOX/` at tool level (S.2.3)
- **Exit Criteria**
  - [x] Attempts to write unhashed INBOX artifacts fail-closed with clear error
  - [x] Pre-commit rejects invalid INBOX changes deterministically

## 1.2 Bucket Enforcement (X3)
- [ ] 1.2.1 Add preflight check: every artifact must belong to exactly one bucket (X3)
- **Exit Criteria**
  - [ ] Violations fail-closed before writes occur

## 1.3 Deprecate Lab MCP Server (Z.1.7)
- [ ] 1.3.1 Mark `THOUGHT/LAB/MCP/server_CATDPT.py` archived/deprecated with clear pointer to canonical server (Z.1.7)
- **Exit Criteria**
  - [ ] No tooling still imports/executes the deprecated server in normal flows

## 1.4 Failure Taxonomy & Recovery Playbooks (ops-grade)
- [ ] 1.4.1 Create `NAVIGATION/OPS/FAILURE_CATALOG.md` listing expected fail-closed errors by subsystem (CAS, ARTIFACTS, RUNS, GC, AUDIT, SKILL_RUNTIME, PACKER)
  - Include: failure code/name, trigger condition, detection signal (exception/exit code), and “safe recovery” steps
- [ ] 1.4.2 Add a deterministic “Recovery” appendix to each major invariant doc:
  - Where receipts live
  - How to re-run verification
  - What to delete vs never delete
- [ ] 1.4.3 Add `NAVIGATION/OPS/SMOKE_RECOVERY.md` with the top 10 recovery flows as copy/paste commands (Windows + WSL where relevant)
- **Exit Criteria**
  - [ ] A new contributor can identify and recover from common failures without tribal knowledge
  - [ ] Recovery steps are deterministic and reference exact commands and artifacts

# Phase 2: CAS + Packer Completion (context cost collapse)
## 2.1 CAS-aware LLM Packer Integration (Z.2.6 + P.2 remainder)
- [x] 2.1.1 Make LITE packs use CAS hashes instead of full file bodies (Z.2.6)
- [x] 2.1.2 Implement CAS garbage collection safety for packer outputs: define GC roots/pins via active packs (P.2.4)
- [x] 2.1.3 Benchmark deduplication savings and pack generation cost; emit reproducible report + fixtures (P.2.5)
- **Exit Criteria**
  - [x] LITE packs are manifest-only and reference `sha256:` blobs
  - [x] GC never deletes a referenced blob (fixture-backed)
  - [x] Dedup benchmark reproducible and stored as an artifact

## 2.2 Pack Consumer (verification + rehydration)
- [ ] 2.2.1 Define Pack Manifest v1 (schema + invariants)
  - Must include: pack_id, scope (AGS/CAT/LAB), bucket list, path→ref mapping (`sha256:`), build metadata, and declared roots/pins
  - Must be canonical-JSON encoded and stored in CAS (manifest itself is addressable)
- [ ] 2.2.2 Implement `pack_consume(manifest_ref, out_dir, *, dry_run=False)` (tool/CLI)
  - Verify manifest integrity (hash, canonical encoding, schema)
  - Verify every referenced blob exists in CAS (or fail-closed)
  - Materialize tree to `out_dir` atomically (write to temp + rename)
  - Enforce strict path safety (no absolute paths, no `..`, no writing outside `out_dir`)
- [ ] 2.2.3 Emit a consumption receipt
  - Inputs: manifest_ref, cas_snapshot_hash
  - Outputs: out_dir tree hash (or deterministic listing hash), verification summary
  - Commands run, exit status
- [ ] 2.2.4 Tests (fixture-backed)
  - Tamper detection: modify manifest bytes or blob bytes → FAIL
  - Determinism: consume twice → identical tree hash/listing
  - Partial CAS: missing blob → FAIL (no partial materialization)
- **Exit Criteria**
  - [ ] Packs are not write-only: they can be consumed and verified deterministically
  - [ ] Any corruption or missing data fails-closed before producing an output tree

## 2.3 Run Bundle Contract (freezing “what is a run”)
- [ ] 2.3.1 Freeze the per-run directory contract
  - Required artifacts: TASK_SPEC, STATUS timeline, OUTPUT_HASHES, receipts
  - Naming conventions, immutability rules, and deterministic ordering requirements
- [ ] 2.3.2 Implement `run_bundle_create(run_id) -> sha256:<hash>`
  - Bundle is a manifest that references run artifacts in CAS (no raw file paths)
  - Bundle manifest is canonical-JSON encoded and stored in CAS
- [ ] 2.3.3 Define rooting and retention semantics (ties into GC)
  - What becomes a root by default (active runs, explicit pins, pack manifests)
  - Minimum retention policy for safety (e.g., never GC pinned runs)
- [ ] 2.3.4 Implement `run_bundle_verify(bundle_ref)` (dry-run verifier)
  - Ensures: all referenced artifacts exist, hashes match, required outputs are reachable
  - Emits deterministic verification receipt
- **Exit Criteria**
  - [ ] “Run = proof-carrying bundle” is explicit and machine-checkable
  - [ ] GC can safely treat bundles/pins as authoritative roots


# Phase 3: CAT Chat Stabilization (make the interface reliable)
- Precondition: If Phase 4.1 is not green, treat Phase 3 as provisional and expect churn.

## 3.1 Router & Fallback Stability (Z.3.1)
- [ ] 3.1.1 Stabilize model router: deterministic selection + explicit fallback chain (Z.3.1)
## 3.2 Memory Integration (Z.3.2)
- [ ] 3.2.1 Implement CAT Chat context window management (Z.3.2)
## 3.3 Tool Binding (Z.3.3)
- [ ] 3.3.1 Ensure MCP tool access from chat is functional and constrained (Z.3.3)
## 3.4 Session Persistence (Z.3.4)
- [ ] 3.4.1 Implement session persistence and resume (Z.3.4)
- **Exit Criteria**
  - [ ] One end-to-end CAT Chat run can: route → use tools → persist → resume with identical behavior

# Phase 4: Catalytic Architecture (restore guarantees)
## 4.1 Catalytic Snapshot & Restore (Z.4.2–Z.4.4)
- [ ] 4.1.1 Pre-run snapshot: hash catalytic state before execution (Z.4.2)
- [ ] 4.1.2 Post-run restoration: verify byte-identical restoration (Z.4.3)
- [ ] 4.1.3 Hard-fail on restoration mismatch (Z.4.4)
- **Exit Criteria**
  - [ ] Catalytic domains restore byte-identical (fixture-backed)
  - [ ] Failure mode is deterministic and fail-closed

# Phase 5: Vector/Symbol Integration (addressability)
## 5.1 Embed Canon, ADRs, and Skill Discovery (Z.5)
- [ ] 5.1.1 Embed all canon files: `LAW/CANON/*` → vectors (Z.5.1)
- [ ] 5.1.2 Embed all ADRs: decisions/* → vectors (Z.5.2)
- [ ] 5.1.3 Store model weights in vector-indexed CAS (Z.5.3)
- [ ] 5.1.4 Semantic skill discovery: find skills by description similarity (Z.5.4)
- [ ] 5.1.5 Cross-reference indexing: link artifacts by embedding distance (Z.5.5)
- **Exit Criteria**
  - [ ] Vector index includes canon + ADRs with deterministic rebuild
  - [ ] Skill discovery returns stable results for fixed corpus

# Phase 6: Cassette Network (Semantic Manifold) (P0 substrate)
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
- **Exit Criteria**
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

# Phase 8: Resident AI (depends on Phase 6)
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

# Phase 10: System Evolution (Ω) (post-substrate)
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

