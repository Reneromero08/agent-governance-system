# CAT Chat Roadmap v2.0

**Last updated:** 2026-01-18
**Scope:** Pending work only - what remains to make CAT Chat production-ready
**Previous:** CAT_CHAT_ROADMAP_1.1.md (archived)
**Design Spec:** `INBOX/reports/V4/01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md`

---

## Core Concepts

**Session Model:** Session = tiny working set (token clean space) + hash pointers to offloaded state.

**Retrieval Order:** CORTEX first (symbols, indexes) -> CAS (exact hash) -> Vectors (approximate fallback).

**Design Decisions:**
- ELO tier integration SKIPPED (conflicts with "ELO as metadata" design decision)
- ELO scores are informational metadata only, not used for ranking modifications

---

## Completed Infrastructure (Reference Only)

The following is DONE and tested - included for context only:

| Component | Files | Tests |
|-----------|-------|-------|
| Router & Fallback | (model router) | deterministic selection + explicit fallback chain |
| Substrate & Indexing | section_extractor.py, section_indexer.py, slice_resolver.py | test_message_cassette.py |
| Symbol Registry | symbol_registry.py, symbol_resolver.py | (integrated) |
| Message Cassette | message_cassette.py, message_cassette_db.py | test_message_cassette.py |
| Planner | planner.py | test_planner.py |
| Bundle Protocol | bundle.py | test_bundle.py, test_bundle_execution.py |
| Receipts & Attestations | receipt.py, attestation.py, merkle_attestation.py | test_receipt.py, test_attestation.py, test_merkle_*.py |
| Trust & Identity | trust_policy.py, validator_identity.py | test_trust_identity_patch.py, test_identity_pinning.py |
| Executor | executor.py, execution_policy.py | test_execution*.py |
| Context Assembly | context_assembler.py, geometric_context_assembler.py | test_context_assembler.py |
| MCP Integration | mcp_integration.py | test_mcp_integration.py |
| Session Capsule | session_capsule.py | (pending - see A) |
| CORTEX Resolver | cortex_expansion_resolver.py | (pending - see A) |

---

## Pending Work

### A. Session Persistence Tests (P0 - Blocking)

**Status:** Infrastructure done, tests pending
**Files:** session_capsule.py, cortex_expansion_resolver.py

- [ ] A.1 Save/resume determinism test
  - Fixture: save session -> resume -> verify assembly hash identical
  - Must prove byte-identical replay
- [ ] A.2 Partial execution resume test
  - Fixture: partial run -> save -> resume -> execution continues identically
  - Must prove no state loss
- [ ] A.3 Tamper detection test
  - Fixture: tampered capsule -> FAIL (hash mismatch)
  - Must prove fail-closed on corruption
- [ ] A.4 Hydration failure test
  - Fixture: missing dependency during CORTEX retrieval -> FAIL
  - Must prove fail-closed on unresolvable symbols

**Exit Criteria:** All 4 fixtures green, determinism proven

---

### B. Documentation Index Cassette (P1)

**Status:** Not started
**Purpose:** Fast, bounded discovery without pasting whole files into prompts

- [ ] B.1 Build `cat_chat_index.db` with deterministic tables:
  - `files` (path, sha256, size)
  - `content` (file_id, normalized text)
  - `content_fts` (FTS5)
  - `indexing_info` (what was indexed, versions)
- [ ] B.2 Query API:
  - `docs search --query "..." --limit N` -> identifiers + bounded snippets only
  - `docs reindex` (full rebuild)
- [ ] B.3 Deterministic ranking:
  - Primary: FTS rank
  - Tie-breaker: (path ASC, file_sha ASC)
- [ ] B.4 Maintenance surfaces:
  - Optional watcher for incremental update
  - CI task for deterministic rebuild
- [ ] B.5 Tests:
  - Deterministic index hash across runs
  - Bounded snippet output
  - Stable ordering

**Exit Criteria:** `docs search` returns bounded, deterministic results

---

### C. Cassette Lifecycle & Maintenance (P2)

**Status:** Not started
**Purpose:** Production-grade cassette management

- [ ] C.1 Retention policy definition:
  - What rows are immutable forever (receipts)
  - What can be compacted (cache tables)
- [ ] C.2 Repair operations (explicit, opt-in):
  - Prune expired leases safely
  - Rebuild derived indices without mutating canonical receipts
- [ ] C.3 `cassette doctor` command:
  - Detects orphaned rows, missing foreign keys, invalid FSM transitions
  - Emits deterministic report without modifying DB

**Exit Criteria:** `cassette doctor` can diagnose all known failure modes

---

### D. Bundle Replay (P2)

**Status:** Not started
**Purpose:** Prove bundles are self-contained and reproducible

- [ ] D.1 Bundle runner contract:
  - Takes `bundle.json` + `artifacts/` only
  - Reproduces step outputs without reading repo files
  - Emits receipts deterministically
- [ ] D.2 Verify-before-run:
  - Runner must verify bundle and hard fail on mismatch
- [ ] D.3 Reproducibility tests:
  - Run twice -> identical receipts
  - Tampered artifact -> fail closed
- [ ] D.4 End-to-end integration fixture:
  - Full build and verify test setup

**Exit Criteria:** Bundle can be replayed offline with identical outputs

---

### E. Discovery Integration (P3 - Optional)

**Status:** Not started
**Purpose:** FTS + vector search for context retrieval

- [ ] E.1 Deterministic FTS search over sections
- [ ] E.2 Embedding retrieval behind governed boundary with hard budgets
- [ ] E.3 Discovery outputs schema:
  - query, hits, ranks, snippet_slices, provenance
  - stable ordering and deterministic tie-breakers
- [ ] E.4 Search-to-slice pipeline:
  - Discovery returns `(section_id, slice)` only
  - Resolver retrieves exact slice content
  - No "full file paste" path exists
- [ ] E.5 Staleness detection:
  - Warn when indexed file hash differs from current
  - Recommend `docs reindex`
- [ ] E.6 Vector governance boundaries:
  - Keep vectors behind governed boundaries
  - No trust-vectors bypass of verification or boundedness

**Exit Criteria:** Discovery returns bounded, deterministic results

---

### F. Specs & Golden Demo (P3)

**Status:** Not started
**Purpose:** Authoritative documentation that matches reality

- [ ] F.1 Authoritative specs:
  - Bundle protocol spec
  - Receipts + chain spec
  - Trust + identity spec
  - Execution policy spec
- [ ] F.2 Runbook: copy-paste runnable on Windows PowerShell
- [ ] F.3 Golden demo from fresh clone:
  - plan request -> execute -> bundle build -> verify
  - bundle run -> receipt chain verify -> Merkle attest -> verify
- [ ] F.4 Packaging hardening:
  - Repo-root execution works with only `PYTHONPATH=THOUGHT\LAB\CAT_CHAT`

**Exit Criteria:** New user can run golden demo from README

---

### G. Measurement & Benchmarking (P3)

**Status:** Not started
**Purpose:** Prove compression wins with numbers

- [ ] G.1 Per-step metrics:
  - Bytes expanded, cache hit rate, reuse rate
  - Plan and bundle size
- [ ] G.2 Determinism regression suite:
  - Index, symbols, receipts, bundles all deterministic
- [ ] G.3 Benchmark scenarios:
  - Patch one function
  - Refactor N files
  - Generate bundles from corpus
- [ ] G.4 Dashboard/reports:
  - Token savings vs baseline
  - Boundedness compliance

**Exit Criteria:** Compression claims backed by reproducible benchmarks

---

### H. Test Matrix Completion (P3)

**Status:** Partially tested, gaps remain
**Purpose:** Ensure complete test coverage for edge cases

- [ ] H.1 Quorum test matrix (Phase 6.13):
  - Quorum pass
  - Insufficient quorum
  - Duplicate validators
  - Identity mismatches
  - Tampered signatures
- [ ] H.2 CLI verifier test matrix (Phase 6.14):
  - Exit codes (all verify commands)
  - JSON shape validation
  - Quiet mode behavior

**Exit Criteria:** All edge cases covered by fixtures

---

### I. ChatDB Integration (P3 - Optional)

**Status:** ChatDB implemented, integration pending
**Purpose:** Wire chat I/O into deterministic execution pipeline

- [ ] I.1 Wire `chatdb directives` into planner/bundler/executor loop
  - Must not introduce nondeterminism
- [ ] I.2 Tail-mode exporter (append-only view)
- [ ] I.3 `--since` plus deterministic paging for huge threads

**Exit Criteria:** ChatDB fully integrated with deterministic guarantees

---

## Priority Summary

| Priority | Phase | Blocker? | Effort |
|----------|-------|----------|--------|
| P0 | A. Session Tests | Yes - proves infrastructure works | Small |
| P1 | B. Docs Index | No | Medium |
| P2 | C. Cassette Lifecycle | No | Medium |
| P2 | D. Bundle Replay | No | Medium |
| P3 | E. Discovery | Optional | Large |
| P3 | F. Specs & Demo | No | Medium |
| P3 | G. Measurement | No | Medium |
| P3 | H. Test Matrix | No | Small |
| P3 | I. ChatDB Integration | Optional | Medium |

**Recommended order:** A -> B -> D -> C -> H -> F -> G -> E -> I

---

## Dependencies

- **Phase 5 (Vector/Symbol):** COMPLETE - provides semantic search
- **Phase 6 (Cassette Network):** COMPLETE - provides manifold infrastructure
- **Phase 7 (Vector ELO):** COMPLETE - provides usage metadata
- **Phase 8 (Resident AI):** COMPLETE - provides geometric reasoning

All dependencies satisfied. CAT Chat work is unblocked.
