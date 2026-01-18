# CAT Chat Roadmap v2.0

**Last updated:** 2026-01-18
**Scope:** Pending work to make CAT Chat a fully catalytic, integrated system
**Previous:** CAT_CHAT_ROADMAP_1.1.md (archived)
**Design Spec:** `INBOX/reports/V4/01-06-2026-21-13_CAT_CHAT_CATALYTIC_CONTINUITY.md`

---

## What is CAT Chat?

**CAT = Catalytic.** This is not just bounded chat - it's chat that operates on catalytic computing principles:

- **Clean Space:** Bounded context window (tokens)
- **Catalytic Space:** Large disk state that MUST restore exactly after use
- **Compression:** Symbols/pointers instead of full content (56,370x for a single symbol)
- **Verification:** Every expansion produces hash-verified receipts
- **Fail-Closed:** Restoration failure = hard exit, never silent

---

## Sandbox Status

CAT Chat lives in `THOUGHT/LAB/` for rapid iteration without main repo governance.

**Current Structure:**
```
THOUGHT/LAB/CAT_CHAT/
    _generated/
        cat_chat.db     # Single consolidated database (all tables)
    catalytic_chat/     # Python package
    tests/
    SCHEMAS/
```

**On Graduation:** `_generated/cat_chat.db` moves to `NAVIGATION/CORTEX/cassettes/cat_chat.db`

---

## Core Concepts

**Session Model:** Session = tiny working set (clean space) + hash pointers to offloaded state (catalytic space).

**Retrieval Order:** Main cassettes (NAVIGATION/CORTEX/) -> Local index -> CAS (exact hash) -> Vectors (approximate fallback).

**Catalytic Invariants:**
1. **INV-CATALYTIC-01 (Restoration):** File states before/after must be identical (or explicitly committed)
2. **INV-CATALYTIC-02 (Verification):** Proof size = O(1) per domain (single Merkle root)
3. **INV-CATALYTIC-03 (Reversibility):** restore(snapshot) = original (byte-identical)
4. **INV-CATALYTIC-04 (Clean Space Bound):** Context uses pointers, not full content
5. **INV-CATALYTIC-05 (Fail-Closed):** Restoration failure = hard exit
6. **INV-CATALYTIC-06 (Determinism):** Identical inputs = identical Merkle root

**Design Decisions:**
- ELO scores are informational metadata only, not used for ranking
- Vectors are fallback only, not primary retrieval path
- All expansions bounded (no `ALL` slices)

---

## Database Schema

Single `_generated/cat_chat.db` contains all tables:

**Index Layer:**
- `sections` - Indexed content from files
- `section_index_meta` - Index metadata
- `symbols` - @symbol -> section mappings
- `expansion_cache` - Cached resolved content

**Cassette Layer (append-only work log):**
- `cassette_meta` - Version info
- `cassette_messages` - Requests (append-only)
- `cassette_jobs` - Work items
- `cassette_steps` - PENDING -> LEASED -> COMMITTED FSM
- `cassette_receipts` - Execution proofs (append-only)
- `cassette_job_budgets` - Token/byte tracking

**Session Layer:**
- `sessions` - Session state
- `session_events` - Hash-chained event log
- `session_working_set` - Active working set
- `session_pointer_set` - Pointer references

---

## Completed Infrastructure

| Component | Files | Status |
|-----------|-------|--------|
| Substrate & Indexing | section_extractor.py, section_indexer.py, slice_resolver.py | Done |
| Symbol Registry | symbol_registry.py, symbol_resolver.py | Done |
| Message Cassette | message_cassette.py, message_cassette_db.py | Done |
| Planner | planner.py | Done |
| Bundle Protocol | bundle.py | Done |
| Receipts & Attestations | receipt.py, attestation.py, merkle_attestation.py | Done |
| Trust & Identity | trust_policy.py, validator_identity.py | Done |
| Executor | executor.py, execution_policy.py | Done |
| Context Assembly | context_assembler.py, geometric_context_assembler.py | Done |
| MCP Integration | mcp_integration.py | Done |
| Geometric Chat | geometric_chat.py | Done |
| Session Capsule | session_capsule.py | Tests pending (A) |
| CORTEX Resolver | cortex_expansion_resolver.py | Tests pending (A) |

---

## Pending Work

### A. Session Persistence Tests (P0 - Blocking)

**Status:** Infrastructure done, tests pending
**Files:** session_capsule.py, cortex_expansion_resolver.py

- [ ] A.1 Save/resume determinism test (byte-identical replay)
- [ ] A.2 Partial execution resume test (no state loss)
- [ ] A.3 Tamper detection test (fail-closed on corruption)
- [ ] A.4 Hydration failure test (fail-closed on unresolvable symbols)

**Exit Criteria:** All 4 fixtures green, determinism proven

---

### B. Main Cassette Network Integration (P0 - Critical)

**Status:** Not started
**Purpose:** Connect to existing cassette infrastructure instead of duplicating

- [ ] B.1 CassetteClient for reading main cassettes:
  - Read from `NAVIGATION/CORTEX/cassettes/*.db`
  - Search across canon.db, governance.db, etc.
  - Respect cassette network conventions
- [ ] B.2 Symbol resolution via main cassettes:
  - Resolve @symbols to content in main cassettes
  - Fall back to local index if not found
- [ ] B.3 Write isolation:
  - Reads: main cassettes (shared)
  - Writes: `_generated/cat_chat.db` (sandbox only)
- [ ] B.4 Graduation path:
  - Document how `cat_chat.db` becomes a main cassette
  - Ensure schema compatibility

**Exit Criteria:** CAT Chat reads from main cassette network, writes locally

---

### C. Semantic Pointer Compression Integration (P1)

**Status:** Not started
**Purpose:** Use SPC instead of verbose @symbols

- [ ] C.1 Codebook sync handshake:
  - Verify codebook_id + SHA256 on session start
  - Fail-closed on mismatch
- [ ] C.2 Pointer resolution:
  - Support SYMBOL_PTR (CJK characters like `法`)
  - Support HASH_PTR (sha256:7cfd0418...)
  - Support COMPOSITE_PTR (`法.驗`, `C3:build`)
- [ ] C.3 Compression metrics:
  - Track pointer vs full-content savings
  - Prove compression claims

**Exit Criteria:** SPC pointers resolve correctly with fail-closed semantics

---

### D. Geometric E-Gating Integration (P1)

**Status:** geometric_chat.py exists, integration pending
**Purpose:** Use E-scoring for relevance filtering

- [ ] D.1 Wire geometric_chat.py into main flow:
  - E-gate before LLM calls (threshold=0.5)
  - Track E-scores in receipts
- [ ] D.2 Context assembly via E-scoring:
  - Rank retrieved content by geometric relevance
  - Bounded expansion with E-gated filtering
- [ ] D.3 Correlation validation:
  - Prove high-E responses are measurably better
  - Track E-score vs response quality

**Exit Criteria:** E-gating operational, correlation measured

---

### E. Vector Fallback Chain (P2)

**Status:** Not started
**Purpose:** Vectors as governed fallback, not primary path

- [ ] E.1 Retrieval order enforcement:
  - 1st: Main cassette FTS
  - 2nd: Local index
  - 3rd: CAS (exact hash)
  - 4th: Vector search (fallback only)
- [ ] E.2 Vector governance boundaries:
  - Hard token budgets on vector retrieval
  - No trust-vectors bypass of verification
- [ ] E.3 ELO as metadata:
  - Track usage patterns
  - Do NOT modify ranking based on ELO

**Exit Criteria:** Vector fallback operational with governance

---

### F. Docs Index (P2)

**Status:** Not started
**Purpose:** Fast, bounded discovery via FTS

- [ ] F.1 FTS tables in cat_chat.db:
  - `docs_files` (path, sha256, size)
  - `docs_content` (file_id, normalized text)
  - `docs_content_fts` (FTS5)
- [ ] F.2 Query API:
  - `docs search --query "..." --limit N`
  - Returns identifiers + bounded snippets only
- [ ] F.3 Deterministic ranking with stable tie-breakers

**Exit Criteria:** `docs search` returns bounded, deterministic results

---

### G. Bundle Replay & Verification (P2)

**Status:** Not started
**Purpose:** Prove bundles are self-contained and reproducible

- [ ] G.1 Bundle runner: takes bundle.json + artifacts only
- [ ] G.2 Verify-before-run: hard fail on mismatch
- [ ] G.3 Reproducibility: run twice -> identical receipts

**Exit Criteria:** Bundle can be replayed offline with identical outputs

---

### H. Specs & Golden Demo (P3)

**Status:** Not started
**Purpose:** Authoritative documentation + runnable demo

- [ ] H.1 Authoritative specs (bundle, receipts, trust, execution)
- [ ] H.2 Runbook: copy-paste runnable on Windows PowerShell
- [ ] H.3 Golden demo from fresh clone

**Exit Criteria:** New user can run golden demo from README

---

### I. Measurement & Benchmarking (P3)

**Status:** Not started
**Purpose:** Prove catalytic compression with numbers

- [ ] I.1 Per-step metrics (bytes expanded, cache hits, reuse rate)
- [ ] I.2 Compression benchmarks vs baseline
- [ ] I.3 Catalytic invariant verification suite

**Exit Criteria:** Compression claims backed by reproducible benchmarks

---

## Priority Summary

| Priority | Phase | Blocker? | Effort |
|----------|-------|----------|--------|
| P0 | A. Session Tests | Yes | Small |
| P0 | B. Cassette Network Integration | Yes | Medium |
| P1 | C. SPC Integration | No | Medium |
| P1 | D. E-Gating Integration | No | Medium |
| P2 | E. Vector Fallback | No | Medium |
| P2 | F. Docs Index | No | Medium |
| P2 | G. Bundle Replay | No | Medium |
| P3 | H. Specs & Demo | No | Medium |
| P3 | I. Measurement | No | Medium |

**Recommended order:** A -> B -> C -> D -> E -> F -> G -> H -> I

---

## Dependencies (All Complete)

- **Phase 5 (Vector/Symbol):** COMPLETE - provides semantic search
- **Phase 6 (Cassette Network):** COMPLETE - provides 9 modular cassettes
- **Phase 7 (Vector ELO):** COMPLETE - provides usage metadata
- **Phase 8 (Resident AI):** COMPLETE - provides geometric reasoning

All dependencies satisfied. Integration work is unblocked.

---

## Graduation Criteria

CAT Chat graduates from LAB to main system when:

1. All P0/P1 items complete
2. Tests pass with main cassette network
3. Catalytic invariants verified
4. Compression claims proven with benchmarks
5. Golden demo works from fresh clone
