# Catalytic Chat — Roadmap

## Purpose
Build a chat substrate where models write compact, structured messages that reference canonical material via **symbols**, and workers expand only **bounded slices** as needed. The substrate accumulates reusable intermediates so repeated work gets cheaper and more reliable.

Note: Testing environment notes do not affect phase completion status unless they invalidate core invariants.

---

## Hard invariants
- [x] **No bulk context stuffing.** Prefer references (symbols/section_ids) over pasted text.
- [x] **No unbounded expansion.** Every expansion must specify a slice and obey budgets.
- [x] **Receipts are mandatory.** Every step records what was expanded and what was produced.
- [x] **Deterministic addressing.** Sections and symbols resolve identically across runs for identical inputs.
- [x] **Discovery ≠ justification.** Vectors/FTS only select candidates; correctness comes from resolved canonical slices + contracts.

---

## Core objects (contract vocabulary)
- **Section**: `(section_id, file_path, heading_path, line_start, line_end, content_hash)`
- **Symbol**: `(symbol_id, target_type, target_ref, default_slice_policy)`
- **Message** (model output): `(intent, refs[], ops[], budgets, required_outputs[])`
- **Expansion**: `(run_id, symbol_id/section_id, slice, content_hash, payload_ref)`
- **Receipt**: `(run_id, step_id, expanded[], actions[], outputs[], status)`

---

## Phase 0 — Freeze scope and interfaces (COMPLETE)
Goal: lock vocabulary and the minimum tool surfaces so implementation cannot drift.

- [x] Create `docs/catalytic-chat/CONTRACT.md` defining: Section, Symbol, Message, Expansion, Receipt.
- [x] Define budgets: `max_symbols`, `max_sections`, `max_bytes_expanded`, `max_expands_per_step`.
- [x] Define error policy: fail-closed on missing symbol, missing slice, budget breach.
- [x] Define receipt schema (append-only) and minimum required fields.
- [x] Define "what counts as canonical sources" (folders + file types).

Exit criteria
- [x] CONTRACT.md exists and is referenced by roadmap.
- [x] A dummy end-to-end walkthrough can be expressed using only contract objects (no prose).

---

## Phase 1 — Substrate + deterministic indexing (COMPLETE)
Goal: build the persistent substrate and deterministic section index.

- [x] Choose substrate mode: `sqlite` (primary) or `jsonl+indexes` (fallback). Document both.
- [x] Implement section extractor over canonical sources:
  - [x] Markdown headings → section ranges
  - [x] Code blocks / code files → section ranges (file-level or function-level if available)
- [x] Emit `SECTION_INDEX` artifact (DB table and/or JSON file).
- [x] Compute stable `content_hash` per section.
- [x] Add incremental rebuild (only re-index changed files).
- [x] Add a CLI: `cortex build` (or equivalent) to build index.

Exit criteria
- [x] Two consecutive builds on unchanged repo produce identical SECTION_INDEX (hash-stable).
- [x] A section can be fetched by `section_id` with correct slice boundaries.

---

## Phase 2 — Symbol registry + bounded resolver (COMPLETE)
Goal: make compact references real and enforce bounded expansion.

- [x] Create symbol registry:
  - [x] `SYMBOLS` artifact mapping `@Symbol` → `section_id` (or file+heading ref)
  - [x] Namespace conventions (`@CANON/...`, `@CONTRACTS/...`, `@TOOLS/...`, etc.)
- [x] Implement resolver API:
  - [x] `resolve(symbol_id, slice, run_id)` → payload (bounded)
  - [x] Slice forms: `lines[a:b]`, `chars[a:b]`, `head(n)`, `tail(n)` (pick one canonical form)
  - [x] Deny `slice=ALL`
- [x] Implement expansion cache:
  - [x] Store expansions by `(run_id, symbol_id, slice, section_content_hash)`
  - [x] Reuse prior expansions within the same run
- [x] Add CLI:
  - [x] `cortex resolve @Symbol --slice ...` --run-id <id>`
  - [ ] `cortex summary section_id` (advisory only)

Exit criteria
- [x] Symbol resolution is deterministic and bounded.
- [x] Expansion cache reuses identical expands within a run.

---

## Phase 3 — Message cassette (LLM-in-substrate communication) (COMPLETE)
Goal: models write structured messages into substrate and workers consume them.

- [x] Add tables / files for messaging:
   - [x] `cassette_messages` (planner + worker requests)
   - [x] `cassette_jobs` / `cassette_steps` (claimable units)
   - [x] `cassette_receipts` (append-only)
- [x] Implement job lifecycle:
   - [x] `post(message)` → job created
   - [x] `claim(job_id, worker_id)` → exclusive lock
   - [x] `complete(job_id, receipt)` → stored + immutable
- [x] Enforce: message payload must be structured (refs/ops/budgets), not prose-only.
- [x] DB-first enforcement via SQLite triggers
- [x] Append-only on messages and receipts
- [x] FSM enforcement for step transitions
- [x] Lease protection (fencing tokens, expiry)
- [x] Deterministic claim selection
- [x] Foreign key integrity
- [x] 21 passing tests including adversarial cases

Exit criteria
- [x] A job can be posted, claimed, executed, and completed with receipts.
- [x] A worker cannot expand beyond budgets.
- [x] Law document: `docs/cat_chat/PHASE_3_LAW.md`
- [x] CLI commands: `cassette verify/post/claim/complete`
- [x] All tests passing
- [x] Execution-agnostic (no model calls, no workers)

---

## Phase 4 — Discovery: FTS + vectors (candidate selection only) (PENDING)
Goal: find the right symbols/sections cheaply and safely.

- [ ] Add FTS index over sections (title + body).
- [ ] Add embeddings table for sections (vectors stored in DB only).
- [ ] Implement `search(query, k)` returning **section_ids/symbol_ids only**.
- [ ] Implement hybrid search: combine FTS + vector scores (bounded).
- [ ] Store retrieval receipts:
  - [ ] query_hash
  - [ ] topK ids
  - [ ] thresholds
  - [ ] timestamp/run_id

Exit criteria
- [ ] Search returns stable candidates for repeated queries on unchanged corpus.
- [ ] No vectors are ever emitted into model prompts (only ids + optionally tiny snippets).

---

## Phase 5 — Translation protocol (minimal executable bundles) (PENDING)
Goal: convert high-level intent into the smallest per-step bundle: refs + bounded expands + ops.

- [ ] Define `Bundle` schema:
  - [ ] intent
  - [ ] refs (symbols)
  - [ ] expand_plan (symbol+slice list)
  - [ ] ops
  - [ ] budgets
- [ ] Implement bundler:
  - [ ] uses discovery to pick candidates
  - [ ] adds only the minimal refs needed
  - [ ] requests explicit expands (sliced) when required
- [ ] Add bundle verifier:
  - [ ] checks budgets
  - [ ] checks all symbols resolvable
  - [ ] checks slice validity
- [ ] Add memoization across steps within a run:
  - [ ] reuse expansions, avoid re-expanding

Exit criteria
- [ ] Same task, same corpus → bundles differ only when corpus changes.
- [ ] Measured prompt payload stays small and bounded per step.

---

## Phase 6 — Measurement and regression harness (PENDING)
Goal: make "catalytic" measurable and prevent regressions.

- [ ] Log per-step metrics:
  - [ ] tokens_in/tokens_out (if available)
  - [ ] bytes_expanded
  - [ ] expands_per_step
  - [ ] reuse_rate
  - [ ] search_k and hit-rate (when ground-truth available)
- [ ] Add regression tests:
  - [ ] determinism tests for SECTION_INDEX + SYMBOLS
  - [ ] budget enforcement tests
  - [ ] receipt completeness tests
- [ ] Add benchmark scenarios:
  - [ ] "find and patch 1 function" task
  - [ ] "refactor N files" task
  - [ ] "generate roadmap from corpus" task

Exit criteria
- [ ] A dashboard (or printed report) shows token and expansion savings over baseline.
- [ ] Regressions fail tests deterministically.

## Phase 6.2 — Receipt Attestation (COMPLETE 2025-12-30)

Goal: add cryptographic signing and verification to bundle execution receipts.

**Related:**
- Provisional ADR: `docs/provisional/ADR-attestation.md`
- Commit plan: `commit-plan-phase-6-2-attestation.md`

### Completed

- [x] **receipt_canonical_bytes(receipt, attestation_override=None)** in `catalytic_chat/receipt.py`
  - Single source of truth for receipt canonicalization
  - Used by signer, verifier, and executor

- [x] **sign_receipt_bytes()** in `catalytic_chat/attestation.py`
  - Ed25519 signing
  - Hex output for public_key and signature

- [x] **verify_receipt_bytes()** in `catalytic_chat/attestation.py`
  - Ed25519 verification
  - Validates hex, lengths, scheme
  - Recomputes canonical bytes with `attestation_override=None`

- [x] **executor.py** enhanced
  - Uses `receipt_canonical_bytes()` for signing
  - Uses `receipt_canonical_bytes()` for writing
  - Returns `receipt_path` and `attestation` fields

- [x] **cli.py** enhanced
  - `bundle run --attest --signing-key <path>` to sign receipts
  - `bundle run --verify-attestation` to verify receipts

- [x] **Tests** in `tests/test_attestation.py`
  - All 6 tests passing
  - Roundtrip sign/verify
  - Tamper detection
  - Input validation (hex, lengths, scheme)
  - Null handling

Exit criteria
- [x] Receipts can be signed with ed25519 private keys
- [x] Receipts can be verified against public keys
- [x] Canonical bytes are deterministic (single source of truth)
- [x] All validation rules enforced (hex-only, length checks, scheme check)
- [x] All tests passing

---

## Phase 6.2.1 — Attestation Stabilization (COMPLETE 2025-12-30)

Goal: fix test environment issue preventing full test suite from passing.

### Completed

- [x] **test_cli_dry_run** fix in `tests/test_planner.py`
  - Added `PYTHONPATH` environment variable to subprocess call
  - Subprocess now finds `catalytic_chat.cli` module correctly
  - Minimal change: only 3 lines added

Exit criteria
- [x] Entire test suite green: 59 passed, 13 skipped
- [x] No test skips added
- [x] Deterministic receipts unchanged
- [x] Attestation still fail-closed
- [x] No CLI or executor regressions

---

## Optional track — No-DB mode (file substrate) (PENDING)
Goal: keep the same contract with JSONL + deterministic files for environments without SQLite.

- [ ] `CORTEX/SECTION_INDEX.json`
- [ ] `CORTEX/SYMBOLS.json`
- [ ] `CORTEX/_generated/summaries/*`
- [ ] `CORTEX/_cache/expansions/*`
- [ ] `CORTEX/_runs/<run_id>/receipts.jsonl`
- [ ] Provide identical CLI surface backed by files

Exit criteria
- [ ] Feature parity for resolve/search (within limits) and receipts.
