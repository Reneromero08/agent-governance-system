<!-- CONTENT_HASH: f1eae976a4e3a1297b70cbbad46a7f4a07b5cba6ed4d0b84be1d97d6206152b5 -->

# Roadmap.md

**Last updated:** 2026-01-02 (America/Phoenix)  
**Format:** `- [ ]` checklist (use `- [x]` for complete)

## Global invariants (non-negotiable)

- [x] Determinism: identical inputs produce identical outputs (IDs, ordering, bytes).
- [x] Boundedness: expansion is slice-based; `ALL` (or equivalent sentinel) is forbidden.
- [x] Fail-closed: missing refs, invalid schema, hash mismatch, ordering violations must hard error.
- [x] Receipts mandatory: every executed step must be recorded and verifiable.
- [x] Canonical JSON: UTF-8, `\n` newlines, `sort_keys=True`, `separators=(",", ":")`, newline at EOF.
- [x] No timestamps, randomness, absolute paths in deterministic artifacts unless explicitly signed and verified.
- [x] Canonical output roots: generated artifacts live under `CORTEX/_generated` (or explicit `--out` dirs), never sprayed across repo.
- [x] Trust is explicit: verification depends on attestation integrity plus pinned trust policy (no silent fallback).
- [x] Versioning policy (SemVer):
  - [x] MAJOR: schema or contract break, or incompatible CLI output changes
  - [x] MINOR: additive schema fields or additive CLI subcommands that preserve existing behavior
  - [x] PATCH: bugfixes, test additions, internal refactors with identical external behavior
- [x] CLI output contract:
  - [x] **stdout** is reserved for machine outputs (JSON payloads, IDs) and must be deterministic when the command is deterministic.
  - [x] **stderr** is reserved for logs, progress, and human diagnostics. No machine parsing required.

---

## Phase 0: Freeze scope and interfaces (COMPLETE)

- [x] Freeze vocabulary: Section, Symbol, Message, Expansion, Receipt, Bundle.
- [x] Freeze determinism requirements and fail-closed error policy.
- [x] Define budgets and the no-unbounded-expansion invariant.

---

## Phase 1: Substrate plus deterministic indexing (COMPLETE)

- [x] Deterministic section extraction across repo content.
- [x] Deterministic hashing (SHA-256) for section identity and integrity.
- [x] Generate `SECTION_INDEX` under `CORTEX/_generated`.
- [x] Implement slicing rules and resolvers (line, char, head, tail).
- [x] Explicitly forbid `ALL` slice.
- [x] CLI supports building the index from repo root.

---

## Phase 2: Symbol registry plus bounded resolver (COMPLETE core)

### Phase 2.1: Symbol Registry (COMPLETE)

- [x] Symbol schema: stable `symbol_id`, `target_type`, `target_ref`, `default_slice`, timestamps.
- [x] Fail-closed validation:
  - [x] `symbol_id` prefix `@`
  - [x] target exists in `SECTION_INDEX`
  - [x] `default_slice` valid and not `ALL`
  - [x] uniqueness enforced
- [x] Dual substrates (SQLite plus JSONL) with deterministic listing.
- [x] CLI: `symbols add|get|list|verify`.

### Phase 2.2: Symbol Resolution plus Expansion Cache (COMPLETE)

- [x] Resolver expands symbols to exact slices only.
- [x] Apply request slice or `default_slice` when unspecified.
- [x] Explicitly reject `slice=ALL`.
- [x] Expansion cache keyed by stable inputs (for example `(run_id, symbol_id, slice, section_content_hash)`).
- [x] Cache hit returns stored payload deterministically.
- [x] Cache miss computes, stores, and returns deterministically.

### Phase 2.X: Vector sandbox (EXPERIMENTAL, optional)

- [x] SQLite vector store with namespace isolation (if present).
- [x] Deterministic top-k ordering with tie-break rules.
- [ ] Keep vectors behind governed boundaries (no trust-vectors bypass of verification or boundedness).

---

## Phase 3: Message cassette (append-only substrate) (COMPLETE)

- [x] Cassette tables for messages, jobs, steps, receipts (or strict equivalent).
- [x] Append-only semantics and durable FSM rules.
- [x] CLI supports cassette operations plus verification.

### Phase 3.3: Cassette Network - Documentation Index Cassette (PENDING if not yet committed)

This is the “documentation index cassette” that powers fast, bounded discovery without pasting whole files into prompts.

- [ ] Build `cat_chat_index.db` (or equivalent) with deterministic tables:
  - [ ] `files` (path, sha256, size, mtime snapshot if allowed by purity rules)
  - [ ] `content` (file_id, normalized text)
  - [ ] `content_fts` (FTS5)
  - [ ] `indexing_info` (what was indexed, versions, settings)
  - [ ] `indexing_info_fts` (FTS5)
- [ ] Query API (CLI or governed skill):
  - [ ] `docs search --query "..." --limit N`
  - [ ] returns only identifiers plus bounded snippet slices, not raw full documents
- [ ] Deterministic ranking rules:
  - [ ] primary: FTS rank
  - [ ] tie-breaker: `(path asc, file_sha asc)`
- [ ] Maintenance surfaces:
  - [ ] `docs reindex` (full rebuild)
  - [ ] optional watcher for incremental update
  - [ ] CI task for deterministic rebuild
- [ ] Tests:
  - [ ] deterministic index bytes (or deterministic hash) across runs
  - [ ] bounded snippet output and stable ordering

### Phase 3.4: Cassette lifecycle, retention, and repair (PENDING)

- [ ] Define retention policy:
  - [ ] what rows are immutable forever (receipts)
  - [ ] what can be compacted (cache tables)
- [ ] Define repair operations (explicit, opt-in):
  - [ ] prune expired leases safely
  - [ ] rebuild derived indices without mutating canonical receipts
- [ ] Add `cassette doctor` command:
  - [ ] detects orphaned rows, missing foreign keys, invalid FSM transitions
  - [ ] can emit a deterministic report without modifying DB

---

## Phase 4: Planner stabilization plus deterministic planning loop (COMPLETE)

### Phase 4.1: Dry-run missing-symbol tolerance plus zero DB writes (COMPLETE)

- [x] `plan request --dry-run` does not hard-fail on missing `@symbol`.
- [x] Missing symbols become steps marked unresolved via existing step structure.
  - [x] Sentinel in `expected_outputs`, for example `{"unresolved_symbols": ["@TEST/MISSING"]}`.
- [x] Dry-run writes nothing to DB:
  - [x] no `cassette_messages` inserts
  - [x] no plan persistence
  - [x] no step persistence
- [x] Tests:
  - [x] `test_plan_request_dry_run_missing_symbol_does_not_fail`

### Phase 4.2: Plan request rerun idempotency (COMPLETE)

- [x] Re-running identical request returns same `job_id`, `message_id`, `plan_hash`, and step ids and order.
- [x] No duplicate writes and no UNIQUE constraint crashes.
- [x] Uses existing idempotency surfaces (no new id tables invented).
- [x] Tests:
  - [x] `test_plan_request_idempotent_rerun_no_unique_constraint`

### Phase 4.5: Discovery (FTS plus vectors) (PENDING, optional)

- [ ] Deterministic FTS search over sections.
- [ ] Embedding retrieval behind governed boundary with hard budgets.
- [ ] Deterministic tie-break ordering and output caps.
- [ ] Discovery never bypasses verification and bounded expansion.

- [ ] Discovery outputs schema:
  - [ ] query, hits, ranks, snippet_slices, provenance
  - [ ] stable ordering and deterministic tie-breakers
- [ ] Search-to-slice pipeline:
  - [ ] discovery returns `(section_id or symbol_id, slice)` only
  - [ ] resolver retrieves exact slice content for the planner
  - [ ] no “full file paste” path exists
- [ ] Staleness detection (optional, recommended):
  - [ ] warn or fail when indexed file hash differs from current file hash
  - [ ] recommend `docs reindex` in a deterministic way


---

## Phase 5: Bundle (Translation Protocol) MVP (COMPLETE core)

- [x] `SCHEMAS/bundle.schema.json` with `additionalProperties: false` everywhere.
- [x] Bundle builder: deterministic construction from a completed job.
- [x] Bundle verifier: schema plus ordering plus hashes plus boundedness (fail-closed).
- [x] Job completeness gate:
  - [x] build fails unless all steps are final (COMMITTED or strict equivalent)
  - [x] receipts exist per step (exactly one per step, or strict equivalent)
- [x] Boundedness:
  - [x] include only artifacts referenced by plan steps (`READ_SYMBOL`, `READ_SECTION` or strict equivalent)
  - [x] include only exact slices used (request slice or default slice)
  - [x] reject any `ALL` slice at build and verify
- [x] Canonical JSON rules enforced for bundle.json.
- [x] Canonical bundle_id algorithm:
  - [x] build pre-manifest with `bundle_id=""`
  - [x] canonical serialize
  - [x] sha256 over bytes
  - [x] set `bundle_id` to hex digest
  - [x] reserialize canonically to final bundle.json
- [x] Deterministic root hash algorithm enforced.
- [x] CLI: `bundle build`, `bundle verify`.
- [ ] If still deferred: full end-to-end integration fixture setup for build and verify tests.


### Phase 5.X: Bundle consumption and replay (PENDING)

- [ ] Define a “bundle runner” contract:
  - [ ] takes `bundle.json` + `artifacts/` only
  - [ ] reproduces the same step outputs without reading repo files
  - [ ] emits receipts deterministically
- [ ] Verify-before-run:
  - [ ] runner must run `bundle verify` (or equivalent) and hard fail on any mismatch
- [ ] Reproducibility tests:
  - [ ] run twice produces identical receipts (or identical receipt hashes)
  - [ ] tampered artifact fails closed

---

## Phase 6: Deterministic execution, receipts, attestations, trust (COMPLETE through 6.14)

### Phase 6.0: Bundle verifier path hardening plus absolute-path support (COMPLETE)

- [x] Bundle path resolution accepts bundle dir, explicit `bundle.json`, absolute and relative paths (Windows-safe).
- [x] Artifact path traversal defenses (reject `..` escapes).
- [x] Fixed bundle_id verification correctness.

### Phase 6.1: Deterministic execution receipts (COMPLETE)

- [x] Canonical receipt schema and canonical bytes routine as single source of truth.
- [x] Executor emits receipts deterministically for executed steps.
- [x] CLI receipt output support.

### Phase 6.2: Ed25519 receipt attestation (COMPLETE)

- [x] Sign and verify receipt bytes using Ed25519 (PyNaCl).
- [x] Deterministic encoding of keys and signatures (hex only).
- [x] Fail-closed on invalid or unknown keys.

### Phase 6.3: Receipt chain hashing plus linkage verification (COMPLETE)

- [x] `receipt_hash` computed from canonical receipt bytes without attestation.
- [x] `parent_receipt_hash` links prior receipt (null at genesis).
- [x] Chain verification fail-closed on tamper or linkage mismatch.

### Phase 6.4: Deterministic Merkle root over receipt hashes (COMPLETE)

- [x] Deterministic Merkle root after full chain validation.
- [x] CLI support requires chain verification before printing Merkle root (stdout purity).

### Phase 6.5: Signed Merkle attestation (COMPLETE)

- [x] Deterministic message to sign: `b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes`.
- [x] CLI: attest and verify flags with strict interactions, fail-closed, stdout purity.

### Phase 6.6: Trust policy (pinned keys plus strict trust) (COMPLETE)

- [x] `trust_policy.schema.json` plus default `TRUST_POLICY.json`.
- [x] Deterministic, validated indexing with uniqueness rules.
- [x] Strict trust modes: receipts and Merkle attestations must chain to pinned keys with correct scope.

### Phase 6.7: Validator identity pinning hardening (COMPLETE, patched)

- [x] Optional identity fields:
  - [x] `validator_id`
  - [x] `build_id`
- [x] Patch 1: identity fields included in signed bytes (not mutable metadata).
- [x] Patch 2: trust lookup is validator_id primary when validator_id is present, and public_key must match that validator entry.
- [x] Tests cover mismatch cases (fail-closed).

### Phase 6.8: Execution policy gate (COMPLETE)

- [x] `execution_policy.schema.json`.
- [x] Policy loader and validator plus CLI composition.
- [x] Executor enforces policy in strict deterministic order:
  - [x] verify bundle
  - [x] execute steps
  - [x] verify chain
  - [x] verify receipt attestation
  - [x] verify Merkle attestation

### Phase 6.9: Stabilization (COMPLETE)

- [x] Remove flakiness in execution-policy tests.
- [x] Verified repeated runs produce identical pytest outcomes.

### Phase 6.10: Receipt chain ordering hardening (COMPLETE)

- [x] Receipt ordering explicit and filesystem-independent.
- [x] Fail-closed ambiguity detection (duplicate indices or hashes, mixed index and null).
- [x] Tests prove Merkle root independent of filesystem order.

### Phase 6.11: Strict monotonic receipt_index (COMPLETE)

- [x] Chain verification enforces:
  - [x] starts at 0
  - [x] contiguous `[0..n-1]`
  - [x] strict monotonic increments (no gaps)

### Phase 6.12: Remove filesystem dependence from receipt_index assignment (COMPLETE)

- [x] No directory scanning to assign receipt_index.
- [x] Executor uses provided receipt_index when chaining, otherwise defaults deterministically.

### Phase 6.13: Multi-validator aggregation (quorum) (COMPLETE)

- [x] Schema: `aggregated_merkle_attestations.schema.json` (`additionalProperties:false`).
- [x] Quorum verification for receipt attestations.
- [x] Quorum verification for Merkle attestations.
- [x] Deterministic ordering rules for validator sets plus duplicate rejection.
- [x] Fail-closed rules:
  - [x] reject duplicate validator_id or public_key (case-insensitive)
  - [x] reject identity to key mismatches vs policy
  - [x] reject any invalid signature in set
  - [x] reject insufficient quorum (count(valid) < threshold)
- [ ] Ensure minimum test matrix exists (quorum pass, insufficient quorum, duplicates, mismatches, tamper).

### Phase 6.14: External verifier UX (COMPLETE)

- [x] Standardized exit codes (single source of truth) across verify commands.
- [x] `--json` machine-readable verification output.
- [x] `--quiet` mode for stdout purity (human logs suppressed, status to stderr).
- [x] Deterministic JSON formatting in `--json` output (canonical bytes, newline at EOF).
- [ ] Ensure minimum test matrix exists (exit codes, json shape, quiet mode).

---

## Phase 6.X: Mechanical chat I/O (PROTOTYPE track)

> This track exists in the consolidated roadmap, but was explicitly excluded from the Phase 0-6 canonical scope. Keep it separate unless you intentionally promote it into Phase 7+.  

- [x] `chat_db.py`: SQLite chat DB (threaded messages, deterministic ordering).
- [x] `chat_translate.py`: deterministic Markdown exporter (code fences preserved).
- [x] CLI:
  - [x] `chatdb init`
  - [x] `chatdb put`
  - [x] `chatdb export`
  - [x] `chatdb directives` (stdout JSON directives)
- [x] Tests: `tests/test_chat_translate.py` (deterministic output; fail-closed validation).

Backlog for chat I/O:

- [ ] Wire `chatdb directives` into planner and bundler and executor loop without introducing nondeterminism.
- [ ] Add tail-mode exporter (append-only view) if desired.
- [ ] Add `--since` plus deterministic paging guarantees for huge threads.

---

## Phase 7: Freeze, spec, and golden demo (PENDING)

- [ ] Authoritative specs that match reality (no drift):
  - [ ] Bundle protocol spec
  - [ ] Receipts plus chain spec
  - [ ] Trust plus identity spec
  - [ ] Execution policy spec
- [ ] Runbook: copy-paste runnable on Windows PowerShell.
- [ ] Golden demo from fresh clone:
  - [ ] plan request
  - [ ] execute
  - [ ] bundle build plus verify
  - [ ] bundle run
  - [ ] receipt chain verify
  - [ ] Merkle attest plus verify
  - [ ] quorum verify (if used)
- [ ] Packaging hardening so repo-root execution works with only `PYTHONPATH=THOUGHT\\LAB\\CAT_CHAT` and no cwd assumptions.

---

## Phase 8: Measurement and benchmarking (PENDING)

- [ ] Per-step metrics: bytes expanded, cache hit rate, reuse rate, plan and bundle size.
- [ ] Determinism regression suite across index, symbols, receipts, bundles.
- [ ] Benchmark scenarios: patch one function, refactor N files, generate bundles from corpus.
- [ ] Dashboard or reports: token savings and boundedness compliance over baselines.

---

## Recommended next work (priority order)

- [ ] Phase 7: freeze specs plus runbook plus golden demo.
- [ ] Decide whether to promote Phase 6.X mechanical chat I/O into Phase 7 plan or keep as prototype track.
- [ ] Phase 4.5 discovery (FTS first; vectors only behind governed, bounded outputs).
- [ ] Phase 8 measurement (prove compression wins with numbers).
