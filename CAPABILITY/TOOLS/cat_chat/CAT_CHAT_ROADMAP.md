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

## Phase 6.3 — Receipt Chain Anchoring (COMPLETE 2025-12-30)

Goal: extend receipt + attestation system to support deterministic, verifiable receipt chaining across executions.

### Completed

- [x] **Chain fields** in receipt schema
  - `parent_receipt_hash`: hash of previous receipt (null for first receipt)
  - `receipt_hash`: SHA256 of canonical receipt bytes without attestation

- [x] **Executor chain support** in `executor.py`
  - `previous_receipt` parameter to `BundleExecutor.__init__()`
  - Loads previous receipt and extracts `receipt_hash` for `parent_receipt_hash`
  - First receipt has `parent_receipt_hash=null`

- [x] **Chain verification** in `receipt.py`
  - `compute_receipt_hash()`: deterministic hash computation excluding `receipt_hash` field
  - `load_receipt()`: load receipt from file
  - `verify_receipt_chain()`: verifies chain linkage and receipt hashes
  - `find_receipt_chain()`: finds and loads all receipts for a run

- [x] **Schema update** in `SCHEMAS/receipt.schema.json`
  - Added `parent_receipt_hash` (string | null)
  - Added `receipt_hash` (string)

- [x] **CLI chain verification** in `cli.py`
  - `--verify-chain` flag to verify full receipt chain
  - Chain linkage verified with attestation signature checks

- [x] **Tests** in `tests/test_receipt_chain.py`
  - `test_receipt_chain_deterministic`: identical inputs produce identical chain
  - `test_receipt_chain_verification_passes`: chain verification succeeds
  - `test_receipt_chain_break_fails`: tamper detection
  - `test_receipt_chain_requires_sequential_order`: reordering fails

Exit criteria
- [x] Identical inputs and execution order produce identical receipt chain bytes
- [x] Fail-closed: any break in chain or hash mismatch hard-fails
- [x] No timestamps, randomness, absolute paths, or environment-dependent data
- [x] Minimal diffs; only extended receipt/verify paths

---

## Phase 6.5 — Signed Merkle Attestation (COMPLETE 2025-12-30)

Goal: add Ed25519 signing and verification of Merkle root with strict stdout purity and deterministic output.

### Completed

- [x] **Schema** in `SCHEMAS/merkle_attestation.schema.json`
  - `scheme`: const `"ed25519"`
  - `merkle_root`: 64 hex chars (SHA256)
  - `public_key`: 64 hex chars (Ed25519)
  - `signature`: 128 hex chars (Ed25519)
  - Optional fields: `run_id`, `job_id`, `bundle_id`, `receipt_count`, `receipt_chain_head_hash`
  - `additionalProperties: false` everywhere

- [x] **Merkle attestation module** in `catalytic_chat/merkle_attestation.py`
  - `sign_merkle_root()`: sign merkle root with Ed25519
    - Message: `b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes` (decoded 32 bytes)
    - Hex output for public_key and signature
  - `verify_merkle_attestation()`: verify signature
    - Validates hex, lengths, scheme
  - `validate_merkle_root_hex()`: strict hex length validation
  - `write_merkle_attestation()`: write canonical JSON with trailing newline
  - `load_merkle_attestation()`: load from file

- [x] **CLI integration** in `cli.py`
  - `--attest-merkle`: sign merkle root and emit attestation (requires `--verify-chain` and `--merkle-key`)
  - `--merkle-key <hex>`: Ed25519 signing key (64 hex chars)
  - `--verify-merkle-attestation <path>`: verify attestation file (requires `--verify-chain`)
  - `--merkle-attestation-out <path>`: write attestation to file (default: stdout)
  - Strict flag interactions:
    - `--print-merkle` and `--attest-merkle` mutually exclusive
    - `--attest-merkle` and `--verify-merkle-attestation` mutually exclusive
  - Stdout purity:
    - `--attest-merkle` without `--merkle-attestation-out`: ONLY JSON + \n to stdout
    - `--attest-merkle` with `--merkle-attestation-out`: status to stdout, JSON to file
    - `--print-merkle`: ONLY merkle_root hex + \n to stdout
  - All errors go to stderr via `sys.stderr.write()`

- [x] **Tests** in `tests/test_merkle_attestation.py`
  - `test_merkle_attestation_sign_verify_roundtrip`: roundtrip succeeds
  - `test_merkle_attestation_rejects_modified_root`: tamper fails
  - `test_merkle_attestation_rejects_invalid_merkle_root_length`: wrong length rejected
  - `test_merkle_attestation_rejects_invalid_signing_key_length`: wrong key length rejected
  - `test_merkle_attestation_rejects_invalid_hex`: non-hex rejected
  - `test_merkle_attestation_verify_rejects_wrong_scheme`: non-ed25519 rejected
  - `test_merkle_attestation_verify_rejects_wrong_key_length`: wrong pub key length rejected
  - `test_merkle_attestation_verify_rejects_wrong_signature_length`: wrong sig length rejected
  - `test_bundle_run_attest_merkle_outputs_deterministic_bytes`: identical outputs across runs
  - `test_bundle_run_verify_merkle_attestation_fails_on_mismatch`: root mismatch fails
  - `test_merkle_attestation_write_load_roundtrip`: file I/O preserves attestation
  - `test_merkle_attestation_load_nonexistent_file`: missing file returns None

Exit criteria
- [x] Deterministic: identical merkle root + key produce identical signature bytes
- [x] Fail-closed: any chain tamper, ordering mismatch, schema mismatch, signature mismatch => non-zero exit
- [x] No timestamps, randomness, absolute paths in outputs
- [x] Signing does NOT change receipt_hash or merkle root computation
- [x] Canonicalization single source of truth: reuses `receipt_canonical_bytes()` and `compute_merkle_root()`
- [x] Stdout purity: `--attest-merkle` prints ONLY JSON + \n when no file output
- [x] All 12 tests passing

---

## Phase 6.6 — Validator Identity Pinning + Trust Policy (COMPLETE 2025-12-30)
Goal: add deterministic, governed trust policy that pins which validator public keys are allowed to attest receipts and Merkle roots.

### Completed

- [x] **Trust policy schema** in `SCHEMAS/trust_policy.schema.json`
  - `policy_version`: const `"1.0.0"`
  - `allow`: array of pinned validator entries
  - Validator entry: `validator_id`, `public_key` (64 hex chars), `schemes` (must include `ed25519`), `scope` (`RECEIPT` or `MERKLE`), `enabled`
  - `additionalProperties: false` everywhere

- [x] **Trust policy loader + verifier** in `catalytic_chat/trust_policy.py`
  - `load_trust_policy_bytes(path)`: read exact bytes, fail if missing
  - `parse_trust_policy(policy_bytes)`: validate against schema using jsonschema
  - `build_trust_index(policy)`: return deterministic index mapping lowercase public_key → entry
  - Enforce uniqueness: `validator_id` and `public_key` must be unique (case-insensitive)
  - `is_key_allowed(index, public_key_hex, scope, scheme)`: check if key is pinned for scope
  - Default policy path: `CORTEX/_generated/TRUST_POLICY.json`
  - CLI override: `--trust-policy <path>`

- [x] **Receipt attestation strict trust** in `catalytic_chat/attestation.py`
  - `verify_receipt_attestation(receipt, trust_index, strict)`
  - If attestation is None/absent: always OK
  - If attestation exists:
    - Always validate signature correctness (existing behavior)
    - If `strict=True`:
      - `trust_index` MUST be provided
      - Public key MUST be pinned with scope including `RECEIPT`
      - Fail-closed with `AttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned
    - If `strict=False`: signature validity only (no trust policy required)

- [x] **Merkle attestation strict trust** in `catalytic_chat/merkle_attestation.py`
  - `verify_merkle_attestation_with_trust(att, merkle_root_hex, trust_index, strict)`
  - Always validate signature and merkle root match
  - If `strict=True`:
    - `trust_index` MUST be provided
    - Public key MUST be pinned with scope including `MERKLE`
    - Fail-closed with `MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned

- [x] **CLI trust commands** in `cli.py`
  - `trust init`: initialize empty `TRUST_POLICY.json` to default location
    - Deterministic content (no timestamps)
    - `allow: []`
    - Output: stderr `[OK] wrote TRUST_POLICY.json`, exit 0
  - `trust verify [--trust-policy <path>]`: validate policy against schema + uniqueness rules
    - Output: stderr `[OK] trust policy valid` or `[FAIL] <reason>`, exit 1
  - `trust show [--trust-policy <path>]`: print canonical JSON summary to stdout ONLY
    - Summary: `policy_version`, `enabled` count, `scopes` (RECEIPT/MERKLE counts)
    - Uses `canonical_json_bytes()` with trailing `\n`

- [x] **Bundle run strict trust flags** in `cli.py`
  - `--trust-policy <path>`: override default policy path
  - `--strict-trust`: enable strict trust verification (fail-closed if policy missing/invalid)
  - `--require-attestation`: receipt attestation MUST be present or fail
  - `--require-merkle-attestation`: merkle attestation MUST be present and valid or fail
  - Default behavior compatible:
    - No `--strict-trust`: do not require policy
    - No require flags: do not require attestations
  - Path traversal defense: no escaping intended directory when loading from bundle context
  - Stdout purity: machine JSON output (e.g., `--attest-merkle` without `--merkle-attestation-out`) contains ONLY JSON + `\n`

- [x] **Tests** in `tests/test_trust_policy.py`
  - `test_trust_policy_schema_and_uniqueness`: validate schema and duplicate detection
    - Duplicate `public_key` (case-insensitive) → verify fails
    - Duplicate `validator_id` → verify fails
    - Valid empty `allow` → passes
  - `test_receipt_attestation_strict_trust_blocks_unknown_key`: strict blocks unpinned keys
    - Generate receipt + attestation with SigningKey
    - Build trust policy without that pubkey
    - `verify_receipt_attestation(strict=True)` → `UNTRUSTED_VALIDATOR_KEY`
    - `verify_receipt_attestation(strict=False)` → passes (signature valid)
  - `test_receipt_attestation_strict_trust_allows_pinned_key`: strict allows pinned keys
    - Same as above but trust policy includes pubkey with `RECEIPT` scope → passes
  - `test_merkle_attestation_strict_trust_blocks_unknown_key_and_allows_pinned`: merkle strict trust
    - Generate merkle root + sign it
    - Verify strict fails without pin, passes with pin and `MERKLE` scope
  - `test_cli_trust_verify`: CLI smoke test
    - Use `subprocess.run` to call `trust verify` with `PYTHONPATH` env
    - Assert exit codes
  - `test_cli_trust_show`: CLI smoke test
    - Verify stdout JSON structure

Exit criteria
- [x] Deterministic: identical inputs + same trust policy → identical results
- [x] Fail-closed: any untrusted/unknown key fails when strict trust enabled
- [x] No timestamps, randomness, absolute paths, or environment-dependent behavior in outputs
- [x] Trust policy schema enforces `policy_version="1.0.0"`, 64-char hex public keys, ed25519 scheme, scopes limited to RECEIPT/MERKLE
- [x] Trust policy loader enforces unique `validator_id` and `public_key` (case-insensitive)
- [x] Receipt attestation verification: always validates signature; strict mode requires pinned key with RECEIPT scope
- [x] Merkle attestation verification: always validates signature + root; strict mode requires pinned key with MERKLE scope
- [x] CLI trust commands (`init`, `verify`, `show`) work deterministically
- [x] Bundle run strict trust flags enforce policy, require-attestation, require-merkle-attestation
- [x] Default behavior compatible (no `--strict-trust` = no policy required)
- [x] Stdout purity: machine JSON commands output ONLY JSON + `\n`, all status to stderr
- [x] All 6 tests passing

---

## Phase 6.7 — Sanity Check Fixes (COMPLETE 2025-12-30)

Goal: fix critical bugs in attestation and merkle verification that were preventing proper signature validation.

**Related:**
- Commit plan: `commit-plan-phase-6-sanity-checks.md`

### Completed

- [x] **Syntax error fix** in `catalytic_chat/attestation.py`
  - Removed duplicate code at lines 280-293
  - Module imports successfully without SyntaxError

- [x] **Receipt verification logic fix** in `catalytic_chat/attestation.py`
  - `verify_receipt_bytes()` now reconstructs signing stub properly
  - Uses `receipt_signed_bytes()` instead of `receipt_canonical_bytes(attestation_override=None)`
  - Ensures verification uses same message structure as signing (VID, BID, PK fields)

- [x] **Merkle verification message exactness** in `catalytic_chat/merkle_attestation.py`
  - `verify_merkle_attestation()` now includes VID, BID, PK fields in message
  - Message: `b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes + b"|VID:" + vid_bytes + b"|BID:" + bid_bytes + b"|PK:" + vk_bytes`

- [x] **Key validation fix** in `catalytic_chat/merkle_attestation.py`
  - Corrected hex length check: `64, 128` hex chars for 32/64 byte keys
  - Previous check was for byte lengths, not hex char lengths

- [x] **Trust identity test fix** in `tests/test_trust_identity_patch.py`
  - `test_trust_lookup_requires_validator_id_match_when_present` now correctly uses mismatched validator_id
  - Changed from mutating attestation to signing with mismatched validator_id directly

Exit criteria
- [x] Syntax errors eliminated (module imports successfully)
- [x] Receipt verification uses correct signing stub
- [x] Merkle verification uses same message as signing
- [x] Key validation correct for hex lengths
- [x] Trust identity test correctly verifies rejection
- [x] Attestation tests passing (6/6)
- [x] Merkle attestation tests passing (12/12)
- [x] Full test suite: 103 passed, 13 skipped

---

## Phase 6.10 — Receipt Chain Ordering Hardening (COMPLETE 2025-12-31)

Goal: implement deterministic, fail-closed receipt chain ordering with explicit sorting keys and ambiguity detection. Remove all reliance on filesystem iteration order.

**Related:**
- Commit plan: `commit-plan-phase-6-10-receipt-chain-ordering.md`

### Completed

- [x] **Schema enhancement** in `SCHEMAS/receipt.schema.json`
  - Added `receipt_index` field (type: integer|null)
  - This field provides explicit sequential ordering for receipt chains
  - Marked as optional (null allowed) for backward compatibility

- [x] **Receipt chain discovery** in `catalytic_chat/receipt.py`
  - `find_receipt_chain()` now uses explicit ordering key with priority:
    1. receipt_index (if present and not null)
    2. receipt_hash (if receipt_index is null)
    3. filename (final fallback only, not used in proper chains)
  - Detects duplicate ordering keys and raises ValueError
  - Removed reliance on filesystem `sorted()` of glob results
  - Detects mixed receipt_index/null state and fails

- [x] **Receipt chain verification** in `catalytic_chat/receipt.py`
  - `verify_receipt_chain()` enhanced with receipt_index monotonic validation
  - Checks all receipts either have receipt_index or all are null
  - Verifies receipt_index is strictly increasing when in use
  - Maintains existing hash linking and attestation verification

- [x] **Signed bytes helper** in `catalytic_chat/receipt.py`
  - `receipt_signed_bytes()` implemented to support attestation flow
  - Extracts attestation fields (scheme, public_key, validator_id, build_id)
  - Builds signing stub with identity fields
  - Ensures identity fields are part of signed message

- [x] **Executor support** in `catalytic_chat/executor.py`
  - Added `"receipt_index": None` to receipt creation
  - Schema-compliant receipts now include receipt_index field

- [x] **Tests** in `tests/test_receipt_chain_ordering.py` (fully rewritten)
  - `test_receipt_chain_sorted_explicitly`: deterministic order regardless of FS order
  - `test_receipt_chain_fails_on_duplicate_receipt_index`: duplicate index detection
  - `test_receipt_chain_fails_on_mixed_receipt_index`: mixed index/null detection
  - `test_merkle_root_independent_of_fs_order`: filesystem independence
  - `test_verify_receipt_chain_strictly_monotonic`: strictly increasing index enforcement

Exit criteria
- [x] Deterministic ordering: identical inputs produce identical receipt order regardless of filesystem creation order
- [x] Fail-closed: duplicate receipt_index values raise ValueError
- [x] Fail-closed: mixed receipt_index/null state raises ValueError
- [x] Filesystem independence: order determined by receipt_index, not creation order
- [x] Strict monotonicity: receipt_index sequence must be strictly increasing
- [x] Minimal diffs: only extended receipt chain code
- [x] All 5 new tests passing
- [x] Full test suite: 118 passed, 13 skipped

---

## Phase 6.8 — Execution Policy Gate (PENDING)

Goal: unify bundle run verification requirements into a deterministic policy that fail-closes correctly, eliminating scattered flag logic.

**Related:**
- Prompt: Phase 6.8 specification (execution policy as single source of truth)

### Pending

- [ ] **Execution policy schema** in `SCHEMAS/execution_policy.schema.json`
  - `policy_version`: const `"1.0.0"`
  - `require_verify_bundle`: boolean
  - `require_verify_chain`: boolean
  - `require_receipt_attestation`: boolean
  - `require_merkle_attestation`: boolean
  - `strict_trust`: boolean
  - `strict_identity`: boolean
  - `trust_policy_path`: string | null
  - `additionalProperties: false` everywhere

- [ ] **Execution policy module** in `catalytic_chat/execution_policy.py`
  - `load_execution_policy(path)`: load and validate policy from file
  - `policy_from_cli_args(args)`: compile CLI flags into policy dict
  - `validate_policy(policy)`: fail-closed with clear errors
  - Rule: If `strict_trust` true → `trust_policy_path` required and valid

- [ ] **CLI policy integration** in `cli.py`
  - `--policy <path>`: load policy file for bundle run
  - Existing flags remain for backwards compatibility
  - All flags compile into single policy dict and validate once

- [ ] **Executor policy enforcement** in `executor.py`
  - Accept policy dict in run/execute entrypoint
  - Enforce in this order (fail-closed):
    a) If `require_verify_bundle`: verify bundle
    b) Execute steps deterministically using bundle artifacts only
    c) If `require_verify_chain`: verify receipt chain and compute merkle root
    d) If `require_receipt_attestation`: ensure each receipt has non-null attestation and verify
    e) If `require_merkle_attestation`: verify merkle attestation

- [ ] **Tests** in `tests/test_execution_policy.py`
  - `test_policy_requires_trust_policy_when_strict_trust`
  - `test_policy_fails_if_receipt_attestation_missing_when_required`
  - `test_policy_fails_if_merkle_attestation_missing_when_required`
  - `test_policy_passes_full_stack_when_all_requirements_met`
  - `test_policy_cli_backcompat_compiles_to_same_policy`

Exit criteria
- [ ] Verification requirements unified into single policy object
- [ ] One decision point for policy enforcement in executor
- [ ] No scattered flag logic outside policy flow
- [ ] Fail-closed on any policy requirement not met
- [ ] Deterministic behavior (no timestamps/randomness/absolute paths)
- [ ] Backwards compatible: existing flags compile to same policy
- [ ] Stdout purity: JSON-only mode outputs only JSON
- [ ] All 5 tests passing

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
