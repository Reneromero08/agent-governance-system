# Catalytic Chat Changelog

All notable changes to Catalytic Chat System are documented in this file.

---

## [Unreleased] - 2025-12-31

### Added - Phase 7 (Compression Protocol Formalization) - COMPLETE
- **`THOUGHT/LAB/CAT_CHAT/PHASE_7_COMPRESSION_SPEC.md`** (320 lines)
  - Authoritative specification for compression protocol
  - Defines compression metric definitions (ratio, numerator/denominator)
  - Component definitions: vector_db_only, symbol_lang, f3, cas
  - Reconstruction procedures
  - Invariants from Phase 6 (canonical JSON, bundle_id, receipts, trust policy, merkle root, execution policy)
  - Threat model (what IS proven vs NOT proven)
  - Verification checklist (8 phases with 20+ checks)
  - Error codes table (28 error codes)
  - Deterministic computation rules (token estimation, artifact ordering, hash computation)
  - Fail-closed behavior

- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/compression_claim.schema.json`** (67 lines)
  - JSON schema with `additionalProperties: false` everywhere
  - Required fields: claim_version, run_id, bundle_id, components, reported_metrics, claim_hash
  - Component definitions (vector_db_only, symbol_lang, f3, cas)
  - Reported metrics (compression_ratio, uncompressed_tokens, compressed_tokens, artifact_count, total_bytes)
  - Optional component metrics (vector_db_tokens, symbol_lang_tokens, f3_tokens, cas_tokens)
  - F3 marked as theoretical (validator fails if included)
  - Stable identifiers, no timestamps

- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/compression_validator.py`** (470 lines)
  - `CompressionValidator` class with 8-phase verification pipeline
  - Entry function: `validate_compression_claim()`
  - Phases: Input validation → Claim schema → Bundle verification → Trust policy → Receipts → Attestations → Metrics → Claim
  - Deterministic metric computation from verified artifacts only
  - 28 error codes with explicit messages
  - Supports strict_trust, strict_identity, require_attestation flags
  - Reuses: BundleVerifier, find_receipt_chain, verify_receipt_chain, receipt_canonical_bytes, canonical_json_bytes

- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py`** (+~50 lines)
  - Added `compress verify` command
  - Command: `python -m catalytic_chat.cli compress verify --bundle <path> --receipts <dir> --trust-policy <path> --claim <json> [--strict-trust] [--strict-identity] [--require-attestation] [--json] [--quiet]`
  - Exit codes: 0 (OK), 1 (verification failed), 2 (invalid input), 3 (internal error)
  - Output modes: Human-readable (default) or JSON (--json, pure stdout with canonical JSON)

- **`THOUGHT/LAB/CAT_CHAT/tests/test_compression_validator.py`** (400+ lines)
  - 5 test functions:
    1. `test_estimate_tokens()` - Token estimation function
    2. `test_compression_verify_passes_on_matching_claim()` - Pass case with matching metrics
    3. `test_compression_verify_fails_on_metric_mismatch()` - Fail on metric mismatch
    4. `test_compression_verify_fails_if_not_strictly_verified()` - Fail on missing receipts
    5. `test_compression_outputs_deterministic()` - Deterministic JSON output
  - Uses existing fixtures from test bundle

### Added - Phase 6.13 — Multi-Validator Aggregation
- **Multi-validator attestations for quorum validation** (RECEIPT + MERKLE)
  - `SCHEMAS/receipt.schema.json` - added optional `attestations[]` array
  - `SCHEMAS/execution_policy.schema.json` - added `receipt_attestation_quorum` and `merkle_attestation_quorum`
  - `SCHEMAS/aggregated_merkle_attestations.schema.json` - new schema for aggregated attestations
  - `catalytic_chat/attestation.py` - added `verify_receipt_attestation_single()` and `verify_receipt_attestations_with_quorum()`
  - `catalytic_chat/merkle_attestation.py` - added `load_aggregated_merkle_attestations()` and `verify_merkle_attestations_with_quorum()`
  - `tests/test_multi_validator_attestations.py` - comprehensive tests for ordering, quorum, backward compatibility
  - Deterministic ordering by (validator_id, public_key.lower(), build_id or "")
  - Reuses existing trust policy (strict_trust, strict_identity)
  - Purely additive: single attestation path unchanged

### Added - Phase 6.14 — External Verifier UX Improvements
- **CI-friendly output modes and machine-readable summaries**
  - `catalytic_chat/cli_output.py` - new module with standardized exit codes and JSON output helpers
  - `catalytic_chat/cli.py` - added `--json` and `--quiet` flags to `bundle verify`, `bundle run`, `trust verify`
  - `catalytic_chat/__main__.py` - entry point for `python -m catalytic_chat.cli`
  - `tests/test_cli_output.py` - tests for JSON stdout purity, exit code classification, deterministic outputs
  - Standardized exit codes: 0 (OK), 1 (verification failed), 2 (invalid input), 3 (internal error)
  - JSON output uses `canonical_json_bytes()` for deterministic field ordering
  - No verification behavior changes; purely additive UX improvements

### Fixed - Phase 6.12 — Receipt Index Determinism (Redo)
- **Executor-derived receipt_index with no caller control**
  - `catalytic_chat/executor.py` - removed `receipt_index` parameter from `__init__()`
  - `catalytic_chat/executor.py` - removed `_find_next_receipt_index()` (no filesystem scanning)
  - `catalytic_chat/executor.py` - always assigns `receipt_index = 0` deterministically
  - `tests/test_receipt_index_propagation.py` - updated tests to verify receipt_index=0 with no caller control
  - receipt_index is executor-derived, no filesystem scanning, pure execution order function

### Added - Phase 6.11 — Receipt Index Propagation
- **Deterministic receipt_index assignment with strict verification**
  - `catalytic_chat/executor.py` - assign receipt_index deterministically (no filesystem scanning)
  - `catalytic_chat/receipt.py` - hardened `verify_receipt_chain()` to enforce contiguity, start at 0, strictly increasing
  - `tests/test_receipt_index_propagation.py` - tests for contiguous index enforcement, gap detection, nonzero start, mixed null/int
  - All receipt_index rules are fail-closed; indices must be exactly [0, 1, 2, ..., n-1] when present

### Added - Phase 6.10 — Receipt Chain Ordering Hardening
- **Deterministic, fail-closed receipt chain ordering**
  - `SCHEMAS/receipt.schema.json` - added `receipt_index` field (integer | null) for explicit sequential ordering
  - `catalytic_chat/receipt.py` - `find_receipt_chain()` enhanced:
    - Explicit ordering key with priority: receipt_index > receipt_hash > filename
    - Duplicate receipt_index detection (raises ValueError)
    - Mixed receipt_index/null detection (raises ValueError)
    - Duplicate receipt_hash detection (raises ValueError)
  - `catalytic_chat/receipt.py` - `verify_receipt_chain()` enhanced:
    - Strict monotonic validation: receipt_index must be strictly increasing
    - Mixed receipt_index/null detection (raises ValueError)
    - All receipts must have receipt_index set or all must be null
  - `catalytic_chat/receipt.py` - `receipt_signed_bytes()` implemented:
    - Extracts identity fields from attestation
    - Builds signing stub with scheme, public_key, validator_id, build_id
    - Ensures identity fields are included in signed message
  - `catalytic_chat/executor.py` - added `"receipt_index": None` to receipt creation
  - `tests/test_receipt_chain_ordering.py` - 5 deterministic ordering tests

### Changed - Phase 6.10-6.12
- `catalytic_chat/receipt.py` - `build_receipt_from_bundle()`:
  - Added `"receipt_index": None` to receipt dict
- `catalytic_chat/receipt.py` - `find_receipt_chain()`:
  - Removed reliance on filesystem `sorted()` of glob results
  - Uses explicit ordering function for deterministic sorting
- `catalytic_chat/receipt.py` - `compute_merkle_root()`:
  - No longer performs internal re-sorting (caller responsibility)
  - Maintains existing behavior for consumption of ordered input

### Security (Phase 6.10-6.12)
- Fail-closed behavior: duplicate receipt_index raises ValueError
- Fail-closed behavior: mixed receipt_index/null raises ValueError
- Strict monotonicity: receipt_index sequence must be strictly increasing
- Filesystem independence: order determined by receipt_index, not creation order

### Tests (Phase 6.10-6.14)
- 118 passed, 13 skipped (full test suite)
- All 5 new receipt chain ordering tests passing
- All existing receipt chain tests passing

---

## [2025-12-30] - Phase 6.5 Complete

### Added - Phase 6.5 — Signed Merkle Attestation
- **Ed25519 signing and verification of receipt chain Merkle root**
  - `catalytic_chat/merkle_attestation.py` module with `sign_merkle_root()` and `verify_merkle_attestation()`
  - `SCHEMAS/merkle_attestation.schema.json` - JSON schema for merkle attestations
    - Strict hex length validation: merkle_root (64), public_key (64), signature (128)
    - `scheme` const `"ed25519"` with `additionalProperties: false`
  - CLI flags for `bundle run`:
    - `--attest-merkle`: sign merkle root after chain verification
    - `--merkle-key <hex>`: Ed25519 signing key (64 hex chars)
    - `--verify-merkle-attestation <path>`: verify attestation against computed root
    - `--merkle-attestation-out <path>`: write attestation to file
  - Strict stdout purity: `--attest-merkle` prints ONLY JSON + \n when no file output specified
  - `--print-merkle` prints ONLY merkle_root hex + \n to stdout
  - Fail-closed verification: chain tamper, ordering violation, root mismatch, signature failure => non-zero exit

### Changed - Phase 6.5
- `catalytic_chat/cli.py` - extended `cmd_bundle_run()` with merkle attestation support
  - Added `verbose_output` flag to suppress status messages when JSON-only output required
  - All error messages now use `sys.stderr.write()` for consistency
  - Mutually exclusive flag enforcement for stdout purity

### Tests - Phase 6.5
- `tests/test_merkle_attestation.py` - 12 comprehensive tests
  - Sign/verify roundtrip
  - Modified root detection
  - Input validation (hex, lengths, scheme)
  - Deterministic output across multiple runs
  - Mismatch detection
  - File I/O roundtrip

### Security - Phase 6.5
- Message to sign: `b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes` (decoded 32 bytes)
- Hex decoding uses `bytes.fromhex()` - NOT ASCII bytes of hex string
- Canonical JSON output using `receipt.canonical_json_bytes()` (sort_keys, separators, UTF-8, \n EOF)

---

## [2025-12-30] - Phase 6.6 Complete

### Added - Phase 6.6 — Validator Identity Pinning + Trust Policy
- **Deterministic, governed trust policy that pins which validator public keys are allowed**
  - `SCHEMAS/trust_policy.schema.json` - JSON schema for trust policy
    - Enforces `policy_version="1.0.0"` (const)
    - Defines `allow` array of pinned validator entries
    - Each entry requires:
      - `validator_id`: stable human label
      - `public_key`: 64-char hex (pattern `^[0-9a-fA-F]{64}$`)
      - `schemes`: array containing `"ed25519"`
      - `scope`: array of `"RECEIPT"` and/or `"MERKLE"`
      - `enabled`: boolean
    - `additionalProperties: false` everywhere

  - `catalytic_chat/trust_policy.py` - Trust policy loader + verifier:
    - `load_trust_policy_bytes(path)`: Reads exact bytes, fails if missing
    - `parse_trust_policy(policy_bytes)`: Validates against schema using jsonschema
    - `build_trust_index(policy)`: Returns deterministic index mapping lowercase public_key → entry
      - Enforces uniqueness of `validator_id` and `public_key` (case-insensitive)
      - Raises `TrustPolicyError` on duplicates
    - `is_key_allowed(index, public_key_hex, scope, scheme)`: Checks if key is allowed for scope
    - `Default path`: `THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/TRUST_POLICY.json`
    - `CLI override`: `--trust-policy <path>` supports absolute paths (never embedded in outputs)

- **Receipt Attestation Strict Trust**
  - `catalytic_chat/attestation.py` - New function:
    ```python
    verify_receipt_attestation(receipt: dict, trust_index: Optional[dict], strict: bool) -> None
    ```
  - **Rules:**
    - If `receipt["attestation"]` is `null`/`None`: Always OK (no trust needed)
    - If attestation exists:
      - Always validate signature correctness (existing behavior preserved)
      - If `strict == True`:
        - `trust_index` MUST be provided
        - Attesting `public_key` MUST be pinned with scope including `"RECEIPT"`
        - Fail-closed with `AttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned
      - If `strict == False`: Signature validity only (no trust policy required)

- **Merkle Attestation Strict Trust**
  - `catalytic_chat/merkle_attestation.py` - New function:
    ```python
    verify_merkle_attestation_with_trust(att: dict, merkle_root_hex: str, trust_index: Optional[dict], strict: bool) -> None
    ```
  - **Rules:**
    - Always validate signature correctness and merkle root match (existing behavior preserved)
    - If `strict == True`:
      - `trust_index` MUST be provided
      - `public_key` MUST be pinned with scope including `"MERKLE"`
      - Fail-closed with `MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned

- **CLI: Trust Commands + Strict Verification Flags**
  - `catalytic_chat/cli.py` - New command group: `python -m catalytic_chat.cli trust {init,verify,show}`:
    - `trust init`:
      - Writes starter `TRUST_POLICY.json` to default path
      - Deterministic content (no timestamps)
      - `allow: []`
      - Output: stderr `[OK] wrote TRUST_POLICY.json`, exit 0
    - `trust verify [--trust-policy <path>]`:
      - Validates policy against schema + uniqueness rules
      - Output: stderr `[OK] trust policy valid` or `[FAIL] <reason>`, exit 1
    - `trust show [--trust-policy <path>]`:
      - Prints canonical JSON summary to stdout ONLY (machine output)
      - Uses `canonical_json_bytes()` with trailing `\n`

  - `bundle run` new flags:
    - `--trust-policy <path>`: Override default policy path
    - `--strict-trust`: Enable strict trust verification (fail-closed if policy missing/invalid)
    - `--require-attestation`: Receipt attestation MUST be present or fail
    - `--require-merkle-attestation`: Merkle attestation MUST be present and valid or fail

  - **Rules:**
    - If `--strict-trust`: Load trust policy, build index; fail-closed if missing/invalid
    - If `--require-attestation`: Fail if receipt attestation absent
    - If `--require-merkle-attestation`: Fail if merkle attestation absent or invalid
    - Default behavior compatible:
      - No `--strict-trust`: Do not require policy
      - No require flags: Do not require attestations
    - Stdout purity: Machine JSON commands (e.g., `--attest-merkle` without `--merkle-attestation-out`) output ONLY JSON + `\n`; all status to stderr

  - **Invariant enforced:** Receipt attestation verification always validates signature; strict mode adds trust check without weakening existing verification

### Tests - Phase 6.6
- `tests/test_trust_policy.py` - Comprehensive test coverage:
  1. `test_trust_policy_schema_and_uniqueness`:
    - Valid empty `allow` policy passes
    - Duplicate `public_key` (case-insensitive) → verify fails
    - Duplicate `validator_id` → verify fails
  2. `test_receipt_attestation_strict_trust_blocks_unknown_key`:
    - Generate receipt + attestation using SigningKey
    - Build trust policy without that pubkey
    - `verify_receipt_attestation(strict=True)` → `UNTRUSTED_VALIDATOR_KEY`
    - `verify_receipt_attestation(strict=False)` → passes (signature valid)
  3. `test_receipt_attestation_strict_trust_allows_pinned_key`:
    - Same as above but trust policy includes pubkey with `RECEIPT` scope
    - `verify_receipt_attestation(strict=True)` → passes
  4. `test_merkle_attestation_strict_trust_blocks_unknown_key_and_allows_pinned_key`:
    - Generate merkle root + sign it
    - Verify strict fails without pin, passes with pin and `MERKLE` scope
  5. `test_cli_trust_verify`:
    - Use `subprocess.run` to call `trust verify --trust-policy <tmpfile>`
    - Assert exit codes
  6. `test_cli_trust_show`:
    - Verify stdout JSON structure and counts

### Verification Results - Phase 6.6
- Full test suite:
  - 84 passed, 13 skipped in 6.45s
- Trust policy tests:
  - 6 passed in 0.83s

### Invariants Enforced - Phase 6.6
- **Determinism**: All modules - Identical inputs + same trust policy → identical results
- **Fail-closed**: All verification paths - Unknown keys fail when strict trust enabled
- **Schema validation**: `trust_policy.py` - Policy must conform to JSON schema
- **Uniqueness**: `trust_policy.py` - No duplicate `validator_id` or `public_key` (case-insensitive)
- **No existing verification weakened**: `attestation.py`, `merkle_attestation.py` - Signature validation always occurs; trust check is additional
- **Stdout purity**: `cli.py` - Machine JSON outputs ONLY JSON + `\n`; status to stderr
- **Path traversal defense**: `cli.py`, `trust_policy.py` - No escaping intended directories when loading policies
- **No timestamps/randomness**: All modules - Deterministic behavior, no environment-dependent data
- **Default compatibility**: `cli.py` - No `--strict-trust` = no policy required; No require flags = no attestations required
- **Minimal diffs**: All changes - Localized to CAT_CHAT, preserved existing behavior

---

## [2025-12-30] - Phase 6.4 Complete

### Added - Phase 6.4 — Receipt Merkle Root
- **Deterministic Merkle root computation for receipt chains**
  - `catalytic_chat/receipt.py` - `compute_merkle_root()`:
    - Pairwise concatenate hex-decoded bytes (left||right)
    - SHA256 on concatenated bytes
    - Duplicate last node at each level if odd count
    - Preserve deterministic ordering (no re-sorting)
  - `catalytic_chat/receipt.py` - `verify_receipt_chain()` now returns Merkle root string
  - `catalytic_chat/cli.py` - `--print-merkle` flag:
    - Requires `--verify-chain` (fail-closed)
    - Print ONLY Merkle root hex to stdout when set
    - Suppress all other output when `--print-merkle` is set

### Tests - Phase 6.4
- `tests/test_merkle_root.py` - 3 Merkle root tests:
  - `test_merkle_root_deterministic`: same hashes produce identical Merkle root
  - `test_merkle_root_changes_on_tamper`: tampering changes root
  - `test_merkle_root_requires_verify_chain`: `--print-merkle` without `--verify-chain` fails

---

## [2025-12-30] - Phase 6.3 Complete

### Added - Phase 6.3 — Receipt Chain Anchoring
- **Deterministic receipt chaining with parent_receipt_hash linkage**
  - `catalytic_chat/receipt.py` - `compute_receipt_hash()`: deterministic hash from canonical bytes (excluding receipt_hash field)
  - `catalytic_chat/receipt.py` - `load_receipt()`: load receipt from JSON file
  - `catalytic_chat/receipt.py` - `verify_receipt_chain()`: verify chain linkage and receipt hashes
  - `catalytic_chat/receipt.py` - `find_receipt_chain()`: find all receipts for a run in execution order
  - `catalytic_chat/executor.py` - added `previous_receipt` parameter to `__init__()`:
    - Sets `parent_receipt_hash` from previous receipt's `receipt_hash`
    - First receipt has `parent_receipt_hash=null`
  - `SCHEMAS/receipt.schema.json` - added chain fields:
    - `parent_receipt_hash`: string | null
    - `receipt_hash`: string
  - `catalytic_chat/cli.py` - added `--verify-chain` flag to `bundle run`:
    - Verifies full receipt chain for a run
    - Outputs chain status and Merkle root

### Tests - Phase 6.3
- `tests/test_receipt_chain.py` - 4 chain tests:
  - Deterministic chain verification
  - Chain break detection
  - Sequential order enforcement

---

## [2025-12-30] - Phase 6.2 Complete

### Added - Phase 6.2 — Receipt Attestation
- **Ed25519 signing and verification of receipts**
  - `catalytic_chat/attestation.py` - `sign_receipt_bytes()`, `verify_receipt_bytes()`
  - `SCHEMAS/receipt.schema.json` - added optional `attestation` object
  - `catalytic_chat/executor.py` - enhanced to support signing:
    - Added `attestation_override` parameter
    - Signs receipt with `attestation=None` (null in signed scope)
  - `catalytic_chat/receipt.py` - `receipt_canonical_bytes()`: single source of truth for receipt canonicalization
    - Used by signer, verifier, and executor
  - `catalytic_chat/cli.py` - added `--attest` flag:
    - `bundle run` now supports `--attest` (requires `--signing-key`)
    - `--verify-attestation` flag for verification
  - `tests/test_attestation.py` - 6 attestation tests:
    - All tests pass
    - Fixed tamper test to verify canonical byte differences

### Security - Phase 6.2
- Single source of truth for canonicalization
- Signing input is canonical receipt bytes with `attestation=null/None`
- Verifying recomputes exact same canonical bytes
- Hex-only for `public_key`/`signature` with validation
- No timestamps, randomness, absolute paths, or env-dependent behavior
- Minimal diffs; changes localized to canonicalization and signing flow

---

## [2025-12-30] - Phase 6.2.1 Complete

### Fixed - Phase 6.2.1 — Attestation Stabilization
- **Fixed test_cli_dry_run subprocess import issue**
  - `tests/test_planner.py` - added 3 lines:
    - Import `os`
    - Copy environment variables and set `PYTHONPATH` to `Path(__file__).parent.parent`
    - Pass `env` parameter to `subprocess.run()`

### Verification - Phase 6.2.1
- All 59 tests now pass (previously 58 passed, 1 failed)
- No new skips added (same 13 skips)
- Deterministic receipts unchanged
- Attestation still fail-closed
- No CLI or executor regressions

---

## [2025-12-29] - Phase 3 Complete

### Added - Phase 3.1 — Message Cassette (DB-first Schema & API)
- **DB Schema (CORTEX/_generated/system3.db): Message cassette with append-only enforcement**
  - `catalytic_chat/message_cassette_db.py` - SQLite database schema:
    - `cassette_messages`: append-only, run_id, source, payload_json
    - `cassette_jobs`: links to messages, ordinal for ordering
    - `cassette_steps`: status (PENDING/LEASED/COMMITTED), lease_owner, lease_expires_at, fencing_token
    - `cassette_receipts`: append-only, links to step/job

  - **DB-Level Enforcement (triggers):**
    - Append-only on messages and receipts (UPDATE/DELETE blocked)
    - FSM enforcement: only PENDING→LEASED→COMMITTED allowed
    - Lease protection: direct lease changes blocked without proper claim
    - Foreign keys: receipts→steps→jobs→messages integrity

- **API (message_cassette.py): Message cassette with deterministic job/step management**
  - `post_message()`: create message/job/step, idempotency support
  - `claim_step()`: deterministic selection (oldest job, lowest ordinal)
  - `complete_step()`: verify lease/token/expiry before completion
  - `verify_cassette()`: check DB state and invariants

- **Tests:**
  - `tests/test_message_cassette.py` - Full test coverage for append-only, FSM enforcement, lease protection, referential integrity, deterministic claim ordering

### Added - Phase 3.2 — Message Cassette CLI Commands
- **New CLI commands for cassette management**
  - `catalytic_chat/cli.py` - Modified to add cassette commands:
    - `cortex cassette verify --run-id <id>`: Verify cassette state
    - `cortex cassette post --json <file> --run-id <id> --source <src>`: Post message to cassette
    - `cortex cassette claim --run-id <id> --worker <id> [--ttl <s>]`: Claim a step
    - `cortex cassette complete --run-id <id> --step <id> --worker <id> --token <n> --receipt <file> --outcome <out>`: Complete a step

  - **No changes to Phase 1/2 commands** (build, verify, get, extract, symbols, resolve)

### Added - Phase 3.3 — Message Cassette Hardening (Adversarial Tests & Verify Strengthening)
- **Adversarial tests (test_message_cassette.py):**
  - `test_steps_delete_allowed_when_persisting_design`: Confirm steps are NOT append-only (design intent)
  - `test_messages_update_delete_blocked_by_triggers`: Direct SQL UPDATE/DELETE blocked by triggers
  - `test_receipts_update_delete_blocked_by_triggers`: Direct SQL UPDATE/DELETE blocked by triggers
  - `test_illegal_fsm_transition_blocked`: Direct SQL FSM jumps (PENDING→COMMITTED, COMMITTED→LEASED) blocked
  - `test_lease_direct_set_blocked`: Direct SQL lease changes without claim blocked
  - `test_complete_fails_on_stale_token_even_if_lease_owner_matches`: Completion with stale token fails
  - `test_complete_fails_on_expired_lease`: Completion with expired lease fails

- **verify_cassette() Strengthening:**
  - Check PRAGMA foreign_keys is ON
  - Check required tables exist
  - Check all 8 required triggers exist by exact name
  - Return exit code 0 on PASS, non-zero on FAIL

- **CLI Output:**
  - `verify`: "PASS: All invariants verified" to stderr on success
  - `verify`: "FAIL: N issue(s) found" to stderr on failure

### Changed - Phase 3 Files
- `tests/test_message_cassette.py` - Added 8 adversarial tests
- `catalytic_chat/message_cassette.py` - Strengthened verify_cassette()
- `catalytic_chat/cli.py` - Cleaned verify output, added exit codes

### Tests - Phase 3
- 8 adversarial tests (all passing)
- Full test coverage for DB schema, FSM enforcement, lease protection
- Referential integrity and deterministic ordering verified

---

## [2025-12-29] - Phase 4 Complete

### Added - Phase 4 (Deterministic Planner)
- **`THOUGHT/LAB/CAT_CHAT/PHASE_4_LAW.md`** (390 lines)
- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/plan_request.schema.json`** (88 lines)
- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/plan_step.schema.json`** (159 lines)
- **`THOUGHT/LAB/CAT_CHAT/SCHEMAS/plan_output.schema.json`** (113 lines)
- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/planner.py`** (427 lines)
- **`THOUGHT/LAB/CAT_CHAT/catalytic_chat/message_cassette.py`** (modified: integrated with planner)
- **`THOUGHT/LAB/CAT_CHAT/tests/test_planner.py`** (443 lines)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_min.json`** (minimal)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_files.json`** (with file reference)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_max_steps_exceeded.json`** (max_steps exceeded)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_max_bytes_exceeded.json`** (max_bytes exceeded)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_max_symbols_exceeded.json`** (max_symbols exceeded)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_slice_all_forbidden.json`** (slice=ALL forbidden)
- **`THOUGHT/LAB/CAT_CHAT/tests/fixtures/plan_request_invalid_symbol.json`** (invalid symbol_id)

### Phase 4 Features
- **Deterministic Compiler**: Request -> Plan (steps with stable IDs and ordering)
- **Budget Enforcement**: max_steps, max_bytes, max_symbols (fail-closed)
- **Symbol Bounds**: slice=ALL forbidden, uses default_slice
- **Phase 3 Integration**: post_request_and_plan() stores request + plan in cassette
- **Idempotency**: Same (run_id, idempotency_key) returns same job_id/steps
- **CLI Commands**: `cortex plan --request-file <json> [--dry-run]`, `cassette plan-verify`

### Tests - Phase 4 (31 tests passing)
- `test_plan_determinism_same_request_same_output` ✅
- `test_plan_determinism_step_ids_stable` ✅
- `test_plan_rejects_too_many_steps` ✅
- `test_plan_rejects_too_many_bytes` ✅
- `test_plan_rejects_too_many_symbols` ✅
- `test_plan_rejects_slice_all_forbidden` ✅
- `test_plan_rejects_invalid_symbol_id` ✅
- `test_plan_idempotency_same_idempotency_key` ✅
- `test_plan_dry_run_does_not_touch_db` ✅
- `test_plan_verify_matches_stored_hash` ✅
- `test_plan_verify_fails_on_mismatch` ✅

### Verified - Phase 4
- **Phase 4 Tests**: All 31 tests passing
- **CLI Verify**: `python -m catalytic_chat.cli plan --request-file tests/fixtures/plan_request_min.json --dry-run` → valid plan

---

## [2025-12-29] - Legacy Information

### Deprecated / Legacy
- **Previous Claude Code triple-write implementation** (misaligned with roadmap)
  - `legacy/chat_db.py` - Database for Claude Code messages
  - `legacy/embedding_engine.py` - Vector embeddings for chat messages
  - `legacy/message_writer.py` - Triple-write to DB + JSONL + MD
  - `catalytic-chat-research.md` (archived) - Claude Code research
  - `catalytic-chat-phase1-implementation-report.md` (archived) - Old phase report
  - `archive/catalytic-chat-roadmap.md` (archived) - Old roadmap
  - `legacy/SYMBOLIC_README.md` - Symbol encoding (different from roadmap "Symbol" concept)

**Note**: Legacy files preserved in `archive/` directory but are not aligned with current CAT_CHAT architecture. See `archive/LEGACY_NOTES.md` for consolidated legacy documentation.

### Legacy Implementation (Phase 1 - Triple-Write)
- **Core Database** (`legacy/chat_db.py`)
  - SQLite database with 4 tables: `chat_messages`, `message_chunks`, `message_vectors`, `message_fts`
  - Hash-based deduplication (SHA-256)
  - Transaction support with context managers
  - Migration system for schema versioning
  - Foreign key constraints for data integrity

- **Embedding Engine** (`legacy/embedding_engine.py`)
  - Vector embeddings using `all-MiniLM-L6-v2` (384 dimensions)
  - Batch processing for efficiency
  - Cosine similarity computation
  - BLOB serialization for SQLite storage

- **Message Writer** (`legacy/message_writer.py`)
  - Triple-write strategy: DB (primary) + JSONL (mechanical) + MD (human)
  - Atomic writes - all three must succeed or none
  - Automatic chunking of long messages (500 tokens per chunk)
  - Embedding generation for all chunks
  - JSONL export in Claude Code format
  - MD export with human-readable formatting

### Testing (Legacy)
- **Unit Tests** (`legacy/tests/test_chat_system.py`)
  - 44 tests across 3 test classes
  - Test coverage: Database, Embedding Engine, Message Writer
  - All tests passing

### Documentation (Legacy)
- Implementation Report: `catalytic-chat-phase1-implementation-report.md` (archived)
- Research: `catalytic-chat-research.md` (archived)
- Roadmap: `catalytic-chat-roadmap.md` (archived)
- ADR: `CONTEXT/decisions/ADR-031-catalytic-chat-triple-write.md` (if exists)

### Legacy Symbolic Encoding (Not Implemented in Current System)
- **Symbolic Chat Encoding** (archived in `legacy/SYMBOLIC_README.md`)
  - Token savings of 30-70% through symbol compression
  - Symbol dictionary with governance/technical terms
  - Auto-encoding of English to symbols on write
  - Auto-decoding of symbols to English on read
  - `simple_symbolic_demo.py` - working demo with 62.5% token savings

**Note**: This symbolic encoding is experimental and NOT part of current CAT_CHAT system. Current CAT_CHAT uses "Symbol" concept differently (symbol registry referencing canonical sections).

---

## Roadmap Progress

### Completed Phases
- ✅ Phase 0: FREEZE scope and interfaces (CONTRACT.md, all schemas, budgets, error policy)
- ✅ Phase 1: Substrate + deterministic indexing (substrate, extractor, indexer, CLI, slice resolver, section retrieval)
- ✅ Phase 2: Symbol registry + bounded resolver (symbol registry, symbol resolver, expansion cache, CLI)
- ✅ Phase 2.5: Experimental vector sandbox (optional exploration, not canonical)
- ✅ Phase 3: Message cassette (message cassette, DB-first enforcement, lease handling, tests)
- ✅ Phase 4: Deterministic planner + governed step pipeline (planner, budget enforcement, symbol bounds)
- ✅ Phase 6.2: Receipt attestation (Ed25519 signing/verification)
- ✅ Phase 6.3: Receipt chain anchoring (parent_receipt_hash linkage)
- ✅ Phase 6.4: Merkle root over receipt chains (deterministic Merkle computation)
- ✅ Phase 6.5: Signed Merkle attestation (Ed25519 signing of Merkle root)
- ✅ Phase 6.10: Receipt chain ordering hardening (explicit receipt_index)
- ✅ Phase 6.11: Receipt index propagation (deterministic index assignment)
- ✅ Phase 6.12: Receipt index determinism redo (executor-derived, no caller control)
- ✅ Phase 6.13: Multi-validator aggregation (quorum validation for attestations)
- ✅ Phase 6.14: External verifier UX improvements (CI-friendly output modes)
- ✅ Phase 7: Compression protocol formalization + validator (specification, schema, validator, CLI, tests)

### Architecture Notes

### Determinism (Phase 6+)
- Canonical JSON bytes: UTF-8, `\n`, lexicographically sorted keys, `separators=(",", ":")`, newline at EOF
- Arrays are explicitly ordered before serialization
- No timestamps or randomness in outputs that must be reproducible
- No OS-dependent path separators inside manifests
- Merkle root computation: deterministic pairwise concatenation of receipt hashes

### Boundedness (Phase 5+)
- Bundle includes only artifacts referenced by plan steps and only the exact slice used
- Any "ALL" sentinel (case-insensitive) is rejected
- No repo-wide dumps
- Metrics computed ONLY from verified bundle/receipts

### Fail-Closed Verification (Phase 6+)
- Missing refs, schema mismatch, hash mismatch, ordering violation, forbidden slice, forbidden content all raise hard errors
- Verification always recomputes hashes and IDs; no trust-by-assertion paths
- All failures have explicit error codes
- No silent failures

### Trust and Identity Pinning (Phase 6+)
- Trust policy is a schema-validated allowlist of validator keys and scopes
- Validator identity can include build_id pinning
- Identity fields are included in signed bytes so pinning is tamper-resistant
- Strict-trust and strict-identity modes enforce quorum and build_id verification

---

## Next Steps (Post-Phase 7)

### Current Status
CAT_CHAT system is feature-complete through Phase 7 with:
- Deterministic reference layer (sections, symbols, resolver)
- Cassette substrate (message cassette, jobs, steps, receipts)
- Deterministic planner and execution loop
- Bundle protocol (build, verify, replay)
- Receipts, chaining, Merkle commitments
- Attestation, trust policy, validator identity hardening
- Multi-validator aggregation (optional)
- Compression protocol specification and validator

### Recommended Next Work
1. **Integration Testing**: End-to-end integration with external systems
2. **Performance Optimization**: Benchmark and optimize token efficiency
3. **Additional Phases**: Any future work defined in roadmap
4. **Documentation**: User guides, tutorials, examples
5. **Deployment**: Production deployment and monitoring

---

**Changelog Last Updated:** 2025-12-31  
**System Version:** CAT_CHAT Phase 7 Complete  
**Repository:** `agent-governance-system/THOUGHT/LAB/CAT_CHAT`
