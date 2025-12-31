# Catalytic Chat Changelog

See: [docs/catalytic-chat/CHANGELOG.md](docs/catalytic-chat/CHANGELOG.md)

---

## [Unreleased] - 2025-12-31
n### Added
- **Phase 6.13 — Multi-Validator Aggregation**: Multi-validator attestations for quorum validation (RECEIPT + MERKLE)
  - `SCHEMAS/receipt.schema.json` - added optional `attestations[]` array
  - `SCHEMAS/execution_policy.schema.json` - added `receipt_attestation_quorum` and `merkle_attestation_quorum`
  - `SCHEMAS/aggregated_merkle_attestations.schema.json` - new schema for aggregated attestations
  - `catalytic_chat/attestation.py` - added `verify_receipt_attestation_single()` and `verify_receipt_attestations_with_quorum()`
  - `catalytic_chat/merkle_attestation.py` - added `load_aggregated_merkle_attestations()` and `verify_merkle_attestations_with_quorum()`
  - `tests/test_multi_validator_attestations.py` - comprehensive tests for ordering, quorum, backward compatibility
  - Deterministic ordering by (validator_id, public_key.lower(), build_id or "")
  - Reuses existing trust policy (strict_trust, strict_identity)
  - Purely additive: single attestation path unchanged
- **Phase 6.14 — External Verifier UX Improvements**: CI-friendly output modes and machine-readable summaries
  - `catalytic_chat/cli_output.py` - new module with standardized exit codes and JSON output helpers
  - `catalytic_chat/cli.py` - added `--json` and `--quiet` flags to `bundle verify`, `bundle run`, `trust verify`
  - `catalytic_chat/__main__.py` - entry point for `python -m catalytic_chat.cli`
  - `tests/test_cli_output.py` - tests for JSON stdout purity, exit code classification, deterministic outputs
  - Standardized exit codes: 0 (OK), 1 (verification failed), 2 (invalid input), 3 (internal error)
  - JSON output uses `canonical_json_bytes()` for deterministic field ordering
  - No verification behavior changes; purely additive UX improvements



### Fixed
- **Phase 6.12 — Receipt Index Determinism (Redo)**: Executor-derived receipt_index with no caller control
  - `catalytic_chat/executor.py` - removed `receipt_index` parameter from `__init__()`
  - `catalytic_chat/executor.py` - removed `_find_next_receipt_index()` (no filesystem scanning)
  - `catalytic_chat/executor.py` - always assigns `receipt_index = 0` deterministically
  - `tests/test_receipt_index_propagation.py` - updated tests to verify receipt_index=0 with no caller control
  - receipt_index is executor-derived, no filesystem scanning, pure execution order function

### Added
- **Phase 6.11 — Receipt Index Propagation**: Deterministic receipt_index assignment with strict verification
  - `catalytic_chat/executor.py` - assign receipt_index deterministically (no filesystem scanning)
  - `catalytic_chat/receipt.py` - hardened `verify_receipt_chain()` to enforce contiguity, start at 0, strictly increasing
  - `tests/test_receipt_index_propagation.py` - tests for contiguous index enforcement, gap detection, nonzero start, mixed null/int
  - All receipt_index rules are fail-closed; indices must be exactly [0, 1, 2, ..., n-1] when present
- **Phase 6.10 — Receipt Chain Ordering Hardening** - Deterministic, fail-closed receipt chain ordering
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
    - `test_receipt_chain_sorted_explicitly`: deterministic order regardless of FS order
    - `test_receipt_chain_fails_on_duplicate_receipt_index`: duplicate index detection
    - `test_receipt_chain_fails_on_mixed_receipt_index`: mixed index/null detection
    - `test_merkle_root_independent_of_fs_order`: filesystem independence
    - `test_verify_receipt_chain_strictly_monotonic`: strictly increasing index enforcement

### Changed
- `catalytic_chat/receipt.py` - `build_receipt_from_bundle()`:
  - Added `"receipt_index": None` to receipt dict
- `catalytic_chat/receipt.py` - `find_receipt_chain()`:
  - Removed reliance on filesystem `sorted()` of glob results
  - Uses explicit ordering function for deterministic sorting
- `catalytic_chat/receipt.py` - `compute_merkle_root()`:
  - No longer performs internal re-sorting (caller responsibility)
  - Maintains existing behavior for consumption of ordered input

### Security
- Fail-closed behavior: duplicate receipt_index raises ValueError
- Fail-closed behavior: mixed receipt_index/null raises ValueError
- Strict monotonicity: receipt_index sequence must be strictly increasing
- Filesystem independence: order determined by receipt_index, not creation order

### Tests
- 118 passed, 13 skipped (full test suite)
- All 5 new receipt chain ordering tests passing
- All existing receipt chain tests passing

---

## [2025-12-30] - Phase 6.5 Complete

### Added
- **Phase 6.5 — Signed Merkle Attestation** - Ed25519 signing and verification of receipt chain Merkle root
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

### Changed
- `catalytic_chat/cli.py` - extended `cmd_bundle_run()` with merkle attestation support
  - Added `verbose_output` flag to suppress status messages when JSON-only output required
  - All error messages now use `sys.stderr.write()` for consistency
  - Mutually exclusive flag enforcement for stdout purity

### Tests
- `tests/test_merkle_attestation.py` - 12 comprehensive tests
  - Sign/verify roundtrip
  - Modified root detection
  - Input validation (hex, lengths, scheme)
  - Deterministic output across multiple runs
  - Mismatch detection
  - File I/O roundtrip

### Security
- Message to sign: `b"CAT_CHAT_MERKLE_V1:" + merkle_root_bytes` (decoded 32 bytes)
- Hex decoding uses `bytes.fromhex()` - NOT ASCII bytes of hex string
- Canonical JSON output using `receipt.canonical_json_bytes()` (sort_keys, separators, UTF-8, \n EOF)

---

## [2025-12-30] - Phase 6.4 Complete
