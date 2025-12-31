# Catalytic Chat Changelog

See: [docs/catalytic-chat/CHANGELOG.md](docs/catalytic-chat/CHANGELOG.md)

---

## [Unreleased] - 2025-12-31

### Added
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
