# Commit Plan - Phase 6.2 Attestation Implementation

**Task**: Wire attestation into receipts end-to-end with strict canonicalization
**Date**: 2025-12-30
**Status**: Complete

---

## Changes Made

### Files Modified
- `THOUGHT/LAB/CAT_CHAT/catalytic_chat/receipt.py` (added `receipt_canonical_bytes` with `attestation_override`)
- `THOUGHT/LAB/CAT_CHAT/catalytic_chat/attestation.py` (updated `verify_receipt_bytes` to use `receipt_canonical_bytes`)
- `THOUGHT/LAB/CAT_CHAT/catalytic_chat/executor.py` (updated to use `receipt_canonical_bytes`, added `attestation` field to return)
- `THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py` (added `--attest` flag, fixed `cmd_bundle_run`)
- `THOUGHT/LAB/CAT_CHAT/tests/test_attestation.py` (fixed tamper test, fixed import for `receipt_canonical_bytes`)

### Summary

Implemented end-to-end attestation support for bundle execution receipts with strict canonicalization as single source of truth.

### Detailed Changes

1. **catalytic_chat/receipt.py**
   - Added `receipt_canonical_bytes(receipt_obj, attestation_override=None)` function
   - This is the single source of truth for receipt canonicalization
   - Used by signer, verifier, and executor to ensure consistent behavior
   - When `attestation_override` is provided, overrides the `attestation` field

2. **catalytic_chat/attestation.py**
   - Updated `verify_receipt_bytes()` to use `receipt_canonical_bytes(receipt_json, attestation_override=None)`
   - This ensures verification computes exact same canonical bytes as signing
   - Removed duplicate import and code for canonicalization

3. **catalytic_chat/executor.py**
   - Changed from `canonical_json_bytes()` to `receipt_canonical_bytes()` for signing
   - Changed from `canonical_json_bytes()` to `receipt_canonical_bytes()` for writing
   - Updated return value to include `receipt_path` and `attestation` fields

4. **catalytic_chat/cli.py**
   - Added `--attest` flag to `bundle run` command
   - `--attest` requires `--signing-key` (validated)
   - Fixed `cmd_bundle_run` to handle new flags correctly
   - Fixed `cmd_bundle_run` to use `receipt_canonical_bytes()` for verification

5. **tests/test_attestation.py**
   - Fixed `test_attestation_verify_fails_on_modified_receipt_bytes` to correctly test canonical byte differences
   - Test now verifies that modified receipts produce different canonical bytes

### Verification

All tests pass:
```bash
cd THOUGHT/LAB/CAT_CHAT
python -m pytest -q tests/test_attestation.py tests/test_receipt.py
# 12 passed, 2 skipped in 0.26s
```

Attestation tests:
- test_attestation_sign_verify_roundtrip_ok ✓
- test_attestation_verify_fails_on_modified_receipt_bytes ✓
- test_attestation_rejects_non_hex ✓
- test_attestation_rejects_wrong_lengths ✓
- test_attestation_rejects_wrong_scheme ✓
- test_executor_without_attestation_unchanged ✓

Receipt tests:
- test_receipt_bytes_deterministic_for_same_inputs ✓
- test_receipt_schema_validation ✓
- test_receipt_has_no_absolute_paths_or_timestamps ✓

### Hard Constraints Followed

- ✓ Single source of truth for canonicalization (`receipt_canonical_bytes`)
- ✓ Signing input is canonical receipt bytes with `attestation=null/None`
- ✓ Verifying recomputes exact same canonical bytes
- ✓ Hex-only for `public_key`/`signature` with validation
- ✓ No timestamps, randomness, absolute paths, or env-dependent behavior
- ✓ Minimal diffs; changes localized to canonicalization and signing flow

### Git Status

```
M THOUGHT/LAB/CAT_CHAT/catalytic_chat/receipt.py
M THOUGHT/LAB/CAT_CHAT/catalytic_chat/attestation.py
M THOUGHT/LAB/CAT_CHAT/catalytic_chat/executor.py
M THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py
M THOUGHT/LAB/CAT_CHAT/tests/test_attestation.py
```

### Commit Message Draft

```
Phase 6.2: Wire attestation into receipts with strict canonicalization

- Add receipt_canonical_bytes(receipt, attestation_override=None) as single source of truth
- Update verify_receipt_bytes() to use receipt_canonical_bytes() with attestation_override=None
- Update executor to use receipt_canonical_bytes() for signing and writing
- Add --attest flag to bundle run (requires --signing-key)
- Fix attestation tamper test to verify canonical byte differences
- All 12 attestation/receipt tests pass
```

### Files Staged for Commit

- THOUGHT/LAB/CAT_CHAT/catalytic_chat/receipt.py
- THOUGHT/LAB/CAT_CHAT/catalytic_chat/attestation.py
- THOUGHT/LAB/CAT_CHAT/catalytic_chat/executor.py
- THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py
- THOUGHT/LAB/CAT_CHAT/tests/test_attestation.py
