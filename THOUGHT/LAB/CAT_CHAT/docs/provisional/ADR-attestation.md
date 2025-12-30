# PROVISIONAL ADR: Receipt Attestation (Phase 6.2)

**Status:** Provisional
**Date:** 2025-12-30
**Confidence:** High
**Impact:** High
**Tags:** [receipt, attestation, ed25519, signing, verification]

## Context

Phase 6.2 completed implementation of cryptographic attestation for bundle execution receipts. Receipts now support optional ed25519 signatures that allow verification of execution authenticity and integrity.

The attestation module (`catalytic_chat/attestation.py`) and executor were enhanced to:
- Sign receipts with ed25519 private keys
- Verify signatures against public keys
- Enforce strict canonicalization for signing and verification

## Decision

We adopt **receipt attestation** as an optional but strongly recommended feature:

### Attestation Format

```json
{
  "attestation": {
    "scheme": "ed25519",
    "public_key": "64-char-hex-string",
    "signature": "128-char-hex-string"
  } | null
}
```

### Canonicalization Contract

**Single source of truth:** `receipt_canonical_bytes(receipt, attestation_override=None)` in `catalytic_chat/receipt.py`

1. **Signing:** MUST canonicalize receipt with `attestation=null/None` before signing
   - Guarantees signature is over deterministic payload
   - Signature covers all receipt fields except the signature itself

2. **Verification:** MUST recompute canonical bytes with `attestation_override=None`
   - Guarantees verification checks exact same bytes as signing
   - Prevents "attestation injection" attacks

### Validation Rules

1. **Hex-only:** `public_key` and `signature` MUST be valid hex strings
   - Reject non-hex with `AttestationError("invalid hex")`

2. **Length enforcement:**
   - `public_key`: MUST be 32 bytes (64 hex chars)
   - `signature`: MUST be 64 bytes (128 hex chars)
   - Reject wrong lengths with `AttestationError("invalid length")`

3. **Scheme enforcement:**
   - Only `scheme="ed25519"` is supported
   - Reject other schemes with `AttestationError("unsupported scheme")`

4. **Null handling:**
   - `attestation=null` means receipt is unsigned
   - Verification MUST succeed (no-op) when `attestation` is `null`

## Implementation Details

### Signer API (`catalytic_chat/attestation.py`)

```python
def sign_receipt_bytes(receipt_bytes: bytes, private_key: bytes) -> Dict[str, str]:
    """
    Input: receipt_bytes MUST be canonical JSON with attestation=null
    Output: {"scheme": "ed25519", "public_key": "hex", "signature": "hex"}
    """
    sk = SigningKey(private_key)  # 32-byte ed25519 seed
    sig = sk.sign(receipt_bytes).signature  # 64 bytes
    vk = sk.verify_key.encode()  # 32 bytes
    return {
        "scheme": "ed25519",
        "public_key": _bytes_to_hex(vk),
        "signature": _bytes_to_hex(sig),
    }
```

### Verifier API (`catalytic_chat/attestation.py`)

```python
def verify_receipt_bytes(receipt_bytes: bytes, attestation: Dict[str, str]) -> None:
    """
    Raises AttestationError on:
    - Non-hex public_key/signature
    - Wrong lengths
    - Wrong scheme
    - Bad signature
    """
    if attestation is None:
        return  # Unsigned receipts are valid

    # Validate hex, lengths, scheme
    # ...

    # Recompute canonical bytes with attestation_override=None
    receipt_json = json.loads(receipt_bytes.decode('utf-8'))
    canonical_bytes = receipt_canonical_bytes(receipt_json, attestation_override=None)

    # Verify signature
    vk = VerifyKey(vk_bytes)
    vk.verify(canonical_bytes, sig_bytes)
```

### CLI Integration (`catalytic_chat/cli.py`)

```bash
# Execute bundle with attestation
python -m catalytic_chat.cli bundle run \
  --bundle <path> \
  --attest \
  --signing-key <key-path> \
  --receipt-out <out-path>

# Verify attestation
python -m catalytic_chat.cli bundle run \
  --bundle <path> \
  --verify-attestation
```

### Executor Flow (`catalytic_chat/executor.py`)

```python
def execute() -> dict:
    # 1. Build receipt with attestation=None
    receipt = {..., "attestation": None}

    # 2. Canonicalize for signing
    receipt_bytes = receipt_canonical_bytes(receipt, attestation_override=None)

    # 3. Sign if key provided
    if self.signing_key:
        receipt["attestation"] = sign_receipt_bytes(receipt_bytes, self.signing_key)

    # 4. Canonicalize for writing (with or without attestation)
    receipt_bytes = receipt_canonical_bytes(receipt)

    # 5. Write to disk
    self.receipt_out.write_bytes(receipt_bytes)

    return {"receipt_path": ..., "attestation": receipt.get("attestation"), ...}
```

## Rationale

### Why Ed25519?

1. **Small keys:** 32-byte private keys, 32-byte public keys
2. **Fast signatures:** 64-byte signatures, fast verification
3. **Deterministic:** No randomness needed (unlike ECDSA)
4. **Well-supported:** `pynacl` library is mature

### Why Separate Attestation Field?

1. **Optional:** Receipts can exist without attestation (for compatibility)
2. **Explicit:** `attestation=null` is clear indication of unsigned state
3. **No schema drift:** Adding attestation doesn't break existing receipt parsers

### Why Strict Canonicalization?

1. **Prevents injection:** Can't modify receipt and keep valid signature
2. **Deterministic verification:** Same input always produces same canonical bytes
3. **Single source of truth:** `receipt_canonical_bytes()` used everywhere prevents drift

## Consequences

### Positive

- Authenticity: Receipts can be proven to come from specific executor
- Integrity: Any modification to receipt after signing is detectable
- Trust: Audit trails become cryptographically verifiable
- Backward compatible: Old receipts without `attestation` still work

### Negative

- **Key management burden:** Users must generate and store 32-byte signing keys
- **No key rotation:** Current design doesn't support key rotation per executor
- **Pynacl dependency:** Adds external dependency (but widely available)

## Related

- **Phase 6.1:** Receipt schema and canonicalization
- **Phase 6.2:** Attestation signing and verification (this ADR)
- **Commit plan:** `commit-plan-phase-6-2-attestation.md`
- **Tests:** `tests/test_attestation.py` (all 6 tests passing)

## Test Coverage

- [x] `test_attestation_sign_verify_roundtrip_ok` - Roundtrip signing/verification
- [x] `test_attestation_verify_fails_on_modified_receipt_bytes` - Tamper detection
- [x] `test_attestation_rejects_non_hex` - Hex validation
- [x] `test_attestation_rejects_wrong_lengths` - Length enforcement
- [x] `test_attestation_rejects_wrong_scheme` - Scheme enforcement
- [x] `test_executor_without_attestation_unchanged` - Null handling

## Next Steps

1. **Accept** this provisional ADR after review
2. **Move** to `docs/adr/` when accepted (if ADR system is added to CAT_CHAT)
3. **Extend** with key rotation support if needed (future)
4. **Consider** attestation for other artifacts (bundles, steps) if needed
