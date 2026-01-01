<!-- CONTENT_HASH: 4a452ea25c5c6b434f959a664d13fdbe49f66522f40127c9c3ca4d265f3326b8 -->

# Verifier Implementation Guide

**STATUS: NON-NORMATIVE**

This document provides implementation notes for `verify_bundle.py` and related verification tooling. **This document does NOT modify SPECTRUM law.** For normative verification rules, see:
- **SPECTRUM-04 v1.1.0**: Canonicalization and root computation rules
- **SPECTRUM-05 v1.0.0**: Complete verification procedure and threat model

---

## Implementation Requirements

### Mandatory Ed25519 Dependency

Conformant verifier implementations MUST fail-closed if the required cryptographic dependencies (e.g., `cryptography` library in Python) are missing or inoperable. In such cases, the verifier MUST return `ALGORITHM_UNSUPPORTED` or an equivalent hard error, as it cannot fulfill the Ed25519 verification requirement defined in SPECTRUM-05 Section 4.6.

**Rationale**: Without cryptographic verification, the verifier cannot provide the security guarantees specified in SPECTRUM-05. Failing open (accepting bundles without signature verification) would violate the threat model.

### Offline Artifact-Only Verification

Verification MUST be possible using ONLY the artifacts stored in the bundle and the actual files on disk. No network access, external databases, or side-channel information (e.g., logs/transcripts) may be used to reach an ACCEPT decision.

**Rationale**: This ensures bundles are self-contained and verifiable without external dependencies, which is critical for reproducibility and auditability.

### Deterministic JSON Canonicalization

Implementations MUST strictly follow the JSON canonicalization rules defined in SPECTRUM-04 v1.1.0 and SPECTRUM-05 Section 4.4/4.5/6.1. Any deviation in whitespace, field ordering, or character encoding will result in root mismatches and rejection.

**Implementation Note**: The `_canonicalize_json` method in `verify_bundle.py` implements this by:
1. Recursively sorting all object keys lexicographically by UTF-8 byte value
2. Encoding to UTF-8 without whitespace
3. Using `separators=(',', ':')` with `ensure_ascii=False` in `json.dumps`

---

## Verifier Modes

### Strict Mode (`strict=True`)

When `strict=True`, the verifier enforces full SPECTRUM-05 compliance including:
- Ed25519 signature verification (Section 4.6)
- Identity verification (Section 4.3)
- Signed payload verification (Section 4.5)

This is the **default and recommended mode** for production verification.

### Non-Strict Mode (`strict=False`)

When `strict=False`, the verifier:
- Skips cryptographic signature verification
- Allows dummy identity data
- Still enforces structural integrity, output hashes, and acceptance gating

**Use Cases**: Legacy compatibility, internal testing, or scenarios where cryptographic verification is handled externally.

**Warning**: Non-strict mode does NOT provide the security guarantees of SPECTRUM-05.

### Proof Checking (`check_proof`)

The `check_proof` parameter controls whether `PROOF.json` verification is required:
- `check_proof=True` (default): Enforces restoration proof (SPECTRUM-05 Section 4.7)
- `check_proof=False`: Skips proof requirement (for simplified testing scenarios)

---

## API Reference

### `verify_bundle_spectrum05(run_dir, strict=True, check_proof=True)`

Stable API for SPECTRUM-05 verification. Returns:
```python
{
    "ok": bool,
    "code": str,  # "OK" or error code from SPECTRUM-05 Section 8.2
    "details": dict,
    "bundle_root": str  # (if ok=True)
}
```

### `verify_chain_spectrum05(run_dirs, strict=True, check_proof=True)`

Stable API for chain verification per SPECTRUM-05 Section 6. Returns same shape as `verify_bundle_spectrum05`, plus `chain_root` on success.

### Legacy APIs (Deprecated)

`verify_bundle(run_dir, strict=False, check_proof=True)` and `verify_chain(run_dirs, strict=False, check_proof=True)` are deprecated wrappers that adapt the SPECTRUM-05 return shape to the legacy `{"valid": bool, "errors": [...]}` format. Default `strict=False` for backward compatibility.

---

## Error Handling

All errors use the frozen error codes from SPECTRUM-05 Section 8.2. See that section for the complete normative error code mapping.

Errors are returned immediately upon detection (fail-fast). There is no error accumulation or "continue on error" mode.

---

## Testing Considerations

When writing tests for the verifier:
1. Use `strict=False` for structural tests that don't need real cryptographic verification
2. Use `check_proof=False` if the test doesn't involve restoration proofs
3. Always use `strict=True` when testing cryptographic acceptance criteria
4. Provide minimal valid artifacts even in non-strict mode to satisfy Phase 1 checks

Example minimal bundle for non-strict testing:
```python
# Required artifacts (even with strict=False, check_proof=False)
- TASK_SPEC.json
- STATUS.json
- OUTPUT_HASHES.json
- VALIDATOR_IDENTITY.json (dummy data OK if strict=False)
- SIGNED_PAYLOAD.json (dummy data OK if strict=False)
- SIGNATURE.json (dummy data OK if strict=False)
- PROOF.json (only if check_proof=True)
```

---

## References

- SPECTRUM-04 v1.1.0: Validator Identity and Signing Law
- SPECTRUM-05 v1.0.0: Verification and Threat Law for Identity-Pinned Acceptance
- `PRIMITIVES/verify_bundle.py`: Reference implementation
