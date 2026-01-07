# SPECTRUM-05: Verification and Threat Law for Identity-Pinned Acceptance

**Version:** 1.0.0
**Status:** Frozen
**Created:** 2025-12-25
**Depends on:** SPECTRUM-04 v1.1.0
**Promoted to Canon:** 2026-01-07

---

## 1. Purpose

SPECTRUM-05 defines the exact verification procedure and threat boundaries for identity-pinned bundle and chain acceptance, fully aligned with SPECTRUM-04 v1.1.0 canonicalization rules.

This specification is LAW. Once frozen, no implementation may deviate. Any ambiguity rejects.

---

## 2. Scope

This specification defines:
- Verification procedure (step-ordered, artifact-only)
- Acceptance gating rules (single source of truth)
- Chain verification rules (all-or-nothing)
- Threat model (what is and is not defended)
- Error semantics (hard rejects only)

This specification does NOT define:
- Key generation or storage
- Signing implementation
- Network protocols
- Revocation mechanisms (out of scope per SPECTRUM-04)

---

## 3. Required Artifacts

Bundle verification requires exactly these artifacts in the run directory:

| Artifact | Purpose |
|----------|---------|
| `TASK_SPEC.json` | Job specification (hashed into bundle root) |
| `STATUS.json` | Execution status (included in bundle root) |
| `OUTPUT_HASHES.json` | Output verification manifest (included in bundle root) |
| `PROOF.json` | Restoration proof (verified=true required) |
| `VALIDATOR_IDENTITY.json` | Public key and validator_id |
| `SIGNED_PAYLOAD.json` | Signed payload content |
| `SIGNATURE.json` | Ed25519 signature |

**Strict presence rule:**
- If any required artifact is missing: REJECT with `ARTIFACT_MISSING`
- If any extra identity/signature/payload artifact exists: REJECT with `ARTIFACT_EXTRA`

---

## 4. Verification Procedure (Normative)

This section defines the exact, step-ordered verification procedure. Steps MUST be executed in order. Any failure terminates verification immediately with REJECT.

### 4.1 Phase 1: Artifact Presence Check

**Step 1.1:** Verify `TASK_SPEC.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "TASK_SPEC.json"}`

**Step 1.2:** Verify `STATUS.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "STATUS.json"}`

**Step 1.3:** Verify `OUTPUT_HASHES.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "OUTPUT_HASHES.json"}`

**Step 1.4:** Verify `PROOF.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "PROOF.json"}`

**Step 1.5:** Verify `VALIDATOR_IDENTITY.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "VALIDATOR_IDENTITY.json"}`

**Step 1.6:** Verify `SIGNED_PAYLOAD.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "SIGNED_PAYLOAD.json"}`

**Step 1.7:** Verify `SIGNATURE.json` exists.
- If missing: REJECT with `ARTIFACT_MISSING`, details: `{"artifact": "SIGNATURE.json"}`

### 4.2 Phase 2: Artifact Parse Check

**Step 2.1:** Parse `TASK_SPEC.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "TASK_SPEC.json"}`

**Step 2.2:** Parse `STATUS.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "STATUS.json"}`

**Step 2.3:** Parse `OUTPUT_HASHES.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "OUTPUT_HASHES.json"}`

**Step 2.4:** Parse `PROOF.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "PROOF.json"}`

**Step 2.5:** Parse `VALIDATOR_IDENTITY.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "VALIDATOR_IDENTITY.json"}`

**Step 2.6:** Parse `SIGNED_PAYLOAD.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "SIGNED_PAYLOAD.json"}`

**Step 2.7:** Parse `SIGNATURE.json` as JSON.
- If parse fails: REJECT with `ARTIFACT_MALFORMED`, details: `{"artifact": "SIGNATURE.json"}`

### 4.3 Phase 3: Identity Verification

**Step 3.1:** Verify `VALIDATOR_IDENTITY.json` has exactly 3 fields: `algorithm`, `public_key`, `validator_id`.
- If extra fields: REJECT with `FIELD_EXTRA`
- If missing fields: REJECT with `FIELD_MISSING`

**Step 3.2:** Verify `algorithm` equals exactly `"ed25519"`.
- If not: REJECT with `ALGORITHM_UNSUPPORTED`

**Step 3.3:** Verify `public_key` is exactly 64 lowercase hex characters.
- If not: REJECT with `KEY_INVALID`

**Step 3.4:** Decode `public_key` from hex to 32 bytes.
- If decode fails: REJECT with `KEY_INVALID`

**Step 3.5:** Compute `sha256(public_key_bytes)` and encode as 64 lowercase hex characters.

**Step 3.6:** Verify computed value equals declared `validator_id`.
- If mismatch: REJECT with `IDENTITY_INVALID`

### 4.4 Phase 4: Bundle Root Computation

**Step 4.1:** Read raw bytes of `TASK_SPEC.json` (as stored on disk).

**Step 4.2:** Compute `task_spec_hash = lowercase_hex(sha256(task_spec_bytes))`.

**Step 4.3:** Extract `hashes` object from parsed `OUTPUT_HASHES.json`.
- If `hashes` field missing: REJECT with `FIELD_MISSING`

**Step 4.4:** Canonicalize `hashes` object per SPECTRUM-04 v1.1.0:
- Sort keys lexicographically by UTF-8 byte value
- No whitespace
- No trailing elements

**Step 4.5:** Canonicalize parsed `STATUS.json` per SPECTRUM-04 v1.1.0:
- Sort keys lexicographically by UTF-8 byte value
- No whitespace
- No trailing elements

**Step 4.6:** Construct bundle preimage JSON:
```
{"output_hashes":<canonicalized_hashes>,"status":<canonicalized_status>,"task_spec_hash":"<task_spec_hash>"}
```
- Field order: `output_hashes`, `status`, `task_spec_hash` (lexicographic)
- No whitespace
- UTF-8 encoding
- No trailing newline

**Step 4.7:** Compute `bundle_root = lowercase_hex(sha256(bundle_preimage_bytes))`.

### 4.5 Phase 5: Signed Payload Verification

**Step 5.1:** Verify `SIGNED_PAYLOAD.json` has exactly 3 fields: `bundle_root`, `decision`, `validator_id`.
- If extra fields: REJECT with `FIELD_EXTRA`
- If missing fields: REJECT with `FIELD_MISSING`

**Step 5.2:** Verify `SIGNED_PAYLOAD.json.bundle_root` equals computed `bundle_root`.
- If mismatch: REJECT with `BUNDLE_ROOT_MISMATCH`

**Step 5.3:** Verify `SIGNED_PAYLOAD.json.decision` equals exactly `"ACCEPT"`.
- If not: REJECT with `DECISION_INVALID`

**Step 5.4:** Verify `SIGNED_PAYLOAD.json.validator_id` equals `VALIDATOR_IDENTITY.json.validator_id`.
- If mismatch: REJECT with `IDENTITY_MISMATCH`

### 4.6 Phase 6: Signature Verification

**Step 6.1:** Verify `SIGNATURE.json` has required fields: `payload_type`, `signature`, `validator_id`.
- If missing fields: REJECT with `SIGNATURE_INCOMPLETE`

**Step 6.2:** Verify `SIGNATURE.json.payload_type` equals exactly `"BUNDLE"`.
- If not: REJECT with `SIGNATURE_MALFORMED`

**Step 6.3:** Verify `SIGNATURE.json.signature` is exactly 128 lowercase hex characters.
- If not: REJECT with `SIGNATURE_MALFORMED`

**Step 6.4:** Verify `SIGNATURE.json.validator_id` equals `VALIDATOR_IDENTITY.json.validator_id`.
- If mismatch: REJECT with `IDENTITY_MISMATCH`

**Step 6.5:** Canonicalize `SIGNED_PAYLOAD.json` per SPECTRUM-04 v1.1.0:
- Sort keys: `bundle_root`, `decision`, `validator_id` (lexicographic order)
- No whitespace
- UTF-8 encoding

**Step 6.6:** Construct signature message:
```
CAT-DPT-SPECTRUM-04-v1:BUNDLE:<canonicalized_signed_payload>
```
- Domain prefix: `CAT-DPT-SPECTRUM-04-v1:` (23 bytes)
- Payload type: `BUNDLE:` (7 bytes)
- Payload: canonicalized JSON (variable length)
- No trailing newline

**Step 6.7:** Decode `SIGNATURE.json.signature` from hex to 64 bytes.
- If decode fails: REJECT with `SIGNATURE_MALFORMED`

**Step 6.8:** Verify Ed25519 signature over signature message using decoded public key.
- If verification fails: REJECT with `SIGNATURE_INVALID`

### 4.7 Phase 7: Proof Verification

**Step 7.1:** Verify `PROOF.json` contains `restoration_result` object.
- If missing: REJECT with `FIELD_MISSING`

**Step 7.2:** Verify `PROOF.json.restoration_result.verified` equals exactly `true` (boolean).
- If not `true`: REJECT with `RESTORATION_FAILED`

### 4.8 Phase 8: Forbidden Artifact Check

**Step 8.1:** Check if `logs/` directory exists in run directory.
- If exists: REJECT with `FORBIDDEN_ARTIFACT`, details: `{"artifact": "logs/"}`

**Step 8.2:** Check if `tmp/` directory exists in run directory.
- If exists: REJECT with `FORBIDDEN_ARTIFACT`, details: `{"artifact": "tmp/"}`

**Step 8.3:** Check if `transcript.json` file exists in run directory.
- If exists: REJECT with `FORBIDDEN_ARTIFACT`, details: `{"artifact": "transcript.json"}`

### 4.9 Phase 9: Output Hash Verification

**Step 9.1:** For each entry `(path, expected_hash)` in `OUTPUT_HASHES.json.hashes`:

**Step 9.1.1:** Resolve `path` relative to project root.

**Step 9.1.2:** Verify file exists at resolved path.
- If not: REJECT with `OUTPUT_MISSING`, details: `{"path": "<path>"}`

**Step 9.1.3:** Read file bytes and compute `actual_hash = "sha256:" + lowercase_hex(sha256(file_bytes))`.

**Step 9.1.4:** Verify `actual_hash` equals `expected_hash`.
- If mismatch: REJECT with `HASH_MISMATCH`, details: `{"path": "<path>", "expected": "<expected>", "actual": "<actual>"}`

### 4.10 Phase 10: Acceptance

If all steps pass: **ACCEPT**

---

## 5. Acceptance Gating Rules (Normative)

### 5.1 Acceptance Condition

A bundle is ACCEPTED if and only if:
1. All verification steps (Phase 1-9) pass
2. Exactly one `VALIDATOR_IDENTITY.json` exists
3. Exactly one `SIGNED_PAYLOAD.json` exists
4. Exactly one `SIGNATURE.json` exists
5. No forbidden artifacts exist
6. All output hashes verify

### 5.2 Mandatory Rejection Conditions

A bundle MUST be REJECTED if any of these conditions are true, even if other checks pass:

| Condition | Error Code |
|-----------|------------|
| Signature cryptographically invalid | `SIGNATURE_INVALID` |
| Signature malformed | `SIGNATURE_MALFORMED` |
| Signature missing | `ARTIFACT_MISSING` |
| Identity validation fails | `IDENTITY_INVALID` |
| validator_id mismatch across artifacts | `IDENTITY_MISMATCH` |
| Multiple identities detected | `IDENTITY_MULTIPLE` |
| Multiple signatures detected | `SIGNATURE_MULTIPLE` |
| Bundle root mismatch | `BUNDLE_ROOT_MISMATCH` |
| Signed payload does not match artifacts | `PAYLOAD_MISMATCH` |
| Decision not "ACCEPT" | `DECISION_INVALID` |
| PROOF.json missing | `ARTIFACT_MISSING` |
| PROOF.json.verified != true | `RESTORATION_FAILED` |
| Forbidden artifact present | `FORBIDDEN_ARTIFACT` |
| Output hash mismatch | `HASH_MISMATCH` |
| Output file missing | `OUTPUT_MISSING` |
| Canonicalization mismatch | `SERIALIZATION_INVALID` |

### 5.3 No Partial Acceptance

There is no partial acceptance. A bundle is either:
- **ACCEPTED**: All conditions met exactly
- **REJECTED**: Any condition not met

There are no warnings, no "accepted with issues", no "conditionally accepted".

---

## 6. Chain Verification Rules (Normative)

### 6.1 Chain Verification Procedure

**Step C.1:** Verify chain is non-empty.
- If empty: REJECT with `CHAIN_EMPTY`

**Step C.2:** Extract run_id for each run directory (final path component).

**Step C.3:** Verify no duplicate run_ids.
- If duplicates: REJECT with `CHAIN_DUPLICATE_RUN`

**Step C.4:** For each run directory in chain order:
- Execute full bundle verification procedure (Phases 1-9)
- If any bundle fails: REJECT entire chain with bundle's error code and `run_id` context

**Step C.5:** Compute bundle_root for each verified bundle.

**Step C.6:** Construct chain preimage JSON per SPECTRUM-04 v1.1.0:
```
{"bundle_roots":[<roots_in_order>],"run_ids":[<ids_in_order>]}
```
- Field order: `bundle_roots`, `run_ids` (lexicographic)
- Arrays preserve chain order
- No whitespace
- UTF-8 encoding
- No trailing newline

**Step C.7:** Compute `chain_root = lowercase_hex(sha256(chain_preimage_bytes))`.

**Step C.8:** If chain signature exists, verify chain_root matches signed chain_root.
- If mismatch: REJECT with `CHAIN_ROOT_MISMATCH`

### 6.2 Chain Validity Conditions

A chain is VALID if and only if:
1. All bundles in the chain are individually ACCEPTED
2. No duplicate run_ids exist
3. Chain order is explicitly provided and respected
4. Chain root computes correctly (if signed chain)

### 6.3 All-or-Nothing Semantics

Chain verification is all-or-nothing:
- If ANY bundle fails verification: REJECT entire chain
- If ANY constraint violated: REJECT entire chain
- Partial chain acceptance is forbidden

---

## 7. Threat Model (Normative)

### 7.1 Threats Defended

SPECTRUM-05 verification defends against:

| Threat | Defense |
|--------|---------|
| **Forged acceptance** | Ed25519 signature binds acceptance to validator public key; forging requires private key |
| **Validator impersonation** | validator_id is cryptographically derived from public_key; cannot be claimed without possessing key |
| **Bundle substitution** | bundle_root binds signature to exact artifact content; any modification invalidates signature |
| **Replay of signed acceptance on modified artifacts** | bundle_root changes if any artifact changes; replayed signature will fail |
| **Ambiguity-based acceptance bypass** | All canonicalization rules are fully specified; no implementation may interpret differently |
| **Hash collision exploitation** | SHA-256 provides 128-bit collision resistance; no practical attacks known |
| **Multiple identity injection** | Exactly one identity artifact allowed; extras cause rejection |
| **Forbidden artifact smuggling** | Explicit check for logs/, tmp/, transcript.json |
| **Proof bypass** | PROOF.json with verified=true is mandatory; missing or false rejects |

### 7.2 Threats NOT Defended

SPECTRUM-05 verification does NOT defend against:

| Threat | Reason |
|--------|--------|
| **Compromise of validator private key** | If private key is stolen, attacker can sign arbitrary bundles; detection requires external revocation |
| **Malicious validator acting within spec** | A validator with a valid key can sign any bundle they choose; governance must constrain validators |
| **External coercion or governance failures** | If a validator is coerced or governance fails to prevent malicious validators, signatures remain valid |
| **Attacks requiring network trust assumptions** | SPECTRUM-05 is artifact-only; any network-based attack is out of scope |
| **Side-channel attacks on signing implementation** | Implementation security is out of scope; only artifact verification is specified |
| **Quantum computing attacks** | Ed25519 is not quantum-resistant; future quantum computers could forge signatures |
| **Pre-image attacks on SHA-256** | While not currently practical, future advances could enable pre-image attacks |

### 7.3 Trust Assumptions

SPECTRUM-05 verification assumes:
1. SHA-256 is collision-resistant and pre-image resistant
2. Ed25519 signatures cannot be forged without the private key
3. Artifact files are read correctly from disk
4. The verifier implementation is correct

If any assumption is violated, verification guarantees do not hold.

---

## 8. Error Semantics (Normative)

### 8.1 Hard Reject Only

All verification failures are hard rejects. There are no:
- Warnings
- Partial acceptance
- Soft failures
- Recovery paths at verification time
- Retry suggestions

A failure is a failure. The bundle is REJECTED.

### 8.2 Error Code Mapping

Each failure class maps to exactly one error code:

| Failure Class | Error Code |
|---------------|------------|
| Artifact not found | `ARTIFACT_MISSING` |
| Artifact not valid JSON | `ARTIFACT_MALFORMED` |
| Extra artifact of identity/signature/payload type | `ARTIFACT_EXTRA` |
| Required field missing | `FIELD_MISSING` |
| Extra field in strict object | `FIELD_EXTRA` |
| Algorithm not ed25519 | `ALGORITHM_UNSUPPORTED` |
| Public key format invalid | `KEY_INVALID` |
| validator_id derivation failed | `IDENTITY_INVALID` |
| validator_id mismatch across artifacts | `IDENTITY_MISMATCH` |
| Multiple identities detected | `IDENTITY_MULTIPLE` |
| Signature format invalid | `SIGNATURE_MALFORMED` |
| Signature missing required fields | `SIGNATURE_INCOMPLETE` |
| Ed25519 verification failed | `SIGNATURE_INVALID` |
| Multiple signatures detected | `SIGNATURE_MULTIPLE` |
| bundle_root mismatch | `BUNDLE_ROOT_MISMATCH` |
| chain_root mismatch | `CHAIN_ROOT_MISMATCH` |
| decision not "ACCEPT" | `DECISION_INVALID` |
| Payload mismatch | `PAYLOAD_MISMATCH` |
| Canonical JSON rules violated | `SERIALIZATION_INVALID` |
| PROOF.json verified != true | `RESTORATION_FAILED` |
| Forbidden artifact present | `FORBIDDEN_ARTIFACT` |
| Output file missing | `OUTPUT_MISSING` |
| Output hash mismatch | `HASH_MISMATCH` |
| Chain has zero runs | `CHAIN_EMPTY` |
| Duplicate run_id in chain | `CHAIN_DUPLICATE_RUN` |

### 8.3 Error Format

Errors MUST be reported in this format:

```json
{
  "code": "<ERROR_CODE>",
  "message": "<human-readable description>",
  "details": {<structured context>}
}
```

**Required fields:**
- `code`: One of the error codes from section 8.2
- `message`: Human-readable description (for debugging)
- `details`: Structured context (artifact name, path, expected/actual values)

### 8.4 No Recovery Paths

At verification time, there are no recovery paths:
- Cannot "fix" a malformed artifact
- Cannot "override" a signature failure
- Cannot "ignore" a missing proof
- Cannot "continue despite" a forbidden artifact

The ONLY path from REJECT is to produce a new, correct bundle.

---

## 9. Interoperability Requirements

### 9.1 Implementation Conformance

Two implementations are conformant if and only if, for identical inputs:

1. **Verification outcome:** Identical ACCEPT/REJECT decision
2. **Error code:** Identical error code on rejection
3. **Bundle root:** Identical 64-character hex string
4. **Chain root:** Identical 64-character hex string
5. **Signature message:** Byte-for-byte identical

### 9.2 No Interpretation

Implementers MUST NOT need to interpret this specification. All rules are:
- Explicit
- Unambiguous
- Complete
- Testable

If any rule appears to require interpretation, that is a defect in this specification.

### 9.3 Divergence Handling

If two implementations produce different results for identical inputs:
- Both implementations are suspect
- The divergence MUST be investigated
- One or both implementations are non-conformant
- Non-conformant implementations MUST NOT be used for verification

---

## 10. References

- [SPECTRUM-02: Adversarial Resume Without Execution History](SPECTRUM-02_RESUME_BUNDLE.md)
- [SPECTRUM-03: Chained Temporal Integrity](SPECTRUM-03_CHAIN_VERIFICATION.md)
- [SPECTRUM-04: Validator Identity and Signing Law (v1.1.0)](SPECTRUM-04_IDENTITY_SIGNING.md)
- [SPECTRUM-06: Restore Runner Semantics](SPECTRUM-06_RESTORE_RUNNER.md)
- RFC 8032: Edwards-Curve Digital Signature Algorithm (EdDSA)

---

## 11. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-25 | Initial frozen specification |

---

*This document is canonical. Implementation MUST match this specification.*
