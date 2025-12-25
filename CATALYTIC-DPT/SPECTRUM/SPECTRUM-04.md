# SPECTRUM-04: Validator Identity and Signing Law

**Version:** 1.1.0
**Status:** Frozen
**Created:** 2025-12-25
**Updated:** 2025-12-25

---

## 1. Purpose

SPECTRUM-04 defines the constitutional rules for validator identity and cryptographic signing that bind bundle/chain acceptance to authority.

This specification is LAW. Once frozen, no implementation may deviate. Any ambiguity rejects.

---

## 2. Scope

This specification defines:
- Validator identity model (key algorithm, encoding, derivation)
- Canonical serialization rules (byte-level, deterministic)
- Signing surface (what is signed, in what order, with what domain separation)
- Signature format (encoding, metadata, malformed detection)
- Artifact binding (which artifacts contain identity/signature data)
- Mutability and rotation rules

This specification does NOT define:
- Implementation details
- Key storage mechanisms
- Network protocols
- User interface considerations
- Revocation mechanisms (explicitly out-of-scope; see section 7.3)

---

## 3. Validator Identity Model

### 3.1 Key Algorithm (Singular)

**Algorithm:** Ed25519

**Rationale:**
- Deterministic signature generation (same message + key = same signature)
- 64-byte signatures, 32-byte public keys (compact)
- No additional random input required
- Widely supported (OpenSSL, libsodium, NaCl, Go, Rust, Python)
- No known practical attacks

**Constraint:** Ed25519 is the ONLY allowed algorithm. No alternatives. No algorithm negotiation. Any artifact specifying a different algorithm MUST be rejected.

### 3.2 Public Key Encoding

**Format:** Raw 32-byte Ed25519 public key, lowercase hex-encoded.

**Canonical representation:**
```
<64 lowercase hexadecimal characters>
```

**Example:**
```
d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a
```

**Constraints:**
- Exactly 64 characters
- Only characters `[0-9a-f]`
- No prefix, no suffix, no separators
- Uppercase hex MUST be rejected
- Any deviation from exactly 64 lowercase hex characters MUST be rejected

### 3.3 Validator ID Derivation

**Formula:**
```
validator_id = sha256(public_key_bytes)
```

Where:
- `public_key_bytes` is the raw 32-byte Ed25519 public key (NOT hex-encoded)
- `sha256` is the SHA-256 hash function
- Result is 32 bytes, represented as 64 lowercase hex characters

**Canonical representation:**
```
<64 lowercase hexadecimal characters>
```

**Properties:**
- **Globally unique:** SHA-256 collision probability negligible
- **Stable across time:** Same public key always produces same validator_id
- **Verifiable offline:** No network required; derivation is deterministic
- **One-way binding:** validator_id does not reveal public key

### 3.4 Validator Identity Object

The complete validator identity is represented as:

```json
{"algorithm":"ed25519","public_key":"<64 hex>","validator_id":"<64 hex>"}
```

**Required fields (in canonical order):**
| Field | Type | Constraint |
|-------|------|------------|
| `algorithm` | string | Exactly `"ed25519"` |
| `public_key` | string | Exactly 64 lowercase hex chars, raw Ed25519 public key |
| `validator_id` | string | Exactly 64 lowercase hex chars, derived via sha256(public_key_bytes) |

**Verification procedure:**
1. Verify exactly 3 fields present with exactly these names. If extra or missing, REJECT.
2. Verify `algorithm == "ed25519"`. If not, REJECT.
3. Decode `public_key` from hex to 32 bytes. If decode fails, REJECT.
4. Compute `sha256(decoded_public_key)` and hex-encode result (lowercase).
5. Verify result equals `validator_id`. If mismatch, REJECT.
6. ACCEPT.

---

## 4. Canonical Serialization Rules (Normative)

This section defines the ONE canonical encoding used for all preimages in SPECTRUM-04.

### 4.1 Encoding

**Character encoding:** UTF-8 (no BOM)

**Newline policy:** No newlines. The canonical form is a single line with no trailing newline.

**Whitespace policy:** No whitespace outside of string values. No spaces after colons or commas.

### 4.2 JSON Encoding Rules

All preimages use JSON encoding with these exact rules:

1. **Object key order:** Keys sorted lexicographically by UTF-8 byte value (ascending). This is ASCII order for ASCII keys.

2. **No whitespace:** No spaces, tabs, or newlines except within string values.

3. **No trailing elements:** No trailing commas in arrays or objects.

4. **Strings:**
   - Double-quoted
   - Escaped per RFC 8259: `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\uXXXX`
   - Control characters (U+0000 through U+001F) MUST be escaped
   - Non-ASCII characters MAY be literal UTF-8 or `\uXXXX` escaped (implementation choice, but once chosen, consistent)

5. **Numbers:**
   - No leading zeros (except `0` itself)
   - No positive sign
   - No trailing decimal zeros
   - Scientific notation only when the number cannot be represented otherwise
   - Integer values MUST NOT have decimal points

6. **Booleans:** Exactly `true` or `false` (lowercase)

7. **Null:** Exactly `null` (lowercase)

8. **Arrays:** Elements in declaration order (not sorted). No trailing comma.

### 4.3 Field Presence Rules

**Strict field enforcement:**
- If any required field is missing: REJECT with `FIELD_MISSING`
- If any extra field is present: REJECT with `FIELD_EXTRA`

There are no optional fields in any preimage structure.

### 4.4 Canonical JSON Examples

**Correct:**
```json
{"a":1,"b":"hello","c":true}
```

**Incorrect (spaces):**
```json
{"a": 1, "b": "hello", "c": true}
```

**Incorrect (wrong key order):**
```json
{"c":true,"b":"hello","a":1}
```

**Incorrect (trailing newline):**
```json
{"a":1}
<newline>
```

---

## 5. Bundle Root Definition (Normative)

### 5.1 Bundle Root Formula

```
bundle_root = lowercase_hex(sha256(bundle_preimage_bytes))
```

Where `bundle_preimage_bytes` is the UTF-8 encoding (no BOM, no trailing newline) of the canonical JSON defined below.

### 5.2 Bundle Preimage Structure

The bundle preimage is a JSON object with exactly these fields in this order:

```json
{"output_hashes":<object>,"status":<object>,"task_spec_hash":"<64 hex>"}
```

**Field definitions:**

| Field | Source | Construction |
|-------|--------|--------------|
| `output_hashes` | `OUTPUT_HASHES.json` | The value of the `hashes` field from OUTPUT_HASHES.json, canonicalized (keys sorted) |
| `status` | `STATUS.json` | The entire STATUS.json content, canonicalized (keys sorted) |
| `task_spec_hash` | `TASK_SPEC.json` | `lowercase_hex(sha256(raw_bytes_of_TASK_SPEC.json))` |

### 5.3 Field Source Details

**output_hashes:**
- Source file: `<run_dir>/OUTPUT_HASHES.json`
- Extract: The value of the `"hashes"` key (an object mapping paths to hashes)
- Canonicalize: Sort keys lexicographically, no whitespace
- Example source: `{"hashes":{"path/a":"sha256:abc...","path/b":"sha256:def..."},...}`
- Example extracted: `{"path/a":"sha256:abc...","path/b":"sha256:def..."}`

**status:**
- Source file: `<run_dir>/STATUS.json`
- Extract: The entire file content as a JSON object
- Canonicalize: Sort keys lexicographically, no whitespace
- Example: `{"cmp01":"pass","completed_at":"2025-12-25T12:00:00Z","status":"success"}`

**task_spec_hash:**
- Source file: `<run_dir>/TASK_SPEC.json`
- Hash input: The raw bytes of the file as stored on disk (NOT canonicalized)
- Hash function: SHA-256
- Output format: 64 lowercase hex characters
- Example: `"a1b2c3d4e5f6..."`

### 5.4 Bundle Preimage Example

Given:
- OUTPUT_HASHES.json contains `{"hashes":{"out/file.txt":"sha256:aaa..."}}`
- STATUS.json contains `{"status":"success","cmp01":"pass"}`
- TASK_SPEC.json hashes to `bbb...`

The bundle preimage is:
```json
{"output_hashes":{"out/file.txt":"sha256:aaa..."},"status":{"cmp01":"pass","status":"success"},"task_spec_hash":"bbb..."}
```

Note: `status` object keys are sorted (`cmp01` before `status`).

---

## 6. Chain Root Definition (Normative)

### 6.1 Chain Root Formula

```
chain_root = lowercase_hex(sha256(chain_preimage_bytes))
```

Where `chain_preimage_bytes` is the UTF-8 encoding (no BOM, no trailing newline) of the canonical JSON defined below.

### 6.2 Chain Preimage Structure

The chain preimage is a JSON object with exactly these fields in this order:

```json
{"bundle_roots":["<64 hex>",...],"run_ids":["<run_id>",...]}}
```

**Field definitions:**

| Field | Type | Construction |
|-------|------|--------------|
| `bundle_roots` | array of strings | Bundle roots in chain order, each 64 lowercase hex chars |
| `run_ids` | array of strings | Run directory names in chain order |

### 6.3 Chain Order Definition

**Chain order** is the order in which runs are passed to the verifier. This order is:
- Explicitly provided by the caller (list of run directories)
- Preserved exactly in both `bundle_roots` and `run_ids` arrays

**run_id** is the directory name of the run (final path component), not the full path.

Example: For `/path/to/CONTRACTS/_runs/run-001`, the run_id is `run-001`.

### 6.4 Chain Preimage Constraints

1. `bundle_roots` and `run_ids` MUST have identical length
2. `bundle_roots[i]` MUST be the bundle_root of the run at `run_ids[i]`
3. No duplicate run_ids allowed: REJECT with `CHAIN_DUPLICATE_RUN`
4. Empty chain (zero runs) is invalid: REJECT with `CHAIN_EMPTY`

### 6.5 Chain Preimage Example

Given chain order: `[run-001, run-002, run-003]`
With bundle roots: `[aaa..., bbb..., ccc...]`

The chain preimage is:
```json
{"bundle_roots":["aaa...","bbb...","ccc..."],"run_ids":["run-001","run-002","run-003"]}
```

---

## 7. Signed Payload Definition (Normative)

### 7.1 Payload Types

SPECTRUM-04 defines ONE payload type: `BUNDLE`

The `CHAIN` and `ACCEPTANCE` types referenced in v1.0.0 are **removed**. Chain signing uses the same `BUNDLE` type applied to a chain root.

**Rationale:** Reducing payload types eliminates ambiguity. A chain signature is simply a bundle signature where the "bundle" is the chain manifest.

### 7.2 Signed Payload Structure

The signed payload is a JSON object with exactly these fields in this order:

```json
{"bundle_root":"<64 hex>","decision":"ACCEPT","validator_id":"<64 hex>"}
```

**Field definitions:**

| Field | Type | Value |
|-------|------|-------|
| `bundle_root` | string | 64 lowercase hex chars (bundle or chain root) |
| `decision` | string | Exactly `"ACCEPT"` (no other values) |
| `validator_id` | string | 64 lowercase hex chars |

### 7.3 Timestamp Removal

**Timestamps are NOT signed.**

The `timestamp` field from v1.0.0 is removed from the signed payload.

**Rationale:**
- Timestamps cannot be standardized without a trusted time source
- Including timestamps would prevent deterministic signature verification
- Observational timestamps can be stored in non-signed metadata

The `SIGNATURE.json` artifact MAY contain a `signed_at` field for informational purposes, but this field is NOT part of the signed payload and NOT verified.

### 7.4 Signature Message Construction

The message signed by Ed25519 is:

```
CAT-DPT-SPECTRUM-04-v1:BUNDLE:<canonical_payload_json>
```

Where:
- `CAT-DPT-SPECTRUM-04-v1:` is the domain separation prefix (23 bytes)
- `BUNDLE:` is the payload type indicator (7 bytes)
- `<canonical_payload_json>` is the UTF-8 encoding of the canonical JSON payload

**Total message structure (no newlines):**
```
CAT-DPT-SPECTRUM-04-v1:BUNDLE:{"bundle_root":"...","decision":"ACCEPT","validator_id":"..."}
```

### 7.5 Source of Truth

**SIGNED_PAYLOAD.json is the source of truth for the payload.**

The verifier:
1. Reads `SIGNED_PAYLOAD.json`
2. Verifies it has exactly the required fields
3. Verifies `bundle_root` matches the computed bundle root
4. Verifies `decision` is exactly `"ACCEPT"`
5. Verifies `validator_id` matches `VALIDATOR_IDENTITY.json`
6. Reconstructs the signature message from SIGNED_PAYLOAD.json content
7. Verifies the signature in `SIGNATURE.json` against this message

---

## 8. Signature Format

### 8.1 Signature Encoding

**Format:** Ed25519 signature, lowercase hex-encoded.

**Canonical representation:**
```
<128 lowercase hexadecimal characters>
```

**Constraints:**
- Exactly 128 characters (64-byte signature)
- Only characters `[0-9a-f]`
- No prefix, no suffix, no separators
- Uppercase hex MUST be rejected

### 8.2 Signature Artifact

**Artifact:** `SIGNATURE.json`

**Location:** `<run_dir>/SIGNATURE.json`

**Required fields (in canonical order):**

```json
{"payload_type":"BUNDLE","signature":"<128 hex>","validator_id":"<64 hex>"}
```

| Field | Type | Constraint |
|-------|------|------------|
| `payload_type` | string | Exactly `"BUNDLE"` |
| `signature` | string | Exactly 128 lowercase hex chars |
| `validator_id` | string | Exactly 64 lowercase hex chars |

**Optional informational field:**
- `signed_at`: ISO 8601 timestamp (NOT verified, NOT part of signature)

### 8.3 Malformed Signature Detection

A signature MUST be rejected if any of the following are true:

| Condition | Error Code |
|-----------|------------|
| `signature` not exactly 128 lowercase hex chars | `SIGNATURE_MALFORMED` |
| `signature` contains uppercase letters | `SIGNATURE_MALFORMED` |
| `signature` contains non-hex characters | `SIGNATURE_MALFORMED` |
| `payload_type` not `"BUNDLE"` | `SIGNATURE_MALFORMED` |
| `validator_id` not exactly 64 lowercase hex chars | `SIGNATURE_MALFORMED` |
| Required field missing | `SIGNATURE_INCOMPLETE` |
| Extra required-like field present | `SIGNATURE_MALFORMED` |

### 8.4 Signature Verification Procedure

1. Parse `SIGNATURE.json`. If malformed, REJECT with `SIGNATURE_MALFORMED`.
2. Verify required fields present. If missing, REJECT with `SIGNATURE_INCOMPLETE`.
3. Parse `SIGNED_PAYLOAD.json`. If malformed or missing, REJECT with `PAYLOAD_MISSING`.
4. Parse `VALIDATOR_IDENTITY.json`. If malformed or missing, REJECT with `IDENTITY_MISSING`.
5. Verify `validator_id` matches across all three artifacts. If mismatch, REJECT with `IDENTITY_MISMATCH`.
6. Compute bundle_root from artifacts per section 5.
7. Verify `SIGNED_PAYLOAD.json.bundle_root` equals computed bundle_root. If mismatch, REJECT with `BUNDLE_ROOT_MISMATCH`.
8. Verify `SIGNED_PAYLOAD.json.decision` equals `"ACCEPT"`. If not, REJECT with `DECISION_INVALID`.
9. Construct signature message: `CAT-DPT-SPECTRUM-04-v1:BUNDLE:` + canonical JSON of SIGNED_PAYLOAD.json
10. Decode signature from hex to 64 bytes.
11. Decode public_key from VALIDATOR_IDENTITY.json hex to 32 bytes.
12. Verify Ed25519 signature. If invalid, REJECT with `SIGNATURE_INVALID`.
13. ACCEPT.

---

## 9. Artifact Binding

### 9.1 Validator Identity Artifact

**Artifact:** `VALIDATOR_IDENTITY.json`

**Location:** `<run_dir>/VALIDATOR_IDENTITY.json`

**Content (canonical form):**
```json
{"algorithm":"ed25519","public_key":"<64 hex>","validator_id":"<64 hex>"}
```

**Validation rules:**
- MUST be present in every accepted bundle
- MUST have exactly 3 fields with exactly these names
- MUST validate per section 3.4
- MUST match `validator_id` in all other artifacts

### 9.2 Signed Payload Artifact

**Artifact:** `SIGNED_PAYLOAD.json`

**Location:** `<run_dir>/SIGNED_PAYLOAD.json`

**Content (canonical form):**
```json
{"bundle_root":"<64 hex>","decision":"ACCEPT","validator_id":"<64 hex>"}
```

**Validation rules:**
- MUST be present in every accepted bundle
- MUST have exactly 3 fields with exactly these names
- `bundle_root` MUST match computed value
- `decision` MUST be exactly `"ACCEPT"`
- `validator_id` MUST match `VALIDATOR_IDENTITY.json`

### 9.3 Signature Artifact

**Artifact:** `SIGNATURE.json`

**Location:** `<run_dir>/SIGNATURE.json`

**Content (canonical form, informational field omitted):**
```json
{"payload_type":"BUNDLE","signature":"<128 hex>","validator_id":"<64 hex>"}
```

**Validation rules:**
- MUST be present in every accepted bundle
- MUST validate per section 8.4
- `validator_id` MUST match `VALIDATOR_IDENTITY.json`

### 9.4 Artifact-Only Verification Requirement

Bundle/chain verification MUST be possible using ONLY:
- Bundle artifacts on disk
- Public key (embedded in `VALIDATOR_IDENTITY.json`)
- Signature (embedded in `SIGNATURE.json`)

Verification MUST NOT require:
- Network access
- External trust store
- Certificate authority
- Key server
- Any resource not present in the bundle

---

## 10. Mutability and Rotation

### 10.1 Immutability Rule

Once a bundle is accepted (signature verified), the following are IMMUTABLE:

| Artifact | Immutability |
|----------|--------------|
| `TASK_SPEC.json` | Byte-for-byte immutable |
| `STATUS.json` | Byte-for-byte immutable |
| `OUTPUT_HASHES.json` | Byte-for-byte immutable |
| `VALIDATOR_IDENTITY.json` | Byte-for-byte immutable |
| `SIGNATURE.json` | Byte-for-byte immutable |
| `SIGNED_PAYLOAD.json` | Byte-for-byte immutable |
| `PROOF.json` | Byte-for-byte immutable |
| All declared outputs | Hash-immutable |

Any modification to immutable artifacts invalidates the signature and MUST cause rejection.

### 10.2 Validator Identity Rotation

**Status:** NOT ALLOWED

Validator identity rotation is explicitly forbidden in SPECTRUM-04.

**Rationale:**
- Simplifies verification (one identity per validator, forever)
- Eliminates rotation-based attacks
- Ensures historical bundles remain verifiable without rotation history

**Rules:**
1. A validator_id is bound to exactly one public key.
2. A public key is bound to exactly one validator_id.
3. This binding is permanent and irrevocable.
4. If a private key is compromised, the validator_id is permanently compromised.
5. A new validator MUST generate a new key pair and thus a new validator_id.

### 10.3 Revocation (Out of Scope)

**Revocation is NOT part of SPECTRUM-04 and MUST NOT be required for acceptance.**

SPECTRUM-04 defines artifact-only verification. Revocation requires external state (a revocation list) which violates the artifact-only principle.

If a deployment requires revocation:
- It MUST be implemented as a layer above SPECTRUM-04
- It MUST NOT modify SPECTRUM-04 artifacts
- It MUST NOT be required for SPECTRUM-04 verification to succeed

A valid SPECTRUM-04 signature remains valid regardless of any external revocation state.

---

## 11. Fail-Closed Rules

### 11.1 Mandatory Rejections

The following conditions MUST result in immediate rejection:

| Condition | Error Code |
|-----------|------------|
| Any ambiguity in identity | `IDENTITY_AMBIGUOUS` |
| Multiple identities in bundle | `IDENTITY_MULTIPLE` |
| Multiple signatures in bundle | `SIGNATURE_MULTIPLE` |
| Multiple public keys in bundle | `KEY_MULTIPLE` |
| Deviation from canonical serialization | `SERIALIZATION_INVALID` |
| Algorithm not `ed25519` | `ALGORITHM_UNSUPPORTED` |
| Partial identity data | `IDENTITY_INCOMPLETE` |
| Partial signature data | `SIGNATURE_INCOMPLETE` |
| Missing `VALIDATOR_IDENTITY.json` | `IDENTITY_MISSING` |
| Missing `SIGNATURE.json` | `SIGNATURE_MISSING` |
| Missing `SIGNED_PAYLOAD.json` | `PAYLOAD_MISSING` |
| validator_id mismatch across artifacts | `IDENTITY_MISMATCH` |
| Signature verification failure | `SIGNATURE_INVALID` |
| bundle_root mismatch | `BUNDLE_ROOT_MISMATCH` |
| chain_root mismatch | `CHAIN_ROOT_MISMATCH` |
| decision not "ACCEPT" | `DECISION_INVALID` |
| Missing required field | `FIELD_MISSING` |
| Extra field in strict object | `FIELD_EXTRA` |
| Duplicate run in chain | `CHAIN_DUPLICATE_RUN` |
| Empty chain | `CHAIN_EMPTY` |

### 11.2 No Heuristics

Verification MUST NOT use heuristics. Every decision is binary:
- ACCEPT (all conditions met exactly)
- REJECT (any condition not met)

There is no "likely valid", "probably authentic", or "close enough".

### 11.3 No Side Channels

Verification MUST NOT depend on:
- Timing information
- File modification dates
- File creation order
- Network reachability
- External services
- Environment variables
- User input during verification

---

## 12. Error Codes (Normative)

| Code | Condition |
|------|-----------|
| `ALGORITHM_UNSUPPORTED` | Algorithm not ed25519 |
| `BUNDLE_ROOT_MISMATCH` | Computed bundle_root differs from declared |
| `CHAIN_DUPLICATE_RUN` | Same run_id appears twice in chain |
| `CHAIN_EMPTY` | Chain has zero runs |
| `CHAIN_ROOT_MISMATCH` | Computed chain_root differs from declared |
| `DECISION_INVALID` | decision field is not "ACCEPT" |
| `FIELD_EXTRA` | Unexpected field present in strict object |
| `FIELD_MISSING` | Required field not present |
| `IDENTITY_AMBIGUOUS` | Cannot determine validator identity |
| `IDENTITY_INCOMPLETE` | Missing required identity fields |
| `IDENTITY_INVALID` | Identity validation failed |
| `IDENTITY_MISMATCH` | validator_id differs across artifacts |
| `IDENTITY_MISSING` | VALIDATOR_IDENTITY.json not found |
| `IDENTITY_MULTIPLE` | Multiple identities detected |
| `KEY_INVALID` | Public key format invalid |
| `KEY_MULTIPLE` | Multiple public keys detected |
| `PAYLOAD_MISSING` | SIGNED_PAYLOAD.json not found |
| `PAYLOAD_MISMATCH` | Payload does not match computed value |
| `SERIALIZATION_INVALID` | Canonical JSON rules violated |
| `SIGNATURE_INCOMPLETE` | Missing required signature fields |
| `SIGNATURE_INVALID` | Ed25519 verification failed |
| `SIGNATURE_MALFORMED` | Signature format invalid |
| `SIGNATURE_MISSING` | SIGNATURE.json not found |
| `SIGNATURE_MULTIPLE` | Multiple signatures detected |

---

## 13. Determinism Proof Checklist

This section provides normative requirements for verifying implementation correctness.

### 13.1 Preimage Templates

**Bundle preimage template:**
```
{"output_hashes":{<sorted_path:hash_pairs>},"status":{<sorted_status_fields>},"task_spec_hash":"<64_hex>"}
```

**Chain preimage template:**
```
{"bundle_roots":[<ordered_roots>],"run_ids":[<ordered_ids>]}
```

**Signed payload template:**
```
{"bundle_root":"<64_hex>","decision":"ACCEPT","validator_id":"<64_hex>"}
```

**Signature message template:**
```
CAT-DPT-SPECTRUM-04-v1:BUNDLE:{"bundle_root":"<64_hex>","decision":"ACCEPT","validator_id":"<64_hex>"}
```

### 13.2 Determinism Requirements

Two implementations are conformant if and only if, for identical inputs:

1. **Canonical JSON:** Byte-for-byte identical output
2. **Bundle root:** Identical 64-character hex string
3. **Chain root:** Identical 64-character hex string
4. **Signature message:** Byte-for-byte identical
5. **Verification outcome:** Identical ACCEPT/REJECT decision
6. **Error code:** Identical error code on rejection

### 13.3 Reject Conditions for Ambiguity

If any of the following cannot be determined unambiguously, REJECT:

- Field order (use lexicographic sort)
- Whitespace (use none)
- Numeric format (use minimal representation)
- String escaping (use RFC 8259 rules)
- Key presence (all required, none optional)
- Array order (preserve declaration order)

### 13.4 Test Vectors (Informative)

Implementations SHOULD verify against these test conditions:

1. Empty output_hashes: `{"output_hashes":{},"status":{...},"task_spec_hash":"..."}`
2. Unicode in paths: Keys with non-ASCII must sort by UTF-8 byte value
3. Numeric edge cases: `0`, `-1`, `1.5` (not `1.50`)
4. Boolean encoding: `true` not `True`, `false` not `False`

---

## 14. References

- SPECTRUM-01: Minimal Durable Execution Artifact
- SPECTRUM-02: Adversarial Resume Without Execution History
- SPECTRUM-03: Chained Temporal Integrity
- RFC 8032: Edwards-Curve Digital Signature Algorithm (EdDSA)
- RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format
- RFC 3339: Date and Time on the Internet: Timestamps

---

## 15. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-25 | Initial frozen specification |
| 1.1.0 | 2025-12-25 | Canonical byte-serialization rules finalized; bundle_root/chain_root preimages fully specified; timestamp removed from signed payload; revocation explicitly out-of-scope; CHAIN/ACCEPTANCE payload types removed (BUNDLE only); determinism proof checklist added |
