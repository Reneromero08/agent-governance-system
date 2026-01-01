---
title: "Commit Plan Phase 6.6 Trust Policy"
section: "report"
author: "System"
priority: "Low"
created: "2025-12-28 12:00"
modified: "2025-12-28 12:00"
status: "Archived"
summary: "Commit plan for trust policy (Archived)"
tags: [commit_plan, trust, archive]
---

<!-- CONTENT_HASH: fc7503cd551b9dce4d5ede1c4ac9ad328d52cc5dd1737916afaa001783542454 -->

# Commit Plan: Phase 6.6 — Validator Identity Pinning + Trust Policy

**Phase:** CAT_CHAT Phase 6.6
**Status:** Implementation Complete
**Date:** 2025-12-30

---

## Summary

Implemented deterministic, governed trust policy that pins which validator public keys are allowed to attest receipts and Merkle roots. Verification fails-closed if an attestation is present and not pinned, or if the trust policy is missing/invalid when strict mode is enabled.

---

## Deliverables Completed

### 1. Trust Policy Schema
**File:** `THOUGHT/LAB/CAT_CHAT/SCHEMAS/trust_policy.schema.json`

- Enforces `policy_version="1.0.0"` (const)
- Defines `allow` array of pinned validator entries
- Each entry requires:
  - `validator_id`: stable human label
  - `public_key`: 64-char hex (pattern `^[0-9a-fA-F]{64}$`)
  - `schemes`: array containing `"ed25519"`
  - `scope`: array of `"RECEIPT"` and/or `"MERKLE"`
  - `enabled`: boolean
- `additionalProperties: false` everywhere

**Invariant enforced:** Schema ensures valid policy structure with proper public key format and scope restrictions.

---

### 2. Trust Policy Loader + Verifier
**File:** `THOUGHT/LAB/CAT_CHAT/catalytic_chat/trust_policy.py`

**Functions:**
- `load_trust_policy_bytes(path)`: Reads exact bytes, fails if missing
- `parse_trust_policy(policy_bytes)`: Validates against schema using jsonschema
- `build_trust_index(policy)`: Returns deterministic index mapping lowercase public_key → entry
  - Enforces uniqueness of `validator_id` and `public_key` (case-insensitive)
  - Raises `TrustPolicyError` on duplicates
- `is_key_allowed(index, public_key_hex, scope, scheme)`: Checks if key is allowed for scope

**Default path:** `THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/TRUST_POLICY.json`
**CLI override:** `--trust-policy <path>` supports absolute paths (never embedded in outputs)

**Invariants enforced:**
- Deterministic: identical inputs + same trust policy → identical results
- Fail-closed: any duplicate fails with error
- No timestamps, randomness, or environment-dependent behavior

---

### 3. Receipt Attestation Strict Trust
**File:** `THOUGHT/LAB/CAT_CHAT/catalytic_chat/attestation.py`

**New function:**
```python
verify_receipt_attestation(receipt: dict, trust_index: Optional[dict], strict: bool) -> None
```

**Rules:**
- If `receipt["attestation"]` is `null`/`None`: Always OK (no trust needed)
- If attestation exists:
  - Always validate signature correctness (existing behavior preserved)
  - If `strict == True`:
    - `trust_index` MUST be provided
    - Attesting `public_key` MUST be pinned with scope including `"RECEIPT"`
    - Fail-closed with `AttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned
  - If `strict == False`: Signature validity only (no trust policy required)

**Invariant enforced:** Receipt attestation verification always validates signature; strict mode adds trust check without weakening existing verification.

---

### 4. Merkle Attestation Strict Trust
**File:** `THOUGHT/LAB/CAT_CHAT/catalytic_chat/merkle_attestation.py`

**New function:**
```python
verify_merkle_attestation_with_trust(att: dict, merkle_root_hex: str, trust_index: Optional[dict], strict: bool) -> None
```

**Rules:**
- Always validate signature correctness and merkle root match (existing behavior preserved)
- If `strict == True`:
  - `trust_index` MUST be provided
  - `public_key` MUST be pinned with scope including `"MERKLE"`
  - Fail-closed with `MerkleAttestationError("UNTRUSTED_VALIDATOR_KEY")` if not pinned

**Invariant enforced:** Merkle attestation verification always validates signature + root; strict mode adds trust check without weakening existing verification.

---

### 5. CLI: Trust Commands + Strict Verification Flags
**File:** `THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py`

**New command group:** `python -m catalytic_chat.cli trust {init,verify,show}`

#### `trust init`
- Writes starter `TRUST_POLICY.json` to default path
- Deterministic content (no timestamps)
- `allow: []`
- Output: stderr `[OK] wrote TRUST_POLICY.json`, exit 0

#### `trust verify [--trust-policy <path>]`
- Validates policy against schema + uniqueness rules
- Output: stderr `[OK] trust policy valid` or `[FAIL] <reason>`, exit 1

#### `trust show [--trust-policy <path>]`
- Prints canonical JSON summary to stdout ONLY (machine output)
```json
{
  "policy_version":"1.0.0",
  "enabled": <count>,
  "scopes": {"RECEIPT": <count>, "MERKLE": <count>}
}
```
- Uses `canonical_json_bytes()` with trailing `\n`

#### `bundle run` new flags
- `--trust-policy <path>`: Override default policy path
- `--strict-trust`: Enable strict trust verification (fail-closed if policy missing/invalid)
- `--require-attestation`: Receipt attestation MUST be present or fail
- `--require-merkle-attestation`: Merkle attestation MUST be present and valid or fail

**Rules:**
- If `--strict-trust`: Load trust policy, build index; fail-closed if missing/invalid
- If `--require-attestation`: Fail if receipt attestation absent
- If `--require-merkle-attestation`: Fail if merkle attestation absent or invalid
- Default behavior compatible:
  - No `--strict-trust`: Do not require policy
  - No require flags: Do not require attestations
- Stdout purity: Machine JSON commands (e.g., `--attest-merkle` without `--merkle-attestation-out`) output ONLY JSON + `\n`; all status to stderr
- Path traversal defense: No escaping intended directory when loading from bundle context

**Invariants enforced:**
- Fail-closed when strict trust enabled
- Default behavior unchanged (backward compatible)
- Stdout purity for machine JSON outputs
- Path traversal defense

---

### 6. Tests
**File:** `THOUGHT/LAB/CAT_CHAT/tests/test_trust_policy.py`

**Tests (6 total, all passing):**

1. `test_trust_policy_schema_and_uniqueness`
   - Valid empty `allow` policy passes
   - Duplicate `public_key` (case-insensitive) → verify fails
   - Duplicate `validator_id` → verify fails

2. `test_receipt_attestation_strict_trust_blocks_unknown_key`
   - Generate receipt + attestation using SigningKey
   - Build trust policy without that pubkey
   - `verify_receipt_attestation(strict=True)` → `UNTRUSTED_VALIDATOR_KEY`
   - `verify_receipt_attestation(strict=False)` → passes (signature valid)

3. `test_receipt_attestation_strict_trust_allows_pinned_key`
   - Same as above but trust policy includes pubkey with `RECEIPT` scope
   - `verify_receipt_attestation(strict=True)` → passes

4. `test_merkle_attestation_strict_trust_blocks_unknown_key_and_allows_pinned`
   - Generate merkle root + sign it
   - Verify strict fails without pin, passes with pin and `MERKLE` scope

5. `test_cli_trust_verify`
   - Use `subprocess.run` to call `trust verify --trust-policy <tmpfile>`
   - Assert exit codes

6. `test_cli_trust_show`
   - Verify stdout JSON structure and counts

**Testing constraints:**
- Use `tmp_path` for writing policy files
- No OS-specific absolute path format assertions
- No new skipped tests

**Invariant enforced:** All tests deterministic, no skipped tests added

---

## Files Changed

| File | Action | Invariant Enforced |
|-------|--------|-------------------|
| `THOUGHT/LAB/CAT_CHAT/SCHEMAS/trust_policy.schema.json` | Created | Schema enforces valid policy structure |
| `THOUGHT/LAB/CAT_CHAT/catalytic_chat/trust_policy.py` | Created | Deterministic indexing, uniqueness enforcement |
| `THOUGHT/LAB/CAT_CHAT/catalytic_chat/attestation.py` | Modified | Receipt attestation strict trust without weakening verification |
| `THOUGHT/LAB/CAT_CHAT/catalytic_chat/merkle_attestation.py` | Modified | Merkle attestation strict trust without weakening verification |
| `THOUGHT/LAB/CAT_CHAT/catalytic_chat/cli.py` | Modified | CLI trust commands, strict flags, stdout purity |
| `THOUGHT/LAB/CAT_CHAT/tests/test_trust_policy.py` | Created | Comprehensive test coverage |
| `THOUGHT/LAB/CAT_CHAT/CORTEX/_generated/TRUST_POLICY.json` | Created | Default empty trust policy |

---

## Verification Results

### Full test suite
```
84 passed, 13 skipped in 6.45s
```

### Trust policy tests
```
6 passed in 0.83s
```

### Trust policy init + verify
```
[OK] wrote TRUST_POLICY.json
[OK] trust policy valid
```

---

## Invariants Summary

| Invariant | Enforced By | Description |
|-----------|--------------|-------------|
| Determinism | All modules | Identical inputs + same trust policy → identical results |
| Fail-closed | All verification paths | Unknown keys fail when strict trust enabled |
| Schema validation | `trust_policy.py` | Policy must conform to JSON schema |
| Uniqueness | `trust_policy.py` | No duplicate `validator_id` or `public_key` (case-insensitive) |
| No existing verification weakened | `attestation.py`, `merkle_attestation.py` | Signature validation always occurs; trust check is additional |
| Stdout purity | `cli.py` | Machine JSON outputs ONLY JSON + `\n`; status to stderr |
| Path traversal defense | `cli.py`, `trust_policy.py` | No escaping intended directories when loading policies |
| No timestamps/randomness | All modules | Deterministic behavior, no environment-dependent data |
| Default compatibility | `cli.py` | No `--strict-trust` = no policy required |
| Minimal diffs | All changes | Localized to CAT_CHAT, preserved existing behavior |

---

## Checklist

- [x] Schema created and validates correctly
- [x] Trust policy loader with uniqueness enforcement
- [x] Receipt attestation strict trust (non-breaking)
- [x] Merkle attestation strict trust (non-breaking)
- [x] CLI trust commands (`init`, `verify`, `show`)
- [x] CLI strict trust flags to `bundle run`
- [x] Stdout purity for machine JSON outputs
- [x] Default empty trust policy created
- [x] Comprehensive tests (6/6 passing)
- [x] All tests passing (84 passed, 13 skipped)
- [x] CAT_CHAT_ROADMAP.md updated with Phase 6.6 complete
- [x] CHANGELOG.md updated with Phase 6.6 entry
- [x] Commit plan document created

---

## Notes

1. **Backward Compatibility:** All existing `verify_receipt_bytes()` and `verify_merkle_attestation()` functions remain unchanged. New strict trust functions (`verify_receipt_attestation`, `verify_merkle_attestation_with_trust`) are additions that do not break existing callers.

2. **No "Best Effort" Fallbacks:** In strict mode, unknown keys always fail with `UNTRUSTED_VALIDATOR_KEY`. There is no fallback to "signature valid but unknown" in strict mode.

3. **Stdout Purity:** The `--attest-merkle` flag without `--merkle-attestation-out` outputs ONLY JSON + `\n` to stdout. All status goes to stderr via `sys.stderr.write()`.

4. **Path Traversal Defense:** When loading policies from bundle context, the implementation ensures no escaping of intended directories. Absolute paths are supported but never embedded into outputs.

5. **Determinism:** All outputs are deterministic. No timestamps, randomness, or environment-dependent data is included in any generated artifacts.
