# Trust Model Specification v1.0.0

## 1. Overview

The **trust policy** defines which validator public keys are authorized to sign receipts and Merkle attestations. The system is fail-closed: unknown keys are rejected.

**Key Properties:**
- Key Pinning: Only pre-approved keys may attest
- Scope-Based: Keys authorized for specific operations (RECEIPT, MERKLE)
- Fail-Closed: Unknown or disabled keys cause verification failure

## 2. Trust Policy Schema

### 2.1 Top-Level Structure

```json
{
  "policy_version": "1.0.0",
  "allow": [...]
}
```

### 2.2 Validator Entry Schema

Each entry in `allow` defines an authorized validator:

```json
{
  "validator_id": "<human_readable_name>",
  "public_key": "<ed25519_hex_64_chars>",
  "build_id": "git:<commit>" | "file:<hash>",
  "schemes": ["ed25519"],
  "scope": ["RECEIPT"] | ["MERKLE"] | ["RECEIPT", "MERKLE"],
  "enabled": true | false
}
```

### 2.3 Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `validator_id` | Yes | Human-readable identifier (must be unique) |
| `public_key` | Yes | Ed25519 public key as 64-char hex (must be unique) |
| `build_id` | No | Expected code identity for strict identity verification |
| `schemes` | Yes | Allowed signature schemes (must include `"ed25519"`) |
| `scope` | Yes | Operations this key may attest: `"RECEIPT"` and/or `"MERKLE"` |
| `enabled` | Yes | Whether this validator is currently active |

## 3. Trust Index

The trust index is built from the policy for efficient lookup:

```python
def build_trust_index(policy):
    allow = policy.get("allow", [])
    validator_ids = set()
    public_keys = set()

    by_public_key = {}
    by_validator_id = {}

    for entry in allow:
        validator_id = entry["validator_id"]
        public_key = entry["public_key"].lower()

        # Detect duplicates
        if validator_id in validator_ids:
            raise TrustPolicyError(f"duplicate validator_id: {validator_id}")
        if public_key in public_keys:
            raise TrustPolicyError(f"duplicate public_key: {entry['public_key']}")

        validator_ids.add(validator_id)
        public_keys.add(public_key)

        by_public_key[public_key] = entry
        by_validator_id[validator_id] = entry

    return {
        "by_public_key": by_public_key,
        "by_validator_id": by_validator_id
    }
```

**Note**: Public keys are normalized to lowercase for comparison.

## 4. Key Authorization Check

```python
def is_key_allowed(index, public_key_hex, scope, scheme="ed25519"):
    """
    Check if public key is allowed for given scope and scheme.

    Args:
        index: Trust index from build_trust_index
        public_key_hex: Public key hex string
        scope: "RECEIPT" or "MERKLE"
        scheme: Signature scheme (default "ed25519")

    Returns:
        True if key is allowed, False otherwise
    """
    public_key_lower = public_key_hex.lower()
    entry = index["by_public_key"].get(public_key_lower)

    if not entry:
        return False  # Unknown key

    if not entry.get("enabled", False):
        return False  # Disabled validator

    if scheme not in entry.get("schemes", []):
        return False  # Unsupported scheme

    if scope not in entry.get("scope", []):
        return False  # Unauthorized scope

    return True
```

## 5. Scope Definitions

| Scope | Description |
|-------|-------------|
| `RECEIPT` | Attestation on individual execution receipts |
| `MERKLE` | Attestation on chain Merkle root |

A validator may be authorized for one or both scopes.

## 6. Strict Trust Mode

When `strict_trust` is enabled in execution policy:

1. **All attestation keys must be in allow list**: Unknown keys cause immediate failure
2. **Key must be enabled**: Disabled keys cause failure
3. **Scheme must match**: Only allowed schemes accepted
4. **Scope must match**: Key must be authorized for the specific operation

```python
def verify_with_strict_trust(attestation, trust_index, scope):
    public_key = attestation["public_key"]

    if not is_key_allowed(trust_index, public_key, scope):
        raise TrustPolicyError(f"Key not authorized for {scope}")

    # Proceed with signature verification
    verify_signature(attestation)
```

## 7. Strict Identity Mode

When `strict_identity` is enabled (requires `strict_trust`):

1. **validator_id must match**: Attestation `validator_id` must equal pinned entry
2. **build_id must match**: If pinned entry has `build_id`, attestation must match

```python
def verify_with_strict_identity(attestation, trust_index):
    public_key = attestation["public_key"].lower()
    entry = trust_index["by_public_key"].get(public_key)

    if not entry:
        raise TrustPolicyError("Unknown key")

    # Verify validator_id matches
    if attestation.get("validator_id") != entry.get("validator_id"):
        raise TrustPolicyError("validator_id mismatch")

    # Verify build_id matches (if pinned)
    pinned_build_id = entry.get("build_id")
    if pinned_build_id:
        if attestation.get("build_id") != pinned_build_id:
            raise TrustPolicyError("build_id mismatch")
```

## 8. Signature Verification

### 8.1 Ed25519 Verification

```python
def verify_ed25519_signature(message_bytes, public_key_hex, signature_hex):
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignature

    public_key = bytes.fromhex(public_key_hex)
    signature = bytes.fromhex(signature_hex)

    verify_key = VerifyKey(public_key)
    try:
        verify_key.verify(message_bytes, signature)
        return True
    except BadSignature:
        return False
```

### 8.2 Receipt Attestation Verification

```python
def verify_receipt_attestation(receipt, trust_index, strict=False, strict_identity=False):
    attestation = receipt.get("attestation")
    if attestation is None:
        raise AttestationError("No attestation present")

    public_key = attestation.get("public_key", "").lower()

    # Check trust policy
    if strict:
        if not is_key_allowed(trust_index, public_key, "RECEIPT"):
            raise AttestationError("Key not authorized for RECEIPT scope")

    if strict_identity:
        verify_with_strict_identity(attestation, trust_index)

    # Verify signature
    message_bytes = receipt_signed_bytes(receipt)
    if not verify_ed25519_signature(message_bytes, public_key, attestation["signature"]):
        raise AttestationError("Invalid signature")
```

## 9. Merkle Attestation

### 9.1 Message Format

The Merkle attestation signs a deterministic message:

```
CAT_CHAT_MERKLE_V1:<root>|VID:<validator_id>|BID:<build_id>|PK:<public_key>
```

Where:
- `<root>` is the Merkle root hex
- `<validator_id>` is the validator identifier
- `<build_id>` is the code identity
- `<public_key>` is the signing public key

### 9.2 Merkle Attestation Schema

```json
{
  "attestation_version": "1.0.0",
  "merkle_root": "<sha256>",
  "message": "CAT_CHAT_MERKLE_V1:...",
  "scheme": "ed25519",
  "public_key": "<hex_64>",
  "signature": "<hex_128>",
  "validator_id": "<string>",
  "build_id": "<string>",
  "run_id": "<string>",
  "job_id": "<string>",
  "bundle_id": "<sha256>"
}
```

### 9.3 Verification

```python
def verify_merkle_attestation(attestation, expected_root, trust_index, strict=False, strict_identity=False):
    # Check Merkle root matches
    if attestation["merkle_root"] != expected_root:
        raise MerkleAttestationError("Merkle root mismatch")

    # Reconstruct expected message
    expected_message = build_merkle_message(
        attestation["merkle_root"],
        attestation["validator_id"],
        attestation["build_id"],
        attestation["public_key"]
    )

    if attestation["message"] != expected_message:
        raise MerkleAttestationError("Message mismatch")

    # Check trust policy
    if strict:
        if not is_key_allowed(trust_index, attestation["public_key"], "MERKLE"):
            raise MerkleAttestationError("Key not authorized for MERKLE scope")

    if strict_identity:
        verify_with_strict_identity(attestation, trust_index)

    # Verify signature over message bytes
    message_bytes = attestation["message"].encode('utf-8')
    if not verify_ed25519_signature(message_bytes, attestation["public_key"], attestation["signature"]):
        raise MerkleAttestationError("Invalid signature")
```

## 10. Loading Trust Policy

```python
def load_trust_policy(path):
    # Read file bytes
    policy_bytes = path.read_bytes()

    # Parse JSON
    policy_text = policy_bytes.decode('utf-8')
    policy = json.loads(policy_text)

    # Validate against schema
    jsonschema.validate(instance=policy, schema=TRUST_POLICY_SCHEMA)

    return policy
```

## 11. Example Trust Policy

```json
{
  "policy_version": "1.0.0",
  "allow": [
    {
      "validator_id": "ci_validator_prod",
      "public_key": "a1b2c3d4e5f6...",
      "build_id": "git:abc123def456...",
      "schemes": ["ed25519"],
      "scope": ["RECEIPT", "MERKLE"],
      "enabled": true
    },
    {
      "validator_id": "dev_validator",
      "public_key": "f6e5d4c3b2a1...",
      "schemes": ["ed25519"],
      "scope": ["RECEIPT"],
      "enabled": true
    }
  ]
}
```

## 12. Implementation Reference

- Source: `catalytic_chat/trust_policy.py`
- Functions: `load_trust_policy_bytes`, `parse_trust_policy`, `build_trust_index`, `is_key_allowed`
- Schema: `SCHEMAS/trust_policy.schema.json`
