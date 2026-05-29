# Receipt Format Specification v1.0.0

## 1. Overview

A **receipt** is an immutable proof of bundle execution. Receipts form a hash-linked chain, enabling verification of execution history and computation of a Merkle root for attestation.

**Key Properties:**
- Immutable: Receipts are write-once
- Chain-Linked: Each receipt references its predecessor
- Verifiable: Receipt hash computed deterministically
- Attestable: Optional cryptographic signatures

## 2. Receipt Schema

### 2.1 Required Fields

```json
{
  "receipt_version": "1.0.0",
  "run_id": "<string>",
  "job_id": "<string>",
  "bundle_id": "<sha256>",
  "plan_hash": "<sha256>",
  "executor_version": "1.0.0",
  "outcome": "SUCCESS" | "FAILURE",
  "error": null | { "code": "<string>", "message": "<string>", "step_id": "<string>" },
  "steps": [...],
  "artifacts": [...],
  "root_hash": "<sha256>",
  "parent_receipt_hash": null | "<sha256>",
  "receipt_hash": "<sha256>",
  "attestation": null | { ... },
  "receipt_index": null | <integer>
}
```

### 2.2 Step Result Schema

Steps are ordered by `(ordinal ASC, step_id ASC)`:

```json
{
  "ordinal": <integer>,
  "step_id": "<string>",
  "op": "READ_SYMBOL" | "READ_SECTION",
  "outcome": "SUCCESS" | "FAILURE" | "SKIPPED",
  "result": null | { ... },
  "error": null | { "code": "<string>", "message": "<string>" }
}
```

### 2.3 Artifact Entry Schema

Artifacts are ordered by `artifact_id ASC`:

```json
{
  "artifact_id": "<sha256_prefix_16>",
  "sha256": "<content_hash>",
  "bytes": <integer>
}
```

## 3. Receipt Hash Computation

The `receipt_hash` is the SHA-256 of the canonical receipt bytes with `receipt_hash` removed and `attestation` set to `null`:

```python
def compute_receipt_hash(receipt):
    receipt_copy = dict(receipt)

    # Remove receipt_hash field
    if "receipt_hash" in receipt_copy:
        del receipt_copy["receipt_hash"]

    # Set attestation to null
    receipt_copy["attestation"] = None

    # Compute canonical bytes with trailing newline
    canonical_bytes = canonical_json(receipt_copy) + "\n"
    return SHA256(canonical_bytes.encode('utf-8'))
```

**Critical**: The hash is computed BEFORE adding the attestation, so the attestation itself is not part of the hash.

## 4. Chain Linking

### 4.1 First Receipt

The first receipt in a chain MUST have:
```json
"parent_receipt_hash": null
```

### 4.2 Subsequent Receipts

Each subsequent receipt MUST reference the previous receipt:
```json
"parent_receipt_hash": "<previous_receipt.receipt_hash>"
```

### 4.3 Receipt Index

If used, `receipt_index` MUST:
- Start at `0` for the first receipt
- Be strictly increasing and contiguous: `0, 1, 2, 3, ...`
- Either ALL receipts have `receipt_index` or ALL have `null`

## 5. Merkle Root Computation

The Merkle root is computed from receipt hashes in execution order:

```python
def compute_merkle_root(receipt_hashes):
    if not receipt_hashes:
        raise ValueError("Cannot compute Merkle root from empty list")

    # Convert to bytes
    level = [bytes.fromhex(h) for h in receipt_hashes]

    # Binary tree construction
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                # Pair exists
                combined = level[i] + level[i + 1]
            else:
                # Odd leaf: duplicate
                combined = level[i] + level[i]
            next_level.append(SHA256(combined))
        level = next_level

    return level[0].hex()
```

**Note**: For odd number of leaves, the last leaf is duplicated.

## 6. Chain Verification Protocol

To verify a receipt chain:

```python
def verify_receipt_chain(receipts, verify_attestation=True):
    if not receipts:
        raise ValueError("Receipt chain cannot be empty")

    receipt_hashes = []

    for i, receipt in enumerate(receipts):
        # 1. Check receipt_hash field exists
        if receipt.get("receipt_hash") is None:
            raise ValueError(f"Receipt {i} missing receipt_hash")

        # 2. Verify chain linkage
        parent = receipt.get("parent_receipt_hash")
        if i == 0:
            if parent is not None:
                raise ValueError("First receipt must have parent=null")
        else:
            prev_hash = receipts[i-1].get("receipt_hash")
            if parent != prev_hash:
                raise ValueError("Chain link broken")

        # 3. Recompute and verify receipt_hash
        computed = compute_receipt_hash(receipt)
        if computed != receipt.get("receipt_hash"):
            raise ValueError("Receipt hash mismatch")

        # 4. Verify attestation if present and required
        if verify_attestation and receipt.get("attestation"):
            verify_attestation_signature(receipt)

        # 5. Verify receipt_index contiguity
        verify_receipt_index(receipt, i, receipts)

        receipt_hashes.append(receipt["receipt_hash"])

    return compute_merkle_root(receipt_hashes)
```

## 7. Attestation Object

### 7.1 Schema

```json
{
  "scheme": "ed25519",
  "public_key": "<hex_64_chars>",
  "signature": "<hex_128_chars>",
  "validator_id": "<string>",
  "build_id": "git:<hash>" | "file:<hash>"
}
```

### 7.2 Field Descriptions

| Field | Description |
|-------|-------------|
| `scheme` | Signature scheme (only `ed25519` supported) |
| `public_key` | Ed25519 public key as 64-char hex (32 bytes) |
| `signature` | Ed25519 signature as 128-char hex (64 bytes) |
| `validator_id` | Human-readable validator identifier |
| `build_id` | Code identity: `git:<commit>` or `file:<hash>` |

### 7.3 Signing Process

```python
def sign_receipt(receipt, private_key):
    # Build signing stub (identity fields, empty signature)
    signing_stub = {
        "scheme": "ed25519",
        "public_key": derive_public_key(private_key).hex(),
        "signature": "",
        "validator_id": get_validator_id(),
        "build_id": get_build_id()
    }

    # Set attestation to stub for signing
    receipt_copy = dict(receipt)
    receipt_copy["attestation"] = signing_stub

    # Compute canonical bytes
    message = canonical_json(receipt_copy) + "\n"

    # Sign
    signature = ed25519_sign(private_key, message.encode('utf-8'))

    # Return complete attestation
    signing_stub["signature"] = signature.hex()
    return signing_stub
```

## 8. Canonical Bytes Functions

### 8.1 For Hash Computation

```python
def receipt_canonical_bytes(receipt, attestation_override=None):
    receipt_copy = dict(receipt)
    if "attestation" in receipt_copy:
        receipt_copy["attestation"] = attestation_override
    return canonical_json(receipt_copy) + "\n"
```

### 8.2 For Signing

```python
def receipt_signed_bytes(receipt):
    receipt_copy = dict(receipt)
    if receipt_copy.get("attestation"):
        # Build signing stub with identity but empty signature
        signing_stub = {
            "scheme": attestation["scheme"],
            "public_key": attestation["public_key"].lower(),
            "signature": "",
            "validator_id": attestation.get("validator_id"),
            "build_id": attestation.get("build_id")
        }
        receipt_copy["attestation"] = signing_stub
    return canonical_json(receipt_copy) + "\n"
```

## 9. File Operations

### 9.1 Writing Receipts

```python
def write_receipt(out_path, receipt):
    receipt_bytes = canonical_json(receipt) + "\n"
    out_path.write_bytes(receipt_bytes.encode('utf-8'))
```

### 9.2 Loading Receipts

```python
def load_receipt(receipt_path):
    if not receipt_path.exists():
        return None
    receipt_bytes = receipt_path.read_bytes()
    receipt_text = receipt_bytes.decode('utf-8').rstrip('\n')
    return json.loads(receipt_text)
```

## 10. Finding Receipt Chains

Receipts for a run are found by glob pattern `{run_id}_*.json`:

```python
def find_receipt_chain(receipts_dir, run_id):
    receipts = []
    for receipt_file in receipts_dir.glob(f"{run_id}_*.json"):
        receipt = load_receipt(receipt_file)
        if receipt and receipt.get("run_id") == run_id:
            receipts.append(receipt)

    # Sort by receipt_index (if present) or receipt_hash
    return sorted(receipts, key=ordering_key)
```

## 11. Implementation Reference

- Source: `catalytic_chat/receipt.py`
- Functions: `build_receipt_from_bundle_run`, `compute_receipt_hash`, `verify_receipt_chain`, `compute_merkle_root`
- Schema: `SCHEMAS/receipt.schema.json`
