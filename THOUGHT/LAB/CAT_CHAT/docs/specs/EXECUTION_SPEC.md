# Execution Semantics Specification v1.0.0

## 1. Overview

The **executor** runs bundles deterministically and emits receipts. Execution is **fail-closed**: any policy violation halts execution immediately.

**Key Properties:**
- Deterministic: Same bundle produces same receipt (excluding attestation)
- Bounded: Executor reads only from bundle artifacts (artifact confinement)
- Fail-Closed: Policy violations halt execution; never continue silently
- Attestable: Optional signing of receipts and Merkle roots

## 2. Execution Policy Schema

```json
{
  "policy_version": "1.0.0",
  "require_verify_bundle": true | false,
  "require_verify_chain": true | false,
  "require_receipt_attestation": true | false,
  "require_merkle_attestation": true | false,
  "strict_trust": true | false,
  "strict_identity": true | false,
  "trust_policy_path": "<path_to_trust_policy.json>",
  "receipt_attestation_quorum": {
    "required": <integer>,
    "scope": "RECEIPT"
  },
  "merkle_attestation_quorum": {
    "required": <integer>,
    "scope": "MERKLE"
  }
}
```

### 2.1 Policy Field Descriptions

| Field | Default | Description |
|-------|---------|-------------|
| `require_verify_bundle` | false | Verify bundle integrity before execution |
| `require_verify_chain` | false | Verify receipt chain and compute Merkle root |
| `require_receipt_attestation` | false | Every receipt must be signed |
| `require_merkle_attestation` | false | Merkle root must be signed |
| `strict_trust` | false | Keys must be in trust policy allow list |
| `strict_identity` | false | validator_id and build_id must match pinned values |
| `trust_policy_path` | null | Path to trust policy file (required if strict modes enabled) |

## 3. Execution Flow

```
1. Load bundle manifest
2. [If require_verify_bundle] Verify bundle integrity
3. Execute each step in ordinal order
4. Build receipt with step results
5. Compute receipt_hash
6. [If signing_key] Sign receipt with attestation
7. Write receipt to output path
8. Enforce post-execution policy gates
```

### 3.1 Step Execution

For each step in `manifest.steps` (ordered by `ordinal ASC, step_id ASC`):

```python
for step in manifest["steps"]:
    step_result = {
        "ordinal": step["ordinal"],
        "step_id": step["step_id"],
        "op": step["op"],
        "outcome": "SUCCESS",
        "result": None,
        "error": None
    }
    steps_results.append(step_result)
```

### 3.2 Receipt Construction

```python
receipt = {
    "receipt_version": "5.0.0",
    "run_id": manifest["run_id"],
    "job_id": manifest["job_id"],
    "bundle_id": manifest["bundle_id"],
    "plan_hash": manifest["plan_hash"],
    "executor_version": "1.0.0",
    "outcome": "SUCCESS",
    "error": None,
    "steps": steps_results,
    "artifacts": artifact_hashes,
    "root_hash": manifest["hashes"]["root_hash"],
    "parent_receipt_hash": previous_receipt_hash,
    "receipt_hash": None,  # Computed after
    "attestation": None,   # Added after signing
    "receipt_index": 0
}

# Compute receipt hash
receipt["receipt_hash"] = compute_receipt_hash(receipt)

# Sign if key provided
if signing_key:
    receipt["attestation"] = sign_receipt_bytes(receipt_bytes, signing_key)
```

## 4. Fail-Closed Behavior

### 4.1 Pre-Execution Failures

| Condition | Result |
|-----------|--------|
| Bundle manifest not found | FAIL with exit code 2 |
| Bundle verification failure | FAIL with exit code 1 |
| Invalid execution policy | FAIL with exit code 2 |
| Missing trust policy (when required) | FAIL with exit code 2 |

### 4.2 Execution Failures

| Condition | Result |
|-----------|--------|
| Step execution error | Record in receipt, step outcome = FAILURE |
| Artifact read failure | Step outcome = FAILURE |

### 4.3 Post-Execution Policy Failures

| Condition | Result |
|-----------|--------|
| Missing attestation when required | FAIL with RuntimeError |
| Trust verification failure | FAIL with RuntimeError |
| Chain verification failure | FAIL with RuntimeError |
| Quorum not met | FAIL with RuntimeError |

## 5. Post-Execution Policy Enforcement

The executor enforces policy gates in this order:

### 5.1 Receipt Attestation Check

```python
if policy.get("require_receipt_attestation", False):
    if receipt.get("attestation") is None:
        raise RuntimeError("Policy violation: receipt attestation required but missing")

    if trust_index is not None:
        verify_receipt_attestation(
            receipt,
            trust_index,
            strict=policy.get("strict_trust", False),
            strict_identity=policy.get("strict_identity", False)
        )
```

### 5.2 Chain Verification

```python
if policy.get("require_verify_chain", False):
    receipts = find_receipt_chain(receipts_dir, run_id)
    if len(receipts) > 0:
        merkle_root = verify_receipt_chain(
            receipts,
            verify_attestation=policy.get("require_receipt_attestation", False)
        )
    else:
        raise RuntimeError("Cannot verify chain - no receipts found")
```

### 5.3 Merkle Attestation Check

```python
if policy.get("require_merkle_attestation", False):
    if merkle_root is None:
        raise RuntimeError("Merkle attestation requires verify_chain")

    if merkle_attestation_path:
        # Verify existing attestation
        verify_merkle_attestation_with_trust(
            attestation,
            merkle_root,
            trust_index,
            strict=policy.get("strict_trust", False),
            strict_identity=policy.get("strict_identity", False)
        )
    elif policy.get("emit_merkle_attestation", False):
        # Generate new attestation
        attestation = sign_merkle_root(merkle_root, signing_key, ...)
        write_merkle_attestation(out_path, attestation)
```

## 6. Exit Codes

| Code | Meaning | Examples |
|------|---------|----------|
| 0 | OK | Execution and all verifications passed |
| 1 | Verification failed | Hash mismatch, trust failure, policy violation, ordering error |
| 2 | Invalid input | Missing file, invalid JSON, schema validation failure |
| 3 | Internal error | Unexpected exception |

## 7. Determinism Requirements

For deterministic execution:

1. **No timestamps**: Receipt contains no timestamps
2. **No randomness**: Step ordering is deterministic
3. **Canonical JSON**: All output uses `sort_keys=True, separators=(",",":")`
4. **Same inputs = same outputs**: Same bundle produces identical receipt hash (excluding attestation)

### 7.1 What Varies

The `attestation` field may vary between executions:
- Signature depends on private key
- build_id may differ if code changes
- validator_id may differ

The `receipt_hash` is computed BEFORE attestation, so it remains deterministic.

## 8. Artifact Confinement

The executor operates under **artifact confinement**:

- Reads ONLY from bundle artifacts (in `artifacts/` directory)
- No access to live repository files
- Bundle is completely self-contained

This ensures reproducibility: anyone with the bundle can verify execution.

## 9. Chain Linking

### 9.1 First Execution

For the first receipt in a chain:
```json
"parent_receipt_hash": null,
"receipt_index": 0
```

### 9.2 Subsequent Executions

When `previous_receipt` is provided:
```python
prev_receipt = load_receipt(previous_receipt)
parent_receipt_hash = prev_receipt["receipt_hash"]
receipt_index = prev_receipt["receipt_index"] + 1
```

## 10. BundleExecutor API

```python
class BundleExecutor:
    def __init__(
        self,
        bundle_dir: Path,
        receipt_out: Optional[Path] = None,
        signing_key: Optional[Path] = None,
        previous_receipt: Optional[Path] = None,
        repo_root: Optional[Path] = None,
        policy: Optional[Mapping[str, Any]] = None
    ):
        """
        Initialize executor.

        Args:
            bundle_dir: Path to bundle directory
            receipt_out: Output path for receipt (default: bundle_dir/receipt.json)
            signing_key: Path to 32-byte Ed25519 private key
            previous_receipt: Path to previous receipt for chain linking
            repo_root: Repository root for build_id computation
            policy: Execution policy dictionary
        """
        pass

    def execute(self) -> dict:
        """
        Execute bundle and write receipt.

        Returns:
            Dictionary with receipt_path, attestation, and receipt fields

        Raises:
            FileNotFoundError: If bundle manifest not found
            RuntimeError: If policy violation occurs
        """
        pass
```

## 11. CLI Usage

```powershell
# Basic execution
python -m catalytic_chat.cli bundle run --bundle "path/to/bundle"

# With signing
python -m catalytic_chat.cli bundle run --bundle "path/to/bundle" --signing-key "path/to/key"

# With policy
python -m catalytic_chat.cli bundle run --bundle "path/to/bundle" --policy "path/to/policy.json"

# Chain linking
python -m catalytic_chat.cli bundle run --bundle "path/to/bundle" --previous-receipt "path/to/prev.json"
```

## 12. Example Execution Policy

```json
{
  "policy_version": "1.0.0",
  "require_verify_bundle": true,
  "require_verify_chain": true,
  "require_receipt_attestation": true,
  "require_merkle_attestation": false,
  "strict_trust": true,
  "strict_identity": false,
  "trust_policy_path": "THOUGHT/LAB/CAT_CHAT/_generated/TRUST_POLICY.json"
}
```

## 13. Implementation Reference

- Source: `catalytic_chat/executor.py`
- Class: `BundleExecutor`
- Schema: `SCHEMAS/execution_policy.schema.json`
