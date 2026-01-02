#!/usr/bin/env python3
"""
Receipt Module (Phase 6.1)

Deterministic receipt emission from bundle execution.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

RECEIPT_VERSION = "1.0.0"
EXECUTOR_VERSION = "1.0.0"

try:
    import jsonschema
except ImportError:
    jsonschema = None

from catalytic_chat.bundle import BundleError


def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    """Convert dict to canonical JSON bytes with trailing newline.
    
    Args:
        obj: Dictionary to serialize
        
    Returns:
        UTF-8 encoded canonical JSON with exactly one trailing newline
    """
    json_str = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return (json_str + "\n").encode('utf-8')


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hex digest.
    
    Args:
        data: Bytes to hash
        
    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(data).hexdigest()


def validate_receipt_schema(receipt: Dict[str, Any]) -> None:
    """Validate receipt against schema.
    
    Args:
        receipt: Receipt dictionary to validate
        
    Raises:
        ValueError: If receipt fails validation
    """
    try:
        import jsonschema
    except ImportError:
        from .bundle import BundleError
        raise BundleError("jsonschema package required for receipt validation")
    
    schema_path = Path(__file__).parent.parent / "SCHEMAS" / "receipt.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Receipt schema not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    jsonschema.validate(instance=receipt, schema=schema)


def build_receipt_from_bundle_run(
    bundle_manifest: Dict[str, Any],
    step_results: List[Dict[str, Any]],
    outcome: str,
    error: Optional[Dict[str, Any]] = None,
    parent_receipt_hash: Optional[str] = None
) -> Dict[str, Any]:
    """Build receipt from bundle execution.

    Args:
        bundle_manifest: Bundle manifest
        step_results: List of step execution results
        outcome: "SUCCESS" or "FAILURE"
        error: Optional error dict with code, message, step_id
        parent_receipt_hash: Hash of previous receipt in chain, or null for first receipt

    Returns:
        Receipt dictionary
    """
    sorted_steps = sorted(step_results, key=lambda x: (x.get("ordinal", 0), x.get("step_id", "")))

    receipt_steps = []
    for step_result in sorted_steps:
        step_receipt = {
            "ordinal": step_result.get("ordinal"),
            "step_id": step_result.get("step_id"),
            "op": step_result.get("op"),
            "outcome": step_result.get("outcome", "SUCCESS")
        }

        if "result" in step_result:
            step_receipt["result"] = step_result["result"]
        else:
            step_receipt["result"] = None

        if "error" in step_result:
            step_receipt["error"] = step_result["error"]
        else:
            step_receipt["error"] = None

        receipt_steps.append(step_receipt)

    sorted_artifacts = sorted(
        bundle_manifest.get("artifacts", []),
        key=lambda x: x.get("artifact_id", "")
    )

    receipt_artifacts = []
    for artifact in sorted_artifacts:
        receipt_artifacts.append({
            "artifact_id": artifact["artifact_id"],
            "sha256": artifact["sha256"],
            "bytes": artifact["bytes"]
        })

    hash_strings = []
    for artifact in sorted_artifacts:
        hash_strings.append(f"{artifact['artifact_id']}:{artifact['sha256']}")
    combined = "\n".join(hash_strings) + "\n"
    root_hash = sha256_hex(combined.encode('utf-8'))

    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "run_id": bundle_manifest.get("run_id"),
        "job_id": bundle_manifest.get("job_id"),
        "bundle_id": bundle_manifest.get("bundle_id"),
        "plan_hash": bundle_manifest.get("plan_hash"),
        "executor_version": EXECUTOR_VERSION,
        "outcome": outcome,
        "error": error,
        "steps": receipt_steps,
        "artifacts": receipt_artifacts,
        "root_hash": root_hash,
        "parent_receipt_hash": parent_receipt_hash,
        "receipt_hash": None,
        "attestation": None,
        "receipt_index": None
    }

    return receipt


def receipt_canonical_bytes(receipt: Dict[str, Any], attestation_override: Any = None) -> bytes:
    """Compute canonical receipt bytes with optional attestation override.

    This is the single source of truth for receipt canonicalization.
    Used by signer, verifier, and executor to ensure consistent behavior.

    Args:
        receipt: Receipt dictionary
        attestation_override: If provided, override the attestation field.
                           Useful for verification where we set to None.

    Returns:
        Canonical JSON bytes with trailing newline
    """
    receipt_copy = dict(receipt)

    if "attestation" in receipt_copy:
        receipt_copy["attestation"] = attestation_override

    return canonical_json_bytes(receipt_copy)


def receipt_signed_bytes(receipt: Dict[str, Any]) -> bytes:
    """Compute canonical receipt bytes for signing with attestation stub.

    Extracts identity fields from attestation (if present) and builds a signing
    stub to ensure those fields are included in the signed message.

    Args:
        receipt: Receipt dictionary with or without attestation

    Returns:
        Canonical JSON bytes with trailing newline
    """
    receipt_copy = dict(receipt)

    if "attestation" in receipt_copy and receipt_copy["attestation"] is not None:
        attestation = receipt_copy["attestation"]

        signing_stub = {
            "scheme": attestation.get("scheme"),
            "public_key": attestation.get("public_key", "").lower() if isinstance(attestation.get("public_key"), str) else attestation.get("public_key"),
            "signature": ""
        }

        if "validator_id" in attestation:
            signing_stub["validator_id"] = attestation["validator_id"]

        if "build_id" in attestation:
            signing_stub["build_id"] = attestation["build_id"]

        receipt_copy["attestation"] = signing_stub

    return canonical_json_bytes(receipt_copy)


def write_receipt(out_path: Path, receipt: Dict[str, Any]) -> None:
    """Write receipt to file as exact bytes.

    Args:
        out_path: Path to write receipt
        receipt: Receipt dictionary
    """
    receipt_bytes = canonical_json_bytes(receipt)
    out_path.write_bytes(receipt_bytes)


def compute_receipt_hash(receipt: Dict[str, Any]) -> str:
    """Compute receipt hash from canonical bytes without attestation or receipt_hash.

    This is the single source of truth for receipt_hash computation.
    Used by executor and verifier for chain linking.

    Args:
        receipt: Receipt dictionary

    Returns:
        SHA256 hex digest of canonical receipt bytes with attestation=None and receipt_hash excluded
    """
    receipt_copy = dict(receipt)

    if "receipt_hash" in receipt_copy:
        del receipt_copy["receipt_hash"]

    canonical_bytes = receipt_canonical_bytes(receipt_copy, attestation_override=None)
    return sha256_hex(canonical_bytes)


def load_receipt(receipt_path: Path) -> Optional[Dict[str, Any]]:
    """Load receipt from file.

    Args:
        receipt_path: Path to receipt file

    Returns:
        Receipt dictionary or None if file doesn't exist
    """
    if not receipt_path.exists():
        return None

    receipt_bytes = receipt_path.read_bytes()
    receipt_text = receipt_bytes.decode('utf-8').rstrip('\n')
    return json.loads(receipt_text)


def compute_merkle_root(receipt_hashes: List[str]) -> str:
    """Compute Merkle root from list of receipt hashes.

    Leaves must follow deterministic ordering (same as verify_receipt_chain input).
    Does not re-sort receipt_hashes internally.

    Args:
        receipt_hashes: List of receipt_hash hex strings in execution order

    Returns:
        Merkle root as SHA256 hex string

    Raises:
        ValueError: If receipt_hashes is empty
    """
    if not receipt_hashes:
        raise ValueError("Cannot compute Merkle root from empty list")

    level = [bytes.fromhex(h) for h in receipt_hashes]

    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                combined = level[i] + level[i + 1]
            else:
                combined = level[i] + level[i]
            next_level.append(hashlib.sha256(combined).digest())
        level = next_level

    return level[0].hex()


def verify_receipt_chain(
    receipts: List[Dict[str, Any]],
    verify_attestation: bool = True
) -> str:
    """Verify receipt chain integrity and return Merkle root.

    Args:
        receipts: List of receipts in execution order
        verify_attestation: If True, verify attestation signatures

    Returns:
        Merkle root of the receipt chain

    Raises:
        ValueError: If chain verification fails
    """
    if not receipts:
        raise ValueError("Receipt chain cannot be empty")

    receipt_hashes = []
    has_receipt_index = None

    for i, receipt in enumerate(receipts):
        receipt_hash = receipt.get("receipt_hash")
        parent_receipt_hash = receipt.get("parent_receipt_hash")
        receipt_index = receipt.get("receipt_index")

        if receipt_hash is None:
            raise ValueError(f"Receipt {i} missing receipt_hash field")

        receipt_hashes.append(receipt_hash)

        if i == 0:
            if parent_receipt_hash is not None:
                raise ValueError(f"First receipt {i} must have parent_receipt_hash=null, got {parent_receipt_hash!r}")
        else:
            if parent_receipt_hash is None:
                raise ValueError(f"Receipt {i} missing parent_receipt_hash field")

            prev_receipt = receipts[i - 1]
            prev_hash = prev_receipt.get("receipt_hash")

            if prev_hash is None:
                raise ValueError(f"Previous receipt {i-1} missing receipt_hash field")

            if parent_receipt_hash != prev_hash:
                raise ValueError(
                    f"Receipt {i} parent_receipt_hash={parent_receipt_hash!r} "
                    f"does not match previous receipt's receipt_hash={prev_hash!r}"
                )

        computed_hash = compute_receipt_hash(receipt)
        if computed_hash != receipt_hash:
            raise ValueError(
                f"Receipt {i} receipt_hash mismatch: "
                f"computed={computed_hash!r}, stored={receipt_hash!r}"
            )

        if verify_attestation:
            attestation = receipt.get("attestation")
            if attestation is not None:
                from .attestation import verify_receipt_bytes
                receipt_bytes = receipt_canonical_bytes(receipt)
                verify_receipt_bytes(receipt_bytes, attestation)

        if i == 0:
            has_receipt_index = receipt_index is not None
            if has_receipt_index and receipt_index != 0:
                raise ValueError(f"receipt_index must start at 0, got {receipt_index}")
        else:
            current_has_index = receipt_index is not None
            if current_has_index != has_receipt_index:
                raise ValueError("All receipts must have receipt_index set or all must be null")

            if receipt_index is not None:
                prev_receipt = receipts[i - 1]
                prev_index = prev_receipt.get("receipt_index")
                if prev_index is None:
                    raise ValueError(f"Previous receipt {i-1} missing receipt_index")
                if receipt_index != prev_index + 1:
                    if receipt_index <= prev_index:
                        raise ValueError(f"receipt_index must be strictly increasing: {prev_index} -> {receipt_index}")
                    else:
                        raise ValueError(f"receipt_index must be contiguous: gap detected between {prev_index} and {receipt_index}")

    return compute_merkle_root(receipt_hashes)


def find_receipt_chain(receipts_dir: Path, run_id: str) -> List[Dict[str, Any]]:
    """Find and load all receipts for a run in execution order.

    Args:
        receipts_dir: Directory containing receipt files
        run_id: Run ID to search for

    Returns:
        List of receipts sorted by receipt_index (if present) or receipt_hash

    Raises:
        ValueError: If duplicate receipt_index, duplicate receipt_hash, or mixed receipt_index/null
    """
    if not receipts_dir.exists():
        return []

    receipts_with_files = []
    for receipt_file in receipts_dir.glob(f"{run_id}_*.json"):
        receipt = load_receipt(receipt_file)
        if receipt and receipt.get("run_id") == run_id:
            receipts_with_files.append((receipt, receipt_file))

    if not receipts_with_files:
        return []

    has_receipt_index = None
    seen_indices = set()
    seen_hashes = set()

    for receipt, receipt_file in receipts_with_files:
        receipt_index = receipt.get("receipt_index")
        receipt_hash = receipt.get("receipt_hash")

        if receipt_hash is None:
            raise ValueError(f"Receipt in {receipt_file.name} missing receipt_hash")

        if receipt_hash in seen_hashes:
            raise ValueError(f"Duplicate receipt_hash: {receipt_hash}")

        seen_hashes.add(receipt_hash)

        if has_receipt_index is None:
            has_receipt_index = receipt_index is not None
        else:
            current_has_index = receipt_index is not None
            if current_has_index != has_receipt_index:
                raise ValueError("All receipts must have receipt_index set or all must be null")

        if receipt_index is not None:
            if receipt_index in seen_indices:
                raise ValueError(f"Duplicate receipt_index: {receipt_index}")
            seen_indices.add(receipt_index)

    def ordering_key(item):
        receipt, receipt_file = item
        receipt_index = receipt.get("receipt_index")
        receipt_hash = receipt.get("receipt_hash")
        filename = receipt_file.name

        if receipt_index is not None:
            return (0, receipt_index)
        elif receipt_hash is not None:
            return (1, receipt_hash)
        else:
            return (2, filename)

    sorted_receipts_with_files = sorted(receipts_with_files, key=ordering_key)
    return [receipt for receipt, _ in sorted_receipts_with_files]
