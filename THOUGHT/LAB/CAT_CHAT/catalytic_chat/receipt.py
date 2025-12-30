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
        "attestation": None
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


def write_receipt(out_path: Path, receipt: Dict[str, Any]) -> None:
    """Write receipt to file as exact bytes.

    Args:
        out_path: Path to write receipt
        receipt: Receipt dictionary
    """
    receipt_bytes = canonical_json_bytes(receipt)
    out_path.write_bytes(receipt_bytes)


def compute_receipt_hash(receipt: Dict[str, Any]) -> str:
    """Compute receipt hash from canonical bytes without attestation.

    This is the single source of truth for receipt_hash computation.
    Used by executor and verifier for chain linking.

    Args:
        receipt: Receipt dictionary

    Returns:
        SHA256 hex digest of canonical receipt bytes with attestation=None
    """
    canonical_bytes = receipt_canonical_bytes(receipt, attestation_override=None)
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


def verify_receipt_chain(
    receipts: List[Dict[str, Any]],
    verify_attestation: bool = True
) -> None:
    """Verify receipt chain integrity.

    Args:
        receipts: List of receipts in execution order
        verify_attestation: If True, verify attestation signatures

    Raises:
        ValueError: If chain verification fails
    """
    if not receipts:
        raise ValueError("Receipt chain cannot be empty")

    for i, receipt in enumerate(receipts):
        receipt_hash = receipt.get("receipt_hash")
        parent_receipt_hash = receipt.get("parent_receipt_hash")

        if receipt_hash is None:
            raise ValueError(f"Receipt {i} missing receipt_hash field")

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


def find_receipt_chain(receipts_dir: Path, run_id: str) -> List[Dict[str, Any]]:
    """Find and load all receipts for a run in execution order.

    Args:
        receipts_dir: Directory containing receipt files
        run_id: Run ID to search for

    Returns:
        List of receipts sorted by execution timestamp in filename
    """
    if not receipts_dir.exists():
        return []

    receipts = []
    for receipt_file in sorted(receipts_dir.glob(f"{run_id}_*.json")):
        receipt = load_receipt(receipt_file)
        if receipt and receipt.get("run_id") == run_id:
            receipts.append(receipt)

    return receipts
