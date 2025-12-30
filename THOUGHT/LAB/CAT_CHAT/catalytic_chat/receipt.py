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
    error: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build receipt from bundle execution.
    
    Args:
        bundle_manifest: Bundle manifest
        step_results: List of step execution results
        outcome: "SUCCESS" or "FAILURE"
        error: Optional error dict with code, message, step_id
        
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
