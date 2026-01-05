#!/usr/bin/env python3
"""
Run Bundle Contract (P.2.3): Freezing "what is a run".

Implements run bundle creation, verification, and GC rooting semantics.
A run bundle is a proof-carrying immutable record of a complete run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from CAPABILITY.CAS import cas as cas_mod
from CAPABILITY.RUNS import records


# ============================================================================
# Constants
# ============================================================================

RUN_BUNDLE_VERSION = "RB1.0"

# Required artifacts in a run bundle
REQUIRED_ARTIFACTS = ["task_spec_hash", "status_hash", "output_hashes_hash"]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RunBundleManifest:
    """
    Immutable manifest for a run bundle.
    
    A run bundle is a proof-carrying record that references all artifacts
    of a complete run via CAS hashes.
    """
    version: str
    run_id: str
    task_spec_hash: str  # CAS hash of TASK_SPEC
    status_hash: str  # CAS hash of final STATUS
    output_hashes_hash: str  # CAS hash of OUTPUT_HASHES list
    receipts: List[str]  # CAS hashes of receipts (optional)
    metadata: Dict[str, Any]  # Optional metadata (timestamps, etc.)


@dataclass
class BundleVerificationReceipt:
    """Receipt for bundle verification operation."""
    bundle_ref: str
    verification_status: str  # "VALID" | "INVALID"
    errors: List[str]
    artifact_status: Dict[str, bool]  # artifact_name → exists_in_cas
    manifest_valid: bool
    all_artifacts_present: bool


# ============================================================================
# Validation
# ============================================================================

def _validate_cas_hash(hash_str: str, field_name: str) -> None:
    """Validate that a string is a valid CAS hash (64 lowercase hex)."""
    if not isinstance(hash_str, str):
        raise ValueError(f"{field_name} must be a string, got {type(hash_str)}")
    if len(hash_str) != 64:
        raise ValueError(f"{field_name} must be 64 characters, got {len(hash_str)}")
    if not all(c in "0123456789abcdef" for c in hash_str):
        raise ValueError(f"{field_name} must be lowercase hex, got {hash_str}")


def _validate_run_id(run_id: str) -> None:
    """Validate run_id format."""
    if not isinstance(run_id, str):
        raise ValueError(f"run_id must be a string, got {type(run_id)}")
    if not run_id:
        raise ValueError("run_id cannot be empty")
    # Allow alphanumeric, dash, underscore
    if not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValueError(f"run_id contains invalid characters: {run_id}")


def _canonical_json_bytes(obj: Any) -> bytes:
    """Encode object to canonical JSON bytes (deterministic)."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


# ============================================================================
# Run Bundle Creation
# ============================================================================

def run_bundle_create(
    run_id: str,
    task_spec_hash: str,
    status_hash: str,
    output_hashes_hash: str,
    *,
    receipts: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a run bundle and store it in CAS.
    
    A run bundle is an immutable manifest that references all artifacts
    of a complete run. The bundle itself is stored in CAS and addressable
    by its hash.
    
    Args:
        run_id: Unique identifier for this run
        task_spec_hash: CAS hash of the task specification
        status_hash: CAS hash of the final status
        output_hashes_hash: CAS hash of the output hashes list
        receipts: Optional list of CAS hashes for receipts
        metadata: Optional metadata dictionary
    
    Returns:
        CAS reference (sha256:<hash>) to the bundle manifest
    
    Raises:
        ValueError: On validation failure
    """
    # Validate inputs
    _validate_run_id(run_id)
    _validate_cas_hash(task_spec_hash, "task_spec_hash")
    _validate_cas_hash(status_hash, "status_hash")
    _validate_cas_hash(output_hashes_hash, "output_hashes_hash")
    
    if receipts is not None:
        for i, receipt_hash in enumerate(receipts):
            _validate_cas_hash(receipt_hash, f"receipts[{i}]")
    
    # Build manifest
    manifest_dict = {
        "version": RUN_BUNDLE_VERSION,
        "run_id": run_id,
        "task_spec_hash": task_spec_hash,
        "status_hash": status_hash,
        "output_hashes_hash": output_hashes_hash,
        "receipts": receipts or [],
        "metadata": metadata or {},
    }
    
    # Canonical encoding
    manifest_bytes = _canonical_json_bytes(manifest_dict)
    
    # Store in CAS
    manifest_hash = cas_mod.cas_put(manifest_bytes)
    
    return f"sha256:{manifest_hash}"


# ============================================================================
# Run Bundle Verification
# ============================================================================

def run_bundle_verify(
    bundle_ref: str,
) -> BundleVerificationReceipt:
    """
    Verify a run bundle (dry-run verifier).
    
    Checks:
    - Bundle manifest exists in CAS
    - Manifest is valid JSON and has correct schema
    - All referenced artifacts exist in CAS
    - Hashes are valid format
    
    Args:
        bundle_ref: CAS reference (sha256:<hash>) to bundle manifest
    
    Returns:
        BundleVerificationReceipt with verification results
    """
    errors = []
    artifact_status = {}
    manifest_valid = False
    all_artifacts_present = False
    
    try:
        # Validate ref format
        if not bundle_ref.startswith("sha256:"):
            errors.append(f"Invalid bundle_ref format: {bundle_ref}")
            return BundleVerificationReceipt(
                bundle_ref=bundle_ref,
                verification_status="INVALID",
                errors=errors,
                artifact_status=artifact_status,
                manifest_valid=False,
                all_artifacts_present=False,
            )
        
        bundle_hash = bundle_ref.split(":", 1)[1]
        _validate_cas_hash(bundle_hash, "bundle_hash")
        
        # Load manifest from CAS
        try:
            manifest_bytes = cas_mod.cas_get(bundle_hash)
        except Exception as e:
            errors.append(f"Bundle manifest not found in CAS: {e}")
            return BundleVerificationReceipt(
                bundle_ref=bundle_ref,
                verification_status="INVALID",
                errors=errors,
                artifact_status=artifact_status,
                manifest_valid=False,
                all_artifacts_present=False,
            )
        
        # Parse manifest
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as e:
            errors.append(f"Invalid JSON in manifest: {e}")
            return BundleVerificationReceipt(
                bundle_ref=bundle_ref,
                verification_status="INVALID",
                errors=errors,
                artifact_status=artifact_status,
                manifest_valid=False,
                all_artifacts_present=False,
            )
        
        # Validate manifest schema
        if manifest.get("version") != RUN_BUNDLE_VERSION:
            errors.append(f"Invalid version: {manifest.get('version')}")
        
        for field in REQUIRED_ARTIFACTS:
            if field not in manifest:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return BundleVerificationReceipt(
                bundle_ref=bundle_ref,
                verification_status="INVALID",
                errors=errors,
                artifact_status=artifact_status,
                manifest_valid=False,
                all_artifacts_present=False,
            )
        
        manifest_valid = True
        
        # Verify all referenced artifacts exist in CAS
        artifacts_to_check = {
            "task_spec": manifest["task_spec_hash"],
            "status": manifest["status_hash"],
            "output_hashes": manifest["output_hashes_hash"],
        }
        
        for artifact_name, artifact_hash in artifacts_to_check.items():
            try:
                _validate_cas_hash(artifact_hash, artifact_name)
                # Check if blob exists
                blob_path = cas_mod._get_object_path(artifact_hash)
                exists = blob_path.exists()
                artifact_status[artifact_name] = exists
                if not exists:
                    errors.append(f"Artifact {artifact_name} not found in CAS: {artifact_hash}")
            except Exception as e:
                artifact_status[artifact_name] = False
                errors.append(f"Invalid {artifact_name} hash: {e}")
        
        # Check receipts (optional)
        for i, receipt_hash in enumerate(manifest.get("receipts", [])):
            try:
                _validate_cas_hash(receipt_hash, f"receipt[{i}]")
                blob_path = cas_mod._get_object_path(receipt_hash)
                exists = blob_path.exists()
                artifact_status[f"receipt[{i}]"] = exists
                if not exists:
                    errors.append(f"Receipt {i} not found in CAS: {receipt_hash}")
            except Exception as e:
                artifact_status[f"receipt[{i}]"] = False
                errors.append(f"Invalid receipt[{i}] hash: {e}")
        
        all_artifacts_present = all(artifact_status.values())
        
        verification_status = "VALID" if (manifest_valid and all_artifacts_present and not errors) else "INVALID"
        
        return BundleVerificationReceipt(
            bundle_ref=bundle_ref,
            verification_status=verification_status,
            errors=errors,
            artifact_status=artifact_status,
            manifest_valid=manifest_valid,
            all_artifacts_present=all_artifacts_present,
        )
        
    except Exception as e:
        errors.append(f"Unexpected error: {type(e).__name__}: {e}")
        return BundleVerificationReceipt(
            bundle_ref=bundle_ref,
            verification_status="INVALID",
            errors=errors,
            artifact_status=artifact_status,
            manifest_valid=manifest_valid,
            all_artifacts_present=all_artifacts_present,
        )


# ============================================================================
# GC Rooting Semantics
# ============================================================================

def get_bundle_roots(bundle_ref: str) -> List[str]:
    """
    Get all CAS hashes that should be treated as GC roots for this bundle.
    
    This includes:
    - The bundle manifest itself
    - task_spec_hash
    - status_hash
    - output_hashes_hash
    - All receipt hashes
    - All hashes referenced in output_hashes
    
    Args:
        bundle_ref: CAS reference to bundle manifest
    
    Returns:
        List of CAS hashes (64 lowercase hex) that are roots
    """
    
    roots = set()
    
    # Add bundle manifest itself
    bundle_hash = bundle_ref.split(":", 1)[1]
    roots.add(bundle_hash)
    
    try:
        # Load manifest
        manifest_bytes = cas_mod.cas_get(bundle_hash)
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        
        # Add direct artifact references
        roots.add(manifest["task_spec_hash"])
        roots.add(manifest["status_hash"])
        roots.add(manifest["output_hashes_hash"])
        
        # Add receipts
        for receipt_hash in manifest.get("receipts", []):
            roots.add(receipt_hash)
        
        # Load and add output hashes
        try:
            output_hashes = records.load_output_hashes(manifest["output_hashes_hash"])
            roots.update(output_hashes)
        except Exception:
            # If we can't load output_hashes, just skip (verification will catch this)
            pass
        
    except Exception:
        # If we can't load the bundle, just return the bundle hash itself
        pass
    
    return sorted(roots)
