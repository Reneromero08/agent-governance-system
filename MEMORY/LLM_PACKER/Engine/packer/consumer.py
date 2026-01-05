#!/usr/bin/env python3
"""
Pack Consumer (P.2.2): Verification + rehydration.

Implements pack_consume() to verify and materialize CAS-addressed pack manifests.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from CAPABILITY.ARTIFACTS.store import load_bytes
from CAPABILITY.CAS import cas as cas_mod


# Constants
P2_MANIFEST_VERSION = "P2.0"


@dataclass
class ConsumptionReceipt:
    """Receipt for pack consumption operation."""
    manifest_ref: str
    cas_snapshot_hash: str
    out_dir: str
    tree_hash: str
    verification_summary: Dict[str, Any]
    commands_run: List[str]
    exit_status: str  # "SUCCESS" | "FAILED"
    errors: List[str]


def _validate_manifest_ref(ref: str) -> str:
    """Validate that ref is a proper sha256: reference."""
    if not ref.startswith("sha256:"):
        raise ValueError(f"PACK_CONSUME_INVALID_REF: must start with 'sha256:', got: {ref}")
    hash_part = ref.split(":", 1)[1]
    if len(hash_part) != 64 or not all(c in "0123456789abcdef" for c in hash_part):
        raise ValueError(f"PACK_CONSUME_INVALID_HASH: {hash_part}")
    return ref


def _validate_manifest_schema(manifest: Dict[str, Any]) -> None:
    """Validate manifest conforms to Pack Manifest v1 schema."""
    # Required top-level fields
    required_fields = ["version", "scope", "entries"]
    for field in required_fields:
        if field not in manifest:
            raise ValueError(f"PACK_CONSUME_MISSING_FIELD: {field}")
    
    # Version check
    if manifest["version"] != P2_MANIFEST_VERSION:
        raise ValueError(f"PACK_CONSUME_VERSION_MISMATCH: expected {P2_MANIFEST_VERSION}, got {manifest['version']}")
    
    # Scope validation
    valid_scopes = {"ags", "lab", "cat"}
    if manifest["scope"] not in valid_scopes:
        raise ValueError(f"PACK_CONSUME_INVALID_SCOPE: {manifest['scope']}")
    
    # Entries validation
    if not isinstance(manifest["entries"], list):
        raise ValueError("PACK_CONSUME_INVALID_ENTRIES: must be a list")
    
    # Validate each entry
    for i, entry in enumerate(manifest["entries"]):
        required_entry_fields = ["path", "ref", "bytes", "kind"]
        for field in required_entry_fields:
            if field not in entry:
                raise ValueError(f"PACK_CONSUME_ENTRY_MISSING_FIELD: entry[{i}].{field}")
        
        # Validate ref format
        _validate_manifest_ref(entry["ref"])
        
        # Validate path safety
        path = Path(entry["path"])
        if path.is_absolute():
            raise ValueError(f"PACK_CONSUME_ABSOLUTE_PATH: {entry['path']}")
        if ".." in path.parts:
            raise ValueError(f"PACK_CONSUME_PATH_TRAVERSAL: {entry['path']}")


def _verify_cas_blobs_exist(manifest: Dict[str, Any], cas_root: Path) -> List[str]:
    """
    Verify all blobs referenced in manifest exist in CAS.
    
    Returns list of missing blob hashes (empty if all present).
    """
    missing = []
    
    for entry in manifest["entries"]:
        ref = entry["ref"]
        hash_hex = ref.split(":", 1)[1]
        
        try:
            blob_path = cas_mod._get_object_path(hash_hex)
            if not blob_path.exists():
                missing.append(hash_hex)
        except Exception:
            missing.append(hash_hex)
    
    return missing


def _compute_tree_hash(out_dir: Path) -> str:
    """
    Compute deterministic hash of materialized tree.
    
    Hash is computed over sorted list of (path, content_hash) pairs.
    """
    entries = []
    
    for item in sorted(out_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(out_dir).as_posix()
            file_hash = hashlib.sha256(item.read_bytes()).hexdigest()
            entries.append(f"{file_hash} {rel_path}")
    
    # Canonical encoding: one entry per line, sorted
    tree_bytes = "\n".join(sorted(entries)).encode("utf-8")
    return hashlib.sha256(tree_bytes).hexdigest()


def pack_consume(
    manifest_ref: str,
    out_dir: Path,
    *,
    dry_run: bool = False,
    cas_root: Optional[Path] = None,
) -> ConsumptionReceipt:
    """
    Consume a CAS-addressed pack manifest and materialize files.
    
    Args:
        manifest_ref: sha256: reference to pack manifest
        out_dir: Directory to materialize files into
        dry_run: If True, verify only (don't write files)
        cas_root: Optional CAS root override (for testing)
    
    Returns:
        ConsumptionReceipt with verification results
    
    Raises:
        ValueError: On any validation failure (fail-closed)
    """
    commands_run = []
    errors = []
    
    if cas_root is None:
        cas_root = cas_mod._CAS_ROOT
    
    try:
        # Step 1: Validate manifest ref format
        _validate_manifest_ref(manifest_ref)
        commands_run.append(f"validate_ref({manifest_ref})")
        
        # Step 2: Load manifest from CAS
        try:
            manifest_bytes = load_bytes(manifest_ref)
            commands_run.append(f"load_bytes({manifest_ref})")
        except Exception as e:
            raise ValueError(f"PACK_CONSUME_MANIFEST_NOT_FOUND: {manifest_ref}: {e}")
        
        # Step 3: Verify manifest integrity (canonical encoding)
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"PACK_CONSUME_INVALID_JSON: {e}")
        
        # Verify canonical encoding (re-encode and compare)
        canonical_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
        if canonical_bytes != manifest_bytes:
            raise ValueError("PACK_CONSUME_NON_CANONICAL_ENCODING")
        
        commands_run.append("verify_canonical_encoding")
        
        # Step 4: Validate manifest schema
        _validate_manifest_schema(manifest)
        commands_run.append("validate_schema")
        
        # Step 5: Verify all referenced blobs exist in CAS
        missing_blobs = _verify_cas_blobs_exist(manifest, cas_root)
        if missing_blobs:
            raise ValueError(f"PACK_CONSUME_MISSING_BLOBS: {len(missing_blobs)} blobs missing: {missing_blobs[:5]}")
        
        commands_run.append(f"verify_blobs_exist(count={len(manifest['entries'])})")
        
        # Step 6: Materialize files (if not dry-run)
        if not dry_run:
            # Atomic materialization: write to temp dir, then rename
            with tempfile.TemporaryDirectory(prefix="pack_consume_") as temp_str:
                temp_dir = Path(temp_str)
                
                # Materialize all files
                for entry in manifest["entries"]:
                    rel_path = entry["path"]
                    ref = entry["ref"]
                    
                    # Load blob from CAS
                    blob_bytes = load_bytes(ref)
                    
                    # Write to temp location
                    dest = temp_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(blob_bytes)
                
                commands_run.append(f"materialize({len(manifest['entries'])} files)")
                
                # Atomic move: rename temp dir to final location
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                
                # If out_dir exists, fail-closed (don't overwrite)
                if out_dir.exists():
                    raise ValueError(f"PACK_CONSUME_OUT_DIR_EXISTS: {out_dir}")
                
                # Rename is atomic on most filesystems
                temp_dir.rename(out_dir)
                commands_run.append(f"atomic_rename({temp_dir} -> {out_dir})")
            
            # Compute tree hash
            tree_hash = _compute_tree_hash(out_dir)
        else:
            tree_hash = ""  # Not computed in dry-run
            commands_run.append("dry_run (no materialization)")
        
        # Build verification summary
        verification_summary = {
            "manifest_version": manifest["version"],
            "scope": manifest["scope"],
            "entry_count": len(manifest["entries"]),
            "total_bytes": sum(e["bytes"] for e in manifest["entries"]),
            "schema_valid": True,
            "encoding_canonical": True,
            "blobs_present": True,
        }
        
        # Compute CAS snapshot hash (deterministic)
        blob_hashes = sorted([e["ref"].split(":", 1)[1] for e in manifest["entries"]])
        cas_snapshot_hash = hashlib.sha256("\n".join(blob_hashes).encode("utf-8")).hexdigest()
        
        return ConsumptionReceipt(
            manifest_ref=manifest_ref,
            cas_snapshot_hash=cas_snapshot_hash,
            out_dir=str(out_dir),
            tree_hash=tree_hash,
            verification_summary=verification_summary,
            commands_run=commands_run,
            exit_status="SUCCESS",
            errors=[],
        )
        
    except ValueError:
        # Re-raise validation errors (already formatted)
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise ValueError(f"PACK_CONSUME_INTERNAL_ERROR: {type(e).__name__}: {e}")
