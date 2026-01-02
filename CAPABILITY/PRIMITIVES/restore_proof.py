"""
CAT-DPT Restoration Proof Validator

Produces and verifies PROOF.json according to proof.schema.json.

Usage:
    from CATALYTIC_DPT.PRIMITIVES.restore_proof import RestorationProofValidator

    validator = RestorationProofValidator(
        proof_schema_path="CONTEXT/schemas/proof.schema.json"
    )

    proof = validator.generate_proof(
        run_id="run_00000001",
        catalytic_domains=["CATALYTIC-DPT/_scratch"],
        pre_state={"CATALYTIC-DPT/_scratch": {"file.txt": "abc..."}},
        post_state={"CATALYTIC-DPT/_scratch": {"file.txt": "abc..."}},
        timestamp="2025-12-25T00:00:00Z"
    )

    # proof is a dict conforming to proof.schema.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator

from .cas_store import CatalyticStore, normalize_relpath
from .merkle import build_manifest_root

class RestorationProofValidator:
    """Generates and validates restoration proofs."""

    def __init__(self, proof_schema_path: str | Path):
        """
        Initialize validator with proof schema.

        Args:
            proof_schema_path: Path to proof.schema.json
        """
        self.proof_schema_path = Path(proof_schema_path)
        self.proof_schema = json.loads(self.proof_schema_path.read_text(encoding="utf-8"))
        self.validator = Draft7Validator(self.proof_schema)

    def generate_proof(
        self,
        run_id: str,
        catalytic_domains: List[str],
        pre_state: Dict[str, Dict[str, str]],
        post_state: Dict[str, Dict[str, str]],
        timestamp: str,
        referenced_artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate restoration proof by comparing pre and post states.

        Args:
            run_id: Unique run identifier
            catalytic_domains: List of domain roots under verification
            pre_state: Map of domain -> {relative_path -> sha256}
            post_state: Map of domain -> {relative_path -> sha256}
            timestamp: ISO 8601 timestamp
            referenced_artifacts: Optional artifact hashes (ledger_hash, jobspec_hash, validator_id)

        Returns:
            PROOF.json dict conforming to proof.schema.json
        """
        # Build domain states
        pre_domain_state = self._build_domain_state(pre_state)
        post_domain_state = self._build_domain_state(post_state)

        # Compute restoration result
        restoration_result = self._compute_restoration_result(pre_state, post_state)

        # Build proof object (without proof_hash)
        proof_partial = {
            "proof_version": "1.0.0",
            "run_id": run_id,
            "timestamp": timestamp,
            "catalytic_domains": sorted(catalytic_domains),  # deterministic order
            "pre_state": pre_domain_state,
            "post_state": post_domain_state,
            "restoration_result": restoration_result,
        }

        if referenced_artifacts is not None:
            proof_partial["referenced_artifacts"] = referenced_artifacts

        # Compute proof_hash from canonical JSON (sorted keys, no whitespace)
        proof_hash = self._compute_proof_hash(proof_partial)
        proof_partial["proof_hash"] = proof_hash

        # Validate against schema
        errors = list(self.validator.iter_errors(proof_partial))
        if errors:
            raise ValueError(f"Generated proof does not conform to schema: {errors[0].message}")

        return proof_partial

    def _build_domain_state(self, state: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Build domain_state structure with domain_root_hash and file_manifest.

        Args:
            state: Map of domain -> {relative_path -> sha256}

        Returns:
            domain_state dict with domain_root_hash and file_manifest
        """
        file_manifest = flatten_domain_state(state)
        domain_root_hash = compute_manifest_root(file_manifest)

        return {
            "domain_root_hash": domain_root_hash,
            "file_manifest": file_manifest,
        }

    def _compute_restoration_result(
        self, pre_state: Dict[str, Dict[str, str]], post_state: Dict[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Compute restoration result by comparing pre and post states.

        Args:
            pre_state: Map of domain -> {relative_path -> sha256}
            post_state: Map of domain -> {relative_path -> sha256}

        Returns:
            restoration_result dict
        """
        pre_files = flatten_domain_state(pre_state)
        post_files = flatten_domain_state(post_state)

        # Detect mismatches
        mismatches: List[Dict[str, Any]] = []

        # Missing files (in pre but not post)
        for path in sorted(set(pre_files.keys()) - set(post_files.keys())):
            mismatches.append({
                "path": path,
                "type": "missing",
                "expected_hash": pre_files[path],
            })

        # Extra files (in post but not pre)
        for path in sorted(set(post_files.keys()) - set(pre_files.keys())):
            mismatches.append({
                "path": path,
                "type": "extra",
                "actual_hash": post_files[path],
            })

        # Hash mismatches (in both but different hash)
        for path in sorted(set(pre_files.keys()) & set(post_files.keys())):
            if pre_files[path] != post_files[path]:
                mismatches.append({
                    "path": path,
                    "type": "hash_mismatch",
                    "expected_hash": pre_files[path],
                    "actual_hash": post_files[path],
                })

        # Determine result
        if not mismatches:
            return {
                "verified": True,
                "condition": "RESTORED_IDENTICAL",
            }
        else:
            # Determine primary failure condition
            has_missing = any(m["type"] == "missing" for m in mismatches)
            has_extra = any(m["type"] == "extra" for m in mismatches)
            has_hash_mismatch = any(m["type"] == "hash_mismatch" for m in mismatches)

            if has_hash_mismatch:
                condition = "RESTORATION_FAILED_HASH_MISMATCH"
            elif has_missing:
                condition = "RESTORATION_FAILED_MISSING_FILES"
            elif has_extra:
                condition = "RESTORATION_FAILED_EXTRA_FILES"
            else:
                condition = "RESTORATION_FAILED_DOMAIN_UNREACHABLE"

            return {
                "verified": False,
                "condition": condition,
                "mismatches": mismatches,
            }

    def _compute_proof_hash(self, proof_partial: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of proof object (canonical JSON, sorted keys, no whitespace).

        Args:
            proof_partial: Proof dict without proof_hash field

        Returns:
            Lowercase hex SHA-256 hash
        """
        canonical_json = json.dumps(proof_partial, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_manifest_root(file_manifest: Dict[str, str]) -> str:
    """
    Compute a deterministic root hash for a flat file manifest.

    - If manifest is non-empty: Merkle root via `build_manifest_root`.
    - If manifest is empty: deterministic sentinel sha256(b"") (keeps backwards compatibility with prior proofs).
    """
    if not file_manifest:
        return hashlib.sha256(b"").hexdigest()
    return build_manifest_root(file_manifest)


def flatten_domain_state(state: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Flatten {domain -> {path -> sha256}} into a single manifest.

    Backwards compatibility:
    - If exactly one domain exists: keys remain the per-domain relative paths.
    - If multiple domains exist: keys are normalized as "<domain>/<path>" to avoid collisions.
    """
    domains = sorted(state.keys())
    if len(domains) == 1:
        domain = domains[0]
        files = state.get(domain, {})
        out: Dict[str, str] = {}
        for path, sha in sorted(files.items(), key=lambda kv: kv[0]):
            out[normalize_relpath(path)] = sha
        return out

    out = {}
    for domain in domains:
        files = state.get(domain, {})
        for path, sha in sorted(files.items(), key=lambda kv: kv[0]):
            combined = normalize_relpath(f"{domain}/{path}")
            out[combined] = sha
    return out


def compute_domain_manifest(domain_path: Path, *, cas: CatalyticStore) -> Dict[str, str]:
    """
    Compute a domain manifest for a directory:
      { normalized_relpath -> bytes_hash }

    bytes_hash is the SHA-256 of file bytes, sourced via CAS (content addressed, idempotent).
    """
    if not domain_path.exists():
        return {}

    items: list[tuple[str, Path]] = []
    for p in domain_path.rglob("*"):
        if p.is_file():
            rel = normalize_relpath(p.relative_to(domain_path))
            items.append((rel, p))

    items.sort(key=lambda t: t[0])
    manifest: Dict[str, str] = {}
    for rel, p in items:
        with open(p, "rb") as f:
            bytes_hash = cas.put_stream(f)
        manifest[rel] = bytes_hash
    return manifest
