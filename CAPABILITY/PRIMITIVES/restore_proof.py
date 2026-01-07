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
from .merkle import build_manifest_root, build_manifest_with_proofs, verify_membership, MerkleProof

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
        include_membership_proofs: bool = False,
        previous_proof_hash: Optional[str] = None,
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
            include_membership_proofs: If True, include per-file Merkle membership proofs
                                       for selective verification without full manifest
            previous_proof_hash: Optional hash of previous proof in chain (SPECTRUM-03).
                                 If provided, this proof links to the previous proof.
                                 Use None for the first proof in a chain.

        Returns:
            PROOF.json dict conforming to proof.schema.json
        """
        # Build domain states (with optional membership proofs)
        pre_domain_state = self._build_domain_state(pre_state, include_membership_proofs)
        post_domain_state = self._build_domain_state(post_state, include_membership_proofs)

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

        # Add chain linkage if provided (SPECTRUM-03)
        if previous_proof_hash is not None:
            proof_partial["previous_proof_hash"] = previous_proof_hash

        # Compute proof_hash from canonical JSON (sorted keys, no whitespace)
        proof_hash = self._compute_proof_hash(proof_partial)
        proof_partial["proof_hash"] = proof_hash

        # Validate against schema
        errors = list(self.validator.iter_errors(proof_partial))
        if errors:
            raise ValueError(f"Generated proof does not conform to schema: {errors[0].message}")

        return proof_partial

    def _build_domain_state(
        self,
        state: Dict[str, Dict[str, str]],
        include_membership_proofs: bool = False,
    ) -> Dict[str, Any]:
        """
        Build domain_state structure with domain_root_hash and file_manifest.

        Args:
            state: Map of domain -> {relative_path -> sha256}
            include_membership_proofs: If True, include per-file Merkle membership proofs

        Returns:
            domain_state dict with domain_root_hash, file_manifest, and optionally membership_proofs
        """
        file_manifest = flatten_domain_state(state)
        domain_root_hash, membership_proofs = compute_manifest_root_with_proofs(
            file_manifest, include_proofs=include_membership_proofs
        )

        result: Dict[str, Any] = {
            "domain_root_hash": domain_root_hash,
            "file_manifest": file_manifest,
        }

        if include_membership_proofs and membership_proofs:
            # Serialize proofs for JSON embedding
            result["membership_proofs"] = {
                path: proof.to_dict() for path, proof in membership_proofs.items()
            }

        return result

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


def compute_manifest_root_with_proofs(
    file_manifest: Dict[str, str],
    include_proofs: bool = False,
) -> tuple[str, Optional[Dict[str, MerkleProof]]]:
    """
    Compute a deterministic root hash for a flat file manifest, optionally with membership proofs.

    Args:
        file_manifest: Map of relative_path -> sha256 hash
        include_proofs: If True, also generate per-file membership proofs

    Returns:
        (root_hash, proofs) where proofs is None if include_proofs=False or manifest is empty,
        otherwise a dict mapping path -> MerkleProof
    """
    if not file_manifest:
        return hashlib.sha256(b"").hexdigest(), None

    if include_proofs:
        root, proofs = build_manifest_with_proofs(file_manifest)
        return root, proofs
    else:
        return build_manifest_root(file_manifest), None


def verify_file_membership(
    path: str,
    bytes_hash: str,
    proof: Dict[str, Any] | MerkleProof,
    expected_root: str,
) -> bool:
    """
    Verify that a specific file was in the manifest that produced the given root.

    This enables selective verification without requiring the full manifest -
    useful for proving "file X was in domain at snapshot time" without revealing other files.

    Args:
        path: The file path (must match the proof's path)
        bytes_hash: The file's SHA-256 content hash
        proof: MerkleProof object or dict representation from PROOF.json
        expected_root: The Merkle root to verify against (from domain_root_hash)

    Returns:
        True if the proof is valid, False otherwise
    """
    return verify_membership(path, bytes_hash, proof, expected_root)


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


# =============================================================================
# SPECTRUM-03: Chain Verification
# =============================================================================


def compute_proof_hash(proof: Dict[str, Any]) -> str:
    """
    Compute the proof_hash of a proof dict.

    This recomputes the hash from the proof contents, excluding the proof_hash field.
    Useful for chain verification.

    Args:
        proof: The proof dict (may include proof_hash field, which is ignored)

    Returns:
        SHA-256 hash of canonical JSON
    """
    proof_without_hash = {k: v for k, v in proof.items() if k != "proof_hash"}
    canonical = json.dumps(proof_without_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_chain(proofs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify a chain of proofs linked by previous_proof_hash.

    Args:
        proofs: List of proof dicts in chain order (oldest first).
                Each proof (except the first) must have previous_proof_hash
                pointing to the proof_hash of the previous proof.

    Returns:
        {
            "ok": bool,
            "code": str,  # "CHAIN_VALID" or error code
            "message": str,
            "chain_length": int,
            "chain_root": str | None,  # proof_hash of first proof
            "chain_head": str | None,  # proof_hash of last proof
            "failed_at_index": int | None,  # index of first failure
        }
    """
    if not proofs:
        return {
            "ok": False,
            "code": "CHAIN_EMPTY",
            "message": "Chain is empty",
            "chain_length": 0,
            "chain_root": None,
            "chain_head": None,
            "failed_at_index": None,
        }

    chain_root = proofs[0].get("proof_hash")

    # First proof should NOT have previous_proof_hash (or it should be None)
    first_previous = proofs[0].get("previous_proof_hash")
    if first_previous is not None:
        return {
            "ok": False,
            "code": "CHAIN_ROOT_HAS_PREVIOUS",
            "message": f"First proof has previous_proof_hash but should not: {first_previous}",
            "chain_length": len(proofs),
            "chain_root": chain_root,
            "chain_head": None,
            "failed_at_index": 0,
        }

    # Verify each proof's hash matches its computed hash
    for i, proof in enumerate(proofs):
        expected_hash = proof.get("proof_hash")
        if not expected_hash:
            return {
                "ok": False,
                "code": "PROOF_HASH_MISSING",
                "message": f"Proof at index {i} has no proof_hash",
                "chain_length": len(proofs),
                "chain_root": chain_root,
                "chain_head": None,
                "failed_at_index": i,
            }

        computed_hash = compute_proof_hash(proof)
        if computed_hash != expected_hash:
            return {
                "ok": False,
                "code": "PROOF_HASH_MISMATCH",
                "message": f"Proof at index {i} has tampered proof_hash. Expected {computed_hash}, got {expected_hash}",
                "chain_length": len(proofs),
                "chain_root": chain_root,
                "chain_head": None,
                "failed_at_index": i,
            }

    # Verify chain linkage
    for i in range(1, len(proofs)):
        current_previous = proofs[i].get("previous_proof_hash")
        expected_previous = proofs[i - 1].get("proof_hash")

        if current_previous is None:
            return {
                "ok": False,
                "code": "CHAIN_LINK_MISSING",
                "message": f"Proof at index {i} has no previous_proof_hash",
                "chain_length": len(proofs),
                "chain_root": chain_root,
                "chain_head": None,
                "failed_at_index": i,
            }

        if current_previous != expected_previous:
            return {
                "ok": False,
                "code": "CHAIN_LINK_MISMATCH",
                "message": f"Proof at index {i} has wrong previous_proof_hash. Expected {expected_previous}, got {current_previous}",
                "chain_length": len(proofs),
                "chain_root": chain_root,
                "chain_head": None,
                "failed_at_index": i,
            }

    chain_head = proofs[-1].get("proof_hash")

    return {
        "ok": True,
        "code": "CHAIN_VALID",
        "message": f"Chain of {len(proofs)} proofs is valid",
        "chain_length": len(proofs),
        "chain_root": chain_root,
        "chain_head": chain_head,
        "failed_at_index": None,
    }


def get_chain_history(
    head_proof: Dict[str, Any],
    proof_loader: Any,  # Callable[[str], Dict[str, Any] | None]
) -> List[Dict[str, Any]]:
    """
    Walk backwards from a head proof to reconstruct the full chain.

    Args:
        head_proof: The most recent proof in the chain
        proof_loader: A callable that takes a proof_hash and returns
                      the proof dict, or None if not found.

    Returns:
        List of proofs from oldest to newest (chain order).
        Returns empty list if chain is broken or incomplete.
    """
    chain = [head_proof]

    current = head_proof
    while True:
        previous_hash = current.get("previous_proof_hash")
        if previous_hash is None:
            # Reached the chain root
            break

        previous_proof = proof_loader(previous_hash)
        if previous_proof is None:
            # Chain is broken - can't find previous proof
            return []

        chain.append(previous_proof)
        current = previous_proof

    # Reverse to get oldest-first order
    chain.reverse()
    return chain
