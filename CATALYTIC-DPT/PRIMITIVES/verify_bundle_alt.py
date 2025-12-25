#!/usr/bin/env python3
"""
Alternate Bundle/Chain Verifier (code-independent)

This module implements an independent verification path for:
- SPECTRUM-05 v1.0.0 bundle verification (10-phase)
- SPECTRUM-05 v1.0.0 chain verification (Section 6)

Constraints:
- Must NOT reuse existing verification logic/functions from verify_bundle.py.
- May share the same error codes (constants) and artifact contracts.
- Deterministic, fail-fast classification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature

    _ED25519_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ED25519_AVAILABLE = False


ERROR_CODES: Dict[str, str] = {
    "OK": "OK",
    "ARTIFACT_MISSING": "ARTIFACT_MISSING",
    "ARTIFACT_MALFORMED": "ARTIFACT_MALFORMED",
    "ARTIFACT_EXTRA": "ARTIFACT_EXTRA",
    "FIELD_MISSING": "FIELD_MISSING",
    "FIELD_EXTRA": "FIELD_EXTRA",
    "ALGORITHM_UNSUPPORTED": "ALGORITHM_UNSUPPORTED",
    "KEY_INVALID": "KEY_INVALID",
    "IDENTITY_INVALID": "IDENTITY_INVALID",
    "IDENTITY_MISMATCH": "IDENTITY_MISMATCH",
    "IDENTITY_MULTIPLE": "IDENTITY_MULTIPLE",
    "SIGNATURE_MALFORMED": "SIGNATURE_MALFORMED",
    "SIGNATURE_INCOMPLETE": "SIGNATURE_INCOMPLETE",
    "SIGNATURE_INVALID": "SIGNATURE_INVALID",
    "SIGNATURE_MULTIPLE": "SIGNATURE_MULTIPLE",
    "BUNDLE_ROOT_MISMATCH": "BUNDLE_ROOT_MISMATCH",
    "CHAIN_ROOT_MISMATCH": "CHAIN_ROOT_MISMATCH",
    "DECISION_INVALID": "DECISION_INVALID",
    "PAYLOAD_MISMATCH": "PAYLOAD_MISMATCH",
    "SERIALIZATION_INVALID": "SERIALIZATION_INVALID",
    "RESTORATION_FAILED": "RESTORATION_FAILED",
    "FORBIDDEN_ARTIFACT": "FORBIDDEN_ARTIFACT",
    "OUTPUT_MISSING": "OUTPUT_MISSING",
    "HASH_MISMATCH": "HASH_MISMATCH",
    "CHAIN_EMPTY": "CHAIN_EMPTY",
    "CHAIN_DUPLICATE_RUN": "CHAIN_DUPLICATE_RUN",
}


class AltBundleVerifier:
    """Independent verifier for SPECTRUM-05 bundles/chains."""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[3]
        else:
            self.project_root = Path(project_root)

    def verify_bundle_spectrum05(self, run_dir: Path, strict: bool = True, check_proof: bool = True) -> Dict:
        if strict and not _ED25519_AVAILABLE:
            return {
                "ok": False,
                "code": ERROR_CODES["ALGORITHM_UNSUPPORTED"],
                "details": {"message": "Ed25519 verification requires 'cryptography' library"},
                "message": "Ed25519 verification requires 'cryptography' library",
            }

        run_dir = Path(run_dir) if isinstance(run_dir, str) else run_dir

        # PHASE 1: Artifact Presence Check
        required_artifacts: Dict[str, Path] = {
            "TASK_SPEC.json": run_dir / "TASK_SPEC.json",
            "STATUS.json": run_dir / "STATUS.json",
            "OUTPUT_HASHES.json": run_dir / "OUTPUT_HASHES.json",
            "VALIDATOR_IDENTITY.json": run_dir / "VALIDATOR_IDENTITY.json",
            "SIGNED_PAYLOAD.json": run_dir / "SIGNED_PAYLOAD.json",
            "SIGNATURE.json": run_dir / "SIGNATURE.json",
        }
        if check_proof:
            required_artifacts["PROOF.json"] = run_dir / "PROOF.json"

        for artifact_name, artifact_path in required_artifacts.items():
            if not artifact_path.exists():
                return {
                    "ok": False,
                    "code": ERROR_CODES["ARTIFACT_MISSING"],
                    "message": f"{artifact_name} missing",
                    "details": {"artifact": artifact_name},
                }

        # PHASE 2: Artifact Parse Check
        parsed: Dict[str, Any] = {}
        for artifact_name, artifact_path in required_artifacts.items():
            try:
                raw = artifact_path.read_bytes()
                if artifact_name == "TASK_SPEC.json":
                    parsed["TASK_SPEC.json_bytes"] = raw
                parsed[artifact_name] = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return {
                    "ok": False,
                    "code": ERROR_CODES["ARTIFACT_MALFORMED"],
                    "message": f"{artifact_name} is not valid JSON",
                    "details": {"artifact": artifact_name, "error": str(e)},
                }

        # PHASE 3: Identity Verification
        identity = parsed["VALIDATOR_IDENTITY.json"]
        validator_id = identity.get("validator_id", "")
        public_key = identity.get("public_key", "")

        if strict:
            required_identity_fields = {"algorithm", "public_key", "validator_id"}
            actual_identity_fields = set(identity.keys())
            if actual_identity_fields != required_identity_fields:
                if len(actual_identity_fields) > len(required_identity_fields):
                    return {
                        "ok": False,
                        "code": ERROR_CODES["FIELD_EXTRA"],
                        "message": "VALIDATOR_IDENTITY.json has extra fields",
                        "details": {"extra": list(actual_identity_fields - required_identity_fields)},
                    }
                return {
                    "ok": False,
                    "code": ERROR_CODES["FIELD_MISSING"],
                    "message": "VALIDATOR_IDENTITY.json missing required fields",
                    "details": {"missing": list(required_identity_fields - actual_identity_fields)},
                }

            if identity.get("algorithm") != "ed25519":
                return {
                    "ok": False,
                    "code": ERROR_CODES["ALGORITHM_UNSUPPORTED"],
                    "message": f"Algorithm '{identity.get('algorithm')}' not supported (must be 'ed25519')",
                    "details": {"actual": identity.get("algorithm")},
                }

            public_key = identity.get("public_key", "")
            if not (
                isinstance(public_key, str)
                and len(public_key) == 64
                and public_key == public_key.lower()
                and all(c in "0123456789abcdef" for c in public_key)
            ):
                return {
                    "ok": False,
                    "code": ERROR_CODES["KEY_INVALID"],
                    "message": "public_key must be exactly 64 lowercase hex characters",
                    "details": {"actual_length": len(public_key) if isinstance(public_key, str) else 0},
                }

            try:
                public_key_bytes = bytes.fromhex(public_key)
                computed_validator_id = hashlib.sha256(public_key_bytes).hexdigest()
                declared_validator_id = identity.get("validator_id", "")
                if computed_validator_id != declared_validator_id:
                    return {
                        "ok": False,
                        "code": ERROR_CODES["IDENTITY_INVALID"],
                        "message": "validator_id does not match sha256(public_key)",
                        "details": {"declared": declared_validator_id, "computed": computed_validator_id},
                    }
            except ValueError as e:
                return {
                    "ok": False,
                    "code": ERROR_CODES["KEY_INVALID"],
                    "message": f"public_key hex decode failed: {e}",
                    "details": {},
                }

            validator_id = declared_validator_id

        # PHASE 4: Bundle Root Computation
        task_spec_bytes: bytes = parsed["TASK_SPEC.json_bytes"]
        output_hashes_obj: Dict[str, Any] = parsed["OUTPUT_HASHES.json"]
        status_obj: Dict[str, Any] = parsed["STATUS.json"]
        bundle_root = _compute_bundle_root(output_hashes_obj, status_obj, task_spec_bytes)

        # PHASE 5: Signed Payload Verification
        signed_payload = parsed["SIGNED_PAYLOAD.json"]
        if strict:
            required_payload_fields = {"bundle_root", "decision", "validator_id"}
            actual_payload_fields = set(signed_payload.keys())
            if actual_payload_fields != required_payload_fields:
                if len(actual_payload_fields) > len(required_payload_fields):
                    return {
                        "ok": False,
                        "code": ERROR_CODES["FIELD_EXTRA"],
                        "message": "SIGNED_PAYLOAD.json has extra fields",
                        "details": {"extra": list(actual_payload_fields - required_payload_fields)},
                    }
                return {
                    "ok": False,
                    "code": ERROR_CODES["FIELD_MISSING"],
                    "message": "SIGNED_PAYLOAD.json missing required fields",
                    "details": {"missing": list(required_payload_fields - actual_payload_fields)},
                }

            if signed_payload.get("bundle_root") != bundle_root:
                return {
                    "ok": False,
                    "code": ERROR_CODES["BUNDLE_ROOT_MISMATCH"],
                    "message": "SIGNED_PAYLOAD.bundle_root does not match computed bundle root",
                    "details": {"declared": signed_payload.get("bundle_root"), "computed": bundle_root},
                }

            if signed_payload.get("decision") != "ACCEPT":
                return {
                    "ok": False,
                    "code": ERROR_CODES["DECISION_INVALID"],
                    "message": f"Decision is '{signed_payload.get('decision')}', must be 'ACCEPT'",
                    "details": {"actual": signed_payload.get("decision")},
                }

            if signed_payload.get("validator_id") != validator_id:
                return {
                    "ok": False,
                    "code": ERROR_CODES["IDENTITY_MISMATCH"],
                    "message": "SIGNED_PAYLOAD.validator_id does not match VALIDATOR_IDENTITY.validator_id",
                    "details": {"signed_payload": signed_payload.get("validator_id"), "identity": validator_id},
                }

        # PHASE 6: Signature Verification
        signature_obj = parsed["SIGNATURE.json"]
        if strict:
            required_sig_fields = {"payload_type", "signature", "validator_id"}
            if not required_sig_fields.issubset(signature_obj.keys()):
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_INCOMPLETE"],
                    "message": "SIGNATURE.json missing required fields",
                    "details": {"missing": list(required_sig_fields - set(signature_obj.keys()))},
                }

            if signature_obj.get("payload_type") != "BUNDLE":
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_MALFORMED"],
                    "message": f"payload_type is '{signature_obj.get('payload_type')}', must be 'BUNDLE'",
                    "details": {"actual": signature_obj.get("payload_type")},
                }

            sig_hex = signature_obj.get("signature", "")
            if not (
                isinstance(sig_hex, str)
                and len(sig_hex) == 128
                and sig_hex == sig_hex.lower()
                and all(c in "0123456789abcdef" for c in sig_hex)
            ):
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_MALFORMED"],
                    "message": "signature must be exactly 128 lowercase hex characters",
                    "details": {"actual_length": len(sig_hex) if isinstance(sig_hex, str) else 0},
                }

            if signature_obj.get("validator_id") != validator_id:
                return {
                    "ok": False,
                    "code": ERROR_CODES["IDENTITY_MISMATCH"],
                    "message": "SIGNATURE.validator_id does not match VALIDATOR_IDENTITY.validator_id",
                    "details": {"signature": signature_obj.get("validator_id"), "identity": validator_id},
                }

        canonical_payload = _canonicalize_json(signed_payload)
        signature_message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload

        if strict:
            try:
                if not _verify_ed25519_signature(public_key, signature_obj.get("signature", ""), signature_message):
                    return {
                        "ok": False,
                        "code": ERROR_CODES["SIGNATURE_INVALID"],
                        "message": "Ed25519 signature verification failed",
                        "details": {},
                    }
            except RuntimeError as e:
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_INVALID"],
                    "message": str(e),
                    "details": {},
                }

        # PHASE 7: Proof Verification
        if check_proof:
            proof = parsed["PROOF.json"]
            if "restoration_result" not in proof:
                return {
                    "ok": False,
                    "code": ERROR_CODES["FIELD_MISSING"],
                    "message": "PROOF.json missing restoration_result",
                    "details": {},
                }

            verified = proof["restoration_result"].get("verified")
            if verified is not True:
                condition = proof["restoration_result"].get("condition", "UNKNOWN")
                return {
                    "ok": False,
                    "code": ERROR_CODES["RESTORATION_FAILED"],
                    "message": f"PROOF restoration failed: {condition}",
                    "details": {"verified": verified, "condition": condition},
                }

        # PHASE 8: Forbidden Artifact Check
        forbidden_checks = [
            ("logs/", run_dir / "logs"),
            ("tmp/", run_dir / "tmp"),
            ("transcript.json", run_dir / "transcript.json"),
        ]
        for artifact_name, artifact_path in forbidden_checks:
            if artifact_path.exists():
                return {
                    "ok": False,
                    "code": ERROR_CODES["FORBIDDEN_ARTIFACT"],
                    "message": f"Forbidden artifact present: {artifact_name}",
                    "details": {"artifact": artifact_name},
                }

        # PHASE 9: Output Hash Verification (fail-fast; preserve declared ordering)
        hashes = output_hashes_obj.get("hashes", {})
        for rel_path, expected_hash in hashes.items():
            rel_path_posix = rel_path.replace("\\", "/")
            abs_path = self.project_root / rel_path_posix
            if not abs_path.exists():
                return {
                    "ok": False,
                    "code": ERROR_CODES["OUTPUT_MISSING"],
                    "message": f"Output file does not exist: {rel_path_posix}",
                    "details": {"path": rel_path_posix},
                }

            actual_hash = f"sha256:{_compute_sha256_file(abs_path)}"
            if actual_hash != expected_hash:
                return {
                    "ok": False,
                    "code": ERROR_CODES["HASH_MISMATCH"],
                    "message": f"Hash mismatch for {rel_path_posix}",
                    "details": {"path": rel_path_posix, "expected": expected_hash, "actual": actual_hash},
                }

        # PHASE 10: Acceptance
        if status_obj.get("status") != "success":
            return {
                "ok": False,
                "code": ERROR_CODES["DECISION_INVALID"],
                "message": f"Run status is '{status_obj.get('status')}', must be 'success'",
                "details": {"actual": status_obj.get("status")},
            }

        if status_obj.get("cmp01") != "pass":
            return {
                "ok": False,
                "code": ERROR_CODES["DECISION_INVALID"],
                "message": f"CMP-01 status is '{status_obj.get('cmp01')}', must be 'pass'",
                "details": {"actual": status_obj.get("cmp01")},
            }

        return {"ok": True, "code": ERROR_CODES["OK"], "details": {}, "bundle_root": bundle_root}

    def verify_chain_spectrum05(self, run_dirs: List[Path], strict: bool = True, check_proof: bool = True) -> Dict:
        if strict and not _ED25519_AVAILABLE:
            return {
                "ok": False,
                "code": ERROR_CODES["ALGORITHM_UNSUPPORTED"],
                "message": "Ed25519 verification requires 'cryptography' library",
                "details": {},
            }

        run_dirs = [Path(d) if isinstance(d, str) else d for d in run_dirs]

        if not run_dirs:
            return {
                "ok": False,
                "code": ERROR_CODES["CHAIN_EMPTY"],
                "message": "Chain verification requires at least one run",
                "details": {},
            }

        run_ids = [d.name for d in run_dirs]
        if len(run_ids) != len(set(run_ids)):
            duplicates = [rid for rid in run_ids if run_ids.count(rid) > 1]
            return {
                "ok": False,
                "code": ERROR_CODES["CHAIN_DUPLICATE_RUN"],
                "message": f"Duplicate run_id(s) in chain: {', '.join(set(duplicates))}",
                "details": {"duplicates": list(set(duplicates))},
            }

        bundle_roots: List[str] = []
        for run_dir in run_dirs:
            result = self.verify_bundle_spectrum05(run_dir, strict=strict, check_proof=check_proof)
            if not result["ok"]:
                result["run_id"] = run_dir.name
                return result

            task_spec_bytes = (run_dir / "TASK_SPEC.json").read_bytes()
            output_hashes_obj = json.loads((run_dir / "OUTPUT_HASHES.json").read_text(encoding="utf-8"))
            status_obj = json.loads((run_dir / "STATUS.json").read_text(encoding="utf-8"))
            bundle_roots.append(_compute_bundle_root(output_hashes_obj, status_obj, task_spec_bytes))

        chain_root = _compute_chain_root(bundle_roots, run_ids)
        return {
            "ok": True,
            "code": ERROR_CODES["OK"],
            "details": {"chain_root": chain_root, "bundle_roots": bundle_roots, "run_ids": run_ids},
            "chain_root": chain_root,
        }


def _compute_sha256_file(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _canonicalize_json(obj: Any) -> bytes:
    canonical_str = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return canonical_str.encode("utf-8")


def _compute_bundle_root(output_hashes_obj: Dict[str, Any], status_obj: Dict[str, Any], task_spec_bytes: bytes) -> str:
    task_spec_hash = hashlib.sha256(task_spec_bytes).hexdigest()
    hashes = output_hashes_obj.get("hashes", {})
    bundle_preimage = {"output_hashes": hashes, "status": status_obj, "task_spec_hash": task_spec_hash}
    return hashlib.sha256(_canonicalize_json(bundle_preimage)).hexdigest()


def _compute_chain_root(bundle_roots: List[str], run_ids: List[str]) -> str:
    chain_preimage = {"bundle_roots": bundle_roots, "run_ids": run_ids}
    return hashlib.sha256(_canonicalize_json(chain_preimage)).hexdigest()


def _verify_ed25519_signature(public_key_hex: str, signature_hex: str, message: bytes) -> bool:
    if not _ED25519_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Ed25519 verification requires 'cryptography' library")

    try:
        pub_bytes = bytes.fromhex(public_key_hex)
    except ValueError as e:
        raise RuntimeError(f"public_key hex decode failed: {e}")

    try:
        sig_bytes = bytes.fromhex(signature_hex)
    except ValueError as e:
        raise RuntimeError(f"signature hex decode failed: {e}")

    try:
        ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes).verify(sig_bytes, message)
        return True
    except InvalidSignature:
        return False
