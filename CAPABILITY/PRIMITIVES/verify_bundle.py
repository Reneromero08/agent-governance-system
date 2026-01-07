#!/usr/bin/env python3
"""
Bundle/Chain Verifier - SPECTRUM-04/05 Enforcement

Enforces SPECTRUM-04 v1.1.0 (canonicalization + identity/signing) and
SPECTRUM-05 v1.0.0 (verification + threat law).

Verification depends ONLY on:
- Bundle artifacts (TASK_SPEC, STATUS, OUTPUT_HASHES, PROOF)
- Identity artifacts (VALIDATOR_IDENTITY, SIGNED_PAYLOAD, SIGNATURE)
- Actual file hashes
- Chain ordering

Verification rejects on:
- Any missing artifacts (ARTIFACT_MISSING)
- Malformed artifacts (ARTIFACT_MALFORMED)
- Identity verification failure (IDENTITY_INVALID, IDENTITY_MISMATCH)
- Signature verification failure (SIGNATURE_INVALID, SIGNATURE_MALFORMED)
- Bundle/chain root mismatch (BUNDLE_ROOT_MISMATCH, CHAIN_ROOT_MISMATCH)
- Hash mismatches (HASH_MISMATCH)
- Forbidden artifacts present (FORBIDDEN_ARTIFACT)
- PROOF.json verified != true (RESTORATION_FAILED)

All errors use SPECTRUM-05 error codes.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

# Ed25519 signature verification
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    ED25519_AVAILABLE = True
except ImportError:
    ED25519_AVAILABLE = False


# SPECTRUM-02 Validator version
VALIDATOR_SEMVER = "1.0.0"
SUPPORTED_VALIDATOR_SEMVERS = {"1.0.0", "1.0.1", "1.1.0"}

# Frozen error codes per SPECTRUM-05 Section 8.2
ERROR_CODES = {
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

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class BundleVerifier:
    """Verifier for SPECTRUM-02 bundles and SPECTRUM-03 chains."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the verifier.

        Args:
            project_root: Root directory of the project. If None, will be derived
                         from this file's location (3 levels up).
        """
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[3]
        else:
            self.project_root = Path(project_root)

    # =========================================================================
    # SPECTRUM-05 FULL VERIFICATION (Identity + Signing)
    # =========================================================================

    def verify_bundle_spectrum05(
        self,
        run_dir: Path,
        strict: bool = True,
        check_proof: bool = True
    ) -> Dict:
        """Verify bundle per SPECTRUM-05 v1.0.0 (10-phase verification).

        This method enforces the complete SPECTRUM-04/05 verification procedure
        including identity verification, signature verification, and canonicalization.

        Args:
            run_dir: Path to the run directory containing the bundle
            strict: If True, enforce hard Ed25519 dependency (default: True)
            check_proof: If False, skip PROOF.json requirement (not recommended)

        Returns:
            {
                "ok": bool,
                "code": str (ERROR_CODES),
                "details": dict,
                "bundle_root": str (optional)
            }
        """
        if strict and not ED25519_AVAILABLE:
            return {
                "ok": False,
                "code": ERROR_CODES["ALGORITHM_UNSUPPORTED"],
                "details": {"message": "Ed25519 verification requires 'cryptography' library"},
                "message": "Ed25519 verification requires 'cryptography' library"
            }

        run_dir = Path(run_dir) if isinstance(run_dir, str) else run_dir

        """
        10-Phase Verification:
        1. Artifact Presence Check
        2. Artifact Parse Check
        3. Identity Verification
        4. Bundle Root Computation
        5. Signed Payload Verification
        6. Signature Verification
        7. Proof Verification
        8. Forbidden Artifact Check
        9. Output Hash Verification
        10. Acceptance Decision
        """
        errors = []
        run_dir = Path(run_dir) if isinstance(run_dir, str) else run_dir

        # =====================================================================
        # PHASE 1: Artifact Presence Check (SPECTRUM-05 Section 4.1)
        # =====================================================================
        required_artifacts = {
            "TASK_SPEC.json": run_dir / "TASK_SPEC.json",
            "STATUS.json": run_dir / "STATUS.json",
            "OUTPUT_HASHES.json": run_dir / "OUTPUT_HASHES.json",
            "VALIDATOR_IDENTITY.json": run_dir / "VALIDATOR_IDENTITY.json",
            "SIGNED_PAYLOAD.json": run_dir / "SIGNED_PAYLOAD.json",
            "SIGNATURE.json": run_dir / "SIGNATURE.json"
        }
        if check_proof:
            required_artifacts["PROOF.json"] = run_dir / "PROOF.json"

        for artifact_name, artifact_path in required_artifacts.items():
            if not artifact_path.exists():
                return {
                    "ok": False,
                    "code": ERROR_CODES["ARTIFACT_MISSING"],
                    "message": f"{artifact_name} missing",
                    "details": {"artifact": artifact_name}
                }

        # =====================================================================
        # PHASE 2: Artifact Parse Check (SPECTRUM-05 Section 4.2)
        # =====================================================================
        parsed = {}
        for artifact_name, artifact_path in required_artifacts.items():
            try:
                with open(artifact_path, 'rb') as f:
                    raw_bytes = f.read()
                    if artifact_name == "TASK_SPEC.json":
                        parsed[artifact_name + "_bytes"] = raw_bytes
                    parsed[artifact_name] = json.loads(raw_bytes.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # Log full error for debugging (server-side only)
                import logging
                logging.debug(f"JSON parse error in {artifact_name}: {e}")

                return {
                    "ok": False,
                    "code": ERROR_CODES["ARTIFACT_MALFORMED"],
                    "message": f"{artifact_name} is not valid JSON",
                    "details": {"artifact": artifact_name}  # No error text - security
                }

        # =====================================================================
        # PHASE 3: Identity Verification (SPECTRUM-05 Section 4.3)
        # =====================================================================
        identity = parsed["VALIDATOR_IDENTITY.json"]
        validator_id = identity.get("validator_id", "")
        public_key = identity.get("public_key", "")

        if strict:
            # Step 3.1: Verify exactly 3 fields
            required_identity_fields = {"algorithm", "public_key", "validator_id"}
            actual_identity_fields = set(identity.keys())
            if actual_identity_fields != required_identity_fields:
                if len(actual_identity_fields) > len(required_identity_fields):
                    return {
                        "ok": False,
                        "code": ERROR_CODES["FIELD_EXTRA"],
                        "message": "VALIDATOR_IDENTITY.json has extra fields",
                        "details": {"extra": list(actual_identity_fields - required_identity_fields)}
                    }
                else:
                    return {
                        "ok": False,
                        "code": ERROR_CODES["FIELD_MISSING"],
                        "message": "VALIDATOR_IDENTITY.json missing required fields",
                        "details": {"missing": list(required_identity_fields - actual_identity_fields)}
                    }

            # Step 3.2: Verify algorithm is exactly "ed25519"
            if identity.get("algorithm") != "ed25519":
                return {
                    "ok": False,
                    "code": ERROR_CODES["ALGORITHM_UNSUPPORTED"],
                    "message": f"Algorithm '{identity.get('algorithm')}' not supported (must be 'ed25519')",
                    "details": {"actual": identity.get("algorithm")}
                }

            # Step 3.3-3.4: Verify public_key is exactly 64 lowercase hex characters
            public_key = identity.get("public_key", "")
            if not (isinstance(public_key, str) and len(public_key) == 64 and public_key == public_key.lower() and all(c in '0123456789abcdef' for c in public_key)):
                return {
                    "ok": False,
                    "code": ERROR_CODES["KEY_INVALID"],
                    "message": "public_key must be exactly 64 lowercase hex characters",
                    "details": {}  # Don't expose actual length - security
                }

            # Step 3.5-3.6: Verify validator_id derivation
            try:
                public_key_bytes = bytes.fromhex(public_key)
                computed_validator_id = self._compute_sha256_bytes(public_key_bytes)
                declared_validator_id = identity.get("validator_id", "")

                if computed_validator_id != declared_validator_id:
                    return {
                        "ok": False,
                        "code": ERROR_CODES["IDENTITY_INVALID"],
                        "message": "validator_id does not match sha256(public_key)",
                        "details": {
                            "declared": declared_validator_id,
                            "computed": computed_validator_id
                        }
                    }
            except ValueError as e:
                return {
                    "ok": False,
                    "code": ERROR_CODES["KEY_INVALID"],
                    "message": f"public_key hex decode failed: {e}",
                    "details": {}
                }

            validator_id = declared_validator_id

        # =====================================================================
        # PHASE 4: Bundle Root Computation (SPECTRUM-05 Section 4.4)
        # =====================================================================
        task_spec_bytes = parsed["TASK_SPEC.json_bytes"]
        output_hashes_obj = parsed["OUTPUT_HASHES.json"]
        status_obj = parsed["STATUS.json"]

        # Compute bundle root
        bundle_root = self._compute_bundle_root(
            output_hashes_obj,
            status_obj,
            task_spec_bytes
        )

        # =====================================================================
        # PHASE 5: Signed Payload Verification (SPECTRUM-05 Section 4.5)
        # =====================================================================
        signed_payload = parsed["SIGNED_PAYLOAD.json"]

        if strict:
            # Step 5.1: Verify exactly 3 fields
            required_payload_fields = {"bundle_root", "decision", "validator_id"}
            actual_payload_fields = set(signed_payload.keys())
            if actual_payload_fields != required_payload_fields:
                if len(actual_payload_fields) > len(required_payload_fields):
                    return {
                        "ok": False,
                        "code": ERROR_CODES["FIELD_EXTRA"],
                        "message": "SIGNED_PAYLOAD.json has extra fields",
                        "details": {"extra": list(actual_payload_fields - required_payload_fields)}
                    }
                else:
                    return {
                        "ok": False,
                        "code": ERROR_CODES["FIELD_MISSING"],
                        "message": "SIGNED_PAYLOAD.json missing required fields",
                        "details": {"missing": list(required_payload_fields - actual_payload_fields)}
                    }

            # Step 5.2: Verify bundle_root matches computed value
            if signed_payload.get("bundle_root") != bundle_root:
                return {
                    "ok": False,
                    "code": ERROR_CODES["BUNDLE_ROOT_MISMATCH"],
                    "message": "SIGNED_PAYLOAD.bundle_root does not match computed bundle root",
                    "details": {
                        "declared": signed_payload.get("bundle_root"),
                        "computed": bundle_root
                    }
                }

            # Step 5.3: Verify decision is exactly "ACCEPT"
            if signed_payload.get("decision") != "ACCEPT":
                return {
                    "ok": False,
                    "code": ERROR_CODES["DECISION_INVALID"],
                    "message": f"Decision is '{signed_payload.get('decision')}', must be 'ACCEPT'",
                    "details": {"actual": signed_payload.get("decision")}
                }

            # Step 5.4: Verify validator_id matches identity
            if signed_payload.get("validator_id") != validator_id:
                return {
                    "ok": False,
                    "code": ERROR_CODES["IDENTITY_MISMATCH"],
                    "message": "SIGNED_PAYLOAD.validator_id does not match VALIDATOR_IDENTITY.validator_id",
                    "details": {
                        "signed_payload": signed_payload.get("validator_id"),
                        "identity": validator_id
                    }
                }

        # =====================================================================
        # PHASE 6: Signature Verification (SPECTRUM-05 Section 4.6)
        # =====================================================================
        signature = parsed["SIGNATURE.json"]

        if strict:
            # Step 6.1: Verify required fields present
            required_sig_fields = {"payload_type", "signature", "validator_id"}
            if not required_sig_fields.issubset(signature.keys()):
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_INCOMPLETE"],
                    "message": "SIGNATURE.json missing required fields",
                    "details": {"missing": list(required_sig_fields - set(signature.keys()))}
                }

            # Step 6.2: Verify payload_type is exactly "BUNDLE"
            if signature.get("payload_type") != "BUNDLE":
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_MALFORMED"],
                    "message": f"payload_type is '{signature.get('payload_type')}', must be 'BUNDLE'",
                    "details": {"actual": signature.get("payload_type")}
                }

            # Step 6.3: Verify signature is exactly 128 lowercase hex characters
            sig_hex = signature.get("signature", "")
            if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and sig_hex == sig_hex.lower() and all(c in '0123456789abcdef' for c in sig_hex)):
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_MALFORMED"],
                    "message": "signature must be exactly 128 lowercase hex characters",
                    "details": {"actual_length": len(sig_hex) if isinstance(sig_hex, str) else 0}
                }

            # Step 6.4: Verify validator_id matches identity
            if signature.get("validator_id") != validator_id:
                return {
                    "ok": False,
                    "code": ERROR_CODES["IDENTITY_MISMATCH"],
                    "message": "SIGNATURE.validator_id does not match VALIDATOR_IDENTITY.validator_id",
                    "details": {
                        "signature": signature.get("validator_id"),
                        "identity": validator_id
                    }
                }

        # Step 6.5-6.6: Construct signature message
        # Domain prefix + payload type + canonical signed payload
        canonical_payload = self._canonicalize_json(signed_payload)
        signature_message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload

        # Step 6.7-6.8: Verify Ed25519 signature
        if strict:
            try:
                sig_valid = self._verify_ed25519_signature(public_key, sig_hex, signature_message)
                if not sig_valid:
                    return {
                        "ok": False,
                        "code": ERROR_CODES["SIGNATURE_INVALID"],
                        "message": "Ed25519 signature verification failed",
                        "details": {}
                    }
            except RuntimeError as e:
                return {
                    "ok": False,
                    "code": ERROR_CODES["SIGNATURE_INVALID"],
                    "message": str(e),
                    "details": {}
                }

        # =====================================================================
        # PHASE 7: Proof Verification (SPECTRUM-05 Section 4.7)
        # =====================================================================
        if check_proof:
            proof = parsed["PROOF.json"]

            # If a canonical proof_hash is present (64 lowercase hex), verify it deterministically.
            # This is a strict integrity check for proofs generated by `PRIMITIVES/restore_proof.py`.
            proof_hash = proof.get("proof_hash")
            if isinstance(proof_hash, str) and _HEX64_RE.fullmatch(proof_hash) is not None:
                proof_without_hash = dict(proof)
                proof_without_hash.pop("proof_hash", None)
                computed = hashlib.sha256(
                    json.dumps(proof_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                # Use constant-time comparison to prevent timing attacks
                import hmac
                if not hmac.compare_digest(computed, proof_hash):
                    return {
                        "ok": False,
                        "code": ERROR_CODES["ARTIFACT_MALFORMED"],
                        "message": "PROOF.json proof_hash mismatch",
                        "details": {"expected": proof_hash, "computed": computed},
                    }

            # Step 7.1: Verify restoration_result object exists
            if "restoration_result" not in proof:
                return {
                    "ok": False,
                    "code": ERROR_CODES["FIELD_MISSING"],
                    "message": "PROOF.json missing restoration_result",
                    "details": {}
                }

            # Step 7.2: Verify verified == true
            verified = proof["restoration_result"].get("verified")
            if verified is not True:
                condition = proof["restoration_result"].get("condition", "UNKNOWN")
                return {
                    "ok": False,
                    "code": ERROR_CODES["RESTORATION_FAILED"],
                    "message": f"PROOF restoration failed: {condition}",
                    "details": {
                        "verified": verified,
                        "condition": condition
                    }
                }

        # =====================================================================
        # PHASE 8: Forbidden Artifact Check (SPECTRUM-05 Section 4.8)
        # =====================================================================
        forbidden_checks = [
            ("logs/", run_dir / "logs"),
            ("tmp/", run_dir / "tmp"),
            ("transcript.json", run_dir / "transcript.json")
        ]

        for artifact_name, artifact_path in forbidden_checks:
            if artifact_path.exists():
                return {
                    "ok": False,
                    "code": ERROR_CODES["FORBIDDEN_ARTIFACT"],
                    "message": f"Forbidden artifact present: {artifact_name}",
                    "details": {"artifact": artifact_name}
                }

        # =====================================================================
        # PHASE 9: Output Hash Verification (SPECTRUM-05 Section 4.9)
        # =====================================================================
        hashes = output_hashes_obj.get("hashes", {})
        for rel_path, expected_hash in hashes.items():
            # Normalize path to POSIX
            rel_path_posix = rel_path.replace("\\", "/")
            abs_path = self.project_root / rel_path_posix

            # Step 9.1.2: Verify file exists
            if not abs_path.exists():
                return {
                    "ok": False,
                    "code": ERROR_CODES["OUTPUT_MISSING"],
                    "message": f"Output file does not exist: {rel_path_posix}",
                    "details": {"path": rel_path_posix}
                }

            # Step 9.1.3-9.1.4: Compute and verify hash
            actual_hash = f"sha256:{self._compute_sha256(abs_path)}"
            # Use constant-time comparison to prevent timing attacks
            import hmac
            if not hmac.compare_digest(actual_hash, expected_hash):
                return {
                    "ok": False,
                    "code": ERROR_CODES["HASH_MISMATCH"],
                    "message": f"Hash mismatch for {rel_path_posix}",
                    "details": {
                        "path": rel_path_posix,
                        "expected": expected_hash,
                        "actual": actual_hash
                    }
                }

        # =====================================================================
        # PHASE 10: Acceptance (SPECTRUM-05 Section 4.10)
        # =====================================================================
        # Acceptance requires STATUS.status == success and STATUS.cmp01 == pass
        # (even if strict=False, this is a semantic bundle requirement)
        if status_obj.get("status") != "success":
            return {
                "ok": False,
                "code": ERROR_CODES["DECISION_INVALID"],
                "message": f"Run status is '{status_obj.get('status')}', must be 'success'",
                "details": {"actual": status_obj.get("status")}
            }

        if status_obj.get("cmp01") != "pass":
            return {
                "ok": False,
                "code": ERROR_CODES["DECISION_INVALID"],
                "message": f"CMP-01 status is '{status_obj.get('cmp01')}', must be 'pass'",
                "details": {"actual": status_obj.get("cmp01")}
            }

        return {
            "ok": True,
            "code": ERROR_CODES["OK"],
            "details": {},
            "bundle_root": bundle_root
        }

    # =========================================================================
    # SINGLE BUNDLE VERIFICATION (SPECTRUM-02 Legacy)
    # =========================================================================

    def verify_bundle(
        self,
        run_dir: Path,
        strict: bool = False,
        check_proof: bool = True
    ) -> Dict:
        """[DEPRECATED] Verify a single bundle. 
        Calls verify_bundle_spectrum05 internal logic.
        """
        result = self.verify_bundle_spectrum05(run_dir, strict=strict, check_proof=check_proof)
        
        # Adapt to legacy return shape
        errors = []
        if not result["ok"]:
            errors.append({
                "code": result["code"],
                "message": result.get("message", "Verification failed"),
                "details": result.get("details", {})
            })
            
        return {
            "valid": result["ok"],
            "errors": errors
        }
    # =========================================================================
    # SPECTRUM-05 CHAIN VERIFICATION (with chain_root computation)
    # =========================================================================

    def verify_chain_spectrum05(
        self,
        run_dirs: List[Path],
        strict: bool = True,
        check_proof: bool = True
    ) -> Dict:
        """Verify chain per SPECTRUM-05 v1.0.0 Section 6.

        Checks:
        1. Chain is non-empty
        2. No duplicate run_ids
        3. Each bundle verifies individually via verify_bundle_spectrum05
        4. Compute chain_root from bundle_roots and run_ids

        Args:
            run_dirs: Ordered list of run directories (chain order is passed order)
            strict: If True, enforce hard Ed25519 dependency (default: True)
            check_proof: If False, skip PROOF.json requirement

        Returns:
            {
                "ok": bool,
                "code": str,
                "details": dict,
                "chain_root": str (optional)
            }

        Error codes:
            CHAIN_EMPTY - Chain has zero runs
            CHAIN_DUPLICATE_RUN - Duplicate run_id in chain
            Plus any code from verify_bundle_spectrum05
        """
        if strict and not ED25519_AVAILABLE:
            return {
                "ok": False,
                "code": ERROR_CODES["ALGORITHM_UNSUPPORTED"],
                "message": "Ed25519 verification requires 'cryptography' library",
                "details": {}
            }

        run_dirs = [Path(d) if isinstance(d, str) else d for d in run_dirs]

        # Step C.1: Verify chain is non-empty
        if not run_dirs:
            return {
                "ok": False,
                "code": ERROR_CODES["CHAIN_EMPTY"],
                "message": "Chain verification requires at least one run",
                "details": {}
            }

        # Step C.2-C.3: Extract run_ids and check for duplicates
        run_ids = [d.name for d in run_dirs]
        if len(run_ids) != len(set(run_ids)):
            duplicates = [rid for rid in run_ids if run_ids.count(rid) > 1]
            return {
                "ok": False,
                "code": ERROR_CODES["CHAIN_DUPLICATE_RUN"],
                "message": f"Duplicate run_id(s) in chain: {', '.join(set(duplicates))}",
                "details": {"duplicates": list(set(duplicates))}
            }

        # Step C.4-C.5: Verify each bundle and collect bundle_roots
        bundle_roots = []
        for run_dir in run_dirs:
            result = self.verify_bundle_spectrum05(run_dir, strict=strict, check_proof=check_proof)
            if not result["ok"]:
                # Add run_id context to the error
                result["run_id"] = run_dir.name
                return result

            # Compute bundle_root for this bundle
            task_spec_path = run_dir / "TASK_SPEC.json"
            output_hashes_path = run_dir / "OUTPUT_HASHES.json"
            status_path = run_dir / "STATUS.json"

            with open(task_spec_path, 'rb') as f:
                task_spec_bytes = f.read()
            with open(output_hashes_path) as f:
                output_hashes_obj = json.load(f)
            with open(status_path) as f:
                status_obj = json.load(f)

            bundle_root = self._compute_bundle_root(
                output_hashes_obj,
                status_obj,
                task_spec_bytes
            )
            bundle_roots.append(bundle_root)

        # Step C.6-C.7: Compute chain_root
        chain_root = self._compute_chain_root(bundle_roots, run_ids)

        return {
            "ok": True,
            "code": ERROR_CODES["OK"],
            "details": {
                "chain_root": chain_root,
                "bundle_roots": bundle_roots,
                "run_ids": run_ids
            },
            "chain_root": chain_root
        }

    # =========================================================================
    # CHAIN VERIFICATION (SPECTRUM-03 Legacy)
    # =========================================================================

    def verify_chain(
        self,
        run_dirs: List[Path],
        strict: bool = False,
        check_proof: bool = True
    ) -> Dict:
        """[DEPRECATED] Verify a chain of bundles (Legacy SPECTRUM-03 wrapper)."""
        result = self.verify_chain_spectrum05(run_dirs, strict=strict, check_proof=check_proof)
        
        # Adapt to legacy return shape
        errors = []
        if not result["ok"]:
            # Combine all details into the error object
            details = result.get("details", {}).copy()
            if "run_id" in result:
                details["run_id"] = result["run_id"]

            errors.append({
                "code": result["code"],
                "message": result.get("message", "Chain verification failed"),
                "details": details,
                "run_id": result.get("run_id")
            })
            
        return {
            "valid": result["ok"],
            "errors": errors
        }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA-256 hash (lowercase)
        """
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def _compute_sha256_bytes(self, data: bytes) -> str:
        """Compute SHA-256 hash of bytes.

        Args:
            data: Bytes to hash

        Returns:
            Lowercase hex digest of SHA-256 hash
        """
        return hashlib.sha256(data).hexdigest()

    def _canonicalize_json(self, obj: Any) -> bytes:
        """Canonicalize JSON object per SPECTRUM-04 v1.1.0.

        Rules:
        - UTF-8 encoding (no BOM)
        - Single-line JSON (no newlines)
        - No whitespace outside string values
        - Keys sorted lexicographically by UTF-8 byte value
        - No trailing newline

        Args:
            obj: JSON-serializable object

        Returns:
            Canonical UTF-8 encoded bytes
        """
        # json.dumps with sort_keys=True, no spaces, ensure_ascii=False for UTF-8
        canonical_str = json.dumps(
            obj,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False
        )
        return canonical_str.encode('utf-8')

    def _compute_bundle_root(
        self,
        output_hashes_obj: Dict,
        status_obj: Dict,
        task_spec_bytes: bytes
    ) -> str:
        """Compute bundle root per SPECTRUM-04 v1.1.0 Section 5.

        Preimage structure:
        {"output_hashes":<canonicalized_hashes>,"status":<canonicalized_status>,"task_spec_hash":"<hex>"}

        Args:
            output_hashes_obj: Parsed OUTPUT_HASHES.json
            status_obj: Parsed STATUS.json
            task_spec_bytes: Raw bytes of TASK_SPEC.json

        Returns:
            Lowercase hex SHA-256 hash (64 characters)
        """
        # Compute task_spec_hash from raw bytes
        task_spec_hash = self._compute_sha256_bytes(task_spec_bytes)

        # Extract hashes field from OUTPUT_HASHES.json
        hashes = output_hashes_obj.get("hashes", {})

        # Construct bundle preimage with lexicographic field order
        bundle_preimage = {
            "output_hashes": hashes,
            "status": status_obj,
            "task_spec_hash": task_spec_hash
        }

        # Canonicalize and hash
        canonical_bytes = self._canonicalize_json(bundle_preimage)
        return self._compute_sha256_bytes(canonical_bytes)

    def _compute_chain_root(
        self,
        bundle_roots: List[str],
        run_ids: List[str]
    ) -> str:
        """Compute chain root per SPECTRUM-04 v1.1.0 Section 6.

        Preimage structure:
        {"bundle_roots":[...],"run_ids":[...]}

        Args:
            bundle_roots: Ordered list of bundle root hashes
            run_ids: Ordered list of run_ids (directory names)

        Returns:
            Lowercase hex SHA-256 hash (64 characters)
        """
        # Construct chain preimage with lexicographic field order
        chain_preimage = {
            "bundle_roots": bundle_roots,
            "run_ids": run_ids
        }

        # Canonicalize and hash
        canonical_bytes = self._canonicalize_json(chain_preimage)
        return self._compute_sha256_bytes(canonical_bytes)

    def _verify_ed25519_signature(
        self,
        public_key_hex: str,
        signature_hex: str,
        message: bytes
    ) -> bool:
        """Verify Ed25519 signature.

        Args:
            public_key_hex: 64-char lowercase hex public key
            signature_hex: 128-char lowercase hex signature
            message: Message bytes that were signed

        Returns:
            True if signature is valid, False otherwise
        """
        if not ED25519_AVAILABLE:
            raise RuntimeError("Ed25519 verification requires cryptography library")

        try:
            # Decode hex to bytes
            public_key_bytes = bytes.fromhex(public_key_hex)
            signature_bytes = bytes.fromhex(signature_hex)

            # Construct Ed25519 public key
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)

            # Verify signature
            public_key.verify(signature_bytes, message)
            return True
        except (ValueError, InvalidSignature):
            return False

def verify_bundle(run_dir: Path, strict: bool = False, check_proof: bool = True) -> dict:
    """Standalone convenience wrapper for BundleVerifier.verify_bundle_spectrum05."""
    verifier = BundleVerifier()
    return verifier.verify_bundle_spectrum05(run_dir, strict=strict, check_proof=check_proof)
