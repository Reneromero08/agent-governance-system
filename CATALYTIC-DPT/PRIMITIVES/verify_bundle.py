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
        strict: bool = False
    ) -> Dict:
        """Verify bundle per SPECTRUM-05 v1.0.0 (10-phase verification).

        This method enforces the complete SPECTRUM-04/05 verification procedure
        including identity verification, signature verification, and canonicalization.

        Args:
            run_dir: Path to the run directory containing the bundle
            strict: Reserved for future use

        Returns:
            {"valid": bool, "errors": [...]}

        Error codes per SPECTRUM-05 Section 8.2:
            ARTIFACT_MISSING, ARTIFACT_MALFORMED, ARTIFACT_EXTRA
            FIELD_MISSING, FIELD_EXTRA
            ALGORITHM_UNSUPPORTED, KEY_INVALID
            IDENTITY_INVALID, IDENTITY_MISMATCH, IDENTITY_MULTIPLE
            SIGNATURE_MALFORMED, SIGNATURE_INCOMPLETE, SIGNATURE_INVALID, SIGNATURE_MULTIPLE
            BUNDLE_ROOT_MISMATCH, DECISION_INVALID, PAYLOAD_MISMATCH
            SERIALIZATION_INVALID, RESTORATION_FAILED
            FORBIDDEN_ARTIFACT, OUTPUT_MISSING, HASH_MISMATCH
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
            "PROOF.json": run_dir / "PROOF.json",
            "VALIDATOR_IDENTITY.json": run_dir / "VALIDATOR_IDENTITY.json",
            "SIGNED_PAYLOAD.json": run_dir / "SIGNED_PAYLOAD.json",
            "SIGNATURE.json": run_dir / "SIGNATURE.json"
        }

        for artifact_name, artifact_path in required_artifacts.items():
            if not artifact_path.exists():
                errors.append({
                    "code": "ARTIFACT_MISSING",
                    "severity": "error",
                    "message": f"{artifact_name} missing",
                    "path": "/",
                    "details": {"artifact": artifact_name}
                })
                return {"valid": False, "errors": errors}

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
                errors.append({
                    "code": "ARTIFACT_MALFORMED",
                    "severity": "error",
                    "message": f"{artifact_name} is not valid JSON",
                    "path": "/",
                    "details": {"artifact": artifact_name, "error": str(e)}
                })
                return {"valid": False, "errors": errors}

        # =====================================================================
        # PHASE 3: Identity Verification (SPECTRUM-05 Section 4.3)
        # =====================================================================
        identity = parsed["VALIDATOR_IDENTITY.json"]

        # Step 3.1: Verify exactly 3 fields
        required_identity_fields = {"algorithm", "public_key", "validator_id"}
        actual_identity_fields = set(identity.keys())
        if actual_identity_fields != required_identity_fields:
            if len(actual_identity_fields) > len(required_identity_fields):
                errors.append({
                    "code": "FIELD_EXTRA",
                    "severity": "error",
                    "message": "VALIDATOR_IDENTITY.json has extra fields",
                    "path": "/VALIDATOR_IDENTITY",
                    "details": {"extra": list(actual_identity_fields - required_identity_fields)}
                })
            else:
                errors.append({
                    "code": "FIELD_MISSING",
                    "severity": "error",
                    "message": "VALIDATOR_IDENTITY.json missing required fields",
                    "path": "/VALIDATOR_IDENTITY",
                    "details": {"missing": list(required_identity_fields - actual_identity_fields)}
                })
            return {"valid": False, "errors": errors}

        # Step 3.2: Verify algorithm is exactly "ed25519"
        if identity.get("algorithm") != "ed25519":
            errors.append({
                "code": "ALGORITHM_UNSUPPORTED",
                "severity": "error",
                "message": f"Algorithm '{identity.get('algorithm')}' not supported (must be 'ed25519')",
                "path": "/VALIDATOR_IDENTITY/algorithm",
                "details": {"actual": identity.get("algorithm")}
            })
            return {"valid": False, "errors": errors}

        # Step 3.3-3.4: Verify public_key is exactly 64 lowercase hex characters
        public_key = identity.get("public_key", "")
        if not (isinstance(public_key, str) and len(public_key) == 64 and public_key.islower() and all(c in '0123456789abcdef' for c in public_key)):
            errors.append({
                "code": "KEY_INVALID",
                "severity": "error",
                "message": "public_key must be exactly 64 lowercase hex characters",
                "path": "/VALIDATOR_IDENTITY/public_key",
                "details": {"actual_length": len(public_key) if isinstance(public_key, str) else 0}
            })
            return {"valid": False, "errors": errors}

        # Step 3.5-3.6: Verify validator_id derivation
        try:
            public_key_bytes = bytes.fromhex(public_key)
            computed_validator_id = self._compute_sha256_bytes(public_key_bytes)
            declared_validator_id = identity.get("validator_id", "")

            if computed_validator_id != declared_validator_id:
                errors.append({
                    "code": "IDENTITY_INVALID",
                    "severity": "error",
                    "message": "validator_id does not match sha256(public_key)",
                    "path": "/VALIDATOR_IDENTITY/validator_id",
                    "details": {
                        "declared": declared_validator_id,
                        "computed": computed_validator_id
                    }
                })
                return {"valid": False, "errors": errors}
        except ValueError as e:
            errors.append({
                "code": "KEY_INVALID",
                "severity": "error",
                "message": f"public_key hex decode failed: {e}",
                "path": "/VALIDATOR_IDENTITY/public_key",
                "details": {}
            })
            return {"valid": False, "errors": errors}

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

        # Step 5.1: Verify exactly 3 fields
        required_payload_fields = {"bundle_root", "decision", "validator_id"}
        actual_payload_fields = set(signed_payload.keys())
        if actual_payload_fields != required_payload_fields:
            if len(actual_payload_fields) > len(required_payload_fields):
                errors.append({
                    "code": "FIELD_EXTRA",
                    "severity": "error",
                    "message": "SIGNED_PAYLOAD.json has extra fields",
                    "path": "/SIGNED_PAYLOAD",
                    "details": {"extra": list(actual_payload_fields - required_payload_fields)}
                })
            else:
                errors.append({
                    "code": "FIELD_MISSING",
                    "severity": "error",
                    "message": "SIGNED_PAYLOAD.json missing required fields",
                    "path": "/SIGNED_PAYLOAD",
                    "details": {"missing": list(required_payload_fields - actual_payload_fields)}
                })
            return {"valid": False, "errors": errors}

        # Step 5.2: Verify bundle_root matches computed value
        if signed_payload.get("bundle_root") != bundle_root:
            errors.append({
                "code": "BUNDLE_ROOT_MISMATCH",
                "severity": "error",
                "message": "SIGNED_PAYLOAD.bundle_root does not match computed bundle root",
                "path": "/SIGNED_PAYLOAD/bundle_root",
                "details": {
                    "declared": signed_payload.get("bundle_root"),
                    "computed": bundle_root
                }
            })
            return {"valid": False, "errors": errors}

        # Step 5.3: Verify decision is exactly "ACCEPT"
        if signed_payload.get("decision") != "ACCEPT":
            errors.append({
                "code": "DECISION_INVALID",
                "severity": "error",
                "message": f"Decision is '{signed_payload.get('decision')}', must be 'ACCEPT'",
                "path": "/SIGNED_PAYLOAD/decision",
                "details": {"actual": signed_payload.get("decision")}
            })
            return {"valid": False, "errors": errors}

        # Step 5.4: Verify validator_id matches identity
        if signed_payload.get("validator_id") != validator_id:
            errors.append({
                "code": "IDENTITY_MISMATCH",
                "severity": "error",
                "message": "SIGNED_PAYLOAD.validator_id does not match VALIDATOR_IDENTITY.validator_id",
                "path": "/SIGNED_PAYLOAD/validator_id",
                "details": {
                    "signed_payload": signed_payload.get("validator_id"),
                    "identity": validator_id
                }
            })
            return {"valid": False, "errors": errors}

        # =====================================================================
        # PHASE 6: Signature Verification (SPECTRUM-05 Section 4.6)
        # =====================================================================
        signature = parsed["SIGNATURE.json"]

        # Step 6.1: Verify required fields present
        required_sig_fields = {"payload_type", "signature", "validator_id"}
        if not required_sig_fields.issubset(signature.keys()):
            errors.append({
                "code": "SIGNATURE_INCOMPLETE",
                "severity": "error",
                "message": "SIGNATURE.json missing required fields",
                "path": "/SIGNATURE",
                "details": {"missing": list(required_sig_fields - set(signature.keys()))}
            })
            return {"valid": False, "errors": errors}

        # Step 6.2: Verify payload_type is exactly "BUNDLE"
        if signature.get("payload_type") != "BUNDLE":
            errors.append({
                "code": "SIGNATURE_MALFORMED",
                "severity": "error",
                "message": f"payload_type is '{signature.get('payload_type')}', must be 'BUNDLE'",
                "path": "/SIGNATURE/payload_type",
                "details": {"actual": signature.get("payload_type")}
            })
            return {"valid": False, "errors": errors}

        # Step 6.3: Verify signature is exactly 128 lowercase hex characters
        sig_hex = signature.get("signature", "")
        if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and sig_hex.islower() and all(c in '0123456789abcdef' for c in sig_hex)):
            errors.append({
                "code": "SIGNATURE_MALFORMED",
                "severity": "error",
                "message": "signature must be exactly 128 lowercase hex characters",
                "path": "/SIGNATURE/signature",
                "details": {"actual_length": len(sig_hex) if isinstance(sig_hex, str) else 0}
            })
            return {"valid": False, "errors": errors}

        # Step 6.4: Verify validator_id matches identity
        if signature.get("validator_id") != validator_id:
            errors.append({
                "code": "IDENTITY_MISMATCH",
                "severity": "error",
                "message": "SIGNATURE.validator_id does not match VALIDATOR_IDENTITY.validator_id",
                "path": "/SIGNATURE/validator_id",
                "details": {
                    "signature": signature.get("validator_id"),
                    "identity": validator_id
                }
            })
            return {"valid": False, "errors": errors}

        # Step 6.5-6.6: Construct signature message
        # Domain prefix + payload type + canonical signed payload
        canonical_payload = self._canonicalize_json(signed_payload)
        signature_message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:" + canonical_payload

        # Step 6.7-6.8: Verify Ed25519 signature
        try:
            sig_valid = self._verify_ed25519_signature(public_key, sig_hex, signature_message)
            if not sig_valid:
                errors.append({
                    "code": "SIGNATURE_INVALID",
                    "severity": "error",
                    "message": "Ed25519 signature verification failed",
                    "path": "/SIGNATURE/signature",
                    "details": {}
                })
                return {"valid": False, "errors": errors}
        except RuntimeError as e:
            errors.append({
                "code": "SIGNATURE_INVALID",
                "severity": "error",
                "message": str(e),
                "path": "/SIGNATURE/signature",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        # =====================================================================
        # PHASE 7: Proof Verification (SPECTRUM-05 Section 4.7)
        # =====================================================================
        proof = parsed["PROOF.json"]

        # Step 7.1: Verify restoration_result object exists
        if "restoration_result" not in proof:
            errors.append({
                "code": "FIELD_MISSING",
                "severity": "error",
                "message": "PROOF.json missing restoration_result",
                "path": "/PROOF/restoration_result",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        # Step 7.2: Verify verified == true
        verified = proof["restoration_result"].get("verified")
        if verified is not True:
            condition = proof["restoration_result"].get("condition", "UNKNOWN")
            errors.append({
                "code": "RESTORATION_FAILED",
                "severity": "error",
                "message": f"PROOF restoration failed: {condition}",
                "path": "/PROOF/restoration_result/verified",
                "details": {
                    "verified": verified,
                    "condition": condition
                }
            })
            return {"valid": False, "errors": errors}

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
                errors.append({
                    "code": "FORBIDDEN_ARTIFACT",
                    "severity": "error",
                    "message": f"Forbidden artifact present: {artifact_name}",
                    "path": "/",
                    "details": {"artifact": artifact_name}
                })
                return {"valid": False, "errors": errors}

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
                errors.append({
                    "code": "OUTPUT_MISSING",
                    "severity": "error",
                    "message": f"Output file does not exist: {rel_path_posix}",
                    "path": f"/OUTPUT_HASHES/hashes/{rel_path_posix}",
                    "details": {"path": rel_path_posix}
                })
                continue

            # Step 9.1.3-9.1.4: Compute and verify hash
            actual_hash = f"sha256:{self._compute_sha256(abs_path)}"
            if actual_hash != expected_hash:
                errors.append({
                    "code": "HASH_MISMATCH",
                    "severity": "error",
                    "message": f"Hash mismatch for {rel_path_posix}",
                    "path": f"/OUTPUT_HASHES/hashes/{rel_path_posix}",
                    "details": {
                        "path": rel_path_posix,
                        "expected": expected_hash,
                        "actual": actual_hash
                    }
                })

        # =====================================================================
        # PHASE 10: Acceptance (SPECTRUM-05 Section 4.10)
        # =====================================================================
        if errors:
            return {"valid": False, "errors": errors}

        return {"valid": True, "errors": []}

    # =========================================================================
    # SINGLE BUNDLE VERIFICATION (SPECTRUM-02 Legacy)
    # =========================================================================

    def verify_bundle(
        self,
        run_dir: Path,
        strict: bool = False,
        check_proof: bool = True
    ) -> Dict:
        """Verify a single SPECTRUM-02 bundle.

        Checks:
        1. TASK_SPEC.json exists
        2. STATUS.json exists with status=success and cmp01=pass
        3. OUTPUT_HASHES.json exists with supported validator_semver
        4. validator_build_id exists and is non-empty
        5. All declared output hashes verify against actual files
        6. PROOF.json exists with verified=true (if check_proof=True)
        7. No forbidden artifacts (logs/, tmp/, transcript.json)

        Args:
            run_dir: Path to the run directory containing the bundle
            strict: If True, perform additional strict checks
            check_proof: If True, require PROOF.json with verified=true

        Returns:
            {"valid": bool, "errors": [...]}

        Error codes:
            BUNDLE_INCOMPLETE - Required artifact missing
            STATUS_NOT_SUCCESS - STATUS.status != "success"
            CMP01_NOT_PASS - STATUS.cmp01 != "pass"
            VALIDATOR_UNSUPPORTED - validator_semver not supported
            VALIDATOR_BUILD_ID_MISSING - validator_build_id missing
            OUTPUT_MISSING - Declared output file does not exist
            HASH_MISMATCH - Computed hash != declared hash
            PROOF_REQUIRED - PROOF.json missing (when check_proof=True)
            RESTORATION_FAILED - PROOF.json.verified != true
            FORBIDDEN_ARTIFACT - logs/, tmp/, or transcript.json exists
        """
        errors = []

        # Ensure run_dir is a Path
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # 1. Check TASK_SPEC.json exists
        task_spec_path = run_dir / "TASK_SPEC.json"
        if not task_spec_path.exists():
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "severity": "error",
                "message": "TASK_SPEC.json missing",
                "path": "/",
                "details": {"expected": str(task_spec_path)}
            })
            return {"valid": False, "errors": errors}

        # 2. Check STATUS.json exists and is valid
        status_path = run_dir / "STATUS.json"
        if not status_path.exists():
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "severity": "error",
                "message": "STATUS.json missing",
                "path": "/",
                "details": {"expected": str(status_path)}
            })
            return {"valid": False, "errors": errors}

        try:
            with open(status_path) as f:
                status = json.load(f)
        except json.JSONDecodeError as e:
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "severity": "error",
                "message": f"STATUS.json invalid JSON: {e}",
                "path": "/",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        if status.get("status") != "success":
            errors.append({
                "code": "STATUS_NOT_SUCCESS",
                "severity": "error",
                "message": f"STATUS.status is '{status.get('status')}', expected 'success'",
                "path": "/status",
                "details": {"actual": status.get("status")}
            })
            return {"valid": False, "errors": errors}

        if status.get("cmp01") != "pass":
            errors.append({
                "code": "CMP01_NOT_PASS",
                "severity": "error",
                "message": f"STATUS.cmp01 is '{status.get('cmp01')}', expected 'pass'",
                "path": "/cmp01",
                "details": {"actual": status.get("cmp01")}
            })
            return {"valid": False, "errors": errors}

        # 3. Check OUTPUT_HASHES.json exists and is valid
        hashes_path = run_dir / "OUTPUT_HASHES.json"
        if not hashes_path.exists():
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "severity": "error",
                "message": "OUTPUT_HASHES.json missing",
                "path": "/",
                "details": {"expected": str(hashes_path)}
            })
            return {"valid": False, "errors": errors}

        try:
            with open(hashes_path) as f:
                output_hashes = json.load(f)
        except json.JSONDecodeError as e:
            errors.append({
                "code": "BUNDLE_INCOMPLETE",
                "severity": "error",
                "message": f"OUTPUT_HASHES.json invalid JSON: {e}",
                "path": "/",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        # 4. Check validator semver is supported
        validator_semver = output_hashes.get("validator_semver")
        if validator_semver not in SUPPORTED_VALIDATOR_SEMVERS:
            errors.append({
                "code": "VALIDATOR_UNSUPPORTED",
                "severity": "error",
                "message": f"validator_semver '{validator_semver}' not supported",
                "path": "/validator_semver",
                "details": {
                    "actual": validator_semver,
                    "supported": list(SUPPORTED_VALIDATOR_SEMVERS)
                }
            })
            return {"valid": False, "errors": errors}

        # 5. Check validator_build_id exists and is non-empty
        validator_build_id = output_hashes.get("validator_build_id")
        if not validator_build_id:
            errors.append({
                "code": "VALIDATOR_BUILD_ID_MISSING",
                "severity": "error",
                "message": "validator_build_id is missing or empty",
                "path": "/validator_build_id",
                "details": {"actual": validator_build_id}
            })
            return {"valid": False, "errors": errors}

        # 6. Verify each hash
        hashes = output_hashes.get("hashes", {})
        for rel_path, expected_hash in hashes.items():
            # Normalize path to POSIX style for consistent comparison
            rel_path_posix = rel_path.replace("\\", "/")
            abs_path = self.project_root / rel_path_posix

            if not abs_path.exists():
                errors.append({
                    "code": "OUTPUT_MISSING",
                    "severity": "error",
                    "message": f"Output file does not exist: {rel_path_posix}",
                    "path": f"/hashes/{rel_path_posix}",
                    "details": {"declared": rel_path_posix, "resolved": str(abs_path)}
                })
                continue

            # Compute actual hash
            actual_hash = f"sha256:{self._compute_sha256(abs_path)}"

            if actual_hash != expected_hash:
                errors.append({
                    "code": "HASH_MISMATCH",
                    "severity": "error",
                    "message": f"Hash mismatch for {rel_path_posix}",
                    "path": f"/hashes/{rel_path_posix}",
                    "details": {
                        "declared": rel_path_posix,
                        "expected": expected_hash,
                        "actual": actual_hash
                    }
                })

        # 7. Check PROOF.json if required
        if check_proof:
            proof_path = run_dir / "PROOF.json"
            if not proof_path.exists():
                errors.append({
                    "code": "PROOF_REQUIRED",
                    "severity": "error",
                    "message": "PROOF.json missing (required for acceptance)",
                    "path": "/",
                    "details": {"expected": str(proof_path)}
                })
            else:
                try:
                    with open(proof_path) as f:
                        proof = json.load(f)

                    verified = proof.get("restoration_result", {}).get("verified")
                    if verified is not True:
                        condition = proof.get("restoration_result", {}).get("condition", "UNKNOWN")
                        errors.append({
                            "code": "RESTORATION_FAILED",
                            "severity": "error",
                            "message": f"PROOF.json restoration failed: {condition}",
                            "path": "/restoration_result/verified",
                            "details": {
                                "verified": verified,
                                "condition": condition
                            }
                        })
                except json.JSONDecodeError as e:
                    errors.append({
                        "code": "PROOF_REQUIRED",
                        "severity": "error",
                        "message": f"PROOF.json invalid JSON: {e}",
                        "path": "/",
                        "details": {}
                    })

        # 8. Check for forbidden artifacts
        forbidden_artifacts = []
        if (run_dir / "logs").exists():
            forbidden_artifacts.append("logs/")
        if (run_dir / "tmp").exists():
            forbidden_artifacts.append("tmp/")
        if (run_dir / "transcript.json").exists():
            forbidden_artifacts.append("transcript.json")

        if forbidden_artifacts:
            errors.append({
                "code": "FORBIDDEN_ARTIFACT",
                "severity": "error",
                "message": f"Forbidden artifacts present: {', '.join(forbidden_artifacts)}",
                "path": "/",
                "details": {"artifacts": forbidden_artifacts}
            })

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # =========================================================================
    # SPECTRUM-05 CHAIN VERIFICATION (with chain_root computation)
    # =========================================================================

    def verify_chain_spectrum05(
        self,
        run_dirs: List[Path],
        strict: bool = False
    ) -> Dict:
        """Verify chain per SPECTRUM-05 v1.0.0 Section 6.

        Checks:
        1. Chain is non-empty
        2. No duplicate run_ids
        3. Each bundle verifies individually via verify_bundle_spectrum05
        4. Compute chain_root from bundle_roots and run_ids

        Args:
            run_dirs: Ordered list of run directories (chain order is passed order)
            strict: Reserved for future use

        Returns:
            {"valid": bool, "errors": [], "chain_root": "<hex>"}

        Error codes (in addition to verify_bundle_spectrum05 codes):
            CHAIN_EMPTY - Chain has zero runs
            CHAIN_DUPLICATE_RUN - Duplicate run_id in chain
        """
        errors = []
        run_dirs = [Path(d) if isinstance(d, str) else d for d in run_dirs]

        # Step C.1: Verify chain is non-empty
        if not run_dirs:
            errors.append({
                "code": "CHAIN_EMPTY",
                "severity": "error",
                "message": "Chain verification requires at least one run",
                "path": "/",
                "details": {}
            })
            return {"valid": False, "errors": errors}

        # Step C.2-C.3: Extract run_ids and check for duplicates
        run_ids = [d.name for d in run_dirs]
        if len(run_ids) != len(set(run_ids)):
            duplicates = [rid for rid in run_ids if run_ids.count(rid) > 1]
            errors.append({
                "code": "CHAIN_DUPLICATE_RUN",
                "severity": "error",
                "message": f"Duplicate run_id(s) in chain: {', '.join(set(duplicates))}",
                "path": "/",
                "details": {"duplicates": list(set(duplicates))}
            })
            return {"valid": False, "errors": errors}

        # Step C.4-C.5: Verify each bundle and collect bundle_roots
        bundle_roots = []
        for run_dir in run_dirs:
            result = self.verify_bundle_spectrum05(run_dir, strict=strict)
            if not result["valid"]:
                # Add run_id context to each error
                for err in result["errors"]:
                    err["run_id"] = run_dir.name
                errors.extend(result["errors"])
                return {"valid": False, "errors": errors}

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
            "valid": True,
            "errors": [],
            "chain_root": chain_root,
            "bundle_roots": bundle_roots,
            "run_ids": run_ids
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
        """Verify a chain of SPECTRUM-02 bundles.

        Checks:
        1. Each bundle verifies individually via verify_bundle
        2. Build registry of available outputs from each run
        3. Validate TASK_SPEC references only earlier outputs or current outputs
        4. No forbidden artifacts in any bundle

        Args:
            run_dirs: Ordered list of run directories (chain order is passed order)
            strict: If True, perform additional strict checks
            check_proof: If True, require PROOF.json with verified=true for each bundle

        Returns:
            {"valid": bool, "errors": [...]}

        Error codes (in addition to verify_bundle codes):
            INVALID_CHAIN_REFERENCE - TASK_SPEC references output not in chain history

        Chain order note:
            Chain order is currently the passed order. Future versions may add
            timestamp parsing from STATUS.json to enforce strict temporal ordering.
        """
        errors = []

        # Normalize paths
        run_dirs = [Path(d) if isinstance(d, str) else d for d in run_dirs]

        # Phase 1: Verify each bundle individually
        for run_dir in run_dirs:
            result = self.verify_bundle(run_dir, strict=strict, check_proof=check_proof)
            if not result["valid"]:
                # Add run_id context to each error
                for err in result["errors"]:
                    err["run_id"] = run_dir.name
                errors.extend(result["errors"])
                # Fail fast on first bundle failure
                return {"valid": False, "errors": errors}

        # Phase 2: Build available output registry and validate references
        available_outputs: Dict[str, str] = {}  # path -> run_id that produced it

        for run_dir in run_dirs:
            run_id = run_dir.name

            # Load OUTPUT_HASHES.json to get this run's outputs
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            try:
                with open(hashes_path) as f:
                    output_hashes = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                errors.append({
                    "code": "BUNDLE_INCOMPLETE",
                    "severity": "error",
                    "message": f"Failed to load OUTPUT_HASHES.json: {e}",
                    "run_id": run_id,
                    "path": "/",
                    "details": {"expected": str(hashes_path)}
                })
                return {"valid": False, "errors": errors}

            # Get this run's declared outputs (keys of hashes dict)
            current_run_outputs = set(output_hashes.get("hashes", {}).keys())

            # Load TASK_SPEC.json to check for references
            task_spec_path = run_dir / "TASK_SPEC.json"
            try:
                with open(task_spec_path) as f:
                    task_spec = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                errors.append({
                    "code": "BUNDLE_INCOMPLETE",
                    "severity": "error",
                    "message": f"Failed to load TASK_SPEC.json: {e}",
                    "run_id": run_id,
                    "path": "/",
                    "details": {"expected": str(task_spec_path)}
                })
                return {"valid": False, "errors": errors}

            # Check references if present (skip validation if not present)
            # References are repo-root-relative POSIX paths
            references = task_spec.get("references", None)
            if references is not None:
                for ref_path in references:
                    # Normalize to POSIX for comparison
                    ref_posix = ref_path.replace("\\", "/")

                    # Check if reference exists in available_outputs OR current run outputs
                    in_earlier = ref_posix in available_outputs
                    in_current = ref_posix in current_run_outputs

                    if not (in_earlier or in_current):
                        errors.append({
                            "code": "INVALID_CHAIN_REFERENCE",
                            "severity": "error",
                            "message": f"Reference '{ref_posix}' not found in chain history or current outputs",
                            "run_id": run_id,
                            "path": "/references",
                            "details": {
                                "reference": ref_posix,
                                "available_outputs": sorted(list(available_outputs.keys())),
                                "current_outputs": sorted(list(current_run_outputs))
                            }
                        })
                        return {"valid": False, "errors": errors}

            # Register this run's outputs into available_outputs
            # Normalize all output paths to POSIX
            for output_path in current_run_outputs:
                output_path_posix = output_path.replace("\\", "/")
                available_outputs[output_path_posix] = run_id

        return {"valid": True, "errors": []}

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
