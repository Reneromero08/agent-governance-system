#!/usr/bin/env python3
"""
Bundle/Chain Verifier (Phase 1 Option #1)

Fail-closed verification of SPECTRUM-02 bundles and SPECTRUM-03 chains.

Verification depends ONLY on:
- Bundle artifacts (TASK_SPEC.json, STATUS.json, OUTPUT_HASHES.json)
- Actual file hashes
- Chain ordering

Verification rejects on:
- Any missing bundle artifacts (BUNDLE_INCOMPLETE)
- Hash mismatches (HASH_MISMATCH)
- Invalid chain references (INVALID_CHAIN_REFERENCE)
- Forbidden artifacts present (FORBIDDEN_ARTIFACT)
- STATUS.status != "success" (STATUS_NOT_SUCCESS)
- STATUS.cmp01 != "pass" (CMP01_NOT_PASS)
- Missing PROOF.json or verified != true (PROOF_REQUIRED, RESTORATION_FAILED)

All errors use structured validation_error format with stable codes.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set


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
    # SINGLE BUNDLE VERIFICATION (SPECTRUM-02)
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
    # CHAIN VERIFICATION (SPECTRUM-03)
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
            Hex digest of SHA-256 hash
        """
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks for large files
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()
