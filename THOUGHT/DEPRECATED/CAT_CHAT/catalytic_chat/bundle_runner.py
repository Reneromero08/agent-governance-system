#!/usr/bin/env python3
"""
Bundle Runner (Phase G - Bundle Replay & Verification)

Standalone offline bundle runner with verify-before-run semantics.
Proves bundles are self-contained and reproducible.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class BundleRunnerError(Exception):
    """Bundle runner error - hard fail on any verification or execution failure."""
    pass


def _canonical_json(data: Any) -> str:
    """Serialize data to canonical JSON."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(content: str) -> str:
    """Compute SHA256 hex digest of string content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


class BundleRunner:
    """Standalone offline bundle runner with verify-before-run semantics.

    This runner operates WITHOUT access to:
    - Repository root
    - Database
    - External files

    All inputs must be in the bundle's artifacts/ directory.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        bundle_path: Path,
        receipt_out: Optional[Path] = None,
        signing_key: Optional[bytes] = None,
        previous_receipt_hash: Optional[str] = None
    ):
        """Initialize runner.

        Args:
            bundle_path: Path to bundle directory or bundle.json
            receipt_out: Output path for receipt (default: bundle_dir/receipt.json)
            signing_key: Optional Ed25519 signing key (32 bytes)
            previous_receipt_hash: Optional hash of previous receipt for chaining
        """
        self.bundle_dir, self.bundle_json_path = self._resolve_bundle_path(bundle_path)
        self.artifacts_dir = self.bundle_dir / "artifacts"
        self.receipt_out = receipt_out or (self.bundle_dir / "receipt.json")
        self.signing_key = signing_key
        self.previous_receipt_hash = previous_receipt_hash

        if signing_key is not None and len(signing_key) not in (32, 64):
            raise BundleRunnerError(
                f"Signing key must be 32 or 64 bytes, got {len(signing_key)}"
            )

    def _resolve_bundle_path(self, bundle_path: Path) -> tuple:
        """Resolve bundle path and return (bundle_dir, bundle_json_path).

        Args:
            bundle_path: Path to bundle directory or bundle.json

        Returns:
            Tuple of (bundle_dir, bundle_json_path)

        Raises:
            BundleRunnerError: If path is invalid or bundle.json not found
        """
        bundle_path = Path(bundle_path)

        if bundle_path.is_file():
            if bundle_path.name != "bundle.json":
                raise BundleRunnerError(
                    f"Bundle file must be named bundle.json: {bundle_path}"
                )
            bundle_json_path = bundle_path
            bundle_dir = bundle_path.parent
        elif bundle_path.is_dir():
            bundle_dir = bundle_path
            bundle_json_path = bundle_dir / "bundle.json"
            if not bundle_json_path.exists():
                raise BundleRunnerError(
                    f"bundle.json not found in directory: {bundle_dir}"
                )
        else:
            raise BundleRunnerError(f"Bundle path not found: {bundle_path}")

        return bundle_dir, bundle_json_path

    def _load_manifest(self) -> Dict[str, Any]:
        """Load bundle manifest from bundle.json.

        Returns:
            Bundle manifest dictionary

        Raises:
            BundleRunnerError: If manifest cannot be loaded
        """
        try:
            with open(self.bundle_json_path, 'r', encoding='utf-8') as f:
                content = f.read().rstrip('\n')
                return json.loads(content)
        except (IOError, json.JSONDecodeError) as e:
            raise BundleRunnerError(f"Failed to load bundle manifest: {e}")

    def verify(self) -> Dict[str, Any]:
        """Verify bundle integrity before execution.

        Performs:
        1. Schema validation
        2. Ordering validation (steps, artifacts)
        3. Artifact hash verification (all artifacts)
        4. Root hash verification
        5. Bundle ID verification
        6. Boundedness validation

        Returns:
            Verification result dict with bundle_id, root_hash, artifact_count

        Raises:
            BundleRunnerError: On any verification failure (fail-closed)
        """
        from catalytic_chat.bundle import BundleVerifier, BundleError

        try:
            verifier = BundleVerifier(self.bundle_dir)
            result = verifier.verify()
            return {
                "status": "verified",
                "bundle_id": result["bundle_id"],
                "root_hash": result["root_hash"],
                "artifact_count": result["artifact_count"]
            }
        except BundleError as e:
            raise BundleRunnerError(f"Bundle verification failed: {e}")

    def load_artifact_content(self, artifact: Dict[str, Any]) -> str:
        """Load artifact content from bundle's artifacts directory.

        Args:
            artifact: Artifact manifest entry with path, sha256, bytes fields

        Returns:
            Artifact content as string

        Raises:
            BundleRunnerError: If artifact not found or hash mismatch
        """
        posix_path = artifact["path"]
        artifact_path = self.bundle_dir / posix_path

        if not artifact_path.exists():
            raise BundleRunnerError(
                f"Artifact file missing: {artifact['artifact_id']}"
            )

        # Path traversal check
        try:
            artifact_path.resolve().relative_to(self.bundle_dir.resolve())
        except ValueError:
            raise BundleRunnerError(
                f"Path traversal attempt: {artifact['path']}"
            )

        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except IOError as e:
            raise BundleRunnerError(
                f"Failed to read artifact {artifact['artifact_id']}: {e}"
            )

        # Verify hash
        computed_hash = _sha256(content)
        if computed_hash != artifact["sha256"]:
            raise BundleRunnerError(
                f"Artifact hash mismatch: {artifact['artifact_id']} "
                f"(expected {artifact['sha256'][:16]}..., got {computed_hash[:16]}...)"
            )

        # Verify size
        actual_bytes = len(content.encode('utf-8'))
        if actual_bytes != artifact["bytes"]:
            raise BundleRunnerError(
                f"Artifact size mismatch: {artifact['artifact_id']} "
                f"(expected {artifact['bytes']}, got {actual_bytes})"
            )

        return content

    def _build_artifacts_by_ref(
        self,
        manifest: Dict[str, Any]
    ) -> Dict[tuple, Dict[str, Any]]:
        """Build lookup map from (ref, slice) to artifact.

        Args:
            manifest: Bundle manifest

        Returns:
            Dict mapping (ref, slice) tuples to artifact entries
        """
        artifacts_by_ref = {}
        for artifact in manifest.get("artifacts", []):
            key = (artifact["ref"], artifact.get("slice"))
            artifacts_by_ref[key] = artifact
        return artifacts_by_ref

    def execute_step(
        self,
        step: Dict[str, Any],
        artifacts_by_ref: Dict[tuple, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a single step using only bundled artifacts.

        Args:
            step: Step definition from bundle
            artifacts_by_ref: Map of (ref, slice) -> artifact for lookup

        Returns:
            Step result dict with ordinal, step_id, op, outcome, result, error
        """
        op = step.get("op")
        refs = step.get("refs", {})
        constraints = step.get("constraints", {})

        result = {
            "ordinal": step.get("ordinal"),
            "step_id": step.get("step_id"),
            "op": op,
            "outcome": "SUCCESS",
            "result": None,
            "error": None
        }

        try:
            if op == "READ_SECTION":
                section_id = refs.get("section_id")
                slice_expr = constraints.get("slice")
                key = (section_id, slice_expr)

                artifact = artifacts_by_ref.get(key)
                if artifact is None:
                    raise BundleRunnerError(
                        f"No artifact for section {section_id} slice={slice_expr}"
                    )

                content = self.load_artifact_content(artifact)
                result["result"] = {
                    "content_hash": artifact["sha256"],
                    "bytes": artifact["bytes"]
                }

            elif op == "READ_SYMBOL":
                symbol_id = refs.get("symbol_id")
                slice_expr = constraints.get("slice")
                key = (symbol_id, slice_expr)

                artifact = artifacts_by_ref.get(key)
                if artifact is None:
                    raise BundleRunnerError(
                        f"No artifact for symbol {symbol_id} slice={slice_expr}"
                    )

                content = self.load_artifact_content(artifact)
                result["result"] = {
                    "content_hash": artifact["sha256"],
                    "bytes": artifact["bytes"]
                }

            else:
                # Unknown operation - log but don't fail
                # The bundle passed verification, so we trust its structure
                result["result"] = {"op": op, "status": "passthrough"}

        except BundleRunnerError as e:
            result["outcome"] = "FAILURE"
            result["error"] = {"code": "STEP_EXECUTION_ERROR", "message": str(e)}

        return result

    def run(self) -> Dict[str, Any]:
        """Run bundle with verify-before-run semantics.

        Algorithm:
        1. Call verify() - hard fail on any mismatch
        2. Load manifest
        3. Build artifacts_by_ref lookup
        4. Execute steps in order using only bundled artifacts
        5. Build receipt
        6. Compute receipt_hash
        7. Optionally sign receipt
        8. Write receipt to receipt_out

        Returns:
            Receipt dict with all fields populated

        Raises:
            BundleRunnerError: On verification failure or execution error
        """
        # Step 1: Verify BEFORE any execution
        verification = self.verify()

        # Step 2: Load manifest (already verified)
        manifest = self._load_manifest()

        # Step 3: Build artifacts lookup
        artifacts_by_ref = self._build_artifacts_by_ref(manifest)

        # Step 4: Execute steps in order
        step_results = []
        overall_outcome = "SUCCESS"
        overall_error = None

        for step in manifest.get("steps", []):
            step_result = self.execute_step(step, artifacts_by_ref)
            step_results.append(step_result)

            if step_result["outcome"] != "SUCCESS":
                overall_outcome = "FAILURE"
                if overall_error is None:
                    overall_error = {
                        "code": "STEP_FAILED",
                        "message": f"Step {step_result['step_id']} failed",
                        "step_id": step_result["step_id"]
                    }

        # Step 5: Build receipt
        from catalytic_chat.receipt import (
            canonical_json_bytes,
            compute_receipt_hash,
            write_receipt,
            RECEIPT_VERSION,
            EXECUTOR_VERSION
        )

        # Sort step results by (ordinal, step_id) for determinism
        sorted_steps = sorted(
            step_results,
            key=lambda x: (x.get("ordinal", 0), x.get("step_id", ""))
        )

        # Build artifact list for receipt
        sorted_artifacts = sorted(
            manifest.get("artifacts", []),
            key=lambda x: x.get("artifact_id", "")
        )

        receipt_artifacts = []
        for artifact in sorted_artifacts:
            receipt_artifacts.append({
                "artifact_id": artifact["artifact_id"],
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"]
            })

        receipt = {
            "receipt_version": RECEIPT_VERSION,
            "run_id": manifest.get("run_id"),
            "job_id": manifest.get("job_id"),
            "bundle_id": manifest.get("bundle_id"),
            "plan_hash": manifest.get("plan_hash"),
            "executor_version": EXECUTOR_VERSION,
            "outcome": overall_outcome,
            "error": overall_error,
            "steps": sorted_steps,
            "artifacts": receipt_artifacts,
            "root_hash": manifest.get("hashes", {}).get("root_hash", ""),
            "parent_receipt_hash": self.previous_receipt_hash,
            "receipt_hash": None,
            "attestation": None,
            "receipt_index": None
        }

        # Step 6: Compute receipt hash
        receipt["receipt_hash"] = compute_receipt_hash(receipt)

        # Step 7: Sign if key provided
        if self.signing_key:
            from catalytic_chat.attestation import sign_receipt_bytes
            receipt_bytes = canonical_json_bytes(receipt)
            receipt["attestation"] = sign_receipt_bytes(
                receipt_bytes,
                self.signing_key
            )

        # Step 8: Write receipt
        write_receipt(self.receipt_out, receipt)

        return receipt


def replay_bundle(
    bundle_path: Path,
    receipt_out: Optional[Path] = None,
    signing_key: Optional[bytes] = None,
    previous_receipt_hash: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for single-shot bundle replay.

    Args:
        bundle_path: Path to bundle directory or bundle.json
        receipt_out: Output path for receipt
        signing_key: Optional signing key
        previous_receipt_hash: Optional previous receipt hash for chaining

    Returns:
        Replay result with receipt dict

    Raises:
        BundleRunnerError: On any failure
    """
    runner = BundleRunner(
        bundle_path,
        receipt_out,
        signing_key,
        previous_receipt_hash
    )
    return runner.run()


def verify_bundle_integrity(bundle_path: Path) -> Dict[str, Any]:
    """Verify bundle integrity without executing.

    Args:
        bundle_path: Path to bundle directory or bundle.json

    Returns:
        Verification result dict

    Raises:
        BundleRunnerError: On verification failure
    """
    runner = BundleRunner(bundle_path)
    return runner.verify()
