#!/usr/bin/env python3
"""
SPECTRUM-03 Chain Verification Tests

Tests for verifying chains of SPECTRUM-02 bundles with reference integrity.

Run: python CATALYTIC-DPT/TESTBENCH/spectrum/test_spectrum03_chain.py
"""

import json
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# Direct file import for MCP/server.py
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = REPO_ROOT / "CATALYTIC-DPT" / "LAB" / "MCP" / "server.py"

import importlib.util
spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server_module)

MCPTerminalServer = mcp_server_module.MCPTerminalServer
VALIDATOR_SEMVER = mcp_server_module.VALIDATOR_SEMVER
get_validator_build_id = mcp_server_module.get_validator_build_id

# Import the new BundleVerifier primitive
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))
from PRIMITIVES.verify_bundle import BundleVerifier


# =============================================================================
# SPECTRUM-03 CHAIN VERIFICATION HELPER
# =============================================================================

def verify_spectrum03_chain(
    run_dirs: List[Path],
    strict_order: bool = False
) -> Dict:
    """Verify a chain of SPECTRUM-02 bundles.

    This is a compatibility wrapper that uses the new BundleVerifier primitive.

    Checks:
    1. Each bundle verifies individually via verify_bundle
    2. Build registry of available outputs from each run
    3. If TASK_SPEC contains "references", validate each against available outputs
    4. No forbidden artifacts (logs/, tmp/, transcript.json)

    Args:
        run_dirs: Ordered list of run directories (chain order is passed order)
        strict_order: Reserved for future timestamp-based ordering

    Returns:
        {"valid": bool, "errors": [...]}

    Note: Chain order is currently the passed order. Future versions may add
          timestamp parsing from STATUS.json to enforce strict temporal ordering.
    """
    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Use the new verify_chain method with check_proof=False to match old behavior
    # (old tests don't create PROOF.json)
    result = verifier.verify_chain(run_dirs, strict=strict_order, check_proof=False)

    return result


# =============================================================================
# TEST CLASS
# =============================================================================

class RunnerSPECTRUM03Chain:
    """Tests for SPECTRUM-03 chain verification."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.server = MCPTerminalServer()
        # Use a temp location under CONTRACTS/_runs for test isolation
        self.test_base = REPO_ROOT / "CONTRACTS" / "_runs" / "_test_spectrum03"

    def run_all(self):
        """Run all SPECTRUM-03 chain tests."""
        print("=" * 60)
        print("SPECTRUM-03: Chain Verification Tests")
        print("=" * 60)
        print(f"REPO_ROOT: {REPO_ROOT}")
        print(f"VALIDATOR_SEMVER: {VALIDATOR_SEMVER}")
        print()

        self.test_chain_accepts_all_verified()
        self.test_chain_rejects_middle_tamper()
        self.test_chain_rejects_missing_bundle_artifact()
        self.test_chain_no_history_dependency()

        # Summary
        print()
        print("=" * 60)
        print(f"PASSED: {self.passed}")
        print(f"FAILED: {self.failed}")
        print("=" * 60)

        if self.errors:
            print("\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")

        return self.failed == 0

    def _assert(self, condition: bool, test_name: str, detail: str = ""):
        """Assert helper."""
        if condition:
            print(f"[PASS] {test_name}")
            self.passed += 1
        else:
            msg = f"{test_name}: {detail}" if detail else test_name
            print(f"[FAIL] {msg}")
            self.failed += 1
            self.errors.append(msg)

    def _compute_sha256(self, content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def _create_minimal_run(
        self,
        run_id: str,
        output_filename: str,
        output_content: bytes,
        references: List[str] = None
    ) -> Path:
        """Create a minimal valid SPECTRUM-02 bundle for testing.

        Args:
            run_id: Run identifier
            output_filename: Name of output file (created under out/)
            output_content: Content for the output file
            references: Optional list of repo-root-relative paths this run references

        Returns:
            Path to the run directory
        """
        run_dir = self.test_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create output directory and file
        out_dir = run_dir / "out"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / output_filename
        output_path.write_bytes(output_content)

        # Compute relative path (repo-root-relative, POSIX style)
        output_rel = f"CONTRACTS/_runs/_test_spectrum03/{run_id}/out/{output_filename}"

        # Create TASK_SPEC.json
        task_spec = {
            "task_id": run_id,
            "outputs": {
                "durable_paths": [output_rel]
            }
        }
        # Add references if provided
        if references:
            task_spec["references"] = references

        task_spec_bytes = json.dumps(task_spec, indent=2).encode("utf-8")
        (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

        # Create TASK_SPEC.sha256 (integrity hash)
        task_spec_hash = self._compute_sha256(task_spec_bytes)
        (run_dir / "TASK_SPEC.sha256").write_text(task_spec_hash)

        # Create STATUS.json with success and cmp01=pass
        status = {
            "status": "success",
            "cmp01": "pass",
            "run_id": run_id,
            "completed_at": "2024-12-24T12:00:00Z"
        }
        (run_dir / "STATUS.json").write_text(json.dumps(status, indent=2))

        # Create OUTPUT_HASHES.json with correct hash
        output_hash = f"sha256:{self._compute_sha256(output_content)}"
        output_hashes = {
            "validator_semver": VALIDATOR_SEMVER,
            "validator_build_id": get_validator_build_id(),
            "generated_at": "2024-12-24T12:00:00Z",
            "hashes": {
                output_rel: output_hash
            }
        }
        (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps(output_hashes, indent=2))

        # Add required SPECTRUM-05 artifacts (dummy data sufficient for Phase 1/2)
        (run_dir / "VALIDATOR_IDENTITY.json").write_text(json.dumps({
            "algorithm": "ed25519",
            "public_key": "0" * 64,
            "validator_id": "0" * 64
        }, indent=2))
        (run_dir / "SIGNED_PAYLOAD.json").write_text(json.dumps({
            "bundle_root": "0" * 64,
            "decision": "ACCEPT",
            "validator_id": "0" * 64
        }, indent=2))
        (run_dir / "SIGNATURE.json").write_text(json.dumps({
            "payload_type": "BUNDLE",
            "signature": "0" * 128,
            "validator_id": "0" * 64
        }, indent=2))

        return run_dir

    def _cleanup_test_base(self):
        """Remove test directories."""
        if self.test_base.exists():
            shutil.rmtree(self.test_base, ignore_errors=True)

    # =========================================================================
    # CHAIN TESTS
    # =========================================================================

    def test_chain_accepts_all_verified(self):
        """Chain should accept when all runs have valid SPECTRUM-02 bundles."""
        try:
            self._cleanup_test_base()

            # Create 3 fake runs with valid bundles
            run1_dir = self._create_minimal_run(
                "chain-run-001",
                "result1.txt",
                b"Output from run 1\n"
            )
            run2_dir = self._create_minimal_run(
                "chain-run-002",
                "result2.txt",
                b"Output from run 2\n"
            )
            run3_dir = self._create_minimal_run(
                "chain-run-003",
                "result3.txt",
                b"Output from run 3\n"
            )

            # Verify chain
            run_dirs = [run1_dir, run2_dir, run3_dir]
            result = verify_spectrum03_chain(run_dirs)

            self._assert(
                result["valid"],
                "test_chain_accepts_all_verified (chain valid)",
                f"Expected valid=True, got errors: {result['errors']}"
            )

            # Also verify each individual bundle passes
            for run_dir in run_dirs:
                bundle_result = self.server.verify_spectrum02_bundle(run_dir)
                self._assert(
                    bundle_result["valid"],
                    f"test_chain_accepts_all_verified ({run_dir.name} bundle valid)",
                    f"Individual bundle failed: {bundle_result['errors']}"
                )

        finally:
            self._cleanup_test_base()

    def test_chain_rejects_middle_tamper(self):
        """Chain should reject when middle run has tampered output."""
        try:
            self._cleanup_test_base()

            # Create 3 runs with valid bundles
            run1_dir = self._create_minimal_run(
                "chain-run-001",
                "result1.txt",
                b"Output from run 1\n"
            )
            run2_dir = self._create_minimal_run(
                "chain-run-002",
                "result2.txt",
                b"Output from run 2\n"
            )
            run3_dir = self._create_minimal_run(
                "chain-run-003",
                "result3.txt",
                b"Output from run 3\n"
            )

            # Tamper with run2's output file (append a single byte)
            run2_output = run2_dir / "out" / "result2.txt"
            with open(run2_output, "ab") as f:
                f.write(b"X")

            # Verify chain - should reject
            run_dirs = [run1_dir, run2_dir, run3_dir]
            result = verify_spectrum03_chain(run_dirs)

            self._assert(
                not result["valid"],
                "test_chain_rejects_middle_tamper (chain invalid)",
                f"Expected valid=False, got valid=True"
            )

            # Check error code and run_id
            if result["errors"]:
                first_error = result["errors"][0]
                self._assert(
                    first_error["code"] == "HASH_MISMATCH",
                    "test_chain_rejects_middle_tamper (error code)",
                    f"Expected HASH_MISMATCH, got {first_error['code']}"
                )
                self._assert(
                    first_error["run_id"] == run2_dir.name,
                    "test_chain_rejects_middle_tamper (error run_id)",
                    f"Expected {run2_dir.name}, got {first_error.get('run_id')}"
                )
            else:
                self._assert(
                    False,
                    "test_chain_rejects_middle_tamper (has errors)",
                    "Expected errors, got none"
                )

        finally:
            self._cleanup_test_base()

    def test_chain_rejects_missing_bundle_artifact(self):
        """Chain should reject when a bundle is missing OUTPUT_HASHES.json."""
        try:
            self._cleanup_test_base()

            # Create 3 runs with valid bundles
            run1_dir = self._create_minimal_run(
                "chain-run-001",
                "result1.txt",
                b"Output from run 1\n"
            )
            run2_dir = self._create_minimal_run(
                "chain-run-002",
                "result2.txt",
                b"Output from run 2\n"
            )
            run3_dir = self._create_minimal_run(
                "chain-run-003",
                "result3.txt",
                b"Output from run 3\n"
            )

            # Delete OUTPUT_HASHES.json from run1
            hashes_path = run1_dir / "OUTPUT_HASHES.json"
            hashes_path.unlink()

            # Verify chain - should reject
            run_dirs = [run1_dir, run2_dir, run3_dir]
            result = verify_spectrum03_chain(run_dirs)

            self._assert(
                not result["valid"],
                "test_chain_rejects_missing_bundle_artifact (chain invalid)",
                f"Expected valid=False, got valid=True"
            )

            # Check error code and run_id
            if result["errors"]:
                first_error = result["errors"][0]
                self._assert(
                    first_error["code"] == "ARTIFACT_MISSING",
                    "test_chain_rejects_missing_bundle_artifact (error code)",
                    f"Expected ARTIFACT_MISSING, got {first_error['code']}"
                )
                self._assert(
                    first_error["run_id"] == run1_dir.name,
                    "test_chain_rejects_missing_bundle_artifact (error run_id)",
                    f"Expected {run1_dir.name}, got {first_error.get('run_id')}"
                )
            else:
                self._assert(
                    False,
                    "test_chain_rejects_missing_bundle_artifact (has errors)",
                    "Expected errors, got none"
                )

        finally:
            self._cleanup_test_base()

    def test_chain_no_history_dependency(self):
        """Chain should accept without forbidden artifacts (logs/, tmp/, transcript.json)."""
        try:
            self._cleanup_test_base()

            # Create 3 runs with valid bundles
            run1_dir = self._create_minimal_run(
                "chain-run-001",
                "result1.txt",
                b"Output from run 1\n"
            )
            run2_dir = self._create_minimal_run(
                "chain-run-002",
                "result2.txt",
                b"Output from run 2\n"
            )
            run3_dir = self._create_minimal_run(
                "chain-run-003",
                "result3.txt",
                b"Output from run 3\n"
            )

            # Assert forbidden artifacts do NOT exist
            run_dirs = [run1_dir, run2_dir, run3_dir]
            for run_dir in run_dirs:
                self._assert(
                    not (run_dir / "logs").exists(),
                    f"test_chain_no_history_dependency ({run_dir.name} no logs/)",
                    "logs/ should not exist"
                )
                self._assert(
                    not (run_dir / "tmp").exists(),
                    f"test_chain_no_history_dependency ({run_dir.name} no tmp/)",
                    "tmp/ should not exist"
                )
                self._assert(
                    not (run_dir / "transcript.json").exists(),
                    f"test_chain_no_history_dependency ({run_dir.name} no transcript.json)",
                    "transcript.json should not exist"
                )

            # Verify chain - should accept
            result = verify_spectrum03_chain(run_dirs)

            self._assert(
                result["valid"],
                "test_chain_no_history_dependency (chain valid)",
                f"Expected valid=True, got errors: {result['errors']}"
            )

        finally:
            self._cleanup_test_base()


def test_spectrum03_chain():
    """Pytest entry point."""
    runner = RunnerSPECTRUM03Chain()
    assert runner.run_all()


if __name__ == "__main__":
    runner = RunnerSPECTRUM03Chain()
    success = runner.run_all()
    sys.exit(0 if success else 1)
