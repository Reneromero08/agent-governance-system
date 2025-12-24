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
REPO_ROOT = Path(__file__).parent.parent.parent.parent
SERVER_PATH = REPO_ROOT / "CATALYTIC-DPT" / "MCP" / "server.py"

import importlib.util
spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server_module)

MCPTerminalServer = mcp_server_module.MCPTerminalServer
VALIDATOR_SEMVER = mcp_server_module.VALIDATOR_SEMVER
get_validator_build_id = mcp_server_module.get_validator_build_id


# =============================================================================
# SPECTRUM-03 CHAIN VERIFICATION HELPER
# =============================================================================

def verify_spectrum03_chain(
    run_dirs: List[Path],
    strict_order: bool = False
) -> Dict:
    """Verify a chain of SPECTRUM-02 bundles.

    Checks:
    1. Each bundle verifies individually via verify_spectrum02_bundle
    2. Build registry of available outputs from each run
    3. If TASK_SPEC contains "references", validate each against available outputs
    4. "No history dependency" is asserted but non-failing (see TODO)

    Args:
        run_dirs: Ordered list of run directories (chain order is passed order)
        strict_order: Reserved for future timestamp-based ordering

    Returns:
        {"valid": bool, "errors": [...]}

    TODO: Chain order is currently the passed order. Add timestamp parsing
          from STATUS.json or ledger to enforce strict temporal ordering.
    """
    errors = []
    server = MCPTerminalServer()

    # Normalize paths
    run_dirs = [Path(d) if isinstance(d, str) else d for d in run_dirs]

    # Phase 1: Verify each bundle individually
    for run_dir in run_dirs:
        result = server.verify_spectrum02_bundle(run_dir)
        if not result["valid"]:
            # Add run_id context to each error
            for err in result["errors"]:
                err["run_id"] = run_dir.name
            errors.extend(result["errors"])
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
                        "message": f"Reference '{ref_posix}' not found in chain history or current outputs",
                        "run_id": run_id,
                        "path": f"/references",
                        "details": {
                            "reference": ref_posix,
                            "available_outputs": list(available_outputs.keys()),
                            "current_outputs": list(current_run_outputs)
                        }
                    })
                    return {"valid": False, "errors": errors}

        # Register this run's outputs into available_outputs
        for output_path in current_run_outputs:
            available_outputs[output_path] = run_id

    # Phase 3: No history dependency assertion (NON-FAILING for now)
    # This is a conceptual check: verification depends only on bundle artifacts
    # and actual file hashes, not execution history.
    # TODO: Consider adding explicit checks for forbidden artifacts (logs/, tmp/)
    #       but per requirements, this is non-failing for now.

    return {"valid": True, "errors": []}


# =============================================================================
# TEST CLASS
# =============================================================================

class TestSPECTRUM03Chain:
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

        # Run tests
        self.test_chain_accepts_all_verified()
        self.test_chain_rejects_middle_tamper()
        self.test_chain_rejects_missing_bundle_artifact()
        self.test_chain_rejects_invalid_reference()
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
                    first_error["code"] == "BUNDLE_INCOMPLETE",
                    "test_chain_rejects_missing_bundle_artifact (error code)",
                    f"Expected BUNDLE_INCOMPLETE, got {first_error['code']}"
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

    def test_chain_rejects_invalid_reference(self):
        """Chain should reject when a run references non-existent output."""
        try:
            self._cleanup_test_base()

            # Create run1 and run2 normally
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

            # Create run3 with invalid reference
            run3_dir = self._create_minimal_run(
                "chain-run-003",
                "result3.txt",
                b"Output from run 3\n",
                references=["CONTRACTS/_runs/_test_spectrum03/chain-run-999/out/nope.txt"]
            )

            # Verify chain - should reject
            run_dirs = [run1_dir, run2_dir, run3_dir]
            result = verify_spectrum03_chain(run_dirs)

            self._assert(
                not result["valid"],
                "test_chain_rejects_invalid_reference (chain invalid)",
                f"Expected valid=False, got valid=True"
            )

            # Check error code and run_id
            if result["errors"]:
                first_error = result["errors"][0]
                self._assert(
                    first_error["code"] == "INVALID_CHAIN_REFERENCE",
                    "test_chain_rejects_invalid_reference (error code)",
                    f"Expected INVALID_CHAIN_REFERENCE, got {first_error['code']}"
                )
                self._assert(
                    first_error["run_id"] == run3_dir.name,
                    "test_chain_rejects_invalid_reference (error run_id)",
                    f"Expected {run3_dir.name}, got {first_error.get('run_id')}"
                )
            else:
                self._assert(
                    False,
                    "test_chain_rejects_invalid_reference (has errors)",
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


if __name__ == "__main__":
    tester = TestSPECTRUM03Chain()
    success = tester.run_all()
    sys.exit(0 if success else 1)
