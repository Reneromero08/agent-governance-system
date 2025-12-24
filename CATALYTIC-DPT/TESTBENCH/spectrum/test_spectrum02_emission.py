#!/usr/bin/env python3
"""
SPECTRUM-02 Bundle Emission Integration Test

Tests that MCP server correctly emits SPECTRUM-02 bundles on successful skill_complete.

Run: python CATALYTIC-DPT/TESTBENCH/spectrum/test_spectrum02_emission.py
"""

import json
import hashlib
import shutil
import sys
from pathlib import Path

# Direct file import for MCP/server.py
REPO_ROOT = Path(__file__).parent.parent.parent.parent
SERVER_PATH = REPO_ROOT / "CATALYTIC-DPT" / "MCP" / "server.py"

import importlib.util
spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)

MCPTerminalServer = mcp_server.MCPTerminalServer
PROJECT_ROOT = mcp_server.PROJECT_ROOT
CONTRACTS_DIR = mcp_server.CONTRACTS_DIR
VALIDATOR_SEMVER = mcp_server.VALIDATOR_SEMVER


class TestSPECTRUM02Emission:
    """Integration tests for SPECTRUM-02 bundle emission."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.server = MCPTerminalServer()

    def run_all(self):
        """Run all SPECTRUM-02 emission tests."""
        print("=" * 60)
        print("SPECTRUM-02: Bundle Emission Integration Tests")
        print("=" * 60)
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"VALIDATOR_SEMVER: {VALIDATOR_SEMVER}")
        print()

        # Core emission tests
        self.test_bundle_emitted_on_success()
        self.test_bundle_verification_accepts_valid()
        self.test_bundle_verification_rejects_tampered()
        self.test_directory_output_hashes_all_files()
        self.test_no_forbidden_artifacts_in_bundle()

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

    def _create_test_run(self, run_id: str, output_content: bytes) -> Path:
        """Create a test run directory with TASK_SPEC and output file."""
        run_dir = CONTRACTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create output file
        output_rel = f"CONTRACTS/_runs/{run_id}/output.txt"
        output_path = run_dir / "output.txt"
        output_path.write_bytes(output_content)

        # Create TASK_SPEC.json
        task_spec = {
            "task_id": run_id,
            "inputs": [],
            "expected_outputs": [output_rel],
            "outputs": {
                "durable_paths": [output_rel]
            },
            "constraints": {}
        }
        task_spec_bytes = json.dumps(task_spec, indent=2).encode('utf-8')
        (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)

        # Create TASK_SPEC.sha256 (integrity hash)
        task_spec_hash = hashlib.sha256(task_spec_bytes).hexdigest()
        (run_dir / "TASK_SPEC.sha256").write_text(task_spec_hash)

        return run_dir

    def _cleanup_run(self, run_id: str):
        """Remove test run directory."""
        run_dir = CONTRACTS_DIR / run_id
        shutil.rmtree(run_dir, ignore_errors=True)

    # =========================================================================
    # EMISSION TESTS
    # =========================================================================

    def test_bundle_emitted_on_success(self):
        """skill_complete should emit OUTPUT_HASHES.json on success."""
        run_id = "spectrum02-emission-test"
        output_content = b"Test output content for hashing\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Call skill_complete
            result = self.server.skill_complete(run_id, "success", {})

            self._assert(
                result["status"] == "success",
                "test_bundle_emitted_on_success (skill_complete success)",
                f"Expected success, got: {result}"
            )

            # Verify OUTPUT_HASHES.json exists
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            self._assert(
                hashes_path.exists(),
                "test_bundle_emitted_on_success (OUTPUT_HASHES.json exists)",
                "OUTPUT_HASHES.json not found"
            )

            if hashes_path.exists():
                with open(hashes_path) as f:
                    output_hashes = json.load(f)

                # Verify structure
                self._assert(
                    "validator_semver" in output_hashes,
                    "test_bundle_emitted_on_success (has validator_semver)",
                    f"Missing validator_semver: {output_hashes}"
                )
                self._assert(
                    "validator_build_id" in output_hashes,
                    "test_bundle_emitted_on_success (has validator_build_id)",
                    f"Missing validator_build_id: {output_hashes}"
                )
                self._assert(
                    "generated_at" in output_hashes,
                    "test_bundle_emitted_on_success (has generated_at)",
                    f"Missing generated_at: {output_hashes}"
                )
                self._assert(
                    "hashes" in output_hashes,
                    "test_bundle_emitted_on_success (has hashes)",
                    f"Missing hashes: {output_hashes}"
                )

                # Verify validator_semver matches server
                self._assert(
                    output_hashes["validator_semver"] == VALIDATOR_SEMVER,
                    "test_bundle_emitted_on_success (validator_semver correct)",
                    f"Expected {VALIDATOR_SEMVER}, got {output_hashes['validator_semver']}"
                )

                # Verify hash content
                expected_hash = "sha256:" + hashlib.sha256(output_content).hexdigest()
                output_rel = f"CONTRACTS/_runs/{run_id}/output.txt"
                self._assert(
                    output_rel in output_hashes["hashes"],
                    "test_bundle_emitted_on_success (output in hashes)",
                    f"Output {output_rel} not in hashes: {output_hashes['hashes'].keys()}"
                )
                if output_rel in output_hashes["hashes"]:
                    self._assert(
                        output_hashes["hashes"][output_rel] == expected_hash,
                        "test_bundle_emitted_on_success (hash correct)",
                        f"Expected {expected_hash}, got {output_hashes['hashes'][output_rel]}"
                    )

            # Verify STATUS.json
            status_path = run_dir / "STATUS.json"
            self._assert(
                status_path.exists(),
                "test_bundle_emitted_on_success (STATUS.json exists)",
                "STATUS.json not found"
            )
            if status_path.exists():
                with open(status_path) as f:
                    status = json.load(f)
                self._assert(
                    status.get("status") == "success" and status.get("cmp01") == "pass",
                    "test_bundle_emitted_on_success (STATUS correct)",
                    f"Expected status=success, cmp01=pass, got: {status}"
                )

        finally:
            self._cleanup_run(run_id)

    def test_bundle_verification_accepts_valid(self):
        """verify_spectrum02_bundle should accept a valid bundle."""
        run_id = "spectrum02-verify-valid"
        output_content = b"Valid bundle test content\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Generate bundle via skill_complete
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                "test_bundle_verification_accepts_valid (skill_complete)",
                f"skill_complete failed: {result}"
            )

            # Verify bundle
            verify_result = self.server.verify_spectrum02_bundle(run_dir)

            self._assert(
                verify_result["valid"],
                "test_bundle_verification_accepts_valid (bundle valid)",
                f"Expected valid, got errors: {verify_result['errors']}"
            )

        finally:
            self._cleanup_run(run_id)

    def test_bundle_verification_rejects_tampered(self):
        """verify_spectrum02_bundle should reject a tampered output."""
        run_id = "spectrum02-verify-tampered"
        output_content = b"Original content before tampering\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Generate bundle via skill_complete
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                "test_bundle_verification_rejects_tampered (skill_complete)",
                f"skill_complete failed: {result}"
            )

            # Tamper with output file (modify 1 byte)
            output_path = run_dir / "output.txt"
            tampered = output_content[:-1] + b"X"
            output_path.write_bytes(tampered)

            # Verify bundle - should reject
            verify_result = self.server.verify_spectrum02_bundle(run_dir)

            self._assert(
                not verify_result["valid"],
                "test_bundle_verification_rejects_tampered (rejected)",
                f"Expected invalid, got valid"
            )

            error_codes = [e["code"] for e in verify_result["errors"]]
            self._assert(
                "HASH_MISMATCH" in error_codes,
                "test_bundle_verification_rejects_tampered (HASH_MISMATCH)",
                f"Expected HASH_MISMATCH, got: {error_codes}"
            )

        finally:
            self._cleanup_run(run_id)

    def test_directory_output_hashes_all_files(self):
        """Directory outputs should hash every file inside."""
        run_id = "spectrum02-dir-output"

        try:
            run_dir = CONTRACTS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create output directory with multiple files
            output_dir = run_dir / "results"
            output_dir.mkdir()
            (output_dir / "file1.txt").write_bytes(b"content1\n")
            (output_dir / "file2.txt").write_bytes(b"content2\n")
            (output_dir / "subdir").mkdir()
            (output_dir / "subdir" / "nested.txt").write_bytes(b"nested content\n")

            # Create TASK_SPEC.json with directory output
            output_rel = f"CONTRACTS/_runs/{run_id}/results"
            task_spec = {
                "task_id": run_id,
                "inputs": [],
                "outputs": {
                    "durable_paths": [output_rel]
                }
            }
            task_spec_bytes = json.dumps(task_spec, indent=2).encode('utf-8')
            (run_dir / "TASK_SPEC.json").write_bytes(task_spec_bytes)
            (run_dir / "TASK_SPEC.sha256").write_text(
                hashlib.sha256(task_spec_bytes).hexdigest()
            )

            # Generate bundle
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                "test_directory_output_hashes_all_files (skill_complete)",
                f"skill_complete failed: {result}"
            )

            # Check OUTPUT_HASHES.json contains all files
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            with open(hashes_path) as f:
                output_hashes = json.load(f)

            hashed_files = set(output_hashes["hashes"].keys())
            expected_files = {
                f"CONTRACTS/_runs/{run_id}/results/file1.txt",
                f"CONTRACTS/_runs/{run_id}/results/file2.txt",
                f"CONTRACTS/_runs/{run_id}/results/subdir/nested.txt",
            }

            self._assert(
                expected_files <= hashed_files,
                "test_directory_output_hashes_all_files (all files hashed)",
                f"Expected {expected_files}, got {hashed_files}"
            )

            # Verify bundle is valid
            verify_result = self.server.verify_spectrum02_bundle(run_dir)
            self._assert(
                verify_result["valid"],
                "test_directory_output_hashes_all_files (bundle valid)",
                f"Expected valid, got errors: {verify_result['errors']}"
            )

        finally:
            self._cleanup_run(run_id)

    def test_no_forbidden_artifacts_in_bundle(self):
        """Bundle should NOT contain logs, tmp, transcripts, etc."""
        run_id = "spectrum02-no-forbidden"
        output_content = b"Clean bundle test\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Generate bundle
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                "test_no_forbidden_artifacts_in_bundle (skill_complete)",
                f"skill_complete failed: {result}"
            )

            # Verify no forbidden artifacts exist
            forbidden = [
                "logs",
                "tmp",
                "chat_history.json",
                "reasoning_trace.json",
                "checkpoints",
                "intermediate_state.json",
            ]

            for artifact in forbidden:
                artifact_path = run_dir / artifact
                self._assert(
                    not artifact_path.exists(),
                    f"test_no_forbidden_artifacts_in_bundle (no {artifact})",
                    f"{artifact} should not exist in bundle"
                )

        finally:
            self._cleanup_run(run_id)


if __name__ == "__main__":
    tester = TestSPECTRUM02Emission()
    success = tester.run_all()
    sys.exit(0 if success else 1)
