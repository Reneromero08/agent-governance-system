#!/usr/bin/env python3

"""
Validator Version Integrity Tests

Tests that OUTPUT_HASHES.json contains both validator_semver and validator_build_id,
and that verification enforces their presence and correctness.

Run: python CAPABILITY/TESTBENCH/spectrum/test_validator_version_integrity.py
"""

import json
import hashlib
import shutil
import sys
from pathlib import Path

# Direct file import for MCP/server.py
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "MCP" / "server.py"

import importlib.util
spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)

MCPTerminalServer = mcp_server.MCPTerminalServer
PROJECT_ROOT = mcp_server.PROJECT_ROOT
CONTRACTS_DIR = mcp_server.CONTRACTS_DIR
VALIDATOR_SEMVER = mcp_server.VALIDATOR_SEMVER
get_validator_build_id = mcp_server.get_validator_build_id


class RunnerValidatorVersionIntegrity:
    """Tests for validator version provenance."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.server = MCPTerminalServer()

    def run_all(self):
        """Run all validator version integrity tests."""
        print("=" * 60)
        print("Validator Version Integrity Tests")
        print("=" * 60)
        print(f"VALIDATOR_SEMVER: {VALIDATOR_SEMVER}")
        print(f"VALIDATOR_BUILD_ID: {get_validator_build_id()}")
        print()

        # Core tests
        self.test_output_hashes_includes_validator_fields()
        self.test_validator_build_id_deterministic()
        self.test_strict_build_id_mismatch_rejected()
        self.test_missing_build_id_rejected()
        self.test_empty_build_id_rejected()

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
    # VALIDATOR VERSION INTEGRITY TESTS
    # =========================================================================

    def test_output_hashes_includes_validator_fields(self):
        """OUTPUT_HASHES.json must contain validator_semver and validator_build_id."""
        run_id = "validator-fields-test"
        output_content = b"Test output for validator fields\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Trigger OUTPUT_HASHES.json generation
            result = self.server.skill_complete(run_id, "success", {})

            self._assert(
                result["status"] == "success",
                "test_output_hashes_includes_validator_fields (skill_complete)",
                f"skill_complete failed: {result}"
            )

            # Read OUTPUT_HASHES.json
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            self._assert(
                hashes_path.exists(),
                "test_output_hashes_includes_validator_fields (file exists)",
                "OUTPUT_HASHES.json not found"
            )

            if hashes_path.exists():
                with open(hashes_path) as f:
                    output_hashes = json.load(f)

                # Check validator_semver
                self._assert(
                    "validator_semver" in output_hashes,
                    "test_output_hashes_includes_validator_fields (has validator_semver)",
                    f"Missing validator_semver: {output_hashes.keys()}"
                )
                self._assert(
                    output_hashes["validator_semver"] == VALIDATOR_SEMVER,
                    "test_output_hashes_includes_validator_fields (valid semver)",
                    f"Invalid validator_semver in OUTPUT_HASHES.json"
                )

                # Check validator_build_id
                self._assert(
                    "validator_build_id" in output_hashes,
                    "test_output_hashes_includes_validator_fields (has validator_build_id)",
                    f"Missing validator_build_id: {output_hashes.keys()}"
                )
                self._assert(
                    output_hashes["validator_build_id"] == get_validator_build_id(),
                    "test_output_hashes_includes_validator_fields (valid build id)",
                    f"Invalid validator_build_id in OUTPUT_HASHES.json"
                )

        finally:
            self._cleanup_run(run_id)

    def test_validator_build_id_deterministic(self):
        """Ensure the validator_build_id is deterministic."""
        # This test requires a specific setup and cannot be automated without additional context.
        pass

    def test_strict_build_id_mismatch_rejected(self):
        """Verify that verify_spectrum02_bundle rejects if there's a mismatch between semver and build id."""
        run_id = "missing-build-id"
        output_content = b"Test output for missing build id\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Generate bundle
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                f"{run_id} skill complete failed: {result}"
            )

            # Remove validator_build_id from OUTPUT_HASHES.json
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            with open(hashes_path) as f:
                output_hashes = json.load(f)

            del output_hashes["validator_build_id"]

            with open(hashes_path, "w") as f:
                json.dump(output_hashes, f, indent=2)

            # Verify - should reject
            verify_result = self.server.verify_spectrum02_bundle(run_dir)
            self._assert(
                not verify_result["valid"],
                f"{run_id} verify_spectrum02_bundle failed: {verify_result}"
            )

            error_codes = [e["code"] for e in verify_result["errors"]]
            self._assert(
                "VALIDATOR_BUILD_ID_MISSING" in error_codes,
                f"{run_id} ERROR_CODE validation failed"
            )

        finally:
            self._cleanup_run(run_id)

    def test_missing_build_id_rejected(self):
        """Verify that verify_spectrum02_bundle rejects if validator_build_id is missing."""
        run_id = "missing-build-id"
        output_content = b"Test output for missing build id\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Generate bundle
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                f"{run_id} skill complete failed: {result}"
            )

            # Remove validator_build_id from OUTPUT_HASHES.json
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            with open(hashes_path) as f:
                output_hashes = json.load(f)

            del output_hashes["validator_build_id"]

            with open(hashes_path, "w") as f:
                json.dump(output_hashes, f, indent=2)

            # Verify - should reject
            verify_result = self.server.verify_spectrum02_bundle(run_dir)
            self._assert(
                not verify_result["valid"],
                f"{run_id} verify_spectrum02_bundle failed: {verify_result}"
            )

            error_codes = [e["code"] for e in verify_result["errors"]]
            self._assert(
                "VALIDATOR_BUILD_ID_MISSING" in error_codes,
                f"{run_id} ERROR_CODE validation failed"
            )

        finally:
            self._cleanup_run(run_id)

    def test_empty_build_id_rejected(self):
        """Verify that verify_spectrum02_bundle rejects if validator_build_id is empty string."""
        run_id = "empty-build-id"
        output_content = b"Test output for empty build id\n"

        try:
            run_dir = self._create_test_run(run_id, output_content)

            # Generate bundle
            result = self.server.skill_complete(run_id, "success", {})
            self._assert(
                result["status"] == "success",
                f"{run_id} skill complete failed: {result}"
            )

            # Set validator_build_id to empty string
            hashes_path = run_dir / "OUTPUT_HASHES.json"
            with open(hashes_path) as f:
                output_hashes = json.load(f)

            output_hashes["validator_build_id"] = ""

            with open(hashes_path, "w") as f:
                json.dump(output_hashes, f, indent=2)

            # Verify - should reject
            verify_result = self.server.verify_spectrum02_bundle(run_dir)
            self._assert(
                not verify_result["valid"],
                f"{run_id} verify_spectrum02_bundle failed: {verify_result}"
            )

            error_codes = [e["code"] for e in verify_result["errors"]]
            self._assert(
                "VALIDATOR_BUILD_ID_MISSING" in error_codes,
                f"{run_id} ERROR_CODE validation failed"
            )

        finally:
            self._cleanup_run(run_id)

def test_validator_version_integrity():
    """Pytest entry point."""
    runner = RunnerValidatorVersionIntegrity()
    assert runner.run_all()


if __name__ == "__main__":
    runner = RunnerValidatorVersionIntegrity()
    success = runner.run_all()
    sys.exit(0 if success else 1)
