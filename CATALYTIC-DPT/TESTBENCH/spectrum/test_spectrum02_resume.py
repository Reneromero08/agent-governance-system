#!/usr/bin/env python3
"""
SPECTRUM-02: Adversarial Resume Without Execution History

Tests that an agent can resume trust using ONLY a compressed durable bundle,
when all prior execution context has been destroyed.

Run: python CATALYTIC-DPT/TESTBENCH/spectrum/test_spectrum02_resume.py
"""

import json
import hashlib
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

# Constants
REPO_ROOT = Path(__file__).parent.parent.parent.parent
SUPPORTED_VALIDATOR_VERSIONS = {"1.0.0", "1.0.1", "1.1.0"}


class SPECTRUM02Verifier:
    """
    SPECTRUM-02 Resume Bundle Verifier.

    Verifies that a durable bundle is valid for resume WITHOUT
    requiring any execution history.
    """

    def __init__(self, bundle_path: Path):
        self.bundle_path = bundle_path
        self.errors = []

    def verify(self) -> dict:
        """
        Verify the SPECTRUM-02 bundle.

        Returns:
            {"valid": bool, "errors": list[dict]}
        """
        self.errors = []

        # Check required artifacts exist
        task_spec_path = self.bundle_path / "TASK_SPEC.json"
        status_path = self.bundle_path / "STATUS.json"
        hashes_path = self.bundle_path / "OUTPUT_HASHES.json"

        if not task_spec_path.exists():
            self._add_error("BUNDLE_INCOMPLETE", "TASK_SPEC.json missing")
            return self._result()

        if not status_path.exists():
            self._add_error("BUNDLE_INCOMPLETE", "STATUS.json missing")
            return self._result()

        if not hashes_path.exists():
            self._add_error("BUNDLE_INCOMPLETE", "OUTPUT_HASHES.json missing")
            return self._result()

        # Parse and verify STATUS.json
        try:
            with open(status_path) as f:
                status = json.load(f)
        except json.JSONDecodeError as e:
            self._add_error("BUNDLE_INCOMPLETE", f"STATUS.json invalid: {e}")
            return self._result()

        if status.get("status") != "success":
            self._add_error("STATUS_NOT_SUCCESS", f"status={status.get('status')}")
            return self._result()

        if status.get("cmp01") != "pass":
            self._add_error("CMP01_NOT_PASS", f"cmp01={status.get('cmp01')}")
            return self._result()

        # Parse and verify OUTPUT_HASHES.json
        try:
            with open(hashes_path) as f:
                output_hashes = json.load(f)
        except json.JSONDecodeError as e:
            self._add_error("BUNDLE_INCOMPLETE", f"OUTPUT_HASHES.json invalid: {e}")
            return self._result()

        validator_version = output_hashes.get("validator_version")
        if validator_version not in SUPPORTED_VALIDATOR_VERSIONS:
            self._add_error(
                "VALIDATOR_UNSUPPORTED",
                f"validator_version={validator_version}, supported={SUPPORTED_VALIDATOR_VERSIONS}"
            )
            return self._result()

        # Verify each output hash
        hashes = output_hashes.get("hashes", {})
        for rel_path, expected_hash in hashes.items():
            abs_path = REPO_ROOT / rel_path

            if not abs_path.exists():
                self._add_error("OUTPUT_MISSING", f"path={rel_path}")
                continue

            # Compute SHA-256
            with open(abs_path, "rb") as f:
                actual_hash = "sha256:" + hashlib.sha256(f.read()).hexdigest()

            if actual_hash != expected_hash:
                self._add_error(
                    "HASH_MISMATCH",
                    f"path={rel_path}, expected={expected_hash[:32]}..., actual={actual_hash[:32]}..."
                )

        return self._result()

    def _add_error(self, code: str, message: str):
        self.errors.append({"code": code, "message": message})

    def _result(self) -> dict:
        return {"valid": len(self.errors) == 0, "errors": self.errors}


class TestSPECTRUM02Resume:
    """Test suite for SPECTRUM-02 adversarial resume."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.test_root = REPO_ROOT / "CONTRACTS" / "_runs" / "_spectrum02_test"

    def run_all(self):
        """Run all SPECTRUM-02 tests."""
        print("=" * 60)
        print("SPECTRUM-02: Adversarial Resume Tests")
        print("=" * 60)
        print(f"REPO_ROOT: {REPO_ROOT}")
        print()

        # Setup
        self._setup()

        try:
            # Core resume tests
            self.test_resume_accepts_verified_bundle()
            self.test_resume_rejects_hash_mismatch()
            self.test_resume_rejects_missing_output()
            self.test_resume_rejects_validator_mismatch()
            self.test_resume_no_history_dependency()

            # Additional edge cases
            self.test_resume_rejects_status_not_success()
            self.test_resume_rejects_cmp01_not_pass()
            self.test_resume_rejects_missing_status()
            self.test_resume_rejects_missing_task_spec()
            self.test_resume_rejects_missing_output_hashes()
        finally:
            # Cleanup
            self._cleanup()

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

    def _setup(self):
        """Create test directory."""
        self.test_root.mkdir(parents=True, exist_ok=True)

    def _cleanup(self):
        """Remove test directory."""
        shutil.rmtree(self.test_root, ignore_errors=True)

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

    def _create_valid_bundle(self, bundle_name: str) -> Path:
        """Create a valid SPECTRUM-02 bundle for testing."""
        bundle_path = self.test_root / bundle_name
        bundle_path.mkdir(parents=True, exist_ok=True)

        # Create output file
        output_rel = f"CONTRACTS/_runs/_spectrum02_test/{bundle_name}/output.txt"
        output_path = bundle_path / "output.txt"
        output_content = b"SPECTRUM-02 test output content\n"
        output_path.write_bytes(output_content)
        output_hash = "sha256:" + hashlib.sha256(output_content).hexdigest()

        # TASK_SPEC.json
        task_spec = {
            "task_id": bundle_name,
            "inputs": [],
            "expected_outputs": [output_rel],
            "constraints": {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        (bundle_path / "TASK_SPEC.json").write_text(json.dumps(task_spec, indent=2))

        # STATUS.json
        status = {
            "status": "success",
            "cmp01": "pass",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "error": None
        }
        (bundle_path / "STATUS.json").write_text(json.dumps(status, indent=2))

        # OUTPUT_HASHES.json
        output_hashes = {
            "validator_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hashes": {
                output_rel: output_hash
            }
        }
        (bundle_path / "OUTPUT_HASHES.json").write_text(json.dumps(output_hashes, indent=2))

        return bundle_path

    # =========================================================================
    # CORE RESUME TESTS
    # =========================================================================

    def test_resume_accepts_verified_bundle(self):
        """Valid SPECTRUM-02 bundle should be accepted WITHOUT history."""
        bundle_path = self._create_valid_bundle("valid-bundle")

        # Verify NO forbidden artifacts exist
        self._assert(
            not (bundle_path / "logs").exists(),
            "test_resume_accepts_verified_bundle (no logs/)",
            "logs/ directory should not exist"
        )
        self._assert(
            not (bundle_path / "tmp").exists(),
            "test_resume_accepts_verified_bundle (no tmp/)",
            "tmp/ directory should not exist"
        )
        self._assert(
            not (bundle_path / "chat_history.json").exists(),
            "test_resume_accepts_verified_bundle (no chat_history)",
            "chat_history.json should not exist"
        )
        self._assert(
            not (bundle_path / "reasoning_trace.json").exists(),
            "test_resume_accepts_verified_bundle (no reasoning_trace)",
            "reasoning_trace.json should not exist"
        )

        # Verify bundle is accepted
        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            result["valid"],
            "test_resume_accepts_verified_bundle (accepted)",
            f"Expected valid, got errors: {result['errors']}"
        )

    def test_resume_rejects_hash_mismatch(self):
        """Modified output byte should cause rejection."""
        bundle_path = self._create_valid_bundle("hash-mismatch")

        # Tamper with output file (modify one byte)
        output_path = bundle_path / "output.txt"
        original = output_path.read_bytes()
        tampered = original[:-1] + b"X"  # Change last byte
        output_path.write_bytes(tampered)

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_hash_mismatch (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "HASH_MISMATCH" in error_codes,
            "test_resume_rejects_hash_mismatch (error code)",
            f"Expected HASH_MISMATCH, got: {error_codes}"
        )

    def test_resume_rejects_missing_output(self):
        """Missing declared output should cause rejection."""
        bundle_path = self._create_valid_bundle("missing-output")

        # Remove the output file
        output_path = bundle_path / "output.txt"
        output_path.unlink()

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_missing_output (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "OUTPUT_MISSING" in error_codes,
            "test_resume_rejects_missing_output (error code)",
            f"Expected OUTPUT_MISSING, got: {error_codes}"
        )

    def test_resume_rejects_validator_mismatch(self):
        """Unsupported validator version should cause rejection."""
        bundle_path = self._create_valid_bundle("validator-mismatch")

        # Modify validator version to unsupported value
        hashes_path = bundle_path / "OUTPUT_HASHES.json"
        hashes = json.loads(hashes_path.read_text())
        hashes["validator_version"] = "99.99.99"  # Unsupported version
        hashes_path.write_text(json.dumps(hashes, indent=2))

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_validator_mismatch (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "VALIDATOR_UNSUPPORTED" in error_codes,
            "test_resume_rejects_validator_mismatch (error code)",
            f"Expected VALIDATOR_UNSUPPORTED, got: {error_codes}"
        )

    def test_resume_no_history_dependency(self):
        """
        Agent MUST resume correctly without ANY history artifacts.

        Explicitly asserts:
        - No logs directory
        - No tmp directory
        - No transcript
        """
        bundle_path = self._create_valid_bundle("no-history")

        # Explicitly verify forbidden artifacts do NOT exist
        forbidden_paths = [
            bundle_path / "logs",
            bundle_path / "tmp",
            bundle_path / "chat_history.json",
            bundle_path / "reasoning_trace.json",
            bundle_path / "checkpoints",
            bundle_path / "intermediate_state.json",
            bundle_path / "debug.log",
            bundle_path / "execution_order.json",
        ]

        for forbidden in forbidden_paths:
            self._assert(
                not forbidden.exists(),
                f"test_resume_no_history_dependency (no {forbidden.name})",
                f"{forbidden.name} should not exist"
            )

        # Verify bundle STILL accepted (no history required)
        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            result["valid"],
            "test_resume_no_history_dependency (still valid)",
            f"Expected valid without history, got errors: {result['errors']}"
        )

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_resume_rejects_status_not_success(self):
        """STATUS.status != 'success' should cause rejection."""
        bundle_path = self._create_valid_bundle("status-failure")

        # Modify status to failure
        status_path = bundle_path / "STATUS.json"
        status = json.loads(status_path.read_text())
        status["status"] = "failure"
        status_path.write_text(json.dumps(status, indent=2))

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_status_not_success (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "STATUS_NOT_SUCCESS" in error_codes,
            "test_resume_rejects_status_not_success (error code)",
            f"Expected STATUS_NOT_SUCCESS, got: {error_codes}"
        )

    def test_resume_rejects_cmp01_not_pass(self):
        """STATUS.cmp01 != 'pass' should cause rejection."""
        bundle_path = self._create_valid_bundle("cmp01-fail")

        # Modify cmp01 to fail
        status_path = bundle_path / "STATUS.json"
        status = json.loads(status_path.read_text())
        status["cmp01"] = "fail"
        status_path.write_text(json.dumps(status, indent=2))

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_cmp01_not_pass (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "CMP01_NOT_PASS" in error_codes,
            "test_resume_rejects_cmp01_not_pass (error code)",
            f"Expected CMP01_NOT_PASS, got: {error_codes}"
        )

    def test_resume_rejects_missing_status(self):
        """Missing STATUS.json should cause rejection."""
        bundle_path = self._create_valid_bundle("missing-status")

        # Remove STATUS.json
        (bundle_path / "STATUS.json").unlink()

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_missing_status (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "BUNDLE_INCOMPLETE" in error_codes,
            "test_resume_rejects_missing_status (error code)",
            f"Expected BUNDLE_INCOMPLETE, got: {error_codes}"
        )

    def test_resume_rejects_missing_task_spec(self):
        """Missing TASK_SPEC.json should cause rejection."""
        bundle_path = self._create_valid_bundle("missing-task-spec")

        # Remove TASK_SPEC.json
        (bundle_path / "TASK_SPEC.json").unlink()

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_missing_task_spec (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "BUNDLE_INCOMPLETE" in error_codes,
            "test_resume_rejects_missing_task_spec (error code)",
            f"Expected BUNDLE_INCOMPLETE, got: {error_codes}"
        )

    def test_resume_rejects_missing_output_hashes(self):
        """Missing OUTPUT_HASHES.json should cause rejection."""
        bundle_path = self._create_valid_bundle("missing-output-hashes")

        # Remove OUTPUT_HASHES.json
        (bundle_path / "OUTPUT_HASHES.json").unlink()

        verifier = SPECTRUM02Verifier(bundle_path)
        result = verifier.verify()

        self._assert(
            not result["valid"],
            "test_resume_rejects_missing_output_hashes (rejected)",
            f"Expected invalid, got valid"
        )

        error_codes = [e["code"] for e in result["errors"]]
        self._assert(
            "BUNDLE_INCOMPLETE" in error_codes,
            "test_resume_rejects_missing_output_hashes (error code)",
            f"Expected BUNDLE_INCOMPLETE, got: {error_codes}"
        )


if __name__ == "__main__":
    tester = TestSPECTRUM02Resume()
    success = tester.run_all()
    sys.exit(0 if success else 1)
