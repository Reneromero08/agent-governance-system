#!/usr/bin/env python3
"""
CMP-01 Validator Tests

Tests the path validation logic in MCP/server.py for:
- Forbidden overlap containment
- Declared output existence post-run

Run: python CATALYTIC-DPT/TESTBENCH/test_cmp01_validator.py
"""

import json
import sys
import shutil
import importlib.util
from pathlib import Path

# Direct file import for MCP/server.py
REPO_ROOT = Path(__file__).parent.parent.parent
SERVER_PATH = REPO_ROOT / "CATALYTIC-DPT" / "MCP" / "server.py"

spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)

# Aliases for convenience
MCPTerminalServer = mcp_server.MCPTerminalServer
PROJECT_ROOT = mcp_server.PROJECT_ROOT

class TestCMP01Validator:
    """Test suite for CMP-01 path validation."""

    def __init__(self):
        self.server = MCPTerminalServer()
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_all(self):
        """Run all tests."""
        print("=" * 60)
        print("CMP-01 Validator Tests")
        print("=" * 60)
        print(f"PROJECT_ROOT: {PROJECT_ROOT}")
        print()

        # Path validation tests
        self.test_traversal_rejection()
        self.test_absolute_path_rejection()
        self.test_forbidden_overlap()
        self.test_durable_outside_root()
        self.test_catalytic_outside_root()
        self.test_valid_catalytic_domain()
        self.test_valid_durable_output()
        self.test_nested_overlap_same_list()

        # Post-run output tests
        self.test_output_missing_post_run()

        # Correctness fixes tests
        self.test_path_overlap_correct_indices()
        self.test_post_run_forbidden_overlap_only_one_error()
        self.test_post_run_absolute_path_error()
        self.test_post_run_traversal_path_error()


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

    def _find_error_code(self, errors: list, code: str) -> bool:
        """Check if error list contains a specific code."""
        return any(e.get("code") == code for e in errors)

    # =========================================================================
    # PATH VALIDATION TESTS
    # =========================================================================

    def test_traversal_rejection(self):
        """Path with '..' traversal must be rejected."""
        task_spec = {
            "catalytic_domains": ["CATALYTIC-DPT/../CANON"],
            "outputs": {"durable_paths": []}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "PATH_CONTAINS_TRAVERSAL"),
            "test_traversal_rejection",
            f"Expected PATH_CONTAINS_TRAVERSAL, got: {result['errors']}"
        )

    def test_absolute_path_rejection(self):
        """Absolute path must be rejected (escapes repo)."""
        task_spec = {
            "catalytic_domains": ["/tmp/foo"],
            "outputs": {"durable_paths": []}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "PATH_ESCAPES_REPO_ROOT"),
            "test_absolute_path_rejection",
            f"Expected PATH_ESCAPES_REPO_ROOT, got: {result['errors']}"
        )

    def test_forbidden_overlap(self):
        """Path overlapping forbidden root must be rejected."""
        task_spec = {
            "catalytic_domains": [],
            "outputs": {"durable_paths": ["CANON/foo.txt"]}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "FORBIDDEN_PATH_OVERLAP"),
            "test_forbidden_overlap",
            f"Expected FORBIDDEN_PATH_OVERLAP, got: {result['errors']}"
        )

    def test_durable_outside_root(self):
        """Durable output outside DURABLE_ROOTS must be rejected."""
        task_spec = {
            "catalytic_domains": [],
            "outputs": {"durable_paths": ["TOOLS/foo.py"]}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "OUTPUT_OUTSIDE_DURABLE_ROOT"),
            "test_durable_outside_root",
            f"Expected OUTPUT_OUTSIDE_DURABLE_ROOT, got: {result['errors']}"
        )

    def test_catalytic_outside_root(self):
        """Catalytic domain outside CATALYTIC_ROOTS must be rejected."""
        task_spec = {
            "catalytic_domains": ["TOOLS/not_tmp/scratch"],
            "outputs": {"durable_paths": []}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"],
            "test_catalytic_outside_root",
            f"Expected validation failure, got: {result}"
        )

    def test_valid_catalytic_domain(self):
        """Valid catalytic domain should pass."""
        task_spec = {
            "catalytic_domains": ["CONTRACTS/_runs/_tmp/test"],
            "outputs": {"durable_paths": []}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            result["valid"],
            "test_valid_catalytic_domain",
            f"Expected valid, got errors: {result['errors']}"
        )

    def test_valid_durable_output(self):
        """Valid durable output path should pass."""
        task_spec = {
            "catalytic_domains": [],
            "outputs": {"durable_paths": ["CONTRACTS/_runs/test-run/output.json"]}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        self._assert(
            result["valid"],
            "test_valid_durable_output",
            f"Expected valid, got errors: {result['errors']}"
        )

    def test_nested_overlap_same_list(self):
        """Paths that contain each other in the same list should be flagged."""
        task_spec = {
            "catalytic_domains": [
                "CONTRACTS/_runs/_tmp/",
                "CONTRACTS/_runs/_tmp/nested/"
            ],
            "outputs": {"durable_paths": []}
        }
        result = self.server._validate_jobspec_paths(task_spec)
        # This should either fail or at minimum dedupe - depends on strictness
        # For now, we flag it as a containment issue
        self._assert(
            not result["valid"],
            "test_nested_overlap_same_list",
            f"Expected overlap failure, got: {result}"
        )

    # =========================================================================
    # POST-RUN OUTPUT EXISTENCE TESTS
    # =========================================================================

    def test_output_missing_post_run(self):
        """Declared output that doesn't exist post-run must fail."""
        # Create a temp run directory with a TASK_SPEC.json
        run_id = "test-output-missing"
        run_dir = PROJECT_ROOT / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {
            "outputs": {
                "durable_paths": [
                    f"CONTRACTS/_runs/{run_id}/should_exist.txt"
                ]
            }
        }

        try:
            # Write TASK_SPEC.json
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            # Do NOT create the output file - simulate missing output

            # Call post-run verification
            result = self.server._verify_post_run_outputs(run_id)

            self._assert(
                not result["valid"] and self._find_error_code(result["errors"], "OUTPUT_MISSING"),
                "test_output_missing_post_run",
                f"Expected OUTPUT_MISSING, got: {result['errors']}"
            )
        finally:
            # Cleanup
            shutil.rmtree(run_dir, ignore_errors=True)

    # =========================================================================
    # CORRECTNESS FIXES TESTS
    # =========================================================================

    def test_path_overlap_correct_indices(self):
        """PATH_OVERLAP should report original indices even when invalid paths are filtered."""
        # First path is invalid (will be filtered), second and third overlap
        task_spec = {
            "catalytic_domains": [
                "/absolute/invalid",  # index 0 - will be filtered (absolute)
                "CONTRACTS/_runs/_tmp/",  # index 1 - valid
                "CONTRACTS/_runs/_tmp/nested/"  # index 2 - overlaps with index 1
            ],
            "outputs": {"durable_paths": []}
        }
        result = self.server._validate_jobspec_paths(task_spec)

        # Find the PATH_OVERLAP error
        overlap_errors = [e for e in result["errors"] if e["code"] == "PATH_OVERLAP"]
        self._assert(
            len(overlap_errors) > 0,
            "test_path_overlap_correct_indices (has overlap error)",
            f"Expected PATH_OVERLAP error, got: {result['errors']}"
        )

        if overlap_errors:
            err = overlap_errors[0]
            # Check that details contain original indices 1 and 2 (not filtered indices 0 and 1)
            self._assert(
                err["details"].get("index_a") == 1 and err["details"].get("index_b") == 2,
                "test_path_overlap_correct_indices (indices)",
                f"Expected indices 1 and 2, got: {err['details']}"
            )
            # Check that path points to smaller index
            self._assert(
                err["path"] == "/catalytic_domains/1",
                "test_path_overlap_correct_indices (path pointer)",
                f"Expected /catalytic_domains/1, got: {err['path']}"
            )

    def test_post_run_forbidden_overlap_only_one_error(self):
        """Forbidden overlap in post-run should produce ONLY FORBIDDEN_PATH_OVERLAP for that entry."""
        run_id = "test-forbidden-only"
        run_dir = PROJECT_ROOT / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # CANON/foo.txt overlaps forbidden root
        task_spec = {
            "outputs": {
                "durable_paths": ["CANON/foo.txt"]
            }
        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            result = self.server._verify_post_run_outputs(run_id)

            # Should have exactly one error: FORBIDDEN_PATH_OVERLAP
            # Should NOT have OUTPUT_OUTSIDE_DURABLE_ROOT or OUTPUT_MISSING for the same entry
            self._assert(
                len(result["errors"]) == 1,
                "test_post_run_forbidden_overlap_only_one_error (count)",
                f"Expected 1 error, got {len(result['errors'])}: {result['errors']}"
            )
            self._assert(
                result["errors"][0]["code"] == "FORBIDDEN_PATH_OVERLAP",
                "test_post_run_forbidden_overlap_only_one_error (code)",
                f"Expected FORBIDDEN_PATH_OVERLAP, got: {result['errors'][0]['code']}"
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_post_run_absolute_path_error(self):
        """Post-run should report error for absolute paths instead of silently skipping."""
        run_id = "test-postrun-absolute"
        run_dir = PROJECT_ROOT / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Use Windows-style absolute path (C:\tmp\...) for cross-platform correctness
        task_spec = {
            "outputs": {
                "durable_paths": ["C:\\tmp\\absolute_path.txt"]
            }

        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            result = self.server._verify_post_run_outputs(run_id)

            self._assert(
                not result["valid"] and self._find_error_code(result["errors"], "PATH_ESCAPES_REPO_ROOT"),
                "test_post_run_absolute_path_error",
                f"Expected PATH_ESCAPES_REPO_ROOT, got: {result['errors']}"
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_post_run_traversal_path_error(self):
        """Post-run should report error for traversal paths instead of silently skipping."""
        run_id = "test-postrun-traversal"
        run_dir = PROJECT_ROOT / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {
            "outputs": {
                "durable_paths": ["CONTRACTS/../CANON/foo.txt"]
            }
        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            result = self.server._verify_post_run_outputs(run_id)

            self._assert(
                not result["valid"] and self._find_error_code(result["errors"], "PATH_CONTAINS_TRAVERSAL"),
                "test_post_run_traversal_path_error",
                f"Expected PATH_CONTAINS_TRAVERSAL, got: {result['errors']}"
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    tester = TestCMP01Validator()
    success = tester.run_all()
    sys.exit(0 if success else 1)

