#!/usr/bin/env python3
"""
CMP-01 Validator Tests

Tests the live path validation logic in CAPABILITY/MCP/validation.py for:
- Forbidden overlap containment
- Declared output existence post-run

Historical note: these tests originally targeted the deprecated prototype
server (THOUGHT/DEPRECATED/MCP_EXPERIMENTAL/server_CATDPT.py). They now run
against the canonical module. Three prototype-only lifecycle tests
(skill_complete / TASK_SPEC.sha256 integrity / STATUS.json) were removed
because that ceremony has no equivalent in the live server.

Run: python CAPABILITY/TESTBENCH/core/test_cmp01_validator.py
"""

import json
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.MCP import validation as mcp_validation
from CAPABILITY.MCP.validation import (
    validate_jobspec_paths,
    verify_post_run_outputs,
)

PROJECT_ROOT = mcp_validation.PROJECT_ROOT


class RunnerCMP01Validator:
    """Test suite for CMP-01 path validation."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_all(self):
        """Run all tests."""
        print("=" * 60)
        print("CMP-01 Validator Tests (live CAPABILITY/MCP/validation.py)")
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

        # Audit hardening tests
        self.test_duplicate_paths_allowed()
        self.test_near_prefix_no_overlap()

        # Symlink escape test
        self.test_symlink_escape_rejected_pre_and_post()

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
            "catalytic_domains": ["CAPABILITY/../LAW/CANON"],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)
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
        result = validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "PATH_ESCAPES_REPO_ROOT"),
            "test_absolute_path_rejection",
            f"Expected PATH_ESCAPES_REPO_ROOT, got: {result['errors']}"
        )

    def test_forbidden_overlap(self):
        """Path overlapping forbidden root must be rejected."""
        task_spec = {
            "catalytic_domains": [],
            "outputs": {"durable_paths": ["LAW/CANON/foo.txt"]}
        }
        result = validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "FORBIDDEN_PATH_OVERLAP"),
            "test_forbidden_overlap",
            f"Expected FORBIDDEN_PATH_OVERLAP, got: {result['errors']}"
        )

    def test_durable_outside_root(self):
        """Durable output outside DURABLE_ROOTS must be rejected."""
        task_spec = {
            "catalytic_domains": [],
            "outputs": {"durable_paths": ["CAPABILITY/TOOLS/foo.py"]}
        }
        result = validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"] and self._find_error_code(result["errors"], "OUTPUT_OUTSIDE_DURABLE_ROOT"),
            "test_durable_outside_root",
            f"Expected OUTPUT_OUTSIDE_DURABLE_ROOT, got: {result['errors']}"
        )

    def test_catalytic_outside_root(self):
        """Catalytic domain outside CATALYTIC_ROOTS must be rejected."""
        task_spec = {
            "catalytic_domains": ["CAPABILITY/TOOLS/not_tmp/scratch"],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)
        self._assert(
            not result["valid"],
            "test_catalytic_outside_root",
            f"Expected validation failure, got: {result}"
        )

    def test_valid_catalytic_domain(self):
        """Valid catalytic domain should pass."""
        task_spec = {
            "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/test"],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)
        self._assert(
            result["valid"],
            "test_valid_catalytic_domain",
            f"Expected valid, got errors: {result['errors']}"
        )

    def test_valid_durable_output(self):
        """Valid durable output path should pass."""
        task_spec = {
            "catalytic_domains": [],
            "outputs": {"durable_paths": ["LAW/CONTRACTS/_runs/test-run/output.json"]}
        }
        result = validate_jobspec_paths(task_spec)
        self._assert(
            result["valid"],
            "test_valid_durable_output",
            f"Expected valid, got errors: {result['errors']}"
        )

    def test_nested_overlap_same_list(self):
        """Paths that contain each other in the same list should be flagged."""
        task_spec = {
            "catalytic_domains": [
                "LAW/CONTRACTS/_runs/_tmp/",
                "LAW/CONTRACTS/_runs/_tmp/nested/"
            ],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)
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
        run_id = "test-output-missing"
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {
            "outputs": {
                "durable_paths": [
                    f"LAW/CONTRACTS/_runs/{run_id}/should_exist.txt"
                ]
            }
        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            # Do NOT create the output file - simulate missing output
            result = verify_post_run_outputs(run_id)

            self._assert(
                not result["valid"] and self._find_error_code(result["errors"], "OUTPUT_MISSING"),
                "test_output_missing_post_run",
                f"Expected OUTPUT_MISSING, got: {result['errors']}"
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    # =========================================================================
    # CORRECTNESS FIXES TESTS
    # =========================================================================

    def test_path_overlap_correct_indices(self):
        """PATH_OVERLAP should report original indices even when invalid paths are filtered."""
        task_spec = {
            "catalytic_domains": [
                "/absolute/invalid",  # index 0 - will be filtered (absolute)
                "LAW/CONTRACTS/_runs/_tmp/",  # index 1 - valid
                "LAW/CONTRACTS/_runs/_tmp/nested/"  # index 2 - overlaps with index 1
            ],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)

        overlap_errors = [e for e in result["errors"] if e["code"] == "PATH_OVERLAP"]
        self._assert(
            len(overlap_errors) > 0,
            "test_path_overlap_correct_indices (has overlap error)",
            f"Expected PATH_OVERLAP error, got: {result['errors']}"
        )

        if overlap_errors:
            err = overlap_errors[0]
            self._assert(
                err["details"].get("index_a") == 1 and err["details"].get("index_b") == 2,
                "test_path_overlap_correct_indices (indices)",
                f"Expected indices 1 and 2, got: {err['details']}"
            )
            self._assert(
                err["path"] == "/catalytic_domains/1",
                "test_path_overlap_correct_indices (path pointer)",
                f"Expected /catalytic_domains/1, got: {err['path']}"
            )

    def test_post_run_forbidden_overlap_only_one_error(self):
        """Forbidden overlap in post-run should produce ONLY FORBIDDEN_PATH_OVERLAP for that entry."""
        run_id = "test-forbidden-only"
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {
            "outputs": {
                "durable_paths": ["LAW/CANON/foo.txt"]
            }
        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            result = verify_post_run_outputs(run_id)

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
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Windows-style absolute path for cross-platform correctness
        task_spec = {
            "outputs": {
                "durable_paths": ["C:\\tmp\\absolute_path.txt"]
            }
        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            result = verify_post_run_outputs(run_id)

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
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {
            "outputs": {
                "durable_paths": ["LAW/CONTRACTS/../CANON/foo.txt"]
            }
        }

        try:
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(task_spec, f)

            result = verify_post_run_outputs(run_id)

            self._assert(
                not result["valid"] and self._find_error_code(result["errors"], "PATH_CONTAINS_TRAVERSAL"),
                "test_post_run_traversal_path_error",
                f"Expected PATH_CONTAINS_TRAVERSAL, got: {result['errors']}"
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    # =========================================================================
    # AUDIT HARDENING TESTS
    # =========================================================================

    def test_duplicate_paths_allowed(self):
        """Exact duplicate paths should be allowed (no PATH_OVERLAP)."""
        task_spec = {
            "catalytic_domains": [
                "LAW/CONTRACTS/_runs/_tmp/",
                "LAW/CONTRACTS/_runs/_tmp/"  # exact duplicate
            ],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)

        overlap_errors = [e for e in result["errors"] if e["code"] == "PATH_OVERLAP"]
        self._assert(
            len(overlap_errors) == 0,
            "test_duplicate_paths_allowed",
            f"Expected no PATH_OVERLAP for exact duplicates, got: {overlap_errors}"
        )

    def test_near_prefix_no_overlap(self):
        """Paths like 'a' and 'ab' should not be considered overlapping."""
        task_spec = {
            "catalytic_domains": [
                "LAW/CONTRACTS/_runs/_tmp/a",
                "LAW/CONTRACTS/_runs/_tmp/ab"  # near-prefix, not overlap
            ],
            "outputs": {"durable_paths": []}
        }
        result = validate_jobspec_paths(task_spec)

        overlap_errors = [e for e in result["errors"] if e["code"] == "PATH_OVERLAP"]
        self._assert(
            len(overlap_errors) == 0,
            "test_near_prefix_no_overlap",
            f"Expected no PATH_OVERLAP for near-prefix paths, got: {overlap_errors}"
        )

    def test_symlink_escape_rejected_pre_and_post(self):
        """Symlink inside allowed root pointing outside PROJECT_ROOT should be rejected."""
        import tempfile
        import os

        # Skip on Windows if symlinks require elevated permissions
        if os.name == "nt":
            try:
                with tempfile.TemporaryDirectory() as test_dir:
                    test_link = Path(test_dir) / "test_link"
                    test_target = Path(test_dir) / "test_target"
                    test_target.mkdir()
                    test_link.symlink_to(test_target, target_is_directory=True)
            except (OSError, NotImplementedError, PermissionError) as e:
                print(f"[SKIP] test_symlink_escape_rejected_pre_and_post: Symlink creation not permitted on Windows ({e})")
                return

        with tempfile.TemporaryDirectory() as tmp_root:
            fake_repo_root = Path(tmp_root) / "repo"
            outside_root = Path(tmp_root) / "outside"

            fake_repo_root.mkdir()
            outside_root.mkdir()
            contracts_runs = fake_repo_root / "LAW" / "CONTRACTS" / "_runs"
            contracts_runs.mkdir(parents=True)

            link_path = contracts_runs / "link"
            try:
                link_path.symlink_to(outside_root, target_is_directory=True)
            except (OSError, PermissionError) as e:
                print(f"[SKIP] test_symlink_escape_rejected_pre_and_post: {e}")
                return

            # Save original module globals
            old_project_root = mcp_validation.PROJECT_ROOT
            old_contracts_dir = mcp_validation.CONTRACTS_DIR

            try:
                # Monkeypatch the validation module's roots; its functions
                # read these globals at call time.
                mcp_validation.PROJECT_ROOT = fake_repo_root
                mcp_validation.CONTRACTS_DIR = contracts_runs

                # === PRE-RUN VALIDATION TEST ===
                task_spec = {
                    "catalytic_domains": [],
                    "outputs": {"durable_paths": ["LAW/CONTRACTS/_runs/link/out.txt"]}
                }
                result = validate_jobspec_paths(task_spec)

                self._assert(
                    not result["valid"],
                    "test_symlink_escape_rejected_pre_and_post (pre-run invalid)",
                    f"Expected invalid, got: {result}"
                )

                escape_errors = [e for e in result["errors"] if e["code"] == "PATH_ESCAPES_REPO_ROOT"]
                self._assert(
                    len(escape_errors) > 0,
                    "test_symlink_escape_rejected_pre_and_post (pre-run PATH_ESCAPES_REPO_ROOT)",
                    f"Expected PATH_ESCAPES_REPO_ROOT, got: {result['errors']}"
                )

                # === POST-RUN VALIDATION TEST ===
                run_id = "test-symlink-escape"
                run_dir = contracts_runs / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                post_run_task_spec = {
                    "outputs": {"durable_paths": ["LAW/CONTRACTS/_runs/link/out.txt"]}
                }
                with open(run_dir / "TASK_SPEC.json", "w") as f:
                    json.dump(post_run_task_spec, f)

                post_result = verify_post_run_outputs(run_id)

                self._assert(
                    not post_result["valid"],
                    "test_symlink_escape_rejected_pre_and_post (post-run invalid)",
                    f"Expected invalid, got: {post_result}"
                )

                post_escape_errors = [e for e in post_result["errors"] if e["code"] == "PATH_ESCAPES_REPO_ROOT"]
                self._assert(
                    len(post_escape_errors) > 0,
                    "test_symlink_escape_rejected_pre_and_post (post-run PATH_ESCAPES_REPO_ROOT)",
                    f"Expected PATH_ESCAPES_REPO_ROOT, got: {post_result['errors']}"
                )

            finally:
                # Restore original globals
                mcp_validation.PROJECT_ROOT = old_project_root
                mcp_validation.CONTRACTS_DIR = old_contracts_dir


def test_cmp01_validator():
    """Pytest entry point."""
    runner = RunnerCMP01Validator()
    assert runner.run_all()


if __name__ == "__main__":
    runner = RunnerCMP01Validator()
    success = runner.run_all()
    sys.exit(0 if success else 1)
