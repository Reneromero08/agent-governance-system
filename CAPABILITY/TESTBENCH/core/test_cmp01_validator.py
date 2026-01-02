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
# Direct file import for MCP/server.py (Testing the Prototype Server Logic for now)
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "MCP_EXPERIMENTAL" / "server_CATDPT.py"

spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server)

# Aliases for convenience
MCPTerminalServer = mcp_server.MCPTerminalServer
PROJECT_ROOT = mcp_server.PROJECT_ROOT

class RunnerCMP01Validator:
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

        # Audit hardening tests
        self.test_duplicate_paths_allowed()
        self.test_near_prefix_no_overlap()
        self.test_task_spec_integrity_success()
        self.test_task_spec_tampered()
        self.test_status_json_on_cmp01_failure()

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
            "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/test"],
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
            "outputs": {"durable_paths": ["LAW/CONTRACTS/_runs/test-run/output.json"]}
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
                "LAW/CONTRACTS/_runs/_tmp/",
                "LAW/CONTRACTS/_runs/_tmp/nested/"
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
                "LAW/CONTRACTS/_runs/_tmp/",  # index 1 - valid
                "LAW/CONTRACTS/_runs/_tmp/nested/"  # index 2 - overlaps with index 1
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
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
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
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
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

            result = self.server._verify_post_run_outputs(run_id)

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
        result = self.server._validate_jobspec_paths(task_spec)
        
        # Should NOT have PATH_OVERLAP for exact duplicates
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
        result = self.server._validate_jobspec_paths(task_spec)
        
        overlap_errors = [e for e in result["errors"] if e["code"] == "PATH_OVERLAP"]
        self._assert(
            len(overlap_errors) == 0,
            "test_near_prefix_no_overlap",
            f"Expected no PATH_OVERLAP for near-prefix paths, got: {overlap_errors}"
        )

    def test_task_spec_integrity_success(self):
        """TASK_SPEC hash match should pass verification."""
        run_id = "test-integrity-ok"
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {"outputs": {"durable_paths": []}}
        task_spec_bytes = json.dumps(task_spec, indent=2).encode('utf-8')
        
        try:
            # Write TASK_SPEC.json and hash (simulating execute_skill)
            with open(run_dir / "TASK_SPEC.json", "wb") as f:
                f.write(task_spec_bytes)
            import hashlib
            with open(run_dir / "TASK_SPEC.sha256", "w") as f:
                f.write(hashlib.sha256(task_spec_bytes).hexdigest())

            # Call skill_complete - should succeed
            result = self.server.skill_complete(run_id, "success", {})
            
            self._assert(
                result["status"] == "success",
                "test_task_spec_integrity_success",
                f"Expected success, got: {result}"
            )
            
            # Check STATUS.json exists with cmp01=pass
            status_path = run_dir / "STATUS.json"
            self._assert(
                status_path.exists(),
                "test_task_spec_integrity_success (STATUS.json exists)",
                "STATUS.json not found"
            )
            if status_path.exists():
                with open(status_path) as f:
                    status_data = json.load(f)
                self._assert(
                    status_data.get("cmp01") == "pass",
                    "test_task_spec_integrity_success (cmp01=pass)",
                    f"Expected cmp01=pass, got: {status_data}"
                )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_task_spec_tampered(self):
        """Modified TASK_SPEC.json should fail with TASK_SPEC_TAMPERED."""
        run_id = "test-integrity-fail"
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {"outputs": {"durable_paths": []}}
        task_spec_bytes = json.dumps(task_spec, indent=2).encode('utf-8')
        
        try:
            # Write TASK_SPEC.json and hash
            with open(run_dir / "TASK_SPEC.json", "wb") as f:
                f.write(task_spec_bytes)
            import hashlib
            with open(run_dir / "TASK_SPEC.sha256", "w") as f:
                f.write(hashlib.sha256(task_spec_bytes).hexdigest())

            # TAMPER: modify TASK_SPEC.json
            tampered = {"outputs": {"durable_paths": []}, "tampered": True}
            with open(run_dir / "TASK_SPEC.json", "w") as f:
                json.dump(tampered, f)

            # Call skill_complete - should fail
            result = self.server.skill_complete(run_id, "success", {})
            
            self._assert(
                result["status"] == "error" and self._find_error_code(result.get("errors", []), "TASK_SPEC_TAMPERED"),
                "test_task_spec_tampered",
                f"Expected TASK_SPEC_TAMPERED error, got: {result}"
            )
            
            # Check STATUS.json exists with cmp01=fail
            status_path = run_dir / "STATUS.json"
            if status_path.exists():
                with open(status_path) as f:
                    status_data = json.load(f)
                self._assert(
                    status_data.get("cmp01") == "fail",
                    "test_task_spec_tampered (STATUS.json cmp01=fail)",
                    f"Expected cmp01=fail, got: {status_data}"
                )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_status_json_on_cmp01_failure(self):
        """CMP-01 output verification failure should write STATUS.json with cmp01=fail."""
        run_id = "test-status-cmp01-fail"
        run_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_spec = {
            "outputs": {
                "durable_paths": [f"LAW/CONTRACTS/_runs/{run_id}/missing.txt"]
            }
        }
        task_spec_bytes = json.dumps(task_spec, indent=2).encode('utf-8')
        
        try:
            with open(run_dir / "TASK_SPEC.json", "wb") as f:
                f.write(task_spec_bytes)
            import hashlib
            with open(run_dir / "TASK_SPEC.sha256", "w") as f:
                f.write(hashlib.sha256(task_spec_bytes).hexdigest())

            # Don't create the declared output - trigger OUTPUT_MISSING
            result = self.server.skill_complete(run_id, "success", {})
            
            self._assert(
                result["status"] == "error",
                "test_status_json_on_cmp01_failure (status=error)",
                f"Expected error status, got: {result}"
            )
            
            # Check STATUS.json
            status_path = run_dir / "STATUS.json"
            self._assert(
                status_path.exists(),
                "test_status_json_on_cmp01_failure (STATUS.json exists)",
                "STATUS.json not found"
            )
            if status_path.exists():
                with open(status_path) as f:
                    status_data = json.load(f)
                self._assert(
                    status_data.get("cmp01") == "fail" and status_data.get("status") == "error",
                    "test_status_json_on_cmp01_failure (cmp01=fail, status=error)",
                    f"Expected cmp01=fail and status=error, got: {status_data}"
                )
            
            # Check ERRORS.json exists
            errors_path = run_dir / "ERRORS.json"
            self._assert(
                errors_path.exists(),
                "test_status_json_on_cmp01_failure (ERRORS.json exists)",
                "ERRORS.json not found"
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_symlink_escape_rejected_pre_and_post(self):
        """Symlink inside allowed root pointing outside PROJECT_ROOT should be rejected."""
        import tempfile
        import os
        
        # Skip on Windows if symlinks require elevated permissions
        if os.name == "nt":
            try:
                # Test if we can create symlinks
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
            
            # Build minimal repo structure
            fake_repo_root.mkdir()
            outside_root.mkdir()
            contracts_runs = fake_repo_root / "LAW" / "CONTRACTS" / "_runs"
            contracts_runs.mkdir(parents=True)
            
            # Create symlink inside allowed root pointing OUTSIDE repo
            link_path = contracts_runs / "link"
            try:
                link_path.symlink_to(outside_root, target_is_directory=True)
            except (OSError, PermissionError) as e:
                print(f"[SKIP] test_symlink_escape_rejected_pre_and_post: {e}")
                return

            # Save original module globals
            old_project_root = mcp_server.PROJECT_ROOT
            old_contracts_dir = mcp_server.CONTRACTS_DIR

            try:
                # Monkeypatch PROJECT_ROOT and CONTRACTS_DIR
                mcp_server.PROJECT_ROOT = fake_repo_root
                mcp_server.CONTRACTS_DIR = contracts_runs

                # Create a fresh server instance (uses module globals)
                test_server = MCPTerminalServer()

                # === PRE-RUN VALIDATION TEST ===
                task_spec = {
                    "catalytic_domains": [],
                    "outputs": {"durable_paths": ["LAW/CONTRACTS/_runs/link/out.txt"]}
                }
                result = test_server._validate_jobspec_paths(task_spec)
                
                self._assert(
                    not result["valid"],
                    "test_symlink_escape_rejected_pre_and_post (pre-run invalid)",
                    f"Expected invalid, got: {result}"
                )
                
                # Check for PATH_ESCAPES_REPO_ROOT
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

                post_result = test_server._verify_post_run_outputs(run_id)
                
                self._assert(
                    not post_result["valid"],
                    "test_symlink_escape_rejected_pre_and_post (post-run invalid)",
                    f"Expected invalid, got: {post_result}"
                )
                
                # Check for PATH_ESCAPES_REPO_ROOT (should be rejected before existence check)
                post_escape_errors = [e for e in post_result["errors"] if e["code"] == "PATH_ESCAPES_REPO_ROOT"]
                self._assert(
                    len(post_escape_errors) > 0,
                    "test_symlink_escape_rejected_pre_and_post (post-run PATH_ESCAPES_REPO_ROOT)",
                    f"Expected PATH_ESCAPES_REPO_ROOT, got: {post_result['errors']}"
                )

            finally:
                # Restore original globals
                mcp_server.PROJECT_ROOT = old_project_root
                mcp_server.CONTRACTS_DIR = old_contracts_dir


def test_cmp01_validator():
    """Pytest entry point."""
    runner = RunnerCMP01Validator()
    assert runner.run_all()


if __name__ == "__main__":
    runner = RunnerCMP01Validator()
    success = runner.run_all()
    sys.exit(0 if success else 1)

