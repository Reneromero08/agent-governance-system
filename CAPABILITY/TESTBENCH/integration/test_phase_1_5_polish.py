#!/usr/bin/env python3
"""
Phase 1.5 Polish: Symlink Bypass Defense, Error Receipts, and Negative Integration Tests

Tests covering:
- Symlink escape protection (repo_digest)
- Symlink/junction bypass protection (write_firewall)
- CLI error receipt emission
- Negative integration test (firewall blocks guarded_writer)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.repo_digest import (
    DigestSpec,
    RepoDigest,
    write_error_receipt,
)
from CAPABILITY.PRIMITIVES.write_firewall import WriteFirewall, FirewallViolation
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter


# ============================================================================
# Symlink Bypass Defense Tests (repo_digest)
# ============================================================================


def test_repo_digest_symlink_escape_blocked():
    """
    Test: Symlink inside repo pointing outside repo is NOT followed.

    Evidence:
    - With followlinks=False, symlinks are not traversed
    - Digest does not include files from symlink targets outside repo
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()
        outside_dir = Path(tmpdir) / "outside"
        outside_dir.mkdir()

        # Create file inside repo
        (repo_root / "inside.txt").write_text("inside", encoding="utf-8")

        # Create file outside repo
        (outside_dir / "outside.txt").write_text("outside", encoding="utf-8")

        # Create symlink inside repo pointing outside
        symlink_path = repo_root / "evil_link"
        try:
            symlink_path.symlink_to(outside_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform/config")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=[],
            tmp_roots=[],
        )

        # Compute digest
        digest = RepoDigest(spec)
        receipt = digest.compute_digest()

        # Verify symlink target is NOT included
        assert "evil_link/outside.txt" not in receipt["file_manifest"], \
            "Symlink escape: files outside repo must not be included in digest"
        assert "inside.txt" in receipt["file_manifest"], \
            "Files inside repo must be included"

        # Verify file count is 1 (only inside.txt)
        assert receipt["file_count"] == 1, "Only inside.txt should be counted"

        print("✓ Symlink escape blocked: digest does not follow symlinks outside repo")


def test_repo_digest_symlink_within_repo_not_followed():
    """
    Test: Symlinks within repo are NOT followed (followlinks=False).

    Evidence:
    - With followlinks=False, even symlinks within repo are not traversed
    - Only the symlink itself may appear (if treated as file), not target content
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        repo_root.mkdir()

        # Create subdirectory with file
        subdir = repo_root / "subdir"
        subdir.mkdir()
        (subdir / "target.txt").write_text("target content", encoding="utf-8")

        # Create symlink to subdir
        symlink_path = repo_root / "link_to_subdir"
        try:
            symlink_path.symlink_to(subdir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform/config")

        spec = DigestSpec(
            repo_root=repo_root,
            exclusions=[],
            durable_roots=[],
            tmp_roots=[],
        )

        # Compute digest
        digest = RepoDigest(spec)
        receipt = digest.compute_digest()

        # Verify symlink target is NOT followed
        # The file should appear once under subdir/, not twice
        assert "subdir/target.txt" in receipt["file_manifest"], \
            "Original file must be included"

        # Symlink itself should not cause double-counting
        file_count = receipt["file_count"]
        assert file_count == 1, f"Expected 1 file, got {file_count} (symlink should not be followed)"

        print("✓ Symlink within repo not followed: followlinks=False prevents traversal")


# ============================================================================
# Write Firewall Symlink Bypass Defense Tests
# ============================================================================


def test_write_firewall_symlink_escape_blocked():
    """
    Test: Firewall blocks writes via symlink pointing outside project_root.

    Evidence:
    - Symlink inside tmp_roots pointing outside project_root
    - Write via symlink is blocked with FIREWALL_PATH_ESCAPE
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()
        outside_dir = Path(tmpdir) / "outside"
        outside_dir.mkdir()

        # Create tmp_roots directory
        tmp_root = project_root / "_tmp"
        tmp_root.mkdir()

        # Create symlink inside tmp_roots pointing outside project_root
        evil_link = tmp_root / "evil_link"
        try:
            evil_link.symlink_to(outside_dir / "target.txt")
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform/config")

        firewall = WriteFirewall(
            tmp_roots=["_tmp"],
            durable_roots=["outputs"],
            project_root=project_root,
        )

        # Attempt write via symlink
        with pytest.raises(FirewallViolation) as exc_info:
            firewall.safe_write("_tmp/evil_link", "malicious data", kind="tmp")

        # Verify error code
        violation = exc_info.value.violation_receipt
        assert violation["error_code"] == "FIREWALL_PATH_ESCAPE", \
            "Symlink escape must be blocked with FIREWALL_PATH_ESCAPE"

        print(f"✓ Firewall symlink escape blocked: {violation['error_code']}")


def test_write_firewall_symlink_domain_crossing_blocked():
    """
    Test: Firewall blocks writes via symlink crossing from tmp to durable domain.

    Evidence:
    - Symlink inside tmp_roots pointing to durable_roots
    - Tmp write via symlink is blocked with FIREWALL_TMP_WRITE_WRONG_DOMAIN
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()

        # Create domains
        tmp_root = project_root / "_tmp"
        tmp_root.mkdir()
        durable_root = project_root / "outputs"
        durable_root.mkdir()

        # Create symlink inside tmp_roots pointing to durable_roots
        sneaky_link = tmp_root / "link_to_durable"
        try:
            sneaky_link.symlink_to(durable_root / "file.txt")
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform/config")

        firewall = WriteFirewall(
            tmp_roots=["_tmp"],
            durable_roots=["outputs"],
            project_root=project_root,
        )

        # Attempt tmp write via symlink to durable domain
        with pytest.raises(FirewallViolation) as exc_info:
            firewall.safe_write("_tmp/link_to_durable", "sneaky data", kind="tmp")

        # Verify error code
        violation = exc_info.value.violation_receipt
        assert violation["error_code"] == "FIREWALL_TMP_WRITE_WRONG_DOMAIN", \
            "Symlink domain crossing must be blocked"

        print(f"✓ Firewall domain crossing blocked: {violation['error_code']}")


# ============================================================================
# CLI Error Receipt Tests
# ============================================================================


def test_cli_error_receipt_emission():
    """
    Test: write_error_receipt emits valid error receipt on exception.

    Evidence:
    - Error receipt written to specified path
    - Receipt contains all required fields (error_code, exception_type, config_snapshot, etc.)
    - Receipt is valid JSON
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        receipt_path = Path(tmpdir) / "error.json"

        config_snapshot = {
            "repo_root": "/tmp/repo",
            "exclusions": [".git"],
            "durable_roots": ["outputs"],
            "tmp_roots": ["_tmp"],
        }

        exception = ValueError("Test error message")

        # Write error receipt
        write_error_receipt(
            operation="test_operation",
            exception=exception,
            error_code="TEST_ERROR_CODE",
            config_snapshot=config_snapshot,
            output_path=receipt_path,
        )

        # Verify receipt exists
        assert receipt_path.exists(), "Error receipt must be written"

        # Load and verify receipt
        receipt = json.loads(receipt_path.read_text())

        assert receipt["verdict"] == "ERROR"
        assert receipt["error_code"] == "TEST_ERROR_CODE"
        assert receipt["operation"] == "test_operation"
        assert receipt["exception_type"] == "ValueError"
        assert receipt["exception_message"] == "Test error message"
        assert "module_version" in receipt
        assert "module_version_hash" in receipt
        assert receipt["config_snapshot"] == config_snapshot

        print(f"✓ CLI error receipt emitted: {receipt['error_code']}")


# ============================================================================
# Negative Integration Test: Firewall Blocks GuardedWriter
# ============================================================================


def test_negative_integration_guarded_writer_blocked_by_firewall():
    """
    Test: GuardedWriter (integration wrapper) is blocked by firewall on illegal write.

    Evidence:
    - GuardedWriter attempts write outside allowed domains
    - Firewall blocks the write
    - FirewallViolation raised with violation receipt
    - This demonstrates that existing integration tools respect firewall policy
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()

        # Initialize GuardedWriter with restricted domains
        writer = GuardedWriter(
            project_root=project_root,
            tmp_roots=["_tmp"],
            durable_roots=["outputs"],
            exclusions=["protected"],
        )

        # Attempt 1: Write to excluded path (should fail)
        protected_dir = project_root / "protected"
        protected_dir.mkdir()

        with pytest.raises(FirewallViolation) as exc_info:
            writer.write_tmp("protected/file.txt", "data")

        violation1 = exc_info.value.violation_receipt
        assert violation1["error_code"] == "FIREWALL_PATH_EXCLUDED", \
            "Write to excluded path must be blocked"

        # Attempt 2: Write to path not in any domain (should fail)
        with pytest.raises(FirewallViolation) as exc_info:
            writer.write_tmp("random_dir/file.txt", "data")

        violation2 = exc_info.value.violation_receipt
        assert violation2["error_code"] == "FIREWALL_PATH_NOT_IN_DOMAIN", \
            "Write outside domains must be blocked"

        # Attempt 3: Durable write before commit gate (should fail)
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir()

        with pytest.raises(FirewallViolation) as exc_info:
            writer.write_durable("outputs/result.json", "data")

        violation3 = exc_info.value.violation_receipt
        assert violation3["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT", \
            "Durable write before commit gate must be blocked"

        # Verify all violations have complete receipts
        for violation in [violation1, violation2, violation3]:
            assert violation["verdict"] == "FAIL"
            assert "policy_snapshot" in violation
            assert "tool_version_hash" in violation

        print("✓ Negative integration test: GuardedWriter correctly blocked by firewall")
        print(f"  - Blocked excluded path: {violation1['error_code']}")
        print(f"  - Blocked path not in domain: {violation2['error_code']}")
        print(f"  - Blocked durable write before commit: {violation3['error_code']}")


# ============================================================================
# Test Runner
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
