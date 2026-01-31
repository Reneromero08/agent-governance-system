#!/usr/bin/env python3
"""
Tests for CatalyticRuntime unit functionality.

Covers:
- Preflight validation of forbidden paths
- Snapshot operations including symlink rejection
- Restoration verification mismatch detection
- Error code usage
"""

import json
import tempfile
import os
from pathlib import Path
import pytest
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.catalytic.catalytic_runtime import (
    CatalyticSnapshot,
    CatalyticRuntime,
    PROJECT_ROOT,
)
from CAPABILITY.PRIMITIVES.catalytic_errors import (
    CatalyticError,
    CAT_005_DOMAIN_VIOLATION,
)


class TestCatalyticSnapshot:
    """Test suite for CatalyticSnapshot."""

    def test_capture_empty_domain(self, tmp_path: Path) -> None:
        """Empty domain should produce empty snapshot."""
        domain = tmp_path / "empty_domain"
        domain.mkdir()

        snapshot = CatalyticSnapshot(domain)
        snapshot.capture()

        assert snapshot.to_dict() == {}

    def test_capture_with_files(self, tmp_path: Path) -> None:
        """Domain with files should capture all file hashes."""
        domain = tmp_path / "domain"
        domain.mkdir()
        (domain / "file1.txt").write_text("content1")
        (domain / "file2.txt").write_text("content2")

        snapshot = CatalyticSnapshot(domain)
        snapshot.capture()

        files = snapshot.to_dict()
        assert len(files) == 2
        assert "file1.txt" in files
        assert "file2.txt" in files
        # Verify hashes are 64 char hex strings
        for h in files.values():
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_capture_nested_directories(self, tmp_path: Path) -> None:
        """Nested directories should be captured with normalized paths."""
        domain = tmp_path / "domain"
        domain.mkdir()
        subdir = domain / "sub" / "nested"
        subdir.mkdir(parents=True)
        (subdir / "deep.txt").write_text("deep content")

        snapshot = CatalyticSnapshot(domain)
        snapshot.capture()

        files = snapshot.to_dict()
        assert len(files) == 1
        # Should use forward slashes
        assert "sub/nested/deep.txt" in files

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require admin on Windows")
    def test_capture_rejects_symlinks(self, tmp_path: Path) -> None:
        """Symlinks in domain should raise CatalyticError."""
        domain = tmp_path / "domain"
        domain.mkdir()
        real_file = domain / "real.txt"
        real_file.write_text("real content")
        symlink = domain / "link.txt"
        symlink.symlink_to(real_file)

        snapshot = CatalyticSnapshot(domain)

        with pytest.raises(CatalyticError) as exc_info:
            snapshot.capture()

        assert exc_info.value.code == CAT_005_DOMAIN_VIOLATION
        assert "Symlink not allowed" in exc_info.value.message

    def test_capture_nonexistent_domain(self, tmp_path: Path) -> None:
        """Nonexistent domain should produce empty snapshot without error."""
        domain = tmp_path / "nonexistent"

        snapshot = CatalyticSnapshot(domain)
        snapshot.capture()

        assert snapshot.to_dict() == {}

    def test_diff_identical_snapshots(self, tmp_path: Path) -> None:
        """Identical snapshots should produce empty diff."""
        domain = tmp_path / "domain"
        domain.mkdir()
        (domain / "file.txt").write_text("content")

        snapshot1 = CatalyticSnapshot(domain)
        snapshot1.capture()

        snapshot2 = CatalyticSnapshot(domain)
        snapshot2.capture()

        diff = snapshot1.diff(snapshot2)

        assert diff["added"] == {}
        assert diff["removed"] == {}
        assert diff["changed"] == {}

    def test_diff_added_file(self, tmp_path: Path) -> None:
        """Added file should appear in diff."""
        domain = tmp_path / "domain"
        domain.mkdir()
        (domain / "original.txt").write_text("original")

        snapshot1 = CatalyticSnapshot(domain)
        snapshot1.capture()

        # Add a new file
        (domain / "new.txt").write_text("new content")

        snapshot2 = CatalyticSnapshot(domain)
        snapshot2.capture()

        diff = snapshot1.diff(snapshot2)

        assert "new.txt" in diff["added"]
        assert diff["removed"] == {}
        assert diff["changed"] == {}

    def test_diff_removed_file(self, tmp_path: Path) -> None:
        """Removed file should appear in diff."""
        domain = tmp_path / "domain"
        domain.mkdir()
        file_to_remove = domain / "to_remove.txt"
        file_to_remove.write_text("will be removed")
        (domain / "stays.txt").write_text("stays")

        snapshot1 = CatalyticSnapshot(domain)
        snapshot1.capture()

        # Remove the file
        file_to_remove.unlink()

        snapshot2 = CatalyticSnapshot(domain)
        snapshot2.capture()

        diff = snapshot1.diff(snapshot2)

        assert diff["added"] == {}
        assert "to_remove.txt" in diff["removed"]
        assert diff["changed"] == {}

    def test_diff_changed_file(self, tmp_path: Path) -> None:
        """Changed file should appear in diff."""
        domain = tmp_path / "domain"
        domain.mkdir()
        changing_file = domain / "changing.txt"
        changing_file.write_text("original content")

        snapshot1 = CatalyticSnapshot(domain)
        snapshot1.capture()

        # Modify the file
        changing_file.write_text("modified content")

        snapshot2 = CatalyticSnapshot(domain)
        snapshot2.capture()

        diff = snapshot1.diff(snapshot2)

        assert diff["added"] == {}
        assert diff["removed"] == {}
        assert "changing.txt" in diff["changed"]
        assert "before" in diff["changed"]["changing.txt"]
        assert "after" in diff["changed"]["changing.txt"]

    def test_from_dict_round_trip(self) -> None:
        """Snapshot should round-trip through dict serialization."""
        original = CatalyticSnapshot(Path("."))
        original.files = {
            "file1.txt": "a" * 64,
            "sub/file2.txt": "b" * 64,
        }

        exported = original.to_dict()
        restored = CatalyticSnapshot.from_dict(exported)

        assert restored.files == original.files


class TestCatalyticRuntimePreflight:
    """Test preflight validation in CatalyticRuntime."""

    def test_preflight_rejects_forbidden_paths(self) -> None:
        """Preflight should reject catalytic domains touching forbidden paths."""
        runtime = CatalyticRuntime(
            run_id="test-forbidden",
            catalytic_domains=["LAW/CANON"],  # Forbidden!
            durable_outputs=["LAW/CONTRACTS/_runs/test"],
            intent="Test forbidden path rejection",
        )

        valid, errors = runtime.preflight_validate()

        assert valid is False
        assert any("PATH_FORBIDDEN" in str(e) or "PATH_NOT_ALLOWED" in str(e) for e in errors)

    def test_preflight_accepts_valid_domains(self) -> None:
        """Preflight should accept valid catalytic domains."""
        runtime = CatalyticRuntime(
            run_id="test-valid",
            catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
            durable_outputs=["LAW/CONTRACTS/_runs/test-valid"],
            intent="Test valid domain acceptance",
        )

        valid, errors = runtime.preflight_validate()

        assert valid is True, f"Expected valid, got errors: {errors}"


class TestCatalyticErrors:
    """Test structured error codes."""

    def test_catalytic_error_basic(self) -> None:
        """CatalyticError should have code, message, and details."""
        error = CatalyticError("CAT-001", "Test error", {"key": "value"})

        assert error.code == "CAT-001"
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert "[CAT-001]" in str(error)

    def test_catalytic_error_to_dict(self) -> None:
        """CatalyticError should serialize to dict."""
        error = CatalyticError("CAT-002", "Schema error", {"field": "proof_version"})

        d = error.to_dict()

        assert d["code"] == "CAT-002"
        assert d["message"] == "Schema error"
        assert d["details"]["field"] == "proof_version"


class TestMasterOverrideRemoved:
    """Verify MASTER_OVERRIDE_PREFLIGHT bypass is removed."""

    def test_no_master_override_in_preflight(self) -> None:
        """MASTER_OVERRIDE_PREFLIGHT should not bypass validation."""
        # Set the environment variable that would have bypassed validation
        old_val = os.environ.get("MASTER_OVERRIDE_PREFLIGHT")
        os.environ["MASTER_OVERRIDE_PREFLIGHT"] = "1"

        try:
            runtime = CatalyticRuntime(
                run_id="test-override-removed",
                catalytic_domains=["LAW/CANON"],  # Forbidden - should still fail
                durable_outputs=["LAW/CONTRACTS/_runs/test"],
                intent="Test override removal",
            )

            valid, errors = runtime.preflight_validate()

            # Should FAIL even with MASTER_OVERRIDE_PREFLIGHT=1
            assert valid is False, "MASTER_OVERRIDE bypass should be removed!"
        finally:
            # Restore original value
            if old_val is None:
                os.environ.pop("MASTER_OVERRIDE_PREFLIGHT", None)
            else:
                os.environ["MASTER_OVERRIDE_PREFLIGHT"] = old_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
