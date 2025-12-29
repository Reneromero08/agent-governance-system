"""
Test runtime write guard (Layer 2).

Validates that runtime writes are enforced at write-time:
- Allowed writes under allowed roots succeed
- Writes to forbidden paths fail closed
- Writes outside allowed roots fail closed
- Path traversal/escape attempts are detected

Run:
    pytest CATALYTIC-DPT/TESTBENCH/test_runtime_guard.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add CATALYTIC-DPT to path
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_path / "CATALYTIC-DPT"))

from PRIMITIVES.fs_guard import FilesystemGuard


@pytest.fixture
def project_root():
    """Create a temporary project root for testing."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fs_guard(project_root):
    """Create filesystem guard with standard allowed roots."""
    return FilesystemGuard(
        allowed_roots=[
            "CONTRACTS/_runs",
            "CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
            "CATALYTIC-DPT/_scratch",
        ],
        forbidden_paths=["CANON", "AGENTS.md", "BUILD", ".git"],
        project_root=project_root,
    )


def test_allowed_write_succeeds(fs_guard, project_root):
    """Test that write to allowed root succeeds."""
    # Create parent directory
    (project_root / "CONTRACTS" / "_runs" / "test").mkdir(parents=True)

    # This should succeed
    valid, error = fs_guard.validate_write_path("CONTRACTS/_runs/test/output.json")

    assert valid is True
    assert error is None


def test_guarded_write_text_succeeds(fs_guard, project_root):
    """Test that guarded_write_text succeeds for allowed path."""
    # Create parent directory
    (project_root / "CONTRACTS" / "_runs" / "test").mkdir(parents=True)

    # This should succeed
    fs_guard.guarded_write_text(
        project_root / "CONTRACTS" / "_runs" / "test" / "output.json",
        "test content"
    )

    # Verify file was written
    assert (project_root / "CONTRACTS" / "_runs" / "test" / "output.json").read_text() == "test content"


def test_forbidden_path_fails(fs_guard):
    """Test that write to forbidden path fails closed."""
    # Attempt to write to CANON (forbidden)
    valid, error = fs_guard.validate_write_path("CANON/test.md")

    assert valid is False
    assert error is not None
    assert error["code"] == "WRITE_GUARD_PATH_FORBIDDEN"
    assert "CANON" in error["message"]


def test_forbidden_path_guarded_write_fails(fs_guard, project_root):
    """Test that guarded write to forbidden path raises exception."""
    # Create CANON directory (forbidden)
    (project_root / "CANON").mkdir()

    # This should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        fs_guard.guarded_write_text(
            project_root / "CANON" / "test.md",
            "forbidden content"
        )

    assert "WRITE_GUARD_PATH_FORBIDDEN" in str(exc_info.value)


def test_write_outside_allowed_roots_fails(fs_guard):
    """Test that write outside allowed roots fails closed."""
    # Attempt to write to a path not under allowed roots
    valid, error = fs_guard.validate_write_path("README.md")

    assert valid is False
    assert error is not None
    assert error["code"] == "WRITE_GUARD_PATH_NOT_ALLOWED"


def test_path_traversal_fails(fs_guard):
    """Test that path traversal is detected and rejected."""
    # Attempt path traversal
    valid, error = fs_guard.validate_write_path("CONTRACTS/_runs/../../../etc/passwd")

    assert valid is False
    assert error is not None
    # Path traversal may be caught as either TRAVERSAL or ESCAPE depending on resolution
    assert error["code"] in ["WRITE_GUARD_PATH_TRAVERSAL", "WRITE_GUARD_PATH_ESCAPE"]


def test_absolute_path_outside_project_fails(fs_guard):
    """Test that absolute path outside project root fails."""
    # Attempt absolute path outside project
    valid, error = fs_guard.validate_write_path("/tmp/output.json")

    assert valid is False
    assert error is not None
    assert error["code"] in ["WRITE_GUARD_PATH_ABSOLUTE", "WRITE_GUARD_PATH_ESCAPE"]


def test_multiple_allowed_roots_work(fs_guard, project_root):
    """Test that all allowed roots are recognized."""
    # Create all allowed root directories
    for root in ["CONTRACTS/_runs", "CORTEX/_generated", "MEMORY/LLM_PACKER/_packs", "CATALYTIC-DPT/_scratch"]:
        (project_root / root).mkdir(parents=True)

    # Test each allowed root
    test_paths = [
        "CONTRACTS/_runs/test/file.json",
        "CORTEX/_generated/index.json",
        "MEMORY/LLM_PACKER/_packs/pack.tar.gz",
        "CATALYTIC-DPT/_scratch/temp.txt",
    ]

    for path in test_paths:
        valid, error = fs_guard.validate_write_path(path)
        assert valid is True, f"Path {path} should be allowed but got error: {error}"
        assert error is None


def test_all_forbidden_paths_rejected(fs_guard):
    """Test that all forbidden paths are rejected."""
    forbidden_paths = ["CANON", "AGENTS.md", "BUILD", ".git"]

    for forbidden in forbidden_paths:
        valid, error = fs_guard.validate_write_path(f"{forbidden}/test.txt")
        assert valid is False, f"Path {forbidden}/test.txt should be forbidden"
        assert error is not None
        assert error["code"] == "WRITE_GUARD_PATH_FORBIDDEN"


def test_guarded_mkdir_succeeds(fs_guard, project_root):
    """Test that guarded_mkdir succeeds for allowed path."""
    # Create parent directory
    (project_root / "CONTRACTS" / "_runs").mkdir(parents=True)

    # This should succeed
    fs_guard.guarded_mkdir(
        project_root / "CONTRACTS" / "_runs" / "test_dir",
        parents=False,
        exist_ok=False
    )

    # Verify directory was created
    assert (project_root / "CONTRACTS" / "_runs" / "test_dir").is_dir()


def test_guarded_mkdir_forbidden_fails(fs_guard, project_root):
    """Test that guarded_mkdir fails for forbidden path."""
    # This should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        fs_guard.guarded_mkdir(
            project_root / "CANON" / "test_dir",
            parents=False,
            exist_ok=False
        )

    assert "WRITE_GUARD" in str(exc_info.value)


def test_guarded_write_bytes_succeeds(fs_guard, project_root):
    """Test that guarded_write_bytes succeeds for allowed path."""
    # Create parent directory
    (project_root / "CONTRACTS" / "_runs" / "test").mkdir(parents=True)

    # This should succeed
    fs_guard.guarded_write_bytes(
        project_root / "CONTRACTS" / "_runs" / "test" / "binary.dat",
        b"\x00\x01\x02\x03"
    )

    # Verify file was written
    assert (project_root / "CONTRACTS" / "_runs" / "test" / "binary.dat").read_bytes() == b"\x00\x01\x02\x03"


def test_path_normalization(fs_guard, project_root):
    """Test that paths are normalized correctly (Windows backslashes)."""
    # Create directory
    (project_root / "CONTRACTS" / "_runs" / "test").mkdir(parents=True)

    # Test with backslashes (Windows-style)
    valid, error = fs_guard.validate_write_path("CONTRACTS\\_runs\\test\\output.json")

    assert valid is True
    assert error is None
