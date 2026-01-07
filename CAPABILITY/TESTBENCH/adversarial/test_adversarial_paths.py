from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
except ImportError:
    pytest.skip("Write firewall not available", allow_module_level=True)

from CAPABILITY.PRIMITIVES.cas_store import normalize_relpath
from CAPABILITY.PRIMITIVES.merkle import build_manifest_root


def test_normalize_relpath_rejects_traversal_and_absolute() -> None:
    bad = [
        "/etc/passwd",
        "//server/share",
        "C:\\Windows\\System32",
        "C:/Windows/System32",
        "../secrets.txt",
        "a/../b",
        "a\\..\\b",
    ]
    for p in bad:
        with pytest.raises(ValueError):
            _ = normalize_relpath(p)


def test_normalize_relpath_normalizes_separators_and_dot() -> None:
    assert normalize_relpath(r"a\\b/./c") == "a/b/c"
    assert normalize_relpath("./a/b") == "a/b"


def test_merkle_rejects_non_normalized_paths() -> None:
    # Non-normalized path should be rejected deterministically.
    with pytest.raises(ValueError, match=r"non-normalized path"):
        _ = build_manifest_root({r"a\\b": "a" * 64})

    # Traversal should be rejected via normalize_relpath.
    with pytest.raises(ValueError):
        _ = build_manifest_root({"../x": "a" * 64})


def test_guarded_writer_rejects_path_traversal() -> None:
    """Test that GuardedWriter firewall rejects path traversal in allowed domains."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()

        # Create directory structure for GuardedWriter domains
        (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True, exist_ok=True)
        (project_root / "LAW" / "CONTRACTS" / "_runs" / "other").mkdir(parents=True, exist_ok=True)

        writer = GuardedWriter(project_root=project_root)

        # Traversal that stays within project but uses .. syntax (should be rejected)
        # This is within the allowed tmp domain but uses .. which is blocked
        with pytest.raises(FirewallViolation) as exc_info:
            writer.write_tmp("LAW/CONTRACTS/_runs/_tmp/../other/secret.txt", "malicious")

        violation = exc_info.value.violation_receipt
        assert violation["error_code"] == "FIREWALL_PATH_TRAVERSAL", \
            f"Path with .. should fail with FIREWALL_PATH_TRAVERSAL"


def test_guarded_writer_rejects_absolute_path_escape() -> None:
    """Test that GuardedWriter firewall rejects absolute paths outside project root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()

        # Create directory structure for GuardedWriter domains
        (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True, exist_ok=True)

        writer = GuardedWriter(project_root=project_root)

        # Absolute path escape attempts
        escape_attempts = [
            "/etc/passwd",
            "/tmp/secret.txt",
        ]

        for bad_path in escape_attempts:
            with pytest.raises(FirewallViolation) as exc_info:
                writer.write_tmp(bad_path, "malicious")

            violation = exc_info.value.violation_receipt
            assert violation["error_code"] == "FIREWALL_PATH_ESCAPE", \
                f"Absolute path '{bad_path}' should fail with FIREWALL_PATH_ESCAPE"

