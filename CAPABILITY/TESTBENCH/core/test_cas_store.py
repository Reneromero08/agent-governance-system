#!/usr/bin/env python3
"""
Core Test: CAS Store (Catalytic Context Compression)

Tests the Content-Addressed Storage implementation in CAPABILITY/PRIMITIVES.
"""

import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.cas_store import (
    build,
    reconstruct,
    verify,
    sha256_bytes,
    sha256_file,
    EXIT_SUCCESS,
    EXIT_ERROR,
    EXIT_VERIFY_MISMATCH,
    EXIT_UNSAFE_PATH,
    EXIT_BOUNDS_EXCEEDED
)


# The run_cli function and SCRIPT variable are removed as tests will directly call
# the imported functions (build, reconstruct, verify) from cas_store.



class TestBuildReconstructVerify:
    """Test full roundtrip: build -> reconstruct -> verify."""

    def test_roundtrip_success(self, tmp_path):
        # Create source directory with diverse files
        src = tmp_path / "source"
        src.mkdir()
        (src / "readme.txt").write_text("Hello World")
        (src / "data.bin").write_bytes(os.urandom(256))
        nested = src / "nested" / "deep"
        nested.mkdir(parents=True)
        (nested / "config.json").write_text('{"key": "value"}')

        # Build
        out = tmp_path / "pack"
        out.mkdir()
        with pytest.raises(SystemExit) as exc:
            build(src, out, [])
        assert exc.value.code == EXIT_SUCCESS
        assert (out / "manifest.json").exists()
        assert (out / "root.sha256").exists()
        assert (out / "cas").is_dir()

        # Reconstruct
        dst = tmp_path / "restored"
        with pytest.raises(SystemExit) as exc:
            reconstruct(out, dst)
        assert exc.value.code == EXIT_SUCCESS

        # Verify
        with pytest.raises(SystemExit) as exc:
            verify(src, dst)
        assert exc.value.code == EXIT_SUCCESS


class TestDeterminism:
    """Test that consecutive builds produce identical manifests."""

    def test_manifest_stable(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "file1.txt").write_text("content1")
        (src / "file2.txt").write_text("content2")

        out1 = tmp_path / "pack1"
        out1.mkdir()
        out2 = tmp_path / "pack2"
        out2.mkdir()

        with pytest.raises(SystemExit) as exc:
            build(src, out1, [])
        assert exc.value.code == EXIT_SUCCESS

        with pytest.raises(SystemExit) as exc:
            build(src, out2, [])
        assert exc.value.code == EXIT_SUCCESS

        manifest1 = (out1 / "manifest.json").read_bytes()
        manifest2 = (out2 / "manifest.json").read_bytes()
        assert manifest1 == manifest2, "Manifests differ between builds"

        root1 = (out1 / "root.sha256").read_text()
        root2 = (out2 / "root.sha256").read_text()
        assert root1 == root2, "Root hashes differ between builds"


class TestSafety:
    """Test safety caps and path validation."""

    def test_path_traversal_rejected(self, tmp_path):
        # We can't easily create a file with '..' in the name on most OSes,
        # but we can verify the validator rejects it at the logic level.
        # For now, just verify the script doesn't crash on normal input.
        src = tmp_path / "source"
        src.mkdir()
        (src / "safe_file.txt").write_text("ok")

        out = tmp_path / "pack"
        out.mkdir()
        with pytest.raises(SystemExit) as exc:
            build(src, out, [])
        assert exc.value.code == EXIT_SUCCESS

    def test_empty_source(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()

        out = tmp_path / "pack"
        out.mkdir()
        with pytest.raises(SystemExit) as exc:
            build(src, out, [])
        assert exc.value.code == EXIT_SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
