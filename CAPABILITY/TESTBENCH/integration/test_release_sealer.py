#!/usr/bin/env python3
"""
Release Sealer Tests (Crypto Safe - Phase 1)

Tests for tamper-evident release sealing to verify:
- seal_repo() creates manifest and signature files
- seal_repo() includes all git-tracked files
- verify_seal() passes for valid seals
- verify_seal() detects tampered files
- verify_seal() detects missing files
- verify_seal() detects invalid signatures
- Sealing is deterministic (same inputs = same hashes)

Exit Criteria:
- All tracked git files are hashed in manifest
- Manifest is signed with Ed25519
- Tampering detected on any file modification
- Missing files detected
- Invalid signatures rejected
- CLI tools exit 0 for PASS, 1 for FAIL
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.signature import (
    generate_keypair,
    save_keypair,
    _bytes_to_hex,
)
from CAPABILITY.PRIMITIVES.release_manifest import (
    FileEntry,
    ReleaseManifest,
    VerificationStatus,
)
from CAPABILITY.PRIMITIVES.release_sealer import (
    seal_repo,
    verify_seal,
    get_tracked_files,
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tmp_keys(tmp_path: Path) -> tuple[Path, Path]:
    """Generate temporary keypair files."""
    private_path = tmp_path / "test.key"
    public_path = tmp_path / "test.pub"

    private_key, public_key = generate_keypair()
    save_keypair(private_key, public_key, private_path, public_path)

    return private_path, public_path


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with tracked files."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    # Create test files
    (repo_dir / "README.md").write_text("# Test Repository\n", encoding="utf-8")
    (repo_dir / "LICENSE").write_text("MIT License\n", encoding="utf-8")

    src_dir = repo_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (src_dir / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")

    # Add and commit
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_dir,
        capture_output=True,
        check=True,
    )

    return repo_dir


# =============================================================================
# MANIFEST TESTS
# =============================================================================


class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_valid_file_entry(self):
        """Valid file entry is created successfully."""
        entry = FileEntry(
            path="src/main.py",
            sha256="a" * 64,
            size=1234,
        )
        assert entry.path == "src/main.py"
        assert entry.sha256 == "a" * 64
        assert entry.size == 1234

    def test_rejects_empty_path(self):
        """Empty path is rejected."""
        with pytest.raises(ValueError, match="path cannot be empty"):
            FileEntry(path="", sha256="a" * 64, size=100)

    def test_rejects_backslashes(self):
        """Backslashes in path are rejected."""
        with pytest.raises(ValueError, match="forward slashes"):
            FileEntry(path="src\\main.py", sha256="a" * 64, size=100)

    def test_rejects_wrong_hash_length(self):
        """Wrong hash length is rejected."""
        with pytest.raises(ValueError, match="64 hex chars"):
            FileEntry(path="test.py", sha256="abc", size=100)

    def test_rejects_negative_size(self):
        """Negative size is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            FileEntry(path="test.py", sha256="a" * 64, size=-1)


class TestReleaseManifest:
    """Tests for ReleaseManifest dataclass."""

    def test_manifest_creation(self):
        """Manifest is created with computed fields."""
        files = [
            FileEntry(path="b.txt", sha256="b" * 64, size=100),
            FileEntry(path="a.txt", sha256="a" * 64, size=200),
        ]
        manifest = ReleaseManifest(
            sealed_at="2025-01-01T00:00:00Z",
            files=files,
            git_commit="abc123",
        )

        # Files should be sorted by path
        assert manifest.files[0].path == "a.txt"
        assert manifest.files[1].path == "b.txt"

        # Computed fields should be set
        assert len(manifest.merkle_root) == 64
        assert len(manifest.manifest_hash) == 64

    def test_manifest_rejects_empty_files(self):
        """Empty files list is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ReleaseManifest(
                sealed_at="2025-01-01T00:00:00Z",
                files=[],
            )

    def test_manifest_rejects_duplicate_paths(self):
        """Duplicate paths are rejected."""
        files = [
            FileEntry(path="a.txt", sha256="a" * 64, size=100),
            FileEntry(path="a.txt", sha256="b" * 64, size=200),
        ]
        with pytest.raises(ValueError, match="duplicate paths"):
            ReleaseManifest(
                sealed_at="2025-01-01T00:00:00Z",
                files=files,
            )

    def test_manifest_json_roundtrip(self):
        """Manifest survives JSON round-trip."""
        files = [
            FileEntry(path="test.py", sha256="a" * 64, size=100),
        ]
        original = ReleaseManifest(
            sealed_at="2025-01-01T00:00:00Z",
            files=files,
            git_commit="abc123",
        )

        json_str = original.to_json()
        parsed = json.loads(json_str)
        restored = ReleaseManifest.from_dict(parsed)

        assert restored.manifest_hash == original.manifest_hash
        assert restored.merkle_root == original.merkle_root
        assert len(restored.files) == len(original.files)

    def test_manifest_detects_hash_mismatch(self):
        """Loading manifest with wrong hash raises error."""
        files = [FileEntry(path="test.py", sha256="a" * 64, size=100)]
        manifest = ReleaseManifest(
            sealed_at="2025-01-01T00:00:00Z",
            files=files,
        )

        data = manifest.to_dict()
        data["manifest_hash"] = "x" * 64  # Wrong hash

        with pytest.raises(ValueError, match="manifest_hash mismatch"):
            ReleaseManifest.from_dict(data)


# =============================================================================
# SEALING TESTS
# =============================================================================


class TestSealRepo:
    """Tests for seal_repo() function."""

    def test_seal_creates_manifest(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """seal_repo() creates RELEASE_MANIFEST.json."""
        private_path, _ = tmp_keys
        receipt = seal_repo(tmp_repo, private_path)

        manifest_path = tmp_repo / MANIFEST_FILENAME
        assert manifest_path.exists()
        assert receipt.file_count > 0

    def test_seal_creates_signature(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """seal_repo() creates RELEASE_MANIFEST.json.sig."""
        private_path, _ = tmp_keys
        seal_repo(tmp_repo, private_path)

        signature_path = tmp_repo / SIGNATURE_FILENAME
        assert signature_path.exists()

        # Verify signature format
        sig_data = json.loads(signature_path.read_text())
        assert "signature" in sig_data
        assert "public_key" in sig_data
        assert sig_data["algorithm"] == "Ed25519"

    def test_seal_includes_all_tracked_files(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """seal_repo() includes all git-tracked files."""
        private_path, _ = tmp_keys
        receipt = seal_repo(tmp_repo, private_path)

        tracked = get_tracked_files(tmp_repo)
        # Exclude manifest/sig files from count
        tracked = [f for f in tracked if f not in (MANIFEST_FILENAME, SIGNATURE_FILENAME)]

        # Receipt count should match
        assert receipt.file_count == len(tracked)

        # Verify manifest contents
        manifest_data = json.loads((tmp_repo / MANIFEST_FILENAME).read_text())
        manifest_paths = [f["path"] for f in manifest_data["files"]]

        for tracked_file in tracked:
            # Normalize path
            normalized = tracked_file.replace("\\", "/")
            assert normalized in manifest_paths, f"Missing: {normalized}"

    def test_seal_deterministic(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """Same repo produces same manifest hash (excluding timestamp)."""
        private_path, _ = tmp_keys

        # First seal
        receipt1 = seal_repo(tmp_repo, private_path)
        manifest1 = json.loads((tmp_repo / MANIFEST_FILENAME).read_text())

        # Remove manifest/sig for re-seal
        (tmp_repo / MANIFEST_FILENAME).unlink()
        (tmp_repo / SIGNATURE_FILENAME).unlink()

        # Second seal
        receipt2 = seal_repo(tmp_repo, private_path)
        manifest2 = json.loads((tmp_repo / MANIFEST_FILENAME).read_text())

        # Merkle roots should match (file hashes are same)
        assert receipt1.merkle_root == receipt2.merkle_root

        # File lists should match
        assert manifest1["files"] == manifest2["files"]

    def test_seal_with_exclusions(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """seal_repo() respects exclude_patterns."""
        private_path, _ = tmp_keys

        receipt = seal_repo(
            tmp_repo,
            private_path,
            exclude_patterns=["src/"],
        )

        manifest_data = json.loads((tmp_repo / MANIFEST_FILENAME).read_text())
        manifest_paths = [f["path"] for f in manifest_data["files"]]

        # src/ files should be excluded
        for path in manifest_paths:
            assert not path.startswith("src/"), f"Should be excluded: {path}"


# =============================================================================
# VERIFICATION TESTS
# =============================================================================


class TestVerifySeal:
    """Tests for verify_seal() function."""

    def test_verify_valid_repo_passes(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() passes for valid sealed repo."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        result = verify_seal(tmp_repo, public_path)

        assert result.passed
        assert result.status == VerificationStatus.PASS
        assert result.verified_files > 0

    def test_verify_without_pubkey(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() works with embedded public key."""
        private_path, _ = tmp_keys
        seal_repo(tmp_repo, private_path)

        # No public key provided - uses embedded key from signature
        result = verify_seal(tmp_repo)

        assert result.passed

    def test_verify_tampered_file_fails(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() detects tampered files."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        # Tamper with a file
        (tmp_repo / "README.md").write_text("# TAMPERED\n", encoding="utf-8")

        result = verify_seal(tmp_repo, public_path)

        assert not result.passed
        assert result.status == VerificationStatus.TAMPERED_FILE
        assert result.failed_path == "README.md"
        assert result.expected_hash is not None
        assert result.actual_hash is not None
        assert result.expected_hash != result.actual_hash

    def test_verify_missing_file_fails(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() detects missing files."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        # Delete a file (but not from git index)
        (tmp_repo / "LICENSE").unlink()

        result = verify_seal(tmp_repo, public_path)

        assert not result.passed
        assert result.status == VerificationStatus.MISSING_FILE
        assert result.failed_path == "LICENSE"

    def test_verify_invalid_signature_fails(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() detects invalid signatures."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        # Generate different keypair
        other_private, other_public = generate_keypair()
        other_pub_path = tmp_repo / "other.pub"
        other_pub_path.write_text(_bytes_to_hex(other_public))

        # Verify with wrong key
        result = verify_seal(tmp_repo, other_pub_path)

        assert not result.passed
        assert result.status == VerificationStatus.INVALID_SIGNATURE

    def test_verify_missing_manifest_fails(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() fails if manifest is missing."""
        _, public_path = tmp_keys

        # No seal performed, manifest doesn't exist
        result = verify_seal(tmp_repo, public_path)

        assert not result.passed
        assert result.status == VerificationStatus.MANIFEST_NOT_FOUND

    def test_verify_missing_signature_fails(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() fails if signature is missing."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        # Delete signature
        (tmp_repo / SIGNATURE_FILENAME).unlink()

        result = verify_seal(tmp_repo, public_path)

        assert not result.passed
        assert result.status == VerificationStatus.SIGNATURE_NOT_FOUND

    def test_verify_corrupted_manifest_fails(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """verify_seal() fails if manifest is corrupted."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        # Corrupt manifest
        (tmp_repo / MANIFEST_FILENAME).write_text("not valid json", encoding="utf-8")

        result = verify_seal(tmp_repo, public_path)

        assert not result.passed
        assert result.status == VerificationStatus.MANIFEST_CORRUPTED


# =============================================================================
# CLI TESTS
# =============================================================================


class TestSealReleaseCLI:
    """Tests for seal_release.py CLI."""

    def test_cli_keygen(self, tmp_path: Path):
        """CLI keygen generates valid keypair."""
        private_path = tmp_path / "new.key"
        public_path = tmp_path / "new.pub"

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "seal_release.py"),
                "--json",
                "keygen",
                "--private-key", str(private_path),
                "--public-key", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"keygen failed: {result.stderr}"
        assert private_path.exists()
        assert public_path.exists()

        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert "key_id" in output

    def test_cli_seal(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """CLI seal creates manifest and signature."""
        private_path, _ = tmp_keys

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "seal_release.py"),
                "--json",
                "seal",
                "--repo-dir", str(tmp_repo),
                "--private-key", str(private_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"seal failed: {result.stderr}"

        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert output["file_count"] > 0

        # Verify files created
        assert (tmp_repo / MANIFEST_FILENAME).exists()
        assert (tmp_repo / SIGNATURE_FILENAME).exists()


class TestVerifyReleaseCLI:
    """Tests for verify_release.py CLI."""

    def test_cli_verify_pass(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """CLI verify returns 0 for valid seal."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "verify_release.py"),
                "--repo-dir", str(tmp_repo),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"verify failed: {result.stderr}"
        assert "[PASS]" in result.stdout

    def test_cli_verify_fail_tampered(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """CLI verify returns 1 for tampered file."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        # Tamper
        (tmp_repo / "README.md").write_text("# TAMPERED\n", encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "verify_release.py"),
                "--repo-dir", str(tmp_repo),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "[FAIL]" in result.stdout
        assert "TAMPERED_FILE" in result.stdout

    def test_cli_verify_json_output(self, tmp_repo: Path, tmp_keys: tuple[Path, Path]):
        """CLI verify outputs valid JSON with --json flag."""
        private_path, public_path = tmp_keys
        seal_repo(tmp_repo, private_path)

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "verify_release.py"),
                "--json",
                "--repo-dir", str(tmp_repo),
                "--pubkey", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["passed"] is True
        assert output["status"] == "PASS"
        assert output["verified_files"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
