#!/usr/bin/env python3
"""
Release Sealer (Crypto Safe - Phase 1)

Tamper-evident sealing system for the AGS repository to defend CCL v1.4
license provisions (Sections 3.6, 3.7, 4.4).

The seal proves "you broke my seal" without preventing access.

Usage:
    from CAPABILITY.PRIMITIVES.release_sealer import seal_repo, verify_seal

    # Seal a repository
    receipt = seal_repo(repo_dir, private_key_path)

    # Verify a sealed repository
    result = verify_seal(repo_dir, public_key_path)
    if result.passed:
        print("Verification PASSED")
    else:
        print(f"Verification FAILED: {result.status}")

Files created:
    - RELEASE_MANIFEST.json: Manifest of all tracked files with hashes
    - RELEASE_MANIFEST.json.sig: Ed25519 signature of the manifest
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cas_store import sha256_file, normalize_path
from .release_manifest import (
    FileEntry,
    ReleaseManifest,
    SealReceipt,
    VerificationReceipt,
    VerificationStatus,
)
from .signature import (
    generate_keypair,
    sign_proof,
    verify_signature,
    SignatureBundle,
    save_keypair,
    load_public_key_file,
)


# =============================================================================
# CONSTANTS
# =============================================================================

RELEASES_DIR = "NAVIGATION/PROOFS/RELEASES"
MANIFEST_FILENAME = "RELEASE_MANIFEST.json"
SIGNATURE_FILENAME = "RELEASE_MANIFEST.json.sig"


def get_release_dir(repo_dir: Path, version: str) -> Path:
    """Get the directory for a specific release version."""
    return repo_dir / RELEASES_DIR / version


def get_manifest_path(repo_dir: Path, version: str) -> Path:
    """Get the manifest path for a specific release version."""
    return get_release_dir(repo_dir, version) / MANIFEST_FILENAME


def get_signature_path(repo_dir: Path, version: str) -> Path:
    """Get the signature path for a specific release version."""
    return get_release_dir(repo_dir, version) / SIGNATURE_FILENAME


# =============================================================================
# GIT HELPERS
# =============================================================================


def get_tracked_files(repo_dir: Path) -> list[str]:
    """
    Get list of all git-tracked files in the repository.

    Uses: git ls-files --cached

    Returns:
        List of repo-relative paths with forward slashes
    """
    result = subprocess.run(
        ["git", "ls-files", "--cached"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )

    files = []
    for line in result.stdout.strip().split("\n"):
        if line:
            # Normalize to forward slashes
            normalized = line.replace("\\", "/")
            files.append(normalized)

    return sorted(files)


def get_git_commit(repo_dir: Path) -> Optional[str]:
    """
    Get current git commit SHA.

    Returns:
        40-char hex commit SHA, or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_git_normalized_hash(repo_dir: Path, rel_path: str, from_working_tree: bool = False) -> Optional[str]:
    """
    Get SHA-256 hash of file content as Git normalizes it.

    Args:
        repo_dir: Repository root
        rel_path: Relative path to file
        from_working_tree: If True, hash the working tree file through git's filters.
                          If False, read from git index (staged content).

    Returns:
        64-char SHA-256 hex hash, or None if file not found
    """
    import hashlib

    if from_working_tree:
        # Hash working tree file through git's filters
        # This catches modifications that haven't been staged
        try:
            result = subprocess.run(
                ["git", "hash-object", "--stdin-paths"],
                cwd=repo_dir,
                input=rel_path.encode(),
                capture_output=True,
                check=True,
            )
            # hash-object returns the git blob hash (SHA-1, 40 chars)
            # We need to get the actual content and SHA-256 it
            blob_sha = result.stdout.decode().strip()

            # Now get the content using the blob hash
            result = subprocess.run(
                ["git", "cat-file", "blob", blob_sha],
                cwd=repo_dir,
                capture_output=True,
                check=True,
            )
            return hashlib.sha256(result.stdout).hexdigest()
        except subprocess.CalledProcessError:
            return None
    else:
        # Read from git index (normalized by git's filters)
        try:
            result = subprocess.run(
                ["git", "show", f":{rel_path}"],
                cwd=repo_dir,
                capture_output=True,
                check=True,
            )
            return hashlib.sha256(result.stdout).hexdigest()
        except subprocess.CalledProcessError:
            return None


# =============================================================================
# SEAL REPOSITORY
# =============================================================================


def seal_repo(
    repo_dir: Path,
    private_key_path: Path,
    version: str,
    *,
    exclude_patterns: Optional[list[str]] = None,
) -> SealReceipt:
    """
    Seal a repository by creating a signed manifest of all tracked files.

    This creates in NAVIGATION/PROOFS/RELEASES/{version}/:
    - RELEASE_MANIFEST.json: Manifest with file hashes
    - RELEASE_MANIFEST.json.sig: Ed25519 signature

    Args:
        repo_dir: Path to the git repository root
        private_key_path: Path to Ed25519 private key file (hex encoded)
        version: Release version (e.g., "v3.9.0")
        exclude_patterns: Optional list of path prefixes to exclude

    Returns:
        SealReceipt with seal details

    Raises:
        FileNotFoundError: If repo_dir or private_key_path don't exist
        subprocess.CalledProcessError: If git commands fail
        ValueError: If no files to seal
    """
    repo_dir = Path(repo_dir).resolve()
    private_key_path = Path(private_key_path).resolve()

    if not repo_dir.is_dir():
        raise FileNotFoundError(f"Repository directory not found: {repo_dir}")

    if not private_key_path.is_file():
        raise FileNotFoundError(f"Private key not found: {private_key_path}")

    # Get tracked files
    tracked_files = get_tracked_files(repo_dir)

    # Apply exclusions
    if exclude_patterns:
        tracked_files = [
            f for f in tracked_files
            if not any(f.startswith(p) for p in exclude_patterns)
        ]

    # Exclude manifest and signature files (in any releases directory)
    tracked_files = [
        f for f in tracked_files
        if not (f.endswith(MANIFEST_FILENAME) or f.endswith(SIGNATURE_FILENAME))
    ]

    if not tracked_files:
        raise ValueError("No files to seal after exclusions")

    # Hash all files
    file_entries = []
    total_bytes = 0

    for rel_path in tracked_files:
        abs_path = repo_dir / rel_path
        if not abs_path.is_file():
            # Skip files that don't exist (e.g., submodules, deleted but cached)
            continue

        # Use git-normalized content hash for cross-platform consistency
        file_hash = get_git_normalized_hash(repo_dir, rel_path)
        if not file_hash:
            # Fallback to raw hash if not in git index
            file_hash = sha256_file(abs_path)
        file_size = abs_path.stat().st_size
        total_bytes += file_size

        # Ensure path uses forward slashes
        normalized_path = normalize_path(rel_path)
        file_entries.append(FileEntry(
            path=normalized_path,
            sha256=file_hash,
            size=file_size,
        ))

    if not file_entries:
        raise ValueError("No files to seal (all tracked files missing or excluded)")

    # Get git commit
    git_commit = get_git_commit(repo_dir)

    # Create manifest
    sealed_at = datetime.now(timezone.utc).isoformat()
    manifest = ReleaseManifest(
        sealed_at=sealed_at,
        files=file_entries,
        git_commit=git_commit,
    )

    # Load private key
    private_key = load_public_key_file(private_key_path)  # Stored as hex

    # Create dict for signing (exclude signature field)
    manifest_dict = manifest.to_dict()

    # Sign the manifest
    signature_bundle = sign_proof(manifest_dict, private_key)

    # Create release directory
    release_dir = get_release_dir(repo_dir, version)
    release_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest
    manifest_path = release_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write signature
    signature_path = release_dir / SIGNATURE_FILENAME
    signature_path.write_text(
        json.dumps(signature_bundle.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return SealReceipt(
        manifest_path=str(manifest_path),
        signature_path=str(signature_path),
        manifest_hash=manifest.manifest_hash,
        merkle_root=manifest.merkle_root,
        file_count=len(file_entries),
        total_bytes=total_bytes,
        git_commit=git_commit,
        sealed_at=sealed_at,
    )


# =============================================================================
# VERIFY SEAL
# =============================================================================


def verify_seal(
    repo_dir: Path,
    version: str,
    public_key_path: Optional[Path] = None,
) -> VerificationReceipt:
    """
    Verify a sealed repository.

    Checks:
    1. RELEASE_MANIFEST.json exists in NAVIGATION/PROOFS/RELEASES/{version}/
    2. RELEASE_MANIFEST.json.sig exists
    3. Signature is valid
    4. All files in manifest exist with correct hashes

    Args:
        repo_dir: Path to the git repository root
        version: Release version to verify (e.g., "v3.9.0")
        public_key_path: Path to Ed25519 public key file (hex encoded).
                        If None, uses the public key from the signature.

    Returns:
        VerificationReceipt with verification result
    """
    repo_dir = Path(repo_dir).resolve()

    manifest_path = get_manifest_path(repo_dir, version)
    signature_path = get_signature_path(repo_dir, version)

    # Check manifest exists
    if not manifest_path.is_file():
        return VerificationReceipt(
            status=VerificationStatus.MANIFEST_NOT_FOUND,
            message=f"Manifest not found: {manifest_path}",
        )

    # Check signature exists
    if not signature_path.is_file():
        return VerificationReceipt(
            status=VerificationStatus.SIGNATURE_NOT_FOUND,
            message=f"Signature not found: {signature_path}",
        )

    # Load manifest
    try:
        manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return VerificationReceipt(
            status=VerificationStatus.MANIFEST_CORRUPTED,
            message=f"Invalid JSON in manifest: {e}",
        )

    # Load signature
    try:
        signature_dict = json.loads(signature_path.read_text(encoding="utf-8"))
        signature_bundle = SignatureBundle.from_dict(signature_dict)
    except (json.JSONDecodeError, KeyError) as e:
        return VerificationReceipt(
            status=VerificationStatus.INVALID_SIGNATURE,
            message=f"Invalid signature format: {e}",
        )

    # Load public key if provided
    public_key = None
    if public_key_path:
        public_key_path = Path(public_key_path).resolve()
        if not public_key_path.is_file():
            return VerificationReceipt(
                status=VerificationStatus.INVALID_SIGNATURE,
                message=f"Public key not found: {public_key_path}",
            )
        try:
            public_key = load_public_key_file(public_key_path)
        except Exception as e:
            return VerificationReceipt(
                status=VerificationStatus.INVALID_SIGNATURE,
                message=f"Failed to load public key: {e}",
            )

    # Verify signature
    if not verify_signature(manifest_dict, signature_bundle, public_key):
        return VerificationReceipt(
            status=VerificationStatus.INVALID_SIGNATURE,
            message="Signature verification failed",
            manifest_hash=manifest_dict.get("manifest_hash"),
        )

    # Parse and validate manifest
    try:
        manifest = ReleaseManifest.from_dict(manifest_dict)
    except ValueError as e:
        return VerificationReceipt(
            status=VerificationStatus.MANIFEST_CORRUPTED,
            message=f"Manifest validation failed: {e}",
        )

    # Verify each file
    verified_count = 0
    for file_entry in manifest.files:
        file_path = repo_dir / file_entry.path

        # Check file exists
        if not file_path.is_file():
            return VerificationReceipt(
                status=VerificationStatus.MISSING_FILE,
                message=f"File missing: {file_entry.path}",
                manifest_hash=manifest.manifest_hash,
                verified_files=verified_count,
                failed_path=file_entry.path,
                expected_hash=file_entry.sha256,
            )

        # Check file hash using working tree content (through git's filters)
        # This detects tampering even if changes aren't staged
        actual_hash = get_git_normalized_hash(repo_dir, file_entry.path, from_working_tree=True)
        if not actual_hash:
            # Fallback to raw hash
            actual_hash = sha256_file(file_path)
        if actual_hash != file_entry.sha256:
            return VerificationReceipt(
                status=VerificationStatus.TAMPERED_FILE,
                message=f"File tampered: {file_entry.path}",
                manifest_hash=manifest.manifest_hash,
                verified_files=verified_count,
                failed_path=file_entry.path,
                expected_hash=file_entry.sha256,
                actual_hash=actual_hash,
            )

        verified_count += 1

    # All checks passed
    return VerificationReceipt(
        status=VerificationStatus.PASS,
        message="All files verified",
        manifest_hash=manifest.manifest_hash,
        verified_files=verified_count,
    )


# =============================================================================
# CLI SELF-TEST
# =============================================================================


if __name__ == "__main__":
    import tempfile
    import shutil

    print("Release Sealer Self-Test")
    print("=" * 50)

    # Create temp directory for test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, capture_output=True, check=True
        )

        # Create some test files
        (tmp_path / "README.md").write_text("# Test\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')\n")

        # Add and commit
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=tmp_path, capture_output=True, check=True
        )

        # Generate keypair
        private_key, public_key = generate_keypair()
        key_path = tmp_path / "test.key"
        pub_path = tmp_path / "test.pub"
        save_keypair(private_key, public_key, key_path, pub_path)

        print("\n1. Sealing repository...")
        receipt = seal_repo(tmp_path, key_path)
        print(f"   {receipt.compact()}")
        print(f"   Manifest: {receipt.manifest_path}")

        print("\n2. Verifying sealed repository...")
        result = verify_seal(tmp_path, pub_path)
        print(f"   {result.compact()}")

        print("\n3. Tampering with a file...")
        (tmp_path / "README.md").write_text("# Tampered\n")

        print("\n4. Verifying after tampering...")
        result = verify_seal(tmp_path, pub_path)
        print(f"   {result.compact()}")
        if result.failed_path:
            print(f"   Failed: {result.failed_path}")
            print(f"   Expected: {result.expected_hash[:16]}...")
            print(f"   Actual:   {result.actual_hash[:16]}...")

    print("\n" + "=" * 50)
    print("Self-test complete!")
