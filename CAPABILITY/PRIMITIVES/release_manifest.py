#!/usr/bin/env python3
"""
Release Manifest Primitives (Crypto Safe - Phase 1)

Dataclasses for tamper-evident release sealing to defend CCL v1.4 license
provisions (Sections 3.6, 3.7, 4.4).

Usage:
    from CAPABILITY.PRIMITIVES.release_manifest import (
        FileEntry,
        ReleaseManifest,
        SealReceipt,
        VerificationReceipt,
        VerificationStatus,
    )

    # Create manifest from files
    files = [FileEntry(path="src/main.py", sha256="abc...", size=1234)]
    manifest = ReleaseManifest(
        sealed_at="2025-01-01T00:00:00Z",
        git_commit="abc123",
        files=files,
    )

Key Principle: The seal proves "you broke my seal" without preventing access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# CANONICAL JSON UTILITIES
# =============================================================================


def canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string.

    Rules:
    - Keys sorted lexicographically
    - No extra whitespace (separators are ',' and ':')
    - UTF-8 safe (ensure_ascii=False)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(obj: Any) -> bytes:
    """Convert object to canonical JSON bytes."""
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hex digest (64 lowercase chars)."""
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# FILE ENTRY
# =============================================================================


@dataclass
class FileEntry:
    """
    A single file in the release manifest.

    Attributes:
        path: Repo-relative path with forward slashes (e.g., "src/main.py")
        sha256: 64-char lowercase hex SHA256 hash of file contents
        size: File size in bytes
    """
    path: str
    sha256: str
    size: int

    def __post_init__(self):
        """Validate file entry fields."""
        if not self.path:
            raise ValueError("path cannot be empty")
        if "\\" in self.path:
            raise ValueError(f"path must use forward slashes: {self.path}")
        if len(self.sha256) != 64:
            raise ValueError(f"sha256 must be 64 hex chars, got {len(self.sha256)}")
        if self.size < 0:
            raise ValueError(f"size must be non-negative, got {self.size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileEntry":
        """Create from dict."""
        return cls(
            path=d["path"],
            sha256=d["sha256"],
            size=d["size"],
        )


# =============================================================================
# RELEASE MANIFEST
# =============================================================================


@dataclass
class ReleaseManifest:
    """
    A tamper-evident manifest of all tracked files in a release.

    Required fields:
        sealed_at: ISO-8601 timestamp when manifest was created
        files: List of FileEntry objects (sorted by path for determinism)

    Optional fields:
        version: Manifest format version (default "1.0.0")
        license: License identifier (default "CCL-v1.4")
        git_commit: Git commit SHA at seal time

    Computed fields (set in __post_init__):
        merkle_root: Merkle root of file hashes
        manifest_hash: SHA256 of manifest (excludes itself)
    """
    sealed_at: str
    files: List[FileEntry]
    version: str = "1.0.0"
    license: str = "CCL-v1.4"
    git_commit: Optional[str] = None

    # Auto-computed fields
    merkle_root: str = field(default="", init=False)
    manifest_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Validate sealed_at format
        if not self.sealed_at:
            raise ValueError("sealed_at cannot be empty")

        # Validate files list
        if not self.files:
            raise ValueError("files list cannot be empty")

        # Sort files by path for determinism
        self.files = sorted(self.files, key=lambda f: f.path)

        # Check for duplicate paths
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate paths in file list")

        # Compute merkle_root
        self.merkle_root = self._compute_merkle_root()

        # Compute manifest_hash (must be last)
        self.manifest_hash = self._compute_manifest_hash()

    def _compute_merkle_root(self) -> str:
        """
        Compute Merkle root from file entries.

        Algorithm (compatible with merkle.py but allows duplicate hashes):
        - Leaf hash = sha256(path + ":" + bytes_hash)
        - Internal hash = sha256(left + right)
        - Odd leaf count: duplicate last leaf

        Note: We compute directly rather than using build_manifest_root()
        because that function rejects duplicate file hashes, but we need
        to allow multiple files with the same content (e.g., empty files).
        """
        # Build leaf hashes (path:hash pairs ensure uniqueness)
        def leaf_hash(path: str, bytes_hash: str) -> str:
            preimage = f"{path}:{bytes_hash}".encode("utf-8")
            return hashlib.sha256(preimage).hexdigest()

        def node_hash(left: str, right: str) -> str:
            preimage = (left + right).encode("ascii")
            return hashlib.sha256(preimage).hexdigest()

        # Files are already sorted by path in __post_init__
        level = [leaf_hash(f.path, f.sha256) for f in self.files]

        # Build tree bottom-up
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])  # Duplicate last leaf for odd count
            level = [node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]

        return level[0]

    def _to_hashable_dict(self) -> Dict[str, Any]:
        """
        Convert to dict for hashing.

        Excludes computed fields (merkle_root, manifest_hash) to ensure
        deterministic hash computation.
        """
        return {
            "version": self.version,
            "sealed_at": self.sealed_at,
            "license": self.license,
            "git_commit": self.git_commit,
            "files": [f.to_dict() for f in self.files],
        }

    def _compute_manifest_hash(self) -> str:
        """Compute SHA256 of manifest (excludes manifest_hash itself)."""
        hashable = self._to_hashable_dict()
        hashable["merkle_root"] = self.merkle_root
        return sha256_hex(canonical_json_bytes(hashable))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dict for JSON serialization."""
        result = self._to_hashable_dict()
        result["merkle_root"] = self.merkle_root
        result["manifest_hash"] = self.manifest_hash
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReleaseManifest":
        """Create from dict."""
        files = [FileEntry.from_dict(f) for f in d["files"]]
        manifest = cls(
            sealed_at=d["sealed_at"],
            files=files,
            version=d.get("version", "1.0.0"),
            license=d.get("license", "CCL-v1.4"),
            git_commit=d.get("git_commit"),
        )

        # Verify computed fields match stored values
        if d.get("merkle_root") and d["merkle_root"] != manifest.merkle_root:
            raise ValueError(
                f"merkle_root mismatch: stored={d['merkle_root']}, computed={manifest.merkle_root}"
            )
        if d.get("manifest_hash") and d["manifest_hash"] != manifest.manifest_hash:
            raise ValueError(
                f"manifest_hash mismatch: stored={d['manifest_hash']}, computed={manifest.manifest_hash}"
            )

        return manifest


# =============================================================================
# VERIFICATION STATUS
# =============================================================================


class VerificationStatus(str, Enum):
    """Verification result status codes."""
    PASS = "PASS"
    MANIFEST_NOT_FOUND = "MANIFEST_NOT_FOUND"
    SIGNATURE_NOT_FOUND = "SIGNATURE_NOT_FOUND"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    TAMPERED_FILE = "TAMPERED_FILE"
    MISSING_FILE = "MISSING_FILE"
    EXTRA_FILE = "EXTRA_FILE"
    MANIFEST_CORRUPTED = "MANIFEST_CORRUPTED"


# =============================================================================
# SEAL RECEIPT
# =============================================================================


@dataclass
class SealReceipt:
    """
    Receipt returned after sealing a repository.

    Attributes:
        manifest_path: Path to RELEASE_MANIFEST.json
        signature_path: Path to RELEASE_MANIFEST.json.sig
        manifest_hash: SHA256 of manifest
        merkle_root: Merkle root of all files
        file_count: Number of files sealed
        total_bytes: Total bytes sealed
        git_commit: Git commit at seal time
        sealed_at: Timestamp of sealing
    """
    manifest_path: str
    signature_path: str
    manifest_hash: str
    merkle_root: str
    file_count: int
    total_bytes: int
    git_commit: Optional[str]
    sealed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "manifest_path": self.manifest_path,
            "signature_path": self.signature_path,
            "manifest_hash": self.manifest_hash,
            "merkle_root": self.merkle_root,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "git_commit": self.git_commit,
            "sealed_at": self.sealed_at,
        }

    def compact(self) -> str:
        """Compact format for CLI display."""
        return (
            f"[SEAL] {self.file_count} files, {self.total_bytes:,} bytes "
            f"-> {self.manifest_hash[:8]}..."
        )

    def verbose(self) -> str:
        """Verbose format for reports."""
        lines = [
            "RELEASE SEAL RECEIPT",
            "-" * 50,
            f"Manifest:     {self.manifest_path}",
            f"Signature:    {self.signature_path}",
            f"Manifest Hash:{self.manifest_hash}",
            f"Merkle Root:  {self.merkle_root}",
            f"File Count:   {self.file_count:,}",
            f"Total Bytes:  {self.total_bytes:,}",
            f"Git Commit:   {self.git_commit or 'N/A'}",
            f"Sealed At:    {self.sealed_at}",
        ]
        return "\n".join(lines)


# =============================================================================
# VERIFICATION RECEIPT
# =============================================================================


@dataclass
class VerificationReceipt:
    """
    Receipt returned after verifying a sealed repository.

    Attributes:
        status: VerificationStatus (PASS or error code)
        message: Human-readable message
        manifest_hash: Hash from manifest (if found)
        verified_files: Number of files verified
        failed_path: Path of first failed file (if any)
        expected_hash: Expected hash (if tampering detected)
        actual_hash: Actual hash (if tampering detected)
    """
    status: VerificationStatus
    message: str
    manifest_hash: Optional[str] = None
    verified_files: int = 0
    failed_path: Optional[str] = None
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None

    @property
    def passed(self) -> bool:
        """True if verification passed."""
        return self.status == VerificationStatus.PASS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "status": self.status.value,
            "passed": self.passed,
            "message": self.message,
            "verified_files": self.verified_files,
        }
        if self.manifest_hash:
            result["manifest_hash"] = self.manifest_hash
        if self.failed_path:
            result["failed_path"] = self.failed_path
        if self.expected_hash:
            result["expected_hash"] = self.expected_hash
        if self.actual_hash:
            result["actual_hash"] = self.actual_hash
        return result

    def compact(self) -> str:
        """Compact format for CLI display."""
        if self.passed:
            return f"[PASS] {self.verified_files} files verified"
        else:
            return f"[FAIL] {self.status.value}: {self.message}"

    def verbose(self) -> str:
        """Verbose format for reports."""
        lines = [
            "VERIFICATION RECEIPT",
            "-" * 50,
            f"Status:       {self.status.value}",
            f"Passed:       {self.passed}",
            f"Message:      {self.message}",
            f"Files Verified: {self.verified_files}",
        ]
        if self.manifest_hash:
            lines.append(f"Manifest Hash:{self.manifest_hash}")
        if self.failed_path:
            lines.append(f"Failed Path:  {self.failed_path}")
        if self.expected_hash:
            lines.append(f"Expected:     {self.expected_hash}")
        if self.actual_hash:
            lines.append(f"Actual:       {self.actual_hash}")
        return "\n".join(lines)


# =============================================================================
# CLI SELF-TEST
# =============================================================================


if __name__ == "__main__":
    print("Release Manifest Primitives Self-Test")
    print("=" * 50)

    # Create sample file entries
    files = [
        FileEntry(path="src/main.py", sha256="a" * 64, size=1234),
        FileEntry(path="README.md", sha256="b" * 64, size=567),
        FileEntry(path="LICENSE", sha256="c" * 64, size=890),
    ]

    print("\nFile Entries:")
    for f in files:
        print(f"  {f.path}: {f.sha256[:8]}... ({f.size} bytes)")

    # Create manifest
    manifest = ReleaseManifest(
        sealed_at=datetime.now(timezone.utc).isoformat(),
        files=files,
        git_commit="abc123def456",
    )

    print("\nManifest:")
    print(f"  Version:      {manifest.version}")
    print(f"  License:      {manifest.license}")
    print(f"  Sealed At:    {manifest.sealed_at}")
    print(f"  Git Commit:   {manifest.git_commit}")
    print(f"  Merkle Root:  {manifest.merkle_root[:16]}...")
    print(f"  Manifest Hash:{manifest.manifest_hash[:16]}...")

    # Test JSON round-trip
    json_str = manifest.to_json()
    parsed = json.loads(json_str)
    restored = ReleaseManifest.from_dict(parsed)

    print("\nJSON round-trip:")
    print(f"  Original hash:  {manifest.manifest_hash[:16]}...")
    print(f"  Restored hash:  {restored.manifest_hash[:16]}...")
    print(f"  Match: {manifest.manifest_hash == restored.manifest_hash}")

    # Test seal receipt
    receipt = SealReceipt(
        manifest_path="RELEASE_MANIFEST.json",
        signature_path="RELEASE_MANIFEST.json.sig",
        manifest_hash=manifest.manifest_hash,
        merkle_root=manifest.merkle_root,
        file_count=len(files),
        total_bytes=sum(f.size for f in files),
        git_commit=manifest.git_commit,
        sealed_at=manifest.sealed_at,
    )

    print("\nSeal Receipt (compact):")
    print(f"  {receipt.compact()}")

    # Test verification receipt
    pass_receipt = VerificationReceipt(
        status=VerificationStatus.PASS,
        message="All files verified",
        manifest_hash=manifest.manifest_hash,
        verified_files=len(files),
    )

    fail_receipt = VerificationReceipt(
        status=VerificationStatus.TAMPERED_FILE,
        message="File content does not match manifest",
        manifest_hash=manifest.manifest_hash,
        verified_files=2,
        failed_path="src/main.py",
        expected_hash="a" * 64,
        actual_hash="x" * 64,
    )

    print("\nVerification Receipts:")
    print(f"  Pass: {pass_receipt.compact()}")
    print(f"  Fail: {fail_receipt.compact()}")

    print("\n" + "=" * 50)
    print("Self-test complete!")
