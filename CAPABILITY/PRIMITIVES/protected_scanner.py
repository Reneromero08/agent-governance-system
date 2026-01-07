#!/usr/bin/env python3
"""
Protected Artifacts Scanner (CRYPTO_SAFE.0)

Scans the working tree for protected artifacts and enforces distribution policies.

Phase: 2.4.2 Protected Artifact Inventory
Dependencies: Phase 2.4.1C (write surface enforcement complete)

This module provides:
- Recursive directory scanning
- Protected artifact detection
- Fail-closed enforcement for public pack modes
- Deterministic scan receipts
"""

import json
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from CAPABILITY.PRIMITIVES.protected_inventory import (
    ProtectedInventory,
    ProtectedPattern,
    get_default_inventory,
    DistributionPolicy
)


@dataclass
class ScanMatch:
    """A single protected artifact match."""

    path: Path
    artifact_class: str
    distribution_policy: str
    description: str
    size_bytes: int
    sha256: Optional[str] = None  # Computed on demand

    def __post_init__(self):
        """Normalize path for determinism."""
        self.path = Path(str(self.path).replace('\\', '/'))

    def to_dict(self) -> dict:
        """Serialize to canonical JSON format."""
        return {
            "path": str(self.path).replace('\\', '/'),
            "artifact_class": self.artifact_class,
            "distribution_policy": self.distribution_policy,
            "description": self.description,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256
        }

    def compute_hash(self, repo_root: Path) -> str:
        """Compute SHA-256 hash of artifact."""
        if self.sha256:
            return self.sha256

        full_path = repo_root / self.path
        sha = hashlib.sha256()
        with open(full_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha.update(chunk)
        self.sha256 = sha.hexdigest()
        return self.sha256


@dataclass
class ScanResult:
    """Result of a protected artifacts scan."""

    verdict: str  # PASS, FAIL, WARN
    matches: List[ScanMatch] = field(default_factory=list)
    context: str = "working"  # working, public, internal
    total_files_scanned: int = 0
    protected_count: int = 0
    violations: List[str] = field(default_factory=list)
    inventory_hash: Optional[str] = None
    scan_timestamp: Optional[str] = None
    repo_root: Optional[Path] = None

    def __post_init__(self):
        """Ensure deterministic ordering."""
        self.matches = sorted(self.matches, key=lambda m: str(m.path))
        self.violations = sorted(self.violations)
        if not self.scan_timestamp:
            self.scan_timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize to canonical JSON format."""
        return {
            "verdict": self.verdict,
            "context": self.context,
            "scan_timestamp": self.scan_timestamp,
            "inventory_hash": self.inventory_hash,
            "total_files_scanned": self.total_files_scanned,
            "protected_count": self.protected_count,
            "matches": [m.to_dict() for m in self.matches],
            "violations": self.violations
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to canonical JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'), sort_keys=True, indent=indent)

    def save_receipt(self, output_path: Path) -> None:
        """Save scan receipt to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    def print_summary(self) -> None:
        """Print human-readable scan summary."""
        print("=" * 60)
        print("PROTECTED ARTIFACTS SCAN SUMMARY")
        print("=" * 60)
        print(f"Verdict: {self.verdict}")
        print(f"Context: {self.context}")
        print(f"Timestamp: {self.scan_timestamp}")
        print(f"Files scanned: {self.total_files_scanned}")
        print(f"Protected artifacts found: {self.protected_count}")

        if self.violations:
            print("\n" + "=" * 60)
            print("VIOLATIONS:")
            print("=" * 60)
            for violation in self.violations:
                print(f"  - {violation}")

        if self.matches:
            print("\n" + "=" * 60)
            print("PROTECTED ARTIFACTS:")
            print("=" * 60)
            for match in self.matches:
                print(f"\n  Path: {match.path}")
                print(f"  Class: {match.artifact_class}")
                print(f"  Policy: {match.distribution_policy}")
                print(f"  Size: {match.size_bytes:,} bytes")

        print("\n" + "=" * 60)


class ProtectedScanner:
    """Scanner for protected artifacts in working tree."""

    def __init__(self, inventory: Optional[ProtectedInventory] = None, repo_root: Optional[Path] = None):
        """Initialize scanner.

        Args:
            inventory: Protected artifacts inventory (default: get_default_inventory())
            repo_root: Repository root path (default: current working directory)
        """
        self.inventory = inventory or get_default_inventory()
        self.repo_root = repo_root or Path.cwd()

    def scan_directory(
        self,
        directory: Optional[Path] = None,
        context: str = "working",
        compute_hashes: bool = False,
        exclusions: Optional[List[str]] = None
    ) -> ScanResult:
        """Scan a directory for protected artifacts.

        Args:
            directory: Directory to scan (default: repo_root)
            context: Distribution context ("working", "public", "internal")
            compute_hashes: Whether to compute SHA-256 hashes
            exclusions: List of glob patterns to exclude

        Returns:
            ScanResult with verdict and matches
        """
        if directory is None:
            directory = self.repo_root

        # Default exclusions (deterministic order)
        if exclusions is None:
            exclusions = sorted([
                ".git/**",
                ".ags-cas/**",  # Exclude CAS from working tree scans (separate inventory)
                "LAW/CONTRACTS/_runs/**",  # Temporary test artifacts
                "**/__pycache__/**",
                "**/*.pyc",
                "**/node_modules/**",
                "**/.pytest_cache/**"
            ])
        else:
            exclusions = sorted(exclusions)

        matches: List[ScanMatch] = []
        violations: List[str] = []
        total_files = 0

        # Scan directory recursively
        for file_path in directory.rglob("*"):
            try:
                if not file_path.is_file():
                    continue
            except (OSError, PermissionError):
                # Skip files that cannot be accessed (symlinks, permission issues, etc.)
                continue

            total_files += 1

            # Check exclusions
            rel_path = file_path.relative_to(directory)
            rel_path_str = str(rel_path).replace('\\', '/')

            excluded = False
            for pattern in exclusions:
                import fnmatch
                if fnmatch.fnmatch(rel_path_str, pattern):
                    excluded = True
                    break

            if excluded:
                continue

            # Check if protected
            matching_patterns = self.inventory.find_matching_patterns(rel_path)
            if not matching_patterns:
                continue

            # Found a protected artifact
            for pattern in matching_patterns:
                match = ScanMatch(
                    path=rel_path,
                    artifact_class=pattern.artifact_class.value,
                    distribution_policy=pattern.distribution_policy.value,
                    description=pattern.description,
                    size_bytes=file_path.stat().st_size
                )

                if compute_hashes:
                    match.compute_hash(directory)

                matches.append(match)

                # Check for violations
                if context == "public":
                    if pattern.distribution_policy == DistributionPolicy.PLAINTEXT_NEVER:
                        violations.append(
                            f"PLAINTEXT_NEVER artifact in public context: {rel_path} "
                            f"({pattern.artifact_class.value})"
                        )
                    elif pattern.distribution_policy == DistributionPolicy.PLAINTEXT_INTERNAL:
                        violations.append(
                            f"PLAINTEXT_INTERNAL artifact in public context: {rel_path} "
                            f"({pattern.artifact_class.value})"
                        )

        # Determine verdict
        if context == "public" and violations:
            verdict = "FAIL"
        elif matches:
            verdict = "WARN"  # Protected artifacts found, but may be acceptable
        else:
            verdict = "PASS"

        result = ScanResult(
            verdict=verdict,
            matches=matches,
            context=context,
            total_files_scanned=total_files,
            protected_count=len(matches),
            violations=violations,
            inventory_hash=self.inventory.hash(),
            repo_root=directory
        )

        return result

    def scan_file_list(
        self,
        files: List[Path],
        context: str = "working",
        compute_hashes: bool = False
    ) -> ScanResult:
        """Scan a specific list of files for protected artifacts.

        Args:
            files: List of file paths (relative to repo_root)
            context: Distribution context
            compute_hashes: Whether to compute SHA-256 hashes

        Returns:
            ScanResult with verdict and matches
        """
        matches: List[ScanMatch] = []
        violations: List[str] = []

        for file_path in sorted(files):
            # Normalize path
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.repo_root)
            else:
                rel_path = file_path

            # Check if protected
            matching_patterns = self.inventory.find_matching_patterns(rel_path)
            if not matching_patterns:
                continue

            # Found a protected artifact
            full_path = self.repo_root / rel_path
            if not full_path.exists():
                violations.append(f"File not found: {rel_path}")
                continue

            for pattern in matching_patterns:
                match = ScanMatch(
                    path=rel_path,
                    artifact_class=pattern.artifact_class.value,
                    distribution_policy=pattern.distribution_policy.value,
                    description=pattern.description,
                    size_bytes=full_path.stat().st_size
                )

                if compute_hashes:
                    match.compute_hash(self.repo_root)

                matches.append(match)

                # Check for violations
                if context == "public":
                    if pattern.distribution_policy == DistributionPolicy.PLAINTEXT_NEVER:
                        violations.append(
                            f"PLAINTEXT_NEVER artifact in public context: {rel_path} "
                            f"({pattern.artifact_class.value})"
                        )
                    elif pattern.distribution_policy == DistributionPolicy.PLAINTEXT_INTERNAL:
                        violations.append(
                            f"PLAINTEXT_INTERNAL artifact in public context: {rel_path} "
                            f"({pattern.artifact_class.value})"
                        )

        # Determine verdict
        if context == "public" and violations:
            verdict = "FAIL"
        elif matches:
            verdict = "WARN"
        else:
            verdict = "PASS"

        result = ScanResult(
            verdict=verdict,
            matches=matches,
            context=context,
            total_files_scanned=len(files),
            protected_count=len(matches),
            violations=violations,
            inventory_hash=self.inventory.hash()
        )

        return result


def scan_working_tree(
    repo_root: Optional[Path] = None,
    context: str = "working",
    compute_hashes: bool = False,
    fail_on_violations: bool = False
) -> ScanResult:
    """Convenience function to scan working tree.

    Args:
        repo_root: Repository root (default: current directory)
        context: Distribution context
        compute_hashes: Whether to compute hashes
        fail_on_violations: Exit with error code if violations found

    Returns:
        ScanResult
    """
    scanner = ProtectedScanner(repo_root=repo_root)
    result = scanner.scan_directory(context=context, compute_hashes=compute_hashes)

    if fail_on_violations and result.verdict == "FAIL":
        result.print_summary()
        sys.exit(1)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan for protected artifacts")
    parser.add_argument("--context", choices=["working", "public", "internal"], default="working",
                        help="Distribution context")
    parser.add_argument("--compute-hashes", action="store_true",
                        help="Compute SHA-256 hashes for all matches")
    parser.add_argument("--fail-on-violations", action="store_true",
                        help="Exit with error code if violations found")
    parser.add_argument("--output", type=Path,
                        help="Save scan receipt to file")
    parser.add_argument("directory", type=Path, nargs="?",
                        help="Directory to scan (default: current directory)")

    args = parser.parse_args()

    # Run scan
    scanner = ProtectedScanner(repo_root=args.directory or Path.cwd())
    result = scanner.scan_directory(
        context=args.context,
        compute_hashes=args.compute_hashes
    )

    # Print summary
    result.print_summary()

    # Save receipt if requested
    if args.output:
        result.save_receipt(args.output)
        print(f"\nReceipt saved to: {args.output}")

    # Exit with appropriate code
    if args.fail_on_violations and result.verdict == "FAIL":
        sys.exit(1)
    elif result.verdict == "WARN":
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)
