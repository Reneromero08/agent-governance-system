"""
CAT-DPT Filesystem Guard

Enforces allowed roots at write-time during execution (Layer 2 runtime guard).
All file writes must go through this guard to ensure contract compliance.

Usage:
    from CATALYTIC_DPT.PRIMITIVES.fs_guard import FilesystemGuard

    guard = FilesystemGuard(
        allowed_roots=["CONTRACTS/_runs", "CORTEX/_generated"],
        forbidden_paths=["CANON", "AGENTS.md", "BUILD", ".git"],
        project_root=Path("/path/to/project")
    )

    # Before any write operation
    valid, error = guard.validate_write_path("CONTRACTS/_runs/output.json")
    if not valid:
        raise RuntimeError(f"Write violation: {error}")

    # Perform the write
    Path("CONTRACTS/_runs/output.json").write_text(data)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FilesystemGuard:
    """Runtime filesystem guard enforcing allowed roots at write-time."""

    def __init__(
        self,
        allowed_roots: List[str],
        forbidden_paths: List[str],
        project_root: Path,
    ):
        """
        Initialize filesystem guard.

        Args:
            allowed_roots: List of allowed root paths (relative to project_root)
            forbidden_paths: List of forbidden paths (relative to project_root)
            project_root: Project root directory
        """
        self.allowed_roots = allowed_roots
        self.forbidden_paths = forbidden_paths
        self.project_root = Path(project_root).resolve()

    def validate_write_path(self, path: str | Path) -> Tuple[bool, Optional[Dict]]:
        """
        Validate a write path before allowing the write operation.

        Rules:
        - Must be relative (or resolve within project_root)
        - Must not contain traversal (..)
        - Must be under an allowed root
        - Must not touch forbidden paths

        Args:
            path: Path to validate (can be relative or absolute)

        Returns:
            Tuple of (valid, error) where error is a validation_error dict or None
        """
        path_obj = Path(path)

        # Convert to string for comparison
        path_str = str(path)

        # Check if absolute - if so, must be within project_root
        if path_obj.is_absolute():
            try:
                resolved = path_obj.resolve()
                # Must be within project_root
                relative_path = resolved.relative_to(self.project_root)
                path_str = str(relative_path).replace("\\", "/")
            except ValueError:
                return False, {
                    "code": "WRITE_GUARD_PATH_ABSOLUTE",
                    "severity": "error",
                    "message": f"Write path {path} is outside project root",
                    "path": ["write_guard", str(path)],
                }
        else:
            # Relative path - resolve to check for escapes
            try:
                resolved = (self.project_root / path).resolve()
                relative_path = resolved.relative_to(self.project_root)
                path_str = str(relative_path).replace("\\", "/")
            except ValueError:
                return False, {
                    "code": "WRITE_GUARD_PATH_ESCAPE",
                    "severity": "error",
                    "message": f"Write path {path} escapes project root",
                    "path": ["write_guard", str(path)],
                }

        # Check if touches forbidden paths (check first - most specific)
        for forbidden in self.forbidden_paths:
            if path_str.startswith(forbidden) or path_str == forbidden:
                return False, {
                    "code": "WRITE_GUARD_PATH_FORBIDDEN",
                    "severity": "error",
                    "message": f"Write path {path} touches forbidden path: {forbidden}",
                    "path": ["write_guard", str(path)],
                }

        # Check for traversal in original path (before resolution)
        if ".." in Path(path).parts:
            return False, {
                "code": "WRITE_GUARD_PATH_TRAVERSAL",
                "severity": "error",
                "message": f"Write path {path} contains traversal (..)",
                "path": ["write_guard", str(path)],
            }

        # Check if under allowed roots
        is_allowed = any(
            path_str.startswith(root) or path_str == root
            for root in self.allowed_roots
        )

        if not is_allowed:
            return False, {
                "code": "WRITE_GUARD_PATH_NOT_ALLOWED",
                "severity": "error",
                "message": f"Write path {path} not under allowed roots: {self.allowed_roots}",
                "path": ["write_guard", str(path)],
            }

        # Valid
        return True, None

    def guarded_write_text(
        self, path: str | Path, content: str, encoding: str = "utf-8"
    ) -> None:
        """
        Write text to a file with guard validation.

        Args:
            path: Path to write
            content: Text content to write
            encoding: Text encoding (default: utf-8)

        Raises:
            RuntimeError: If write path violates contract
        """
        valid, error = self.validate_write_path(path)
        if not valid:
            raise RuntimeError(
                f"[WRITE_GUARD] {error['code']}: {error['message']}"
            )

        Path(path).write_text(content, encoding=encoding)

    def guarded_write_bytes(self, path: str | Path, content: bytes) -> None:
        """
        Write bytes to a file with guard validation.

        Args:
            path: Path to write
            content: Binary content to write

        Raises:
            RuntimeError: If write path violates contract
        """
        valid, error = self.validate_write_path(path)
        if not valid:
            raise RuntimeError(
                f"[WRITE_GUARD] {error['code']}: {error['message']}"
            )

        Path(path).write_bytes(content)

    def guarded_mkdir(self, path: str | Path, parents: bool = False, exist_ok: bool = False) -> None:
        """
        Create directory with guard validation.

        Args:
            path: Directory path to create
            parents: Create parent directories if needed
            exist_ok: Don't raise if directory exists

        Raises:
            RuntimeError: If path violates contract
        """
        valid, error = self.validate_write_path(path)
        if not valid:
            raise RuntimeError(
                f"[WRITE_GUARD] {error['code']}: {error['message']}"
            )

        Path(path).mkdir(parents=parents, exist_ok=exist_ok)
