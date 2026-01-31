"""
Centralized Path Normalization for Catalytic Operations

Provides consistent cross-platform path handling with security validations.
All catalytic code should use these utilities instead of ad-hoc normalization.
"""

import os
from pathlib import Path
from typing import Union

from CAPABILITY.PRIMITIVES.catalytic_errors import (
    CatalyticError,
    CAT_005_DOMAIN_VIOLATION,
    CAT_008_PATH_INVALID,
)


def normalize_catalytic_path(path: Union[str, Path], *, allow_absolute: bool = False) -> str:
    """
    Canonical path normalization for catalytic operations.

    - Converts to forward slashes (cross-platform)
    - Rejects symlinks
    - Validates no parent traversal (..)
    - Handles Windows drive letters

    Args:
        path: Path to normalize (string or Path object)
        allow_absolute: If False, rejects absolute paths

    Returns:
        Normalized path string with forward slashes

    Raises:
        CatalyticError: If path violates constraints
    """
    path_obj = Path(path) if isinstance(path, str) else path

    # Check for symlinks (security)
    if path_obj.exists() and path_obj.is_symlink():
        raise CatalyticError(
            CAT_005_DOMAIN_VIOLATION,
            f"Symlinks not allowed: {path}",
            {"path": str(path), "reason": "symlink"},
        )

    # Convert to string with forward slashes
    path_str = str(path).replace(os.sep, "/")

    # Check for parent traversal
    parts = path_str.split("/")
    if ".." in parts:
        raise CatalyticError(
            CAT_008_PATH_INVALID,
            f"Parent traversal not allowed: {path}",
            {"path": str(path), "reason": "parent_traversal"},
        )

    # Check for absolute paths (if not allowed)
    if not allow_absolute:
        # Handle Windows drive letters (C:/) and Unix absolute paths (/)
        if path_obj.is_absolute():
            raise CatalyticError(
                CAT_008_PATH_INVALID,
                f"Absolute paths not allowed: {path}",
                {"path": str(path), "reason": "absolute_path"},
            )

    return path_str


def normalize_relpath(path: Union[str, Path]) -> str:
    """
    Normalize a relative path for consistent cross-platform comparison.

    This is a simpler version that just normalizes separators without
    security checks. Use normalize_catalytic_path() for untrusted input.

    Args:
        path: Relative path to normalize

    Returns:
        Normalized path string with forward slashes
    """
    return str(path).replace(os.sep, "/").replace("\\", "/")


def validate_path_in_root(path: Path, root: Path) -> bool:
    """
    Validate that a path is contained within a root directory.

    Uses resolve() to handle symlinks and normalize the path.

    Args:
        path: Path to validate
        root: Root directory that must contain path

    Returns:
        True if path is within root, False otherwise
    """
    try:
        resolved_path = path.resolve()
        resolved_root = root.resolve()
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def reject_symlink(path: Path) -> None:
    """
    Raise an error if the path is a symlink.

    Args:
        path: Path to check

    Raises:
        CatalyticError: If path is a symlink
    """
    if path.is_symlink():
        raise CatalyticError(
            CAT_005_DOMAIN_VIOLATION,
            f"Symlink not allowed in catalytic domain: {path}",
            {"path": str(path), "reason": "symlink"},
        )


def safe_relative_to(path: Path, base: Path) -> str:
    """
    Safely compute relative path with forward-slash normalization.

    Args:
        path: Path to make relative
        base: Base path to compute relative from

    Returns:
        Normalized relative path string

    Raises:
        ValueError: If path is not relative to base
    """
    rel = path.relative_to(base)
    return normalize_relpath(rel)
