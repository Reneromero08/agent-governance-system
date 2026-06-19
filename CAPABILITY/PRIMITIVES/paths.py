"""
Shared repo-root and path utilities for AGS.

Provides a single source of truth for repo-root resolution and host-independent
interpretation of repository-relative path strings. A Windows path must remain
absolute when validated on Linux, and backslashes in a relative path must remain
separators rather than becoming literal filename characters.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Union

_ANCHOR_FILE = "AGENTS.md"
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_cached_root: Path | None = None


def repo_root() -> Path:
    """Return the repository root directory, cached after first discovery."""
    global _cached_root
    if _cached_root is not None:
        return _cached_root

    current = Path(__file__).resolve().parent
    while True:
        if (current / _ANCHOR_FILE).is_file():
            _cached_root = current
            return _cached_root
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        "Could not locate repo root (no %s found in ancestors)" % _ANCHOR_FILE
    )


def portable_path_text(path: Union[str, Path]) -> str:
    """Return *path* with repository separators normalized to ``/``."""
    return str(path).replace("\\", "/")


def is_portable_absolute(path: Union[str, Path]) -> bool:
    """Recognize POSIX, drive-qualified Windows, and UNC absolute paths.

    ``Path.is_absolute()`` follows the current host OS and therefore treats
    ``C:\\tmp`` as relative on Linux. Governance validation must interpret the
    declared syntax, not the machine running the check.
    """
    text = portable_path_text(path)
    return text.startswith("/") or text.startswith("//") or bool(_WINDOWS_DRIVE.match(text))


def portable_parts(path: Union[str, Path]) -> tuple[str, ...]:
    """Return separator-normalized path components without collapsing ``..``."""
    return tuple(part for part in portable_path_text(path).split("/") if part not in ("", "."))


def normalize_relpath(path: Union[str, Path]) -> str:
    """Normalize and validate a repository-relative path.

    Enforces forward slashes, no absolute syntax, no traversal, no leading
    ``./``, and no trailing slash. Empty paths normalize to ``""``.
    """
    text = portable_path_text(path)
    if is_portable_absolute(text):
        raise ValueError(f"Absolute path not allowed: {text}")

    parts: list[str] = []
    for part in portable_parts(text):
        if part == "..":
            raise ValueError(f"Path traversal ('..') not allowed: {text}")
        parts.append(part)
    return "/".join(parts)


def resolve_under_root(
    relpath: Union[str, Path],
    *,
    root: Path | None = None,
) -> Path:
    """Resolve *relpath* under *root* and verify component-safe containment."""
    project_root = (root or repo_root()).resolve()
    normed = normalize_relpath(relpath)
    resolved = (project_root / Path(*normed.split("/"))).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repo root: {relpath} -> {resolved}") from exc
    return resolved
