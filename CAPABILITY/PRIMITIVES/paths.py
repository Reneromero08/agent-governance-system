"""
Shared repo-root and path utilities for AGS.

Provides a single source of truth for REPO_ROOT resolution so callers
no longer need ``Path(__file__).resolve().parents[N]`` with varying N.

Usage::
    from CAPABILITY.PRIMITIVES.paths import repo_root, normalize_relpath
    PROJECT_ROOT = repo_root()
"""

from pathlib import Path
from typing import Union

# -- Repo root detection (cached) ------------------------------------------

_ANCHOR_FILE = "AGENTS.md"
_cached_root = None  # type: Path | None


def repo_root() -> Path:
    """Return the repository root directory.

    Walks up from *this* file until a directory containing ``AGENTS.md``
    is found.  The result is cached after the first call.
    Raises ``FileNotFoundError`` if the anchor cannot be located.
    """
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


# -- Path normalisation (delegates to cas_store) ----------------------------

def normalize_relpath(path: Union[str, "Path"]) -> str:
    """Normalise a repo-relative path.

    Delegates to ``cas_store.normalize_relpath`` which enforces:
    forward slashes, no leading ``./``, no ``..``, no trailing ``/``.
    """
    from CAPABILITY.PRIMITIVES.cas_store import normalize_relpath as _nr
    return _nr(str(path))


# -- Safe resolution under repo root ---------------------------------------

def resolve_under_root(relpath: Union[str, "Path"]) -> Path:
    """Resolve *relpath* under the repo root and verify containment.

    Returns the resolved ``Path``.  Raises ``ValueError`` if the
    resolved path escapes the repository tree.
    """
    root = repo_root()
    normed = normalize_relpath(relpath)
    resolved = (root / normed).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            "Path escapes repo root: %s -> %s" % (relpath, resolved)
        )
    return resolved
