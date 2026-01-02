import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    # normalize_relpath ALWAYS converts \ to /
    assert normalize_relpath(r"a\123/./456") == "a/123/456"
    assert normalize_relpath("./a/b") == "a/b"

def test_merkle_rejects_non_normalized_paths() -> None:
    # Non-normalized path should be rejected (contains \ but should be /)
    with pytest.raises(ValueError, match=r"non-normalized path"):
        _ = build_manifest_root({r"a\123": "a" * 64})

    # Traversal should be rejected via normalize_relpath internally
    with pytest.raises(ValueError):
        _ = build_manifest_root({"../x": "a" * 64})
