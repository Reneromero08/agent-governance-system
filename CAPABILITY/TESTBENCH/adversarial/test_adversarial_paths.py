import sys
from pathlib import Path

# Correct the path calculation to use `parents[3]` instead of `parents[2]`.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Ensure that all local relative imports are updated for this new location.
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
    assert normalize_relpath(r"a\123/./456") == r"a\123/456"
    assert normalize_relpath("./a/b") == "a/b"


def test_merkle_rejects_non_normalized_paths() -> None:
    # Non-normalized path should be rejected deterministically.
    with pytest.raises(ValueError, match=r"non-normalized path"):
        _ = build_manifest_root({r"a\123": "a" * 64})

    # Traversal should be rejected via normalize_relpath.
    with pytest.raises(ValueError):
        _ = build_manifest_root({"../x": "a" * 64})
