from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.cas_store import normalize_relpath
from PRIMITIVES.merkle import build_manifest_root


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
    assert normalize_relpath(r"a\\b/./c") == "a/b/c"
    assert normalize_relpath("./a/b") == "a/b"


def test_merkle_rejects_non_normalized_paths() -> None:
    # Non-normalized path should be rejected deterministically.
    with pytest.raises(ValueError, match=r"non-normalized path"):
        _ = build_manifest_root({r"a\\b": "a" * 64})

    # Traversal should be rejected via normalize_relpath.
    with pytest.raises(ValueError):
        _ = build_manifest_root({"../x": "a" * 64})

