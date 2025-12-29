import hashlib
import sys
from pathlib import Path

import pytest

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
# sys.path cleanup

from CAPABILITY.PRIMITIVES.merkle import build_manifest_root, verify_manifest_root


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _leaf_hash(path: str, bytes_hash: str) -> str:
    return hashlib.sha256(f"{path}:{bytes_hash}".encode("utf-8")).hexdigest()


def _node_hash(left: str, right: str) -> str:
    return hashlib.sha256((left + right).encode("ascii")).hexdigest()


def test_deterministic_root_across_runs() -> None:
    manifest = {"a.txt": _sha256_hex("a"), "b.txt": _sha256_hex("b"), "c.txt": _sha256_hex("c")}
    r1 = build_manifest_root(manifest)
    r2 = build_manifest_root(manifest)
    assert r1 == r2


def test_path_ordering_independent_of_input_order() -> None:
    manifest_a = {"b.txt": _sha256_hex("b"), "a.txt": _sha256_hex("a"), "c.txt": _sha256_hex("c")}
    manifest_b = {"c.txt": _sha256_hex("c"), "b.txt": _sha256_hex("b"), "a.txt": _sha256_hex("a")}
    assert build_manifest_root(manifest_a) == build_manifest_root(manifest_b)


def test_duplicate_path_rejection_via_non_normalized_collision() -> None:
    # Two raw keys that would normalize to the same path must be rejected.
    # Dict preserves both because the raw strings differ.
    manifest = {"a/b": _sha256_hex("x"), "a//b": _sha256_hex("y")}
    with pytest.raises(ValueError):
        build_manifest_root(manifest)


def test_duplicate_hash_under_different_paths_rejected() -> None:
    h = _sha256_hex("same")
    manifest = {"a.txt": h, "b.txt": h}
    with pytest.raises(ValueError):
        build_manifest_root(manifest)


def test_odd_leaf_padding_correctness() -> None:
    # 3 leaves -> pad to 4 by duplicating last leaf.
    manifest = {"a": _sha256_hex("a"), "b": _sha256_hex("b"), "c": _sha256_hex("c")}
    root = build_manifest_root(manifest)

    ha = _leaf_hash("a", manifest["a"])
    hb = _leaf_hash("b", manifest["b"])
    hc = _leaf_hash("c", manifest["c"])
    hcc = hc
    lvl1 = [_node_hash(ha, hb), _node_hash(hc, hcc)]
    expected = _node_hash(lvl1[0], lvl1[1])
    assert root == expected


def test_empty_manifest_rejected() -> None:
    with pytest.raises(ValueError):
        build_manifest_root({})


def test_verify_manifest_root_true_false() -> None:
    manifest = {"a.txt": _sha256_hex("a"), "b.txt": _sha256_hex("b")}
    root = build_manifest_root(manifest)
    assert verify_manifest_root(manifest, root) is True
    assert verify_manifest_root(manifest, "0" * 64) is False


def test_reject_non_normalized_paths() -> None:
    manifest = {"./a.txt": _sha256_hex("a")}
    with pytest.raises(ValueError):
        build_manifest_root(manifest)


def test_reject_invalid_hashes_and_expected_root_format() -> None:
    manifest = {"a.txt": "not-a-hash"}
    with pytest.raises(ValueError):
        build_manifest_root(manifest)

    good_manifest = {"a.txt": _sha256_hex("a")}
    with pytest.raises(ValueError):
        verify_manifest_root(good_manifest, "A" * 64)

