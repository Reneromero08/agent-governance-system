import hashlib
import io
import sys
from pathlib import Path

import pytest

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.cas_store import CatalyticStore, normalize_relpath


def _sha256_file_hex(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def test_put_bytes_same_bytes_same_hash(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "cas")
    data = b"hello world"
    h1 = store.put_bytes(data)
    h2 = store.put_bytes(data)
    assert h1 == h2
    assert h1 == hashlib.sha256(data).hexdigest()


def test_put_bytes_idempotent_does_not_change_bytes(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "cas")
    data = b"x" * 1024
    h = store.put_bytes(data)
    obj_path = (tmp_path / "cas" / "objects" / h[0:2] / h[2:4] / h)
    before = obj_path.read_bytes()
    _ = store.put_bytes(data)
    after = obj_path.read_bytes()
    assert before == after == data


def test_put_stream_matches_put_bytes(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "cas")
    data = b"stream-me" * 1000
    hb = store.put_bytes(data)
    hs = store.put_stream(io.BytesIO(data), chunk_size=257)
    assert hs == hb


def test_large_stream_roundtrip(tmp_path: Path) -> None:
    store_root = tmp_path / "cas"
    store = CatalyticStore(store_root)

    big_path = tmp_path / "big.bin"
    pattern = b"0123456789abcdef" * 4096  # 64 KiB
    target_size = 20 * 1024 * 1024  # 20 MiB

    written = 0
    h = hashlib.sha256()
    with open(big_path, "wb") as f:
        while written < target_size:
            take = pattern if (target_size - written) >= len(pattern) else pattern[: (target_size - written)]
            f.write(take)
            h.update(take)
            written += len(take)

    expected_hash = h.hexdigest()

    with open(big_path, "rb") as src:
        got_hash = store.put_stream(src, chunk_size=1024 * 1024)
    assert got_hash == expected_hash

    out_path = tmp_path / "out.bin"
    with open(out_path, "wb") as out:
        store.get_stream(got_hash, out, chunk_size=1024 * 1024)

    assert _sha256_file_hex(out_path) == expected_hash
    assert out_path.stat().st_size == big_path.stat().st_size


def test_deterministic_object_path_across_instances(tmp_path: Path) -> None:
    store_root = tmp_path / "cas"
    data = b"deterministic"

    s1 = CatalyticStore(store_root)
    h = s1.put_bytes(data)

    expected_path = store_root / "objects" / h[0:2] / h[2:4] / h
    assert expected_path.exists()

    s2 = CatalyticStore(store_root)
    h2 = s2.put_bytes(data)
    assert h2 == h
    assert expected_path.read_bytes() == data


def test_reject_invalid_hash_and_missing_hash(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "cas")

    with pytest.raises(ValueError):
        store.get_bytes("not-a-hash")

    with pytest.raises(ValueError):
        store.get_bytes("A" * 64)  # uppercase not allowed

    valid_missing = "0" * 64
    with pytest.raises(FileNotFoundError):
        store.get_bytes(valid_missing)


def test_normalize_relpath_accepts_and_normalizes() -> None:
    assert normalize_relpath(r"a\b\c") == "a/b/c"
    assert normalize_relpath("a/./b/./c") == "a/b/c"
    assert normalize_relpath("./a/b") == "a/b"
    assert normalize_relpath("a//b///c") == "a/b/c"
    assert normalize_relpath(".") == "."


def test_normalize_relpath_rejects_absolute_and_traversal() -> None:
    with pytest.raises(ValueError):
        normalize_relpath("/abs/path")

    with pytest.raises(ValueError):
        normalize_relpath(r"\\server\share\file")

    with pytest.raises(ValueError):
        normalize_relpath(r"C:\abs\path")

    with pytest.raises(ValueError):
        normalize_relpath("C:/abs/path")

    with pytest.raises(ValueError):
        normalize_relpath("../nope")

    with pytest.raises(ValueError):
        normalize_relpath("a/../b")

