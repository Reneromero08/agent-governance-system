from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
# sys.path cleanup

from CAPABILITY.PRIMITIVES.cas_store import CatalyticStore
from CAPABILITY.PRIMITIVES.hash_toolbelt import hash_describe, hash_read_text


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_cas_corruption_detected_by_hash_toolbelt() -> None:
    cas_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "cas" / "CAS"
    _rm(cas_root)
    cas = CatalyticStore(cas_root)

    h = cas.put_bytes(b"hello world\n")
    path = cas.object_path(h)
    assert path.exists()

    # Baseline: read/describe succeeds.
    _ = hash_describe(store=cas, hash_hex=h, max_bytes=64)

    # Corrupt 1 byte (hostile disk corruption).
    data = path.read_bytes()
    corrupted = bytearray(data)
    corrupted[0] ^= 0x01
    path.write_bytes(bytes(corrupted))

    with pytest.raises(ValueError, match=r"CAS_OBJECT_INTEGRITY_MISMATCH"):
        _ = hash_describe(store=cas, hash_hex=h, max_bytes=64)


def test_cas_truncation_detected_by_hash_toolbelt() -> None:
    cas_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "cas_trunc" / "CAS"
    _rm(cas_root)
    cas = CatalyticStore(cas_root)

    h = cas.put_bytes(b"abcdef" * 100)
    path = cas.object_path(h)
    assert path.exists()

    # Truncate object mid-stream.
    original = path.read_bytes()
    corrupted = bytearray(original[5:])
    path.write_bytes(bytes(corrupted))

    with pytest.raises(ValueError, match=r"CAS_OBJECT_INTEGRITY_MISMATCH"):
        _ = hash_read_text(store=cas, hash_hex=h, max_bytes=32)


def test_cas_partial_write_never_treated_as_present() -> None:
    cas_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "adversarial" / "cas_partial" / "CAS"
    _rm(cas_root)
    cas = CatalyticStore(cas_root)

    expected_bytes = b"partial-write-sim\n"
    h = hashlib.sha256(expected_bytes).hexdigest()
    final_path = cas.object_path(h)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Crash-simulate: only a temp file exists; final object path must still be missing.
    tmp_path = final_path.with_name(final_path.name + ".tmp.crash")
    tmp_path.write_bytes(expected_bytes[:5])
    with pytest.raises(FileNotFoundError):
        _ = hash_read_text(store=cas, hash_hex=h, max_bytes=64)

    # Crash-simulate: a file exists at the final path but is incomplete; must fail closed on integrity.
    final_path.write_bytes(expected_bytes[:5])
    with pytest.raises(ValueError, match=r"CAS_OBJECT_INTEGRITY_MISMATCH"):
        _ = hash_read_text(store=cas, hash_hex=h, max_bytes=64)
