from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

# Add CAPABILITY to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.cas_store import CatalyticStore
from CAPABILITY.PRIMITIVES.hash_toolbelt import (
    hash_ast,
    hash_describe,
    hash_grep,
    hash_read_text,
)


def test_read_enforces_max_bytes(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    data = b"a" * 10000
    h = store.put_bytes(data)
    out = hash_read_text(store=store, hash_hex=h, max_bytes=123, start=0, end=None)
    header, body = out.split("\n", 1)
    assert "bytes_returned=123" in header
    assert len(body.encode("utf-8")) <= 123


def test_read_range_and_clamp(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    data = b"0123456789"
    h = store.put_bytes(data)
    out = hash_read_text(store=store, hash_hex=h, max_bytes=100, start=3, end=8)
    assert out.split("\n", 1)[1] == "34567"


def test_grep_bounds(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    text = (b"line0\n" + b"needle here\n") * 2000  # big
    h = store.put_bytes(text)
    matches = hash_grep(store=store, hash_hex=h, pattern="needle", max_bytes=200, max_matches=2)
    assert len(matches) <= 2
    for m in matches:
        assert "needle" in m.snippet


def test_describe_deterministic_and_bounded(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    data = b"hello world\n" * 1000
    h = store.put_bytes(data)
    a = hash_describe(store=store, hash_hex=h, max_bytes=64)
    b = hash_describe(store=store, hash_hex=h, max_bytes=64)
    assert a == b
    assert '"bytes_preview_len":64' in a


def test_ast_outline_and_truncation(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    src = b"import os\n\nclass A:\n  def m(self):\n    pass\n\n" + b"def f():\n  return 1\n"
    h = store.put_bytes(src)
    out = hash_ast(store=store, hash_hex=h, max_bytes=4096, max_nodes=2, max_depth=2)
    assert '"truncated":true' in out
    assert "TRUNCATED" in out


def test_ast_unsupported_binary(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    # Deterministic binary-like bytes: invalid UTF-8 sequences.
    data = bytes([0, 255, 254, 253]) * 50
    h = store.put_bytes(data)
    with pytest.raises(ValueError) as e:
        hash_ast(store=store, hash_hex=h, max_bytes=256, max_nodes=10, max_depth=3)
    assert str(e.value) == "UNSUPPORTED_AST_FORMAT"


def test_invalid_hash_rejects(tmp_path: Path) -> None:
    store = CatalyticStore(tmp_path / "CAS")
    with pytest.raises(ValueError):
        hash_read_text(store=store, hash_hex="A" * 65)
