from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MEMORY.LLM_PACKER.Engine import packer


def _dir_digest(root: Path) -> str:
    hasher = hashlib.sha256()
    # Sort files deterministically for digest.
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rel == "meta/PROVENANCE.json":
            continue
        rel_bytes = rel.encode("utf-8")
        data = p.read_bytes()
        hasher.update(rel_bytes + b"\0")
        hasher.update(hashlib.sha256(data).digest())
        hasher.update(len(data).to_bytes(8, "big"))
    return hasher.hexdigest()


def test_packer_determinism_catalytic_dpt() -> None:
    import os
    os.environ["LLM_PACKER_DETERMINISTIC_TIMESTAMP"] = "2025-01-01T00:00:00"
    
    packs_root = Path("MEMORY/LLM_PACKER/_packs").resolve()
    tmp_root = packs_root / "_system" / "_tmp_test_phase3_packing_hygiene"
    out_a = tmp_root / "pack-a"
    out_b = tmp_root / "pack-b"

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        packer.make_pack(
            scope_key="catalytic-dpt",
            mode="full",
            profile="full",
            split_lite=False,
            out_dir=out_a,
            combined=False,
            stamp=None,
            zip_enabled=False,
            max_total_bytes=50 * 1024 * 1024,
            max_entry_bytes=2 * 1024 * 1024,
            max_entries=50_000,
            allow_duplicate_hashes=None,
        )
        packer.make_pack(
            scope_key="catalytic-dpt",
            mode="full",
            profile="full",
            split_lite=False,
            out_dir=out_b,
            combined=False,
            stamp=None,
            zip_enabled=False,
            max_total_bytes=50 * 1024 * 1024,
            max_entry_bytes=2 * 1024 * 1024,
            max_entries=50_000,
            allow_duplicate_hashes=None,
        )

        assert _dir_digest(out_a) == _dir_digest(out_b)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


@pytest.mark.parametrize(
    "limits,expected",
    [
        ({"max_total_bytes": 1, "max_entry_bytes": 2 * 1024 * 1024, "max_entries": 50_000}, "PACK_LIMIT_EXCEEDED:max_total_bytes"),
        ({"max_total_bytes": 50 * 1024 * 1024, "max_entry_bytes": 1, "max_entries": 50_000}, "PACK_LIMIT_EXCEEDED:max_entry_bytes"),
        ({"max_total_bytes": 50 * 1024 * 1024, "max_entry_bytes": 2 * 1024 * 1024, "max_entries": 1}, "PACK_LIMIT_EXCEEDED:max_entries"),
    ],
)
def test_packer_limits_fail_closed(limits: dict, expected: str) -> None:
    packs_root = Path("MEMORY/LLM_PACKER/_packs").resolve()
    tmp_root = packs_root / "_system" / "_tmp_test_phase3_packing_hygiene_limits"
    out_dir = tmp_root / "pack"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    try:
        with pytest.raises(ValueError, match=re.escape(expected)):
            packer.make_pack(
                scope_key="catalytic-dpt",
                mode="full",
                profile="full",
                split_lite=False,
                out_dir=out_dir,
                combined=False,
                stamp=None,
                zip_enabled=False,
                max_total_bytes=int(limits["max_total_bytes"]),
                max_entry_bytes=int(limits["max_entry_bytes"]),
                max_entries=int(limits["max_entries"]),
                allow_duplicate_hashes=None,
            )
        assert not out_dir.exists()
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


def test_repo_state_manifest_validation_and_dedup() -> None:
    # Logic is now internal to build_state_manifest and enforce_included_repo_limits
    from MEMORY.LLM_PACKER.Engine.packer.core import enforce_included_repo_limits, PackLimits
    
    a = {"path": "a.txt", "hash": "0" * 64, "size": 1}
    b = {"path": "b.txt", "hash": "1" * 64, "size": 2}
    entries = [a, b]
    
    limits = PackLimits(max_total_bytes=100, max_entry_bytes=100, max_entries=100, allow_duplicate_hashes=False)
    enforce_included_repo_limits(entries, limits)

    with pytest.raises(ValueError, match=r"PACK_LIMIT_EXCEEDED:duplicate_hashes"):
        enforce_included_repo_limits(
            [{"path": "a.txt", "hash": "f" * 64, "size": 1}, {"path": "b.txt", "hash": "f" * 64, "size": 1}],
            limits
        )


def test_verify_manifest_detects_tamper(tmp_path: Path) -> None:
    # Packer now puts meta/ and repo/ inside archive/pack.zip in production, 
    # but verify_manifest still supports a flat pack_dir for testing.
    # Actually verify_manifest in core.py checks for meta/REPO_STATE.json.
    
    pack_dir = tmp_path / "pack"
    (pack_dir / "meta").mkdir(parents=True)
    (pack_dir / "repo").mkdir(parents=True)

    rel = "hello.txt"
    data = b"hello\n"
    (pack_dir / "repo" / rel).write_bytes(data)

    repo_state = {
        "canon_version": "x",
        "grammar_version": "1.0",
        "scope": "test",
        "files": [
            {"path": rel, "hash": hashlib.sha256(data).hexdigest(), "size": len(data)},
        ],
    }
    
    from MEMORY.LLM_PACKER.Engine.packer.core import write_json
    write_json(pack_dir / "meta" / "REPO_STATE.json", repo_state)

    ok, errors = packer.verify_manifest(pack_dir)
    assert ok, errors

    (pack_dir / "repo" / rel).write_bytes(b"HELLO\n")
    ok, errors = packer.verify_manifest(pack_dir)
    assert not ok
    assert any("Hash mismatch" in e for e in errors)

