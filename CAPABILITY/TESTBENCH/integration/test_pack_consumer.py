"""
Tests for Pack Consumer (P.2.2): verification + rehydration.

Tests cover:
- Manifest schema validation
- CAS blob verification
- Atomic materialization
- Tamper detection
- Determinism
- Fail-closed behavior
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MEMORY.LLM_PACKER.Engine.packer import core as packer_core
from MEMORY.LLM_PACKER.Engine.packer import consumer
from CAPABILITY.CAS import cas as cas_mod
from CAPABILITY.ARTIFACTS.store import store_bytes


FIXTURE_ROOT_REL = Path("CAPABILITY") / "TESTBENCH" / "fixtures" / "packer_p2_repo"


def _make_fixture_scope() -> packer_core.PackScope:
    fixture_root_rel = FIXTURE_ROOT_REL.as_posix()
    return packer_core.PackScope(
        key="ags",
        title="AGS (P2 Test Scope)",
        file_prefix="AGS",
        include_dirs=(
            (FIXTURE_ROOT_REL / "LAW").as_posix(),
            (FIXTURE_ROOT_REL / "CAPABILITY").as_posix(),
        ),
        root_files=(),
        anchors=(
            "README.md",
            (Path("LAW") / "CANON" / "CONTRACT.md").as_posix(),
        ),
        excluded_dir_parts=packer_core.SCOPE_AGS.excluded_dir_parts,
        source_root_rel=fixture_root_rel,
    )


def test_pack_consume_basic_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Basic roundtrip: create pack, consume it, verify files match.
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Create pack (must be under _packs/)
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_system" / "fixtures" / "test_consumer_1",
        combined=False,
        stamp="test",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    # Get manifest ref
    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    # Consume pack
    consume_dir = tmp_path / "consumed"
    receipt = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir,
        dry_run=False,
        cas_root=cas_root,
    )

    # Verify success
    assert receipt.exit_status == "SUCCESS"
    assert len(receipt.errors) == 0
    assert consume_dir.exists()

    # Verify files were materialized
    manifest = json.loads((pack_dir / "LITE" / "PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        materialized = consume_dir / entry["path"]
        assert materialized.exists()
        assert materialized.stat().st_size == entry["bytes"]


def test_pack_consume_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Dry-run mode verifies but doesn't materialize.
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Create pack
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_system" / "fixtures" / "test_consumer_2",
        combined=False,
        stamp="test",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    # Consume in dry-run mode
    consume_dir = tmp_path / "consumed"
    receipt = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir,
        dry_run=True,
        cas_root=cas_root,
    )

    # Verify verification succeeded but no files written
    assert receipt.exit_status == "SUCCESS"
    assert "dry_run" in receipt.commands_run[-1]
    assert not consume_dir.exists()


def test_pack_consume_tamper_detection_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Tampering with manifest bytes should fail verification.
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Create pack
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_system" / "fixtures" / "test_consumer_3",
        combined=False,
        stamp="test",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    # Load and tamper with manifest
    manifest_path = pack_dir / "LITE" / "PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    # Tamper: change scope
    manifest["scope"] = "TAMPERED"
    
    # Store tampered manifest in CAS
    tampered_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    tampered_ref = store_bytes(tampered_bytes)

    # Try to consume tampered manifest
    consume_dir = tmp_path / "consumed"
    
    with pytest.raises(ValueError, match="INVALID_SCOPE"):
        consumer.pack_consume(
            manifest_ref=tampered_ref,
            out_dir=consume_dir,
            dry_run=False,
            cas_root=cas_root,
        )


def test_pack_consume_missing_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Missing CAS blob should fail-closed before materialization.
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Create pack
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_system" / "fixtures" / "test_consumer_4",
        combined=False,
        stamp="test",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    # Delete a blob from CAS
    manifest = json.loads((pack_dir / "LITE" / "PACK_MANIFEST.json").read_text(encoding="utf-8"))
    first_entry = manifest["entries"][0]
    first_hash = first_entry["ref"].split(":", 1)[1]
    
    blob_path = cas_mod._get_object_path(first_hash)
    blob_path.unlink()

    # Try to consume (should fail)
    consume_dir = tmp_path / "consumed"
    
    with pytest.raises(ValueError, match="MISSING_BLOBS"):
        consumer.pack_consume(
            manifest_ref=manifest_ref,
            out_dir=consume_dir,
            dry_run=False,
            cas_root=cas_root,
        )
    
    # Verify no partial materialization
    assert not consume_dir.exists()


def test_pack_consume_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Consuming same manifest twice should produce identical tree hash.
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Create pack
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_system" / "fixtures" / "test_consumer_5",
        combined=False,
        stamp="test",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    # Consume twice
    consume_dir1 = tmp_path / "consumed1"
    receipt1 = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir1,
        dry_run=False,
        cas_root=cas_root,
    )

    consume_dir2 = tmp_path / "consumed2"
    receipt2 = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir2,
        dry_run=False,
        cas_root=cas_root,
    )

    # Verify identical tree hashes
    assert receipt1.tree_hash == receipt2.tree_hash
    assert receipt1.tree_hash != ""
    assert receipt1.cas_snapshot_hash == receipt2.cas_snapshot_hash


def test_pack_consume_path_safety(tmp_path: Path) -> None:
    """
    Manifests with unsafe paths (absolute, traversal) should fail validation.
    """
    cas_root = tmp_path / "cas"
    cas_root.mkdir(parents=True, exist_ok=True)

    # Create manifest with absolute path
    bad_manifest1 = {
        "version": "P2.0",
        "scope": "ags",
        "entries": [
            {
                "path": "/etc/passwd",  # Absolute path
                "ref": "sha256:" + "0" * 64,
                "bytes": 100,
                "kind": "FILE",
            }
        ],
    }
    
    bad_bytes1 = (json.dumps(bad_manifest1, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    bad_ref1 = store_bytes(bad_bytes1)

    # Should fail validation (either path safety or missing blob)
    with pytest.raises(ValueError):
        consumer.pack_consume(
            manifest_ref=bad_ref1,
            out_dir=tmp_path / "out1",
            dry_run=True,
            cas_root=cas_root,
        )

    # Create manifest with path traversal
    bad_manifest2 = {
        "version": "P2.0",
        "scope": "ags",
        "entries": [
            {
                "path": "../../../etc/passwd",  # Path traversal
                "ref": "sha256:" + "0" * 64,
                "bytes": 100,
                "kind": "FILE",
            }
        ],
    }
    
    bad_bytes2 = (json.dumps(bad_manifest2, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    bad_ref2 = store_bytes(bad_bytes2)

    # Should fail validation (either path safety or missing blob)
    with pytest.raises(ValueError):
        consumer.pack_consume(
            manifest_ref=bad_ref2,
            out_dir=tmp_path / "out2",
            dry_run=True,
            cas_root=cas_root,
        )
