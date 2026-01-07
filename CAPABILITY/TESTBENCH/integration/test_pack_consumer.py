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
import uuid

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MEMORY.LLM_PACKER.Engine.packer import core as packer_core
from MEMORY.LLM_PACKER.Engine.packer import consumer
from MEMORY.LLM_PACKER.Engine.packer.firewall_writer import PackerWriter
from CAPABILITY.CAS import cas as cas_mod
from CAPABILITY.ARTIFACTS.store import store_bytes
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter


FIXTURE_ROOT_REL = Path("CAPABILITY") / "TESTBENCH" / "fixtures" / "packer_p2_repo"


def _make_cas_test_writer(tmp_path: Path) -> GuardedWriter:
    """Create a GuardedWriter configured for CAS operations in test temp directories."""
    writer = GuardedWriter(
        project_root=tmp_path,
        tmp_roots=["_tmp"],
        durable_roots=["cas", "runs", "pack", "consumed", "consumed1", "consumed2", "out1", "out2"],
        exclusions=[],
    )
    writer.open_commit_gate()
    return writer


def _make_test_writer(tmp_path: Path) -> PackerWriter:
    """Create a test-safe PackerWriter that bypasses firewall for tests."""
    class NoOpFirewallWriter(PackerWriter):
        """PackerWriter that bypasses firewall checks for tests."""
        def __init__(self):
            self.project_root = REPO_ROOT

        def write_json(self, path, payload, *, kind="durable", indent=None, canonical=False):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if canonical:
                text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
            else:
                text = json.dumps(payload, indent=indent or 2, sort_keys=True, ensure_ascii=False) + "\n"
            path.write_text(text, encoding="utf-8")

        def write_text(self, path, content, *, kind="durable", encoding="utf-8"):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)

        def write_bytes(self, path, data, *, kind="durable"):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        def mkdir(self, path, *, kind="durable", parents=True, exist_ok=True):
            Path(path).mkdir(parents=parents, exist_ok=exist_ok)

        def rename(self, src, dst):
            Path(src).rename(Path(dst))

        def unlink(self, path):
            Path(path).unlink()

        def commit(self):
            pass

    return NoOpFirewallWriter()


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


def _setup_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Common test setup: create writers, dirs, and monkeypatch CAS."""
    test_writer = _make_test_writer(tmp_path)
    cas_test_writer = _make_cas_test_writer(tmp_path)
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setattr(cas_mod, "_custom_writer", cas_test_writer)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    return test_writer, cas_root, runs_dir


def test_pack_consume_basic_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Basic roundtrip: create pack, consume it, verify files match."""
    test_writer, cas_root, runs_dir = _setup_test_env(tmp_path, monkeypatch)

    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_test" / f"consumer_{uuid.uuid4().hex[:8]}",
        combined=False,
        stamp=f"test_c1_{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
        writer=test_writer,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    consume_dir = tmp_path / "consumed"
    receipt = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir,
        dry_run=False,
        cas_root=cas_root,
        writer=test_writer,
    )

    assert receipt.exit_status == "SUCCESS"
    assert len(receipt.errors) == 0
    assert consume_dir.exists()

    manifest = json.loads((pack_dir / "LITE" / "PACK_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"]:
        materialized = consume_dir / entry["path"]
        assert materialized.exists()
        assert materialized.stat().st_size == entry["bytes"]


def test_pack_consume_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run mode verifies but doesn't materialize."""
    test_writer, cas_root, runs_dir = _setup_test_env(tmp_path, monkeypatch)

    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_test" / f"consumer_{uuid.uuid4().hex[:8]}",
        combined=False,
        stamp=f"test_c2_{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
        writer=test_writer,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    consume_dir = tmp_path / "consumed"
    receipt = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir,
        dry_run=True,
        cas_root=cas_root,
        writer=test_writer,
    )

    assert receipt.exit_status == "SUCCESS"
    assert "dry_run" in receipt.commands_run[-1]
    assert not consume_dir.exists()


def test_pack_consume_tamper_detection_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tampering with manifest bytes should fail verification."""
    test_writer, cas_root, runs_dir = _setup_test_env(tmp_path, monkeypatch)

    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_test" / f"consumer_{uuid.uuid4().hex[:8]}",
        combined=False,
        stamp=f"test_c3_{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
        writer=test_writer,
    )

    manifest_path = pack_dir / "LITE" / "PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scope"] = "TAMPERED"

    tampered_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
    tampered_ref = store_bytes(tampered_bytes)

    consume_dir = tmp_path / "consumed"

    with pytest.raises(ValueError, match="INVALID_SCOPE"):
        consumer.pack_consume(
            manifest_ref=tampered_ref,
            out_dir=consume_dir,
            dry_run=False,
            cas_root=cas_root,
            writer=test_writer,
        )


def test_pack_consume_missing_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CAS blob should fail-closed before materialization."""
    test_writer, cas_root, runs_dir = _setup_test_env(tmp_path, monkeypatch)

    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_test" / f"consumer_{uuid.uuid4().hex[:8]}",
        combined=False,
        stamp=f"test_c4_{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
        writer=test_writer,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    manifest = json.loads((pack_dir / "LITE" / "PACK_MANIFEST.json").read_text(encoding="utf-8"))
    first_entry = manifest["entries"][0]
    first_hash = first_entry["ref"].split(":", 1)[1]

    blob_path = cas_mod._get_object_path(first_hash)
    blob_path.unlink()

    consume_dir = tmp_path / "consumed"

    with pytest.raises(ValueError, match="MISSING_BLOBS"):
        consumer.pack_consume(
            manifest_ref=manifest_ref,
            out_dir=consume_dir,
            dry_run=False,
            cas_root=cas_root,
            writer=test_writer,
        )

    assert not consume_dir.exists()


def test_pack_consume_determinism(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Consuming same manifest twice should produce identical tree hash."""
    test_writer, cas_root, runs_dir = _setup_test_env(tmp_path, monkeypatch)

    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=packer_core.PACKS_ROOT / "_test" / f"consumer_{uuid.uuid4().hex[:8]}",
        combined=False,
        stamp=f"test_c5_{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
        writer=test_writer,
    )

    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_ref = run_refs["manifest_ref"]

    consume_dir1 = tmp_path / "consumed1"
    receipt1 = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir1,
        dry_run=False,
        cas_root=cas_root,
        writer=test_writer,
    )

    consume_dir2 = tmp_path / "consumed2"
    receipt2 = consumer.pack_consume(
        manifest_ref=manifest_ref,
        out_dir=consume_dir2,
        dry_run=False,
        cas_root=cas_root,
        writer=test_writer,
    )

    assert receipt1.tree_hash == receipt2.tree_hash
    assert receipt1.tree_hash != ""
    assert receipt1.cas_snapshot_hash == receipt2.cas_snapshot_hash


def test_pack_consume_path_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifests with unsafe paths (absolute, traversal) should fail validation."""
    test_writer, cas_root, _ = _setup_test_env(tmp_path, monkeypatch)

    bad_manifest1 = {
        "version": "P2.0",
        "scope": "ags",
        "entries": [
            {
                "path": "/etc/passwd",
                "ref": "sha256:" + "0" * 64,
                "bytes": 100,
                "kind": "FILE",
            }
        ],
    }

    bad_bytes1 = (json.dumps(bad_manifest1, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    bad_ref1 = store_bytes(bad_bytes1)

    with pytest.raises(ValueError):
        consumer.pack_consume(
            manifest_ref=bad_ref1,
            out_dir=tmp_path / "out1",
            dry_run=True,
            cas_root=cas_root,
            writer=test_writer,
        )

    bad_manifest2 = {
        "version": "P2.0",
        "scope": "ags",
        "entries": [
            {
                "path": "../../../etc/passwd",
                "ref": "sha256:" + "0" * 64,
                "bytes": 100,
                "kind": "FILE",
            }
        ],
    }

    bad_bytes2 = (json.dumps(bad_manifest2, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    bad_ref2 = store_bytes(bad_bytes2)

    with pytest.raises(ValueError):
        consumer.pack_consume(
            manifest_ref=bad_ref2,
            out_dir=tmp_path / "out2",
            dry_run=True,
            cas_root=cas_root,
            writer=test_writer,
        )
