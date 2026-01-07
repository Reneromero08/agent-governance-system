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
from CAPABILITY.RUNS.records import load_output_hashes
from CAPABILITY.CAS import cas as cas_mod
from CAPABILITY.ARTIFACTS import store as store_mod
from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter


FIXTURE_ROOT_REL = Path("CAPABILITY") / "TESTBENCH" / "fixtures" / "packer_p2_repo"
FIXTURE_BODY_MARKER = "P2_BODY_MARKER_XYZ"


def _make_test_writer(project_root: Path) -> GuardedWriter:
    """Create a GuardedWriter configured for test temp directories."""
    writer = GuardedWriter(
        project_root=project_root,
        tmp_roots=["_tmp"],
        durable_roots=["cas", "runs", "CAS", "pack", "packs"],  # Dirs used by tests
        exclusions=[],  # No exclusions in test mode
    )
    writer.open_commit_gate()  # Tests need durable writes enabled
    return writer


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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_p2_determinism_manifest_and_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create test writer that allows tmp_path
    test_writer = _make_test_writer(tmp_path)
    monkeypatch.setattr(cas_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(store_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    scope = _make_fixture_scope()
    monkeypatch.setitem(packer_core.SCOPES, "ags", scope)

    out1 = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-p2-1"
    out2 = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-p2-2"
    det_stamp = f"pytest-p2-{uuid.uuid4().hex[:6]}"

    pack1 = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out1,
        combined=False,
        stamp=det_stamp,
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )
    pack2 = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out2,
        combined=False,
        stamp=det_stamp,
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    manifest1 = (pack1 / "LITE" / "PACK_MANIFEST.json").read_bytes()
    manifest2 = (pack2 / "LITE" / "PACK_MANIFEST.json").read_bytes()
    assert manifest1 == manifest2

    refs1 = _read_json(pack1 / "LITE" / "RUN_REFS.json")
    refs2 = _read_json(pack2 / "LITE" / "RUN_REFS.json")
    assert refs1 == refs2
    assert refs1["manifest_ref"].startswith("sha256:")

    required = load_output_hashes(refs1["output_hashes_ref"])
    assert isinstance(required, list)
    assert len(required) >= 2  # manifest + at least one payload


def test_p2_lite_manifest_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create test writer that allows tmp_path
    test_writer = _make_test_writer(tmp_path)
    monkeypatch.setattr(cas_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(store_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    out_dir = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-p2-lite-only"
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out_dir,
        combined=False,
        stamp=f"pytest-p2-{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    lite_dir = pack_dir / "LITE"
    assert (lite_dir / "PACK_MANIFEST.json").exists()
    assert (lite_dir / "RUN_REFS.json").exists()

    for p in lite_dir.rglob("*"):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        assert FIXTURE_BODY_MARKER not in text

    # Back-compat smoke: SPLIT still contains bodies
    split_law = pack_dir / "SPLIT" / "AGS-01_LAW.md"
    assert split_law.exists()
    assert FIXTURE_BODY_MARKER in split_law.read_text(encoding="utf-8", errors="replace")


def test_p2_manifest_ordering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create test writer that allows tmp_path
    test_writer = _make_test_writer(tmp_path)
    monkeypatch.setattr(cas_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(store_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    out_dir = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-p2-ordering"
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out_dir,
        combined=False,
        stamp=f"pytest-p2-{uuid.uuid4().hex[:6]}",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    manifest = json.loads((pack_dir / "LITE" / "PACK_MANIFEST.json").read_text(encoding="utf-8"))
    paths = [e["path"] for e in manifest["entries"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))


def test_p2_invalid_ref_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create test writer that allows tmp_path
    test_writer = _make_test_writer(tmp_path)
    monkeypatch.setattr(cas_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(store_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    def bad_store_file(_: str) -> str:
        return "sha256:NOTAHEX"

    monkeypatch.setattr(store_mod, "store_file", bad_store_file)

    out_dir = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-p2-bad-ref"
    with pytest.raises(ValueError, match="PACK_P2_INVALID_REF"):
        packer_core.make_pack(
            scope_key="ags",
            mode="full",
            profile="full",
            split_lite=True,
            out_dir=out_dir,
            combined=False,
            stamp=f"pytest-p2-{uuid.uuid4().hex[:6]}",
            zip_enabled=False,
            max_total_bytes=5 * 1024 * 1024,
            max_entry_bytes=2 * 1024 * 1024,
            max_entries=10_000,
            allow_duplicate_hashes=True,
            p2_runs_dir=runs_dir,
            p2_cas_root=cas_root,
        )


def test_p2_root_completeness_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create test writer that allows tmp_path
    test_writer = _make_test_writer(tmp_path)
    monkeypatch.setattr(cas_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(store_mod, "_custom_writer", test_writer)
    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    original_write_roots = packer_core._write_run_roots

    def drop_one_required_root(path: Path, *, roots: list[str], writer=None) -> None:
        # Find OUTPUT_HASHES record ref among roots, then drop one required hash.
        required: list[str] = []
        for candidate in roots:
            try:
                required = load_output_hashes(candidate)
                break
            except Exception:
                continue
        if required:
            roots = [h for h in roots if h != required[-1]]
        original_write_roots(path, roots=roots)

    monkeypatch.setattr(packer_core, "_write_run_roots", drop_one_required_root)

    out_dir = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-p2-missing-root"
    with pytest.raises(ValueError, match="PACK_P2_ROOT_AUDIT_FAIL"):
        packer_core.make_pack(
            scope_key="ags",
            mode="full",
            profile="full",
            split_lite=True,
            out_dir=out_dir,
            combined=False,
            stamp=f"pytest-p2-{uuid.uuid4().hex[:6]}",
            zip_enabled=False,
            max_total_bytes=5 * 1024 * 1024,
            max_entry_bytes=2 * 1024 * 1024,
            max_entries=10_000,
            allow_duplicate_hashes=True,
            p2_runs_dir=runs_dir,
            p2_cas_root=cas_root,
        )
