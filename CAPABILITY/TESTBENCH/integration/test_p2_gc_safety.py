"""
Test P.2.4: CAS garbage collection safety for packer outputs.

Ensures that GC never deletes a blob referenced by active packs.
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
from CAPABILITY.CAS import cas as cas_mod
from CAPABILITY.GC import gc


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


def test_gc_never_deletes_pack_referenced_blobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GC must never delete blobs referenced by active packs.
    
    Setup:
    1. Create a pack with CAS-addressed manifest
    2. Store some unreferenced blobs in CAS
    3. Run GC (dry-run)
    4. Verify: referenced blobs are marked reachable, unreferenced blobs are candidates
    5. Run GC (apply mode)
    6. Verify: referenced blobs still exist, unreferenced blobs deleted
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Step 1: Create a pack
    out_dir = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-gc-safety"
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out_dir,
        combined=False,
        stamp="pytest-gc",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    # Step 2: Get referenced hashes from pack manifest
    manifest_path = pack_dir / "LITE" / "PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    referenced_hashes = set()
    for entry in manifest["entries"]:
        ref = entry["ref"]
        assert ref.startswith("sha256:")
        hash_hex = ref.split(":", 1)[1]
        referenced_hashes.add(hash_hex)
    
    # Also get manifest hash itself
    run_refs = json.loads((pack_dir / "LITE" / "RUN_REFS.json").read_text(encoding="utf-8"))
    manifest_hash = run_refs["manifest_ref"].split(":", 1)[1]
    referenced_hashes.add(manifest_hash)

    # Step 3: Store an unreferenced blob
    from CAPABILITY.ARTIFACTS.store import store_bytes
    unreferenced_ref = store_bytes(b"UNREFERENCED_BLOB_FOR_GC_TEST")
    unreferenced_hash = unreferenced_ref.split(":", 1)[1]
    
    # Verify it exists
    unreferenced_path = cas_mod._get_object_path(unreferenced_hash)
    assert unreferenced_path.exists()

    # Step 4: Run GC dry-run
    gc_result_dry = gc.gc_collect(dry_run=True, allow_empty_roots=False, cas_root=cas_root, runs_dir=runs_dir)
    
    assert gc_result_dry["mode"] == "dry_run"
    assert gc_result_dry["roots_count"] > 0
    assert gc_result_dry["reachable_hashes_count"] > 0
    assert gc_result_dry["errors"] == []
    
    # Unreferenced blob should be a candidate
    candidates = [h["hash"] for h in gc_result_dry["skipped_hashes"]]
    assert unreferenced_hash in candidates
    
    # Referenced blobs should NOT be candidates
    for ref_hash in referenced_hashes:
        assert ref_hash not in candidates, f"Referenced hash {ref_hash} should not be a GC candidate"

    # Step 5: Run GC apply mode
    gc_result_apply = gc.gc_collect(dry_run=False, allow_empty_roots=False, cas_root=cas_root, runs_dir=runs_dir)
    
    assert gc_result_apply["mode"] == "apply"
    assert gc_result_apply["errors"] == []
    
    # Step 6: Verify referenced blobs still exist
    for ref_hash in referenced_hashes:
        ref_path = cas_mod._get_object_path(ref_hash)
        assert ref_path.exists(), f"Referenced blob {ref_hash} was deleted by GC!"
    
    # Unreferenced blob should be deleted
    assert not unreferenced_path.exists(), "Unreferenced blob was not deleted by GC"
    assert unreferenced_hash in gc_result_apply["deleted_hashes"]


def test_gc_respects_run_roots_from_packer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that packer writes RUN_ROOTS.json and GC respects it.
    
    This is the integration test for P.2.4: GC roots are defined via active packs.
    """
    cas_root = tmp_path / "cas"
    runs_dir = tmp_path / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cas_mod, "_CAS_ROOT", cas_root)
    monkeypatch.setitem(packer_core.SCOPES, "ags", _make_fixture_scope())

    # Create pack
    out_dir = packer_core.PACKS_ROOT / "_system" / "fixtures" / "pytest-gc-roots"
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out_dir,
        combined=False,
        stamp="pytest-gc",
        zip_enabled=False,
        max_total_bytes=5 * 1024 * 1024,
        max_entry_bytes=2 * 1024 * 1024,
        max_entries=10_000,
        allow_duplicate_hashes=True,
        p2_runs_dir=runs_dir,
        p2_cas_root=cas_root,
    )

    # Verify RUN_ROOTS.json was written
    run_roots_path = runs_dir / "RUN_ROOTS.json"
    assert run_roots_path.exists()
    
    run_roots_data = json.loads(run_roots_path.read_text(encoding="utf-8"))
    assert isinstance(run_roots_data, list)
    assert len(run_roots_data) > 0
    
    # All roots should be valid SHA-256 hashes
    for root in run_roots_data:
        assert isinstance(root, str)
        assert len(root) == 64
        assert all(c in "0123456789abcdef" for c in root)
    
    # GC should enumerate these roots
    gc_result = gc.gc_collect(dry_run=True, allow_empty_roots=False, cas_root=cas_root, runs_dir=runs_dir)
    
    assert gc_result["roots_count"] == len(set(run_roots_data))
    assert gc_result["errors"] == []
    assert "RUN_ROOTS.json" in str(gc_result["root_sources"])
