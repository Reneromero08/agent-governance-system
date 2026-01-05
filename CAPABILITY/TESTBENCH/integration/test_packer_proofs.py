
import sys
import json
import shutil
from pathlib import Path
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MEMORY.LLM_PACKER.Engine.packer import core as packer_core

def test_packer_proofs_integration(tmp_path, monkeypatch):
    # Setup: Copy fixture repo to tmp_path
    fixture_src = PROJECT_ROOT / "CAPABILITY/TESTBENCH/fixtures/packer_p2_repo"
    repo_root = tmp_path / "repo"
    shutil.copytree(fixture_src, repo_root)
    
    # Setup: Create PROOFS directory and dummy CONFIG
    proofs_dir = repo_root / "NAVIGATION/PROOFS"
    proofs_dir.mkdir(parents=True, exist_ok=True)
    
    (proofs_dir / "PROOF_SUITE.json").write_text(
        json.dumps({"commands": [[sys.executable, "-c", "print('proof_run')"]]}),
        encoding="utf-8"
    )
    
    # Patch SCOPE_AGS to point to tmp repo and include NAVIGATION
    # We must use the key "ags" because make_pack(scope_key="ags") looks up SCOPES["ags"]
    test_scope = packer_core.PackScope(
        key="ags",
        title="AGS Test",
        file_prefix="AGS",
        include_dirs=("LAW", "CAPABILITY", "NAVIGATION"),
        root_files=("README.md",),
        anchors=(),
        excluded_dir_parts=frozenset({".git"}),
        source_root_rel=".",
    )
    
    monkeypatch.setitem(packer_core.SCOPES, "ags", test_scope)
    
    # Required for packing to work if these dirs are in include_dirs
    (repo_root / "CAPABILITY").mkdir(exist_ok=True)
    (repo_root / "NAVIGATION").mkdir(exist_ok=True)
    
    # Run Packer
    packs_root = repo_root / "MEMORY/LLM_PACKER/_packs"
    packs_root.mkdir(parents=True, exist_ok=True)
    out_dir = packs_root / "out_pack"

    # Patch packer_core config to accept temp paths
    monkeypatch.setattr(packer_core, "PACKS_ROOT", packs_root)
    monkeypatch.setattr(packer_core, "SYSTEM_DIR", packs_root / "_system")
    monkeypatch.setattr(packer_core, "EXTERNAL_ARCHIVE_DIR", packs_root / "_archive")
    monkeypatch.setattr(packer_core, "FIXTURE_PACKS_DIR", packs_root / "_system/fixtures")
    monkeypatch.setattr(packer_core, "STATE_DIR", packs_root / "_system/_state")
    
    pack_dir = packer_core.make_pack(
        scope_key="ags",
        mode="full",
        profile="full",
        split_lite=True,
        out_dir=out_dir,
        combined=True,
        stamp="PROOF_TEST",
        zip_enabled=False,
        max_total_bytes=10*1024*1024,
        max_entry_bytes=2*1024*1024,
        max_entries=1000,
        allow_duplicate_hashes=True,
        project_root=repo_root, 
        skip_proofs=False,
    )
    
    # --- ASSERTIONS ---
    
    # 1. Dispersed artifacts generated in source
    manifest_path = repo_root / "NAVIGATION/PROOFS/PROOF_MANIFEST.json"
    assert manifest_path.exists(), "PROOF_MANIFEST.json not generated"
    
    green_state = repo_root / "NAVIGATION/PROOFS/GREEN_STATE.json"
    assert green_state.exists(), "GREEN_STATE.json missing"
    
    cat_log = repo_root / "NAVIGATION/PROOFS/CATALYTIC/PROOF_LOG.txt"
    assert cat_log.exists(), "CATALYTIC/PROOF_LOG.txt missing"
    assert "proof_run" in cat_log.read_text("utf-8")
    
    # 2. Pack content copied
    repo_proof_manifest = pack_dir / "repo/NAVIGATION/PROOFS/PROOF_MANIFEST.json"
    assert repo_proof_manifest.exists(), "Proof artifacts not copied to pack repo/"
    
    # 3. SPLIT output
    split_proofs = pack_dir / "SPLIT/AGS-04_PROOFS.md"
    assert split_proofs.exists(), "AGS-04_PROOFS.md missing"
    content = split_proofs.read_text("utf-8")
    assert "PROOF_MANIFEST.json" in content
    
    # Verify renumbering implicitly by checking existence
    assert (pack_dir / "SPLIT/AGS-01_LAW.md").exists()
    assert (pack_dir / "SPLIT/AGS-03_NAVIGATION.md").exists()
    # Check that NAVIGATION does NOT contain PROOFS
    nav_content = (pack_dir / "SPLIT/AGS-03_NAVIGATION.md").read_text("utf-8")
    assert "NAVIGATION/PROOFS" not in nav_content
    
    # 4. LITE output
    lite_proofs = pack_dir / "LITE/PROOFS.json"
    assert lite_proofs.exists(), "LITE/PROOFS.json missing"
    lite_data = json.loads(lite_proofs.read_text("utf-8"))
    assert lite_data["overall_status"] == "PASS"
    assert "proof_files" in lite_data

def test_packer_skip_proofs(tmp_path, monkeypatch):
    # Setup similar to above
    fixture_src = PROJECT_ROOT / "CAPABILITY/TESTBENCH/fixtures/packer_p2_repo"
    repo_root = tmp_path / "repo_skip"
    shutil.copytree(fixture_src, repo_root)
    
    # Ensure PROOFS dir exists so we can check it's NOT populated
    (repo_root / "NAVIGATION/PROOFS").mkdir(parents=True, exist_ok=True)
    
    test_scope = packer_core.PackScope(
        key="ags",
        title="AGS Test",
        file_prefix="AGS",
        include_dirs=("LAW", "CAPABILITY", "NAVIGATION"),
        root_files=("README.md",),
        anchors=(),
        excluded_dir_parts=frozenset({".git"}),
        source_root_rel=".",
    )
    monkeypatch.setitem(packer_core.SCOPES, "ags", test_scope)
    (repo_root / "CAPABILITY").mkdir(exist_ok=True)
    (repo_root / "NAVIGATION").mkdir(exist_ok=True)

    out_dir = repo_root / "MEMORY/LLM_PACKER/_packs/out_pack_skip"
    packs_root = repo_root / "MEMORY/LLM_PACKER/_packs"
    packs_root.mkdir(parents=True, exist_ok=True)
    
    monkeypatch.setattr(packer_core, "PACKS_ROOT", packs_root)
    monkeypatch.setattr(packer_core, "SYSTEM_DIR", packs_root / "_system")
    monkeypatch.setattr(packer_core, "EXTERNAL_ARCHIVE_DIR", packs_root / "_archive")
    monkeypatch.setattr(packer_core, "FIXTURE_PACKS_DIR", packs_root / "_system/fixtures")
    monkeypatch.setattr(packer_core, "STATE_DIR", packs_root / "_system/_state")

    packer_core.make_pack(
        scope_key="ags", 
        mode="full", 
        profile="full", 
        split_lite=False, 
        out_dir=out_dir, 
        combined=False, 
        stamp="SKIP_TEST", 
        zip_enabled=False, 
        max_total_bytes=10**7, 
        max_entry_bytes=2**20, 
        max_entries=1000, 
        allow_duplicate_hashes=True, 
        project_root=repo_root, 
        skip_proofs=True
    )
    
    # Assert _LATEST artifacts NOT created
    assert not (repo_root / "NAVIGATION/PROOFS/PROOF_MANIFEST.json").exists()
