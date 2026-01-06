#!/usr/bin/env python3
"""
Phase 2.4.1C.1 LLM_PACKER Commit Gate Test

Tests that the WriteFirewall + PackerWriter properly enforces the commit gate:
- Tmp writes succeed without commit
- Durable writes fail before commit
- Durable writes succeed after writer.commit()
"""

import pytest
from pathlib import Path
import tempfile
import shutil

# Import the packer firewall components
REPO_ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(REPO_ROOT))

from MEMORY.LLM_PACKER.Engine.packer.firewall_writer import PackerWriter
from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation


def test_llm_packer_commit_gate():
    """Test that LLM_Packer commit gate works correctly."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        # Create packer directory structure
        packs_root = project_root / "MEMORY" / "LLM_PACKER" / "_packs"
        packs_root.mkdir(parents=True)
        tmp_root = packs_root / "_tmp"
        tmp_root.mkdir()
        
        # Initialize writer with firewall
        writer = PackerWriter(project_root=project_root)
        
        # Test data
        test_data = {"test": "data", "phase": "2.4.1C.1"}
        test_binary = b"binary test data"
        
        # 1. Tmp writes should succeed without commit
        tmp_file = tmp_root / "tmp_test.json"
        writer.write_json(tmp_file, test_data, kind="tmp")
        assert tmp_file.exists(), "Tmp write should succeed without commit"
        
        tmp_bin_file = tmp_root / "tmp_test.bin"
        writer.write_bytes(tmp_bin_file, test_binary, kind="tmp")
        assert tmp_bin_file.exists(), "Tmp binary write should succeed without commit"
        
        # 2. Durable writes should fail before commit
        durable_file = packs_root / "durable_test.json"
        with pytest.raises(FirewallViolation, match="commit.*gate"):
            writer.write_json(durable_file, test_data, kind="durable")
        
        durable_bin_file = packs_root / "durable_test.bin"
        with pytest.raises(FirewallViolation, match="commit.*gate"):
            writer.write_bytes(durable_bin_file, test_binary, kind="durable")
        
        # 3. After commit, durable writes should succeed
        writer.commit()
        
        writer.write_json(durable_file, test_data, kind="durable")
        assert durable_file.exists(), "Durable write should succeed after commit"
        
        writer.write_bytes(durable_bin_file, test_binary, kind="durable")
        assert durable_bin_file.exists(), "Durable binary write should succeed after commit"
        
        # Verify content
        import json
        with open(durable_file) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data, "Written JSON data should match"
        
        with open(durable_bin_file, "rb") as f:
            loaded_binary = f.read()
        assert loaded_binary == test_binary, "Written binary data should match"


def test_llm_packer_mkdir_commit_gate():
    """Test that mkdir operations respect commit gate."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        # Create packer directory structure
        packs_root = project_root / "MEMORY" / "LLM_PACKER" / "_packs"
        packs_root.mkdir(parents=True)
        tmp_root = packs_root / "_tmp"
        tmp_root.mkdir()
        
        # Initialize writer with firewall
        writer = PackerWriter(project_root=project_root)
        
        # 1. Tmp mkdir should succeed without commit
        tmp_dir_path = tmp_root / "tmp_subdir"
        writer.mkdir(tmp_dir_path, kind="tmp")
        assert tmp_dir_path.exists(), "Tmp mkdir should succeed without commit"
        
        # 2. Durable mkdir should fail before commit
        durable_dir_path = packs_root / "durable_subdir"
        with pytest.raises(FirewallViolation, match="commit.*gate"):
            writer.mkdir(durable_dir_path, kind="durable")
        
        # 3. After commit, durable mkdir should succeed
        writer.commit()
        writer.mkdir(durable_dir_path, kind="durable")
        assert durable_dir_path.exists(), "Durable mkdir should succeed after commit"


def test_llm_packer_violation_receipt():
    """Test that violation receipts are properly generated."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        project_root = tmp_path / "project"
        project_root.mkdir()
        
        # Create packer directory structure
        packs_root = project_root / "MEMORY" / "LLM_PACKER" / "_packs"
        packs_root.mkdir(parents=True)
        
        # Initialize writer with firewall
        writer = PackerWriter(project_root=project_root)
        
        # Attempt durable write without commit to trigger violation
        durable_file = packs_root / "durable_test.json"
        test_data = {"test": "violation"}
        
        try:
            writer.write_json(durable_file, test_data, kind="durable")
            pytest.fail("Expected FirewallViolation")
        except FirewallViolation as e:
            # Extract violation receipt
            receipt = writer.get_violation_receipt(e)
            
            # Verify receipt structure
            assert "error_code" in receipt
            assert "operation" in receipt
            assert "path" in receipt
            assert "message" in receipt
            assert "firewall_version" in receipt
            
            # Verify specific values
            assert receipt["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"
            assert receipt["operation"] == "write"
            assert "commit gate" in receipt["message"]
            assert receipt["kind"] == "durable"
