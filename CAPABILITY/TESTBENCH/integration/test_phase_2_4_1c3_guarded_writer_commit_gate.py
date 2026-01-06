#!/usr/bin/env python3
"""
Phase 2.4.1C.3 GuardedWriter Commit Gate Test

Tests that GuardedWriter properly enforces commit-gate semantics:
- tmp writes succeed without commit
- durable writes fail before commit  
- durable writes succeed after commit
"""

import json
import sys
import tempfile
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    pytest.skip("GuardedWriter not available", allow_module_level=True)


def test_guarded_writer_commit_gate():
    """
    Test GuardedWriter commit-gate enforcement.
    
    Evidence:
    - tmp write succeeds without commit gate
    - durable write fails before commit gate
    - durable write succeeds after commit gate
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()
        
        # Create directory structure for GuardedWriter domains
        (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True, exist_ok=True)
        (project_root / "LAW" / "CONTRACTS" / "_runs").mkdir(parents=True, exist_ok=True)
        
        # Initialize GuardedWriter
        writer = GuardedWriter(project_root=project_root)
        
        # Test 1: tmp write should succeed without commit
        writer.write_tmp("LAW/CONTRACTS/_runs/_tmp/test.json", '{"status": "tmp"}')
        
        # Verify tmp file exists
        tmp_file = project_root / "LAW/CONTRACTS/_runs/_tmp/test.json"
        assert tmp_file.exists(), "Tmp file should exist after write_tmp"
        assert json.loads(tmp_file.read_text()) == {"status": "tmp"}
        
        # Test 2: durable write should fail before commit
        with pytest.raises(FirewallViolation) as exc_info:
            writer.write_durable("LAW/CONTRACTS/_runs/result.json", '{"status": "durable"}')
        
        violation = exc_info.value.violation_receipt
        assert violation["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT", \
            "Durable write before commit should fail with FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"
        
        # Verify durable file does not exist
        durable_file = project_root / "LAW/CONTRACTS/_runs/result.json"
        assert not durable_file.exists(), "Durable file should not exist before commit"
        
        # Test 3: durable write should succeed after commit
        writer.open_commit_gate()
        writer.write_durable("LAW/CONTRACTS/_runs/result.json", '{"status": "durable"}')
        
        # Verify durable file exists
        assert durable_file.exists(), "Durable file should exist after commit"
        assert json.loads(durable_file.read_text()) == {"status": "durable"}
        
        print("✓ GuardedWriter commit-gate test passed")
        print(f"  - Tmp write succeeded: {tmp_file.exists()}")
        print(f"  - Durable write blocked before commit: {violation['error_code']}")
        print(f"  - Durable write succeeded after commit: {durable_file.exists()}")


def test_guarded_writer_mkdir_operations():
    """
    Test GuardedWriter mkdir operations with commit gate.
    
    Evidence:
    - tmp mkdir succeeds without commit
    - durable mkdir fails before commit
    - durable mkdir succeeds after commit
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir) / "repo"
        project_root.mkdir()
        
        # Create directory structure for GuardedWriter domains
        (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True, exist_ok=True)
        (project_root / "LAW" / "CONTRACTS" / "_runs").mkdir(parents=True, exist_ok=True)
        
        writer = GuardedWriter(project_root=project_root)
        
        # Test 1: tmp mkdir should succeed
        writer.mkdir_tmp("LAW/CONTRACTS/_runs/_tmp/subdir")
        tmp_dir = project_root / "LAW/CONTRACTS/_runs/_tmp/subdir"
        assert tmp_dir.exists(), "Tmp directory should exist after mkdir_tmp"
        assert tmp_dir.is_dir(), "Tmp path should be a directory"
        
        # Test 2: durable mkdir should fail before commit
        with pytest.raises(FirewallViolation) as exc_info:
            writer.mkdir_durable("LAW/CONTRACTS/_runs/subdir")
        
        violation = exc_info.value.violation_receipt
        assert violation["error_code"] == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT", \
            "Durable mkdir before commit should fail"
        
        durable_dir = project_root / "LAW/CONTRACTS/_runs/subdir"
        assert not durable_dir.exists(), "Durable directory should not exist before commit"
        
        # Test 3: durable mkdir should succeed after commit
        writer.open_commit_gate()
        writer.mkdir_durable("LAW/CONTRACTS/_runs/subdir")
        assert durable_dir.exists(), "Durable directory should exist after commit"
        assert durable_dir.is_dir(), "Durable path should be a directory"
        
        print("✓ GuardedWriter mkdir commit-gate test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
