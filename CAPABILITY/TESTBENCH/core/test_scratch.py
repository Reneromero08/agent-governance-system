#!/usr/bin/env python3
"""
Core Test: Catalytic Scratch Layer

Tests the isolation and restoration guarantees of CatalyticScratch.
"""

import os
import shutil
import sys
from pathlib import Path
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.scratch import CatalyticScratch

def test_scratch_isolation_and_restoration(tmp_path):
    # 1. Setup canonical source
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Invariant content")
    (source / "src").mkdir()
    (source / "src" / "main.py").write_text("print('original')")
    
    initial_hash = CatalyticScratch.compute_hash(source)
    
    # 2. Workspace for scratch
    scratch = tmp_path / "scratch"
    
    # 3. Use scratch layer
    with CatalyticScratch(source, scratch) as scratch_dir:
        assert scratch_dir.exists()
        assert (scratch_dir / "README.md").read_text() == "Invariant content"
        
        # Perform destructive operations in scratch
        (scratch_dir / "src" / "main.py").write_text("print('corrupted')")
        (scratch_dir / "README.md").unlink()
        (scratch_dir / "new_file.txt").write_text("noise")
        
        # Verify source remains untouched
        assert (source / "README.md").exists()
        assert (source / "src" / "main.py").read_text() == "print('original')"

    # 4. Post-session verification
    assert not scratch.exists()
    assert CatalyticScratch.compute_hash(source) == initial_hash
    assert (source / "README.md").read_text() == "Invariant content"

def test_scratch_detects_source_tamper(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("safe")
    
    scratch = tmp_path / "scratch"
    
    with pytest.raises(RuntimeError, match="INTEGRITY FAILURE"):
        with CatalyticScratch(source, scratch):
            # TAMPER with SOURCE while scratch is active
            (source / "file.txt").write_text("TAMPERED")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
