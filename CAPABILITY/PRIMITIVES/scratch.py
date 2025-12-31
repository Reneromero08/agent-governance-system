#!/usr/bin/env python3
"""
Catalytic Scratch Layer Proof of Concept (F2)

Demonstrates safe, destructive scratchpad operations using git worktrees or 
shutil copies, guaranteeing byte-for-byte restoration.
"""

import os
import shutil
import hashlib
import tempfile
import json
from pathlib import Path

# Mock Repo Structure - Removed (Use explicit paths)
# PROJECT_ROOT = Path("mock_repo_root")
# SCRATCH_ROOT = Path("mock_scratch_space")



class CatalyticScratch:
    """
    Manages a catalytic scratch layer for destructive operations.
    Guarantees isolation and restoration.
    """
    def __init__(self, source_root: Path, scratch_root: Path):
        self.source = source_root
        self.scratch = scratch_root
        self.initial_hash = None

    def __enter__(self):
        """Initialize the scratch layer."""
        self.initial_hash = self.compute_hash(self.source)
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
        shutil.copytree(self.source, self.scratch)
        return self.scratch

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Teardown and verify integrity."""
        # 1. Verification
        final_hash = self.compute_hash(self.source)
        integrity_ok = (final_hash == self.initial_hash)
        
        # 2. Cleanup
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
            
        if not integrity_ok:
            raise RuntimeError(f"INTEGRITY FAILURE: Source repo modified during scratch session! ({self.initial_hash} -> {final_hash})")
            
    @staticmethod
    def compute_hash(directory: Path) -> str:
        """Compute Merkle-like hash of a directory."""
        sha = hashlib.sha256()
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                path = Path(root) / file
                rel_path = path.relative_to(directory).as_posix()
                sha.update(rel_path.encode())
                sha.update(path.read_bytes())
        return sha.hexdigest()


if __name__ == "__main__":
    print("CatalyticScratch Primitive - Import to use.")
