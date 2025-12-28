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

# Mock Repo Structure
PROJECT_ROOT = Path("mock_repo_root")
SCRATCH_ROOT = Path("mock_scratch_space")

def setup_environment():
    """Create a mock repository with some state."""
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.mkdir()
    
    # Create some "canonical" files
    (PROJECT_ROOT / "README.md").write_text("# Mock Repo\nInvariant content.")
    (PROJECT_ROOT / "src").mkdir()
    (PROJECT_ROOT / "src" / "main.py").write_text("print('Hello World')")
    
    print(f"[Setup] Created mock repo at {PROJECT_ROOT}")

def compute_hash(directory):
    """Compute Merkle-like hash of a directory."""
    sha = hashlib.sha256()
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            path = Path(root) / file
            # Hash filename
            rel_path = path.relative_to(directory).as_posix()
            sha.update(rel_path.encode())
            # Hash content
            sha.update(path.read_bytes())
    return sha.hexdigest()

def operations_phase(scratch_dir):
    """Perform DESTRUCTIVE operations in the scratch layer."""
    print(f"[Operate] Running destructive tasks in {scratch_dir}...")
    
    # 1. Modify existing file
    (scratch_dir / "src" / "main.py").write_text("print('Hacked!')")
    
    # 2. Delete file
    (scratch_dir / "README.md").unlink()
    
    # 3. Create garbage
    (scratch_dir / "entropy.tmp").write_text("Temporary noise")
    
    print("[Operate] Destruction complete.")

def restore_phase(original_dir, scratch_dir):
    """
    In a real git worktree, we would just `git checkout .` or delete the worktree.
    Here, we simulate restoration by verifying the original inputs were untouched
    and the scratch output yielded a specific artifact.
    """
    # Define what we WANTED to keep (the artifact)
    artifact = (scratch_dir / "src" / "main.py").read_text()
    
    print(f"[Restore] Extracted artifact: {artifact.strip()}")
    print("[Restore] Discarding scratch layer...")
    shutil.rmtree(scratch_dir)

def main():
    setup_environment()
    
    # 1. Capture State (Hash)
    initial_hash = compute_hash(PROJECT_ROOT)
    print(f"[State] Initial Hash: {initial_hash}")
    
    # 2. Create Scratch Layer (Copy for POC, Worktree for Prod)
    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT)
    shutil.copytree(PROJECT_ROOT, SCRATCH_ROOT)
    print(f"[Catalyst] Created scratch layer at {SCRATCH_ROOT}")
    
    # 3. Execute Destructive Work
    operations_phase(SCRATCH_ROOT)
    
    # 4. Verify original repo is UNTOUCHED (Isolation check)
    current_hash = compute_hash(PROJECT_ROOT)
    if current_hash != initial_hash:
        print("[CRITICAL] FAIL: Main repo was tainted!")
        exit(1)
    else:
        print("[Success] Main repo integrity maintained.")
        
    # 5. Restore/Cleanup
    restore_phase(PROJECT_ROOT, SCRATCH_ROOT)
    
    print("[Done] Experiment complete.")

if __name__ == "__main__":
    main()
