#!/usr/bin/env python3
"""
Rebuild system1.db by indexing all markdown files in the repo.

Per ADR-027: System 1 should contain "deterministic chunks of all repo files"
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from CORTEX.system1_builder import System1DB

# Directories to index
INDEX_DIRS = [
    "CANON",
    "CONTEXT",
    "SKILLS",
    "CATALYTIC-DPT",
    "CONTRACTS",
    "INBOX",
]

# Directories to EXCLUDE
EXCLUDE_DIRS = [
    "_generated",
    "__pycache__",
    ".git",
    "node_modules",
    "BUILD",
    "MEMORY",  # LLM packer packs are too heavy
    ".pytest_cache",
]


def should_index(path: Path) -> bool:
    """Check if path should be indexed."""
    parts = path.parts
    for exclude in EXCLUDE_DIRS:
        if exclude in parts:
            return False
    return True


import subprocess
import os

def get_tracked_files(root: Path) -> set:
    """Return set of tracked files (absolute paths). Returns None if git fails."""
    try:
        cmd = ["git", "ls-files", "-z"]
        result = subprocess.run(cmd, cwd=root, capture_output=True)
        if result.returncode != 0:
            return None
        paths = set()
        for p in result.stdout.split(b'\0'):
            if p:
                paths.add(root / os.fsdecode(p))
        return paths
    except Exception:
        return None

def reindex_all():
    print("🚀 Reindexing all repo files into system1.db...")
    import os
    
    # Initialize DB (creates schema)
    db = System1DB()
    
    tracked_files = get_tracked_files(PROJECT_ROOT)
    if tracked_files is not None:
        print(f"ℹ️  Git filter active: {len(tracked_files)} tracked files found")
    else:
        print("⚠️  Git filter inactive (using filesystem only)")

    count = 0
    for dir_name in INDEX_DIRS:
        target_dir = PROJECT_ROOT / dir_name
        if not target_dir.exists():
            print(f"  ⚠️  Directory not found: {dir_name}")
            continue
            
        for md_file in target_dir.rglob("*.md"):
            if not should_index(md_file):
                continue
            
            # Git Filter
            if tracked_files is not None and md_file.resolve() not in tracked_files:
                # Skip untracked files
                continue
                
            rel_path = md_file.relative_to(PROJECT_ROOT).as_posix()
                
            rel_path = md_file.relative_to(PROJECT_ROOT).as_posix()
            try:
                content = md_file.read_text(encoding='utf-8')
                db.add_file(rel_path, content)
                count += 1
            except Exception as e:
                print(f"  ❌ Failed to index {rel_path}: {e}")
    
    db.close()
    print(f"✅ Indexed {count} files into system1.db")


if __name__ == "__main__":
    reindex_all()
