#!/usr/bin/env python3
"""
System 1 Verify Skill (Lane C1)

Ensures system1.db is in sync with the repository.

Per ADR-027, System 1 should contain "deterministic chunks of ALL repo files".
This script verifies:
1. All indexed directories have their files in the DB
2. Content hashes match
3. No orphaned entries (files in DB but not on disk)
"""

import sys
import sqlite3
import hashlib
from pathlib import Path

# Add project root to path for absolute resolution
PROJECT_ROOT = Path(__file__).resolve().parents[4]

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
DB_PATH = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"

# Directories that SHOULD be indexed per ADR-027 and build_system1.py
INDEX_DIRS = ["."]

# Directories to EXCLUDE from verification
EXCLUDE_DIRS = [
    "_generated",
    "__pycache__",
    ".git",
    "node_modules",
    "BUILD",
    "MEMORY",
    ".pytest_cache",
    "meta",
    "LAW/CONTRACTS/_runs",
]


def should_verify(path: Path) -> bool:
    """Check if path should be verified."""
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

def verify_system1():
    """Verify system1.db matches repository state."""
    if not DB_PATH.exists():
        print("FAIL: system1.db does not exist")
        return False
        
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # Git Filter
    tracked_files = get_tracked_files(PROJECT_ROOT)
    if tracked_files is not None:
         print(f"[INFO] Git filter active: {len(tracked_files)} tracked files found")

    
    # Get all indexed files from DB
    cursor = conn.execute("SELECT path, content_hash FROM files")
    db_files = {row['path']: row['content_hash'] for row in cursor}
    
    # Get all markdown files from indexed directories
    repo_files = {}
    for dir_name in INDEX_DIRS:
        target_dir = PROJECT_ROOT / dir_name
        if not target_dir.exists():
            continue
            
        for md_file in target_dir.rglob("*.md"):
            if not should_verify(md_file):
                continue

            # Git Filter
            if tracked_files is not None and md_file.resolve() not in tracked_files:
                continue

            rel_path = md_file.relative_to(PROJECT_ROOT).as_posix()
            try:
                content = md_file.read_text(encoding='utf-8')
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                repo_files[rel_path] = content_hash
            except Exception:
                pass  # Skip unreadable files
    
    # Check for missing files (in repo but not in DB)
    missing = set(repo_files.keys()) - set(db_files.keys())
    if missing:
        print(f"[FAIL] {len(missing)} files not indexed:")
        for f in sorted(missing)[:20]:
            print(f"  - {f}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        conn.close()
        return False
    
    # Check for orphaned entries (in DB but not on disk)
    orphaned = set(db_files.keys()) - set(repo_files.keys())
    if orphaned:
        print(f"[WARN] {len(orphaned)} orphaned entries in DB:")
        for f in sorted(orphaned)[:10]:
            print(f"  - {f}")
        if len(orphaned) > 10:
            print(f"  ... and {len(orphaned) - 10} more")
    
    # Check for hash mismatches
    mismatches = []
    for path in repo_files:
        if path in db_files and db_files[path] != repo_files[path]:
            mismatches.append(path)
    
    if mismatches:
        print(f"[FAIL] {len(mismatches)} files have changed:")
        for f in sorted(mismatches):
            print(f"  - {f}")
        conn.close()
        return False
    
    conn.close()
    
    print(f"[PASS] system1.db is in sync")
    print(f"  - {len(repo_files)} files verified")
    if orphaned:
        print(f"  - {len(orphaned)} orphaned (run build_system1.py to clean)")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 2:
        # Fixture mode: run.py <input> <output>
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        
        success = verify_system1()
        
        if not GuardedWriter:
            print("Error: GuardedWriter not available")
            sys.exit(1)

        writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"]) 
        writer.open_commit_gate()
        
        writer.mkdir_durable(str(output_path.parent))
        writer.write_durable(str(output_path), json.dumps({"success": success, "description": "basic test"}))
            
        sys.exit(0 if success else 1)

    success = verify_system1()
    sys.exit(0 if success else 1)
