#!/usr/bin/env python3
"""
System 1 Verify Skill (Lane C1)

Ensures system1.db is in sync with the repository.
Verifies:
1. All CANON files are indexed
2. Content hashes match
3. No orphaned entries in DB
"""

import sys
import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path("CORTEX/system1.db")
CANON_DIR = Path("CANON")

def verify_system1():
    """Verify system1.db matches repository state."""
    if not DB_PATH.exists():
        print("❌ FAIL: system1.db does not exist")
        return False
        
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    # Get all indexed files
    cursor = conn.execute("SELECT path, content_hash FROM files")
    db_files = {row['path']: row['content_hash'] for row in cursor}
    
    # Get all CANON markdown files
    canon_files = {}
    for md_file in CANON_DIR.rglob("*.md"):
        rel_path = md_file.relative_to(Path.cwd()).as_posix()
        content = md_file.read_text(encoding='utf-8')
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        canon_files[rel_path] = content_hash
    
    # Check for missing files
    missing = set(canon_files.keys()) - set(db_files.keys())
    if missing:
        print(f"❌ FAIL: {len(missing)} files not indexed:")
        for f in sorted(missing):
            print(f"  - {f}")
        return False
    
    # Check for stale files
    stale = set(db_files.keys()) - set(canon_files.keys())
    if stale:
        print(f"⚠️  WARN: {len(stale)} orphaned entries in DB:")
        for f in sorted(stale):
            print(f"  - {f}")
    
    # Check for hash mismatches
    mismatches = []
    for path in canon_files:
        if path in db_files and db_files[path] != canon_files[path]:
            mismatches.append(path)
    
    if mismatches:
        print(f"❌ FAIL: {len(mismatches)} files have changed:")
        for f in sorted(mismatches):
            print(f"  - {f}")
        return False
    
    conn.close()
    
    print(f"✅ PASS: system1.db is in sync")
    print(f"  - {len(canon_files)} files verified")
    print(f"  - {len(stale)} orphaned (cleanup recommended)")
    return True

if __name__ == "__main__":
    success = verify_system1()
    sys.exit(0 if success else 1)
