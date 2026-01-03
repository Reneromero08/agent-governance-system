#!/usr/bin/env python3
"""
Add/Update Content Hash Tool

Adds or updates the <!-- CONTENT_HASH: <sha256> --> marker in markdown files.
If YAML frontmatter exists, the hash is placed immediately after.
Otherwise, it is placed at the very top.
"""

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Optional

def compute_sha256_for_hashing(content: str) -> str:
    """Compute SHA-256 hash of content, excluding any existing hash markers."""
    # Remove any existing hash markers to make the hash stable
    # Pattern matches both <!-- CONTENT_HASH: ... --> and CONTENT_HASH: ...
    content_clean = re.sub(r'<!-- CONTENT_HASH: [a-f0-9]{64} -->\n?', '', content)
    content_clean = re.sub(r'^CONTENT_HASH: [a-f0-9]{64}\n?', '', content_clean, flags=re.MULTILINE)
    
    return hashlib.sha256(content_clean.encode('utf-8')).hexdigest()

def add_hash_to_file(file_path: Path, dry_run: bool = False) -> bool:
    """Add or update the content hash in a markdown file."""
    if not file_path.exists() or not file_path.is_file():
        print(f"[ERR] File not found: {file_path}")
        return False

    try:
        content = file_path.read_text(encoding="utf-8")
        h = compute_sha256_for_hashing(content)
        hash_marker = f"<!-- CONTENT_HASH: {h} -->"

        # Check for existing hash
        if f"CONTENT_HASH: {h}" in content:
            # Already has the correct hash
            return True

        # Remove old hashes if they exist
        new_content = re.sub(r'<!-- CONTENT_HASH: [a-f0-9]{64} -->\n?', '', content)
        new_content = re.sub(r'^CONTENT_HASH: [a-f0-9]{64}\n?', '', new_content, flags=re.MULTILINE)
        
        # Determine where to insert
        # 1. After YAML frontmatter
        yaml_match = re.match(r'^---\n(.*?)\n---\n?', new_content, re.DOTALL)
        if yaml_match:
            insert_pos = yaml_match.end()
            # Ensure there's a newline after the marker if we insert it
            result = new_content[:insert_pos] + "\n" + hash_marker + "\n" + new_content[insert_pos:].lstrip()
        else:
            # 2. At the top
            result = hash_marker + "\n\n" + new_content.lstrip()

        if dry_run:
            print(f"[DRY] Would update hash for {file_path}")
            return True

        file_path.write_text(result, encoding="utf-8")
        print(f"[OK] Updated hash for {file_path}")
        return True

    except Exception as e:
        print(f"[ERR] Failed to process {file_path}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add or update CONTENT_HASH in markdown files.")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    
    args = parser.parse_args()
    
    success = True
    for f in args.files:
        path = Path(f).resolve()
        if not add_hash_to_file(path, args.dry_run):
            success = False
            
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
