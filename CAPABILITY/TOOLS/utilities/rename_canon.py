#!/usr/bin/env python3
"""
Canonical Renamer Tool

Renames files to the AGS canonical format: MM-DD-YYYY-HH-MM_TITLE.ext
"""

import os
import sys
import argparse
import re
from pathlib import Path
from datetime import datetime

def is_canonical(filename):
    """Check if filename matches canonical pattern."""
    # Pattern: Digit{2}-Digit{2}-Digit{4}-Digit{2}-Digit{2}_TITLE.ext
    pattern = r"^\d{2}-\d{2}-\d{4}-\d{2}-\d{2}_.+\..+$"
    return bool(re.match(pattern, filename))

def sanitize_title(name):
    """Convert filename to UPPER_CASE_SNAKE_CASE."""
    name = name.upper()
    name = re.sub(r'[^A-Z0-9\.]', '_', name)  # Replace non-alphanumeric with _
    name = re.sub(r'_+', '_', name)          # Collapse multiple underscores
    return name

def rename_to_canonical(path, dry_run=False, verbose=False):
    """Rename a single file to canonical format."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return

    if is_canonical(path.name):
        if verbose:
            print(f"[SKIP] Already canonical: {path.name}")
        return

    # Get timestamp (mtime as proxy for creation if ctime unreliable)
    # In strict mode, we might want ctime, but mtime is safer for "last state" snapshots
    ts = path.stat().st_mtime
    dt = datetime.fromtimestamp(ts)
    date_prefix = dt.strftime("%m-%d-%Y-%H-%M")

    # Sanitize base name
    clean_name = sanitize_title(path.stem)
    ext = path.suffix.lower() # Keep extension lowercase for readability? Or upper?
    # Policy says TITLE must be ALL CAPS. Usually extensions are lowercase. 
    # But let's follow existing: .md
    
    new_name = f"{date_prefix}_{clean_name}{ext}"
    new_path = path.with_name(new_name)

    if new_path.exists():
        if verbose:
            print(f"[SKIP] Target exists: {new_name}")
        return

    if dry_run:
        print(f"[DRY] Rename: {path.name} -> {new_name}")
    else:
        try:
            path.rename(new_path)
            print(f"[OK] Renamed: {path.name} -> {new_name}")
        except Exception as e:
            print(f"[ERR] Failed to rename {path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Rename files to canonical INBOX format.")
    parser.add_argument("--target", required=True, help="Directory to process")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show skipped files")

    args = parser.parse_args()

    root = Path(args.target).resolve()
    if not root.exists():
        print(f"Error: Target path {root} does not exist.")
        sys.exit(1)

    if root.is_file():
        rename_to_canonical(root, args.dry_run, args.verbose)
    else:
        for p in root.rglob("*") if args.recursive else root.iterdir():
            if p.is_file():
                 rename_to_canonical(p, args.dry_run, args.verbose)

if __name__ == "__main__":
    main()
