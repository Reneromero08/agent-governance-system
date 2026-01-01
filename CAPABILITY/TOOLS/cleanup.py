#!/usr/bin/env python3
"""
Repository Hygiene Tool (Cleanup)

A rigorous cleaner for the Agent Governance System.
Removes:
1. Python cache artifacts (__pycache__, *.pyc, *.pyo, *.pyd)
2. Temporary files (*.tmp, *.bak, *.swp)
3. Stray logs (*.log) outside of designated log directories
4. Empty directories (optional)
5. Aggressive mode: Cleans old runs from LAW/CONTRACTS/_runs/

Usage:
    python CAPABILITY/TOOLS/cleanup.py [--dry-run] [--aggressive] [--verbose]
"""

import argparse
import os
import shutil
import time
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFE_EXTENSIONS = {".tmp", ".bak", ".swp", ".swo", ".log"}
SAFE_DIRS = {"__pycache__", ".pytest_cache", "_tmp", "_cache"}
PROTECTED_DIRS = {".git", ".venv", ".kilocode", "node_modules"}

# Allowed Log Locations (Logs here are preserved unless --aggressive)
ALLOWED_LOG_ROOTS = [
    PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
]

def is_protected(path: Path) -> bool:
    """Check if path is in a protected directory."""
    for part in path.parts:
        if part in PROTECTED_DIRS:
            return True
    return False

def is_allowed_log_location(path: Path) -> bool:
    """Check if file is inside an allowed log root."""
    for root in ALLOWED_LOG_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False

def clean_file(path: Path, dry_run: bool = True, verbose: bool = False):
    """Delete a file if allowed."""
    if is_protected(path):
        return

    # Check for stray logs
    if path.suffix == ".log" and is_allowed_log_location(path):
        if verbose:
            print(f"[SKIP] log in allowed location: {path.relative_to(PROJECT_ROOT)}")
        return

    if dry_run:
        print(f"[DRY-RUN] Would delete file: {path.relative_to(PROJECT_ROOT)}")
    else:
        try:
            path.unlink()
            if verbose:
                print(f"[DELETED] {path.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"[ERROR] deleting {path}: {e}")

def clean_dir(path: Path, dry_run: bool = True, verbose: bool = False):
    """Delete a directory recursively."""
    if is_protected(path):
        return

    if dry_run:
        print(f"[DRY-RUN] Would delete dir:  {path.relative_to(PROJECT_ROOT)}")
    else:
        try:
            shutil.rmtree(path)
            if verbose:
                print(f"[DELETED] {path.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"[ERROR] deleting {path}: {e}")

def cleanup(dry_run: bool, aggressive: bool, verbose: bool):
    """Run the cleanup protocol."""
    print(f"Running Cleanup Protocol (Dry Run: {dry_run}, Aggressive: {aggressive})...")
    
    count_files = 0
    count_dirs = 0

    # 1. Walk and clean standard artifacts
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        
        # Skip protected dirs to avoid recursion overhead
        dirs[:] = [d for d in dirs if d not in PROTECTED_DIRS]

        # Clean Directories (__pycache__, etc)
        for d in list(dirs):
            dir_path = root_path / d
            if d in SAFE_DIRS:
                clean_dir(dir_path, dry_run, verbose)
                dirs.remove(d) # Don't descend into deleted dir
                count_dirs += 1

        # Clean Files
        for f in files:
            file_path = root_path / f
            if file_path.suffix in SAFE_EXTENSIONS:
                clean_file(file_path, dry_run, verbose)
                count_files += 1

    # 2. Aggressive Cleanup (Runs)
    if aggressive:
        runs_dir = PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs"
        if runs_dir.exists():
            print(f"\nScanning {runs_dir} for stale runs...")
            # Logic: Delete directories older than 7 days? Or just all non-keep?
            # For now, just listing common test patterns to purge.
            # Real implementation would need care.
            # Current simplified logic: Purge know test patterns.
            patterns = ["pipeline-*", "test-run-*", "add_docstrings_*", "_tmp", "_cache"]
            
            for pat in patterns:
                for target in runs_dir.glob(pat):
                    if target.is_dir():
                        clean_dir(target, dry_run, verbose)
                        count_dirs += 1
                    elif target.is_file():
                        clean_file(target, dry_run, verbose)
                        count_files += 1

    print(f"\nCleanup Complete.")
    print(f"  Files targeted: {count_files}")
    print(f"  Dirs targeted:  {count_dirs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGS Repository Hygiene Tool")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    parser.add_argument("--confirm", action="store_true", help="Execute deletion (negates --dry-run)")
    parser.add_argument("--aggressive", action="store_true", help="Clean up old run artifacts")
    parser.add_argument("--verbose", action="store_true", help="Print details")
    
    args = parser.parse_args()
    
    # Default to dry-run unless confirmed
    is_dry_run = not args.confirm
    if args.dry_run:
        is_dry_run = True

    cleanup(is_dry_run, args.aggressive, args.verbose)
