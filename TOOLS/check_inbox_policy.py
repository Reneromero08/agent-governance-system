#!/usr/bin/env python3
"""
INBOX Policy Check

Enforces that all human-readable documents are stored in INBOX directory
and contain content hashes.

This script is run by the pre-commit hook.
"""

import hashlib
import os
import re
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# INBOX directory
INBOX_DIR = PROJECT_ROOT / "INBOX"

# Files to check (all staged files)
STAGED_FILES_PATH = os.environ.get("STAGED_FILES", "")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def has_content_hash(file_path: Path) -> bool:
    """Check if file has content hash comment."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_lines = f.readlines()[:5]
            for line in first_lines:
                if "CONTENT_HASH:" in line:
                    return True
        return False
    except Exception:
        return False


def is_inbox_file(file_path: Path) -> bool:
    """Check if file is in INBOX directory."""
    try:
        file_path.resolve().relative_to(INBOX_DIR.resolve())
        return True
    except ValueError:
        return False


def get_staged_files() -> list[Path]:
    """Get list of staged files for git commit."""
    if STAGED_FILES_PATH:
        path = Path(STAGED_FILES_PATH)
        if path.exists() and path.is_file():
            import json
            with open(path, "r") as f:
                data = json.load(f)
                return [Path(f) for f in data.get("staged_files", [])]

    # Fallback: use git status
    try:
        import subprocess
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        if result.returncode == 0:
            return [PROJECT_ROOT / f for f in result.stdout.strip().split("\n") if f]
    except Exception:
        pass

    return []


def check_inbox_policy() -> dict:
    """Check INBOX policy for all relevant files."""
    staged_files = get_staged_files()

    violations = []
    valid_count = 0

    for file_path in staged_files:
        # Skip non-markdown files
        if file_path.suffix != ".md":
            continue

        # Skip files in exempt locations
        rel_path = file_path.relative_to(PROJECT_ROOT)
        parts = str(rel_path).split(os.sep)

        # Exempt directories
        if any(part in ["CANON", "CONTEXT", "SKILLS", "TOOLS", "CORTEX", "CONTRACTS", "CATALYTIC-DPT", "MEMORY", "BUILD", ".git"] for part in parts):
            continue

        # Check if file is in INBOX
        if not is_inbox_file(file_path):
            violations.append({
                "file": str(rel_path),
                "error": "Human-readable document outside INBOX",
                "fix": f"Move to INBOX/reports/ or appropriate INBOX subdirectory"
            })
            continue

        # Check for content hash
        if not has_content_hash(file_path):
            violations.append({
                "file": str(file_path.relative_to(INBOX_DIR)),
                "error": "Missing content hash",
                "fix": "Add <!-- CONTENT_HASH: <sha256> --> to top of file"
            })
            continue

        valid_count += 1

    return {
        "violations": violations,
        "valid_count": valid_count
    }


def main():
    """Run INBOX policy check."""
    result = check_inbox_policy()

    if result["violations"]:
        print("[ERROR] INBOX Policy Violations:")
        for v in result["violations"]:
            print(f"  - {v['file']}: {v['error']}")
            print(f"    Fix: {v['fix']}")
        print()
        print(f"[INFO] Valid INBOX files: {result['valid_count']}")
        sys.exit(1)
    else:
        print(f"[OK] All {result['valid_count']} INBOX documents compliant")
        sys.exit(0)


if __name__ == "__main__":
    main()
