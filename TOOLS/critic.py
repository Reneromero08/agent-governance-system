#!/usr/bin/env python3

"""
Critic script for AGS.

This script analyzes the repository for potential governance violations. It can
be run manually, as a pre-commit hook, or in CI. Checks include:

- Ensuring CANON changes have corresponding CHANGELOG entries
- Verifying fixtures exist for skills
- Checking that invariants are not violated
- Detecting raw filesystem access patterns in skills

Exit codes:
- 0: All checks passed
- 1: Violations found
"""

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANON_DIR = PROJECT_ROOT / "CANON"
SKILLS_DIR = PROJECT_ROOT / "SKILLS"
CHANGELOG_PATH = CANON_DIR / "CHANGELOG.md"

# Patterns that indicate raw filesystem access (forbidden in skills per INV-003)
RAW_FS_PATTERNS = [
    r'\bos\.walk\b',
    r'\bos\.listdir\b',
    r'\bglob\.glob\b',
    r'\bPath\([\'"][^\'"]',  # Path with hardcoded string (not from cortex)
    r'\.rglob\(',
    r'\.glob\(',
]


def get_changed_files() -> list[str]:
    """Get list of files changed in the current commit/staging."""
    try:
        # Check staged files first
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')
        
        # Fall back to last commit diff
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "--name-only"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
    except Exception:
        pass
    return []


def check_canon_changelog(changed_files: list[str]) -> list[str]:
    """Check that CANON changes have corresponding CHANGELOG entries."""
    violations = []
    canon_changes = [f for f in changed_files if f.startswith("CANON/") and f != "CANON/CHANGELOG.md"]
    changelog_changed = any(f == "CANON/CHANGELOG.md" for f in changed_files)
    
    if canon_changes and not changelog_changed:
        violations.append(
            f"CANON files changed ({', '.join(canon_changes)}) but CHANGELOG.md was not updated"
        )
    return violations


def check_skill_fixtures() -> list[str]:
    """Check that all skills have fixtures."""
    violations = []
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        fixtures_dir = skill_dir / "fixtures"
        if not fixtures_dir.exists() or not any(fixtures_dir.iterdir()):
            violations.append(f"Skill '{skill_dir.name}' has no fixtures")
    return violations


def check_raw_fs_access() -> list[str]:
    """Check for raw filesystem access in skill code (violates INV-003)."""
    violations = []
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        for py_file in skill_dir.glob("*.py"):
            if py_file.name == "validate.py":
                continue  # Validators may need fs access
            content = py_file.read_text(errors="ignore")
            for pattern in RAW_FS_PATTERNS:
                if re.search(pattern, content):
                    # Check if it's in artifact-escape-hatch or pack-validate (allowed)
                    if skill_dir.name in ("artifact-escape-hatch", "pack-validate"):
                        continue
                    violations.append(
                        f"Skill '{skill_dir.name}/{py_file.name}' may use raw filesystem access (pattern: {pattern})"
                    )
                    break
    return violations


def check_skill_manifests() -> list[str]:
    """Check that all skills have SKILL.md manifests."""
    violations = []
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        manifest = skill_dir / "SKILL.md"
        if not manifest.exists():
            violations.append(f"Skill '{skill_dir.name}' missing SKILL.md manifest")
    return violations


def main() -> int:
    print("[critic] Running governance checks...")
    
    all_violations = []
    
    # Get changed files for context-aware checks
    changed_files = get_changed_files()
    
    # Run all checks
    all_violations.extend(check_canon_changelog(changed_files))
    all_violations.extend(check_skill_fixtures())
    all_violations.extend(check_raw_fs_access())
    all_violations.extend(check_skill_manifests())
    
    if all_violations:
        print(f"\n[critic] Found {len(all_violations)} violation(s):\n")
        for v in all_violations:
            print(f"  ✗ {v}")
        print()
        return 1
    
    print("[critic] All checks passed ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())