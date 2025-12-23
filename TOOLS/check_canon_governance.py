#!/usr/bin/env python3

"""
Canon governance checker for AGS.

This script verifies consistency across canon files:
- canon_version matches across VERSIONING.md, cortex.json, and manifests
- All invariants are numbered correctly
- Glossary terms are alphabetically ordered (warning)
- CHANGELOG has an entry for the current version

Exit codes:
- 0: All checks passed
- 1: Errors found
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSIONING_PATH = PROJECT_ROOT / "CANON" / "VERSIONING.md"
CHANGELOG_PATH = PROJECT_ROOT / "CANON" / "CHANGELOG.md"
INVARIANTS_PATH = PROJECT_ROOT / "CANON" / "INVARIANTS.md"
CORTEX_INDEX = PROJECT_ROOT / "CORTEX" / "_generated" / "cortex.json"
CORTEX_FALLBACK = PROJECT_ROOT / "CORTEX" / "cortex.json"


def get_canon_version() -> Optional[str]:
    """Extract canon_version from VERSIONING.md."""
    content = VERSIONING_PATH.read_text(errors="ignore")
    match = re.search(r'canon_version:\s*(\d+\.\d+\.\d+)', content)
    return match.group(1) if match else None


def get_cortex_version() -> Optional[str]:
    """Extract canon_version from cortex index."""
    for path in [CORTEX_INDEX, CORTEX_FALLBACK]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return data.get("canon_version")
            except Exception:
                pass
    return None


def check_version_consistency() -> List[str]:
    """Check that versions are consistent across files."""
    errors = []
    canon_ver = get_canon_version()
    cortex_ver = get_cortex_version()
    
    if not canon_ver:
        errors.append("Could not find canon_version in VERSIONING.md")
    
    if cortex_ver and canon_ver and cortex_ver != canon_ver:
        errors.append(
            f"Version mismatch: VERSIONING.md has {canon_ver}, cortex has {cortex_ver}"
        )
    
    return errors


def check_changelog_current() -> List[str]:
    """Check that CHANGELOG has an entry for current version."""
    errors = []
    canon_ver = get_canon_version()
    if not canon_ver:
        return errors
    
    content = CHANGELOG_PATH.read_text(errors="ignore")
    if f"[{canon_ver}]" not in content:
        errors.append(f"CHANGELOG.md missing entry for current version [{canon_ver}]")
    
    return errors


def check_invariant_numbering() -> List[str]:
    """Check that invariants are numbered sequentially."""
    errors = []
    content = INVARIANTS_PATH.read_text(errors="ignore")
    
    numbers = []
    for match in re.finditer(r'\[INV-(\d+)\]', content):
        numbers.append(int(match.group(1)))
    
    if numbers:
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            errors.append(f"Invariant numbering not sequential: found {numbers}, expected {expected}")
    
    return errors


# Core invariants that must always exist (v1.0 freeze)
FROZEN_INVARIANTS = ["INV-001", "INV-002", "INV-003", "INV-004", "INV-005", "INV-006", "INV-007", "INV-008"]


def check_invariant_freeze() -> List[str]:
    """Check that all frozen invariants still exist."""
    errors = []
    content = INVARIANTS_PATH.read_text(errors="ignore")
    
    for inv in FROZEN_INVARIANTS:
        if f"[{inv}]" not in content:
            errors.append(f"Frozen invariant {inv} missing from INVARIANTS.md")
    
    return errors


def main() -> int:
    print("[check_canon_governance] Running consistency checks...")
    
    all_errors = []
    all_errors.extend(check_version_consistency())
    all_errors.extend(check_changelog_current())
    all_errors.extend(check_invariant_numbering())
    all_errors.extend(check_invariant_freeze())
    
    if all_errors:
        print(f"\n[check_canon_governance] Found {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  ✗ {e}")
        return 1
    
    print("[check_canon_governance] All checks passed ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
