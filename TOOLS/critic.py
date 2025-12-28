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
from typing import List
import schema_validator  # New import

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANON_DIR = PROJECT_ROOT / "CANON"
SKILLS_DIR = PROJECT_ROOT / "SKILLS"
CONTEXT_DIR = PROJECT_ROOT / "CONTEXT"
CHANGELOG_PATH = CANON_DIR / "CHANGELOG.md"

# Allowed output roots per CONTRACT Rule 6
ALLOWED_OUTPUT_ROOTS = {
    str(PROJECT_ROOT / "CONTRACTS" / "_runs"),
    str(PROJECT_ROOT / "CORTEX" / "_generated"),
    str(PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs"),
}

# Patterns that indicate raw filesystem access (forbidden in skills per INV-003)
RAW_FS_PATTERNS = [
    r'\bos\.walk\b',
    r'\bos\.listdir\b',
    r'\bglob\.glob\b',
    r'\bPath\([\'"][^\'"]',  # Path with hardcoded string (not from cortex)
    r'\.rglob\(',
    r'\.glob\(',
]


def get_changed_files() -> List[str]:
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


def check_canon_changelog(changed_files: List[str]) -> List[str]:
    """Check that CANON changes have corresponding CHANGELOG entries."""
    violations = []
    canon_changes = [f for f in changed_files if f.startswith("CANON/") and f != "CANON/CHANGELOG.md"]
    changelog_changed = any(f == "CANON/CHANGELOG.md" for f in changed_files)
    
    if canon_changes and not changelog_changed:
        violations.append(
            f"CANON files changed ({', '.join(canon_changes)}) but CHANGELOG.md was not updated"
        )
    return violations


def check_skill_fixtures() -> List[str]:
    """Check that all skills have fixtures."""
    violations = []
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        fixtures_dir = skill_dir / "fixtures"
        if not fixtures_dir.exists() or not any(fixtures_dir.iterdir()):
            violations.append(f"Skill '{skill_dir.name}' has no fixtures")
    return violations


def check_raw_fs_access() -> List[str]:
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
                    if skill_dir.name in ("artifact-escape-hatch", "pack-validate", "llm-packer-smoke", "cas-integrity-check"):
                        continue
                    violations.append(
                        f"Skill '{skill_dir.name}/{py_file.name}' may use raw filesystem access (pattern: {pattern})"
                    )
                    break
    return violations


def check_skill_manifests() -> List[str]:
    """Check that all skills have SKILL.md manifests."""
    violations = []
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        manifest = skill_dir / "SKILL.md"
        if not manifest.exists():
            violations.append(f"Skill '{skill_dir.name}' missing SKILL.md manifest")
    return violations


def check_schema_validation() -> List[str]:
    """Check that all Law-Like files (ADRs, Skills, Styles) are schema-valid."""
    violations = []
    
    # Check ADRs
    adr_dir = PROJECT_ROOT / "CONTEXT" / "decisions"
    adr_schema = PROJECT_ROOT / "MCP" / "schemas" / "governance" / "adr.schema.json"
    if adr_dir.exists():
        for adr_path in adr_dir.glob("ADR-*.md"):
            errors = schema_validator.validate_file(str(adr_path), str(adr_schema))
            if errors:
                violations.extend(errors)
                
    # Check Skills
    skill_schema = PROJECT_ROOT / "MCP" / "schemas" / "governance" / "skill.schema.json"
    for skill_path in SKILLS_DIR.iterdir():
        if not skill_path.is_dir() or skill_path.name.startswith("_"):
            continue
        manifest = skill_path / "SKILL.md"
        if manifest.exists():
            errors = schema_validator.validate_file(str(manifest), str(skill_schema))
            if errors:
                violations.extend(errors)
                
    # Check Styles
    style_dir = PROJECT_ROOT / "CONTEXT" / "preferences"
    style_schema = PROJECT_ROOT / "MCP" / "schemas" / "governance" / "style.schema.json"
    if style_dir.exists():
        for style_path in style_dir.glob("STYLE-*.md"):
            if style_path.name.endswith("-template.md"):
                continue
            errors = schema_validator.validate_file(str(style_path), str(style_schema))
            if errors:
                violations.extend(errors)
                
    return violations


def check_log_output_roots() -> List[str]:
    """Check that logging code complies with ADR-015 (logs under CONTRACTS/_runs/)."""
    violations = []

    # Patterns that indicate logging to disallowed locations
    disallowed_patterns = [
        (r'["\']LOGS/', "References LOGS/ directory (should be CONTRACTS/_runs/<purpose>_logs/)"),
        (r'["\']MCP/logs/', "References MCP/logs/ (should be CONTRACTS/_runs/mcp_logs/)"),
    ]

    # Scan Python files that may write logs
    scan_paths = [
        PROJECT_ROOT / "TOOLS",
        PROJECT_ROOT / "MCP",
        PROJECT_ROOT / "SKILLS",
    ]

    for scan_dir in scan_paths:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            # Skip entrypoint (it's explicitly allowed to redirect logs)
            if "entrypoint" in py_file.name:
                continue
            content = py_file.read_text(errors="ignore")
            for pattern, description in disallowed_patterns:
                if re.search(pattern, content):
                    violations.append(
                        f"{py_file.relative_to(PROJECT_ROOT)}: {description}"
                    )
                    break  # Only report once per file

    # Check canon docs for outdated references (except CHANGELOG which documents history)
    canon_files = [
        CANON_DIR / "CRISIS.md",
        CANON_DIR / "STEWARDSHIP.md",
    ]
    
    # Check for bare excepts (see CANON/STEWARDSHIP.md "No Bare Excepts" rule)
    bare_excepts_patterns = [
        (r'except:\s*$', 'Uses bare except: keyword without specifying exception type'),
        (r'except\s*:\s*$', 'Uses bare except: keyword with colon separator'),
    ]

    for canon_file in canon_files:
        if not canon_file.exists():
            continue
        content = canon_file.read_text(errors="ignore")
        # Check for LOGS/ references (but skip ADR context)
        if re.search(r'`LOGS/[^`]+`', content):
            violations.append(
                f"{canon_file.relative_to(PROJECT_ROOT)}: References LOGS/ (should use CONTRACTS/_runs/)"
            )
        # Check for MCP/logs/ references (but skip ADR context)
        if re.search(r'`MCP/logs/[^`]+`', content):
            violations.append(
                f"{canon_file.relative_to(PROJECT_ROOT)}: References MCP/logs/ (should use CONTRACTS/_runs/)"
            )

    # For CHANGELOG, only check Unreleased section (not historical releases)
    changelog_path = CANON_DIR / "CHANGELOG.md"
    if changelog_path.exists():
        content = changelog_path.read_text(errors="ignore")
        # Extract only Unreleased section
        unreleased_match = re.search(r'## \[Unreleased\](.*?)(?=## \[)', content, re.DOTALL)
        if unreleased_match:
            unreleased_section = unreleased_match.group(1)
            if re.search(r'`LOGS/[^`]+`', unreleased_section):
                violations.append(
                    f"CANON/CHANGELOG.md [Unreleased]: References LOGS/ (should use CONTRACTS/_runs/)"
                )
            if re.search(r'`MCP/logs/[^`]+`', unreleased_section):
                violations.append(
                    f"CANON/CHANGELOG.md [Unreleased]: References MCP/logs/ (should use CONTRACTS/_runs/)"
                )

    return violations


def check_context_edits(changed_files: List[str]) -> List[str]:
    """Check that CONTEXT record edits are not made (ADR-016 enforcement).

    Editing existing CONTEXT records requires explicit user instruction AND task intent.
    New files (appends) are allowed. Modifications to existing files are flagged.
    """
    violations = []
    context_files = {
        str(CONTEXT_DIR / "decisions"),
        str(CONTEXT_DIR / "rejected"),
        str(CONTEXT_DIR / "preferences"),
    }

    for changed_file in changed_files:
        # Check if file is in CONTEXT subdirectories
        file_path = PROJECT_ROOT / changed_file
        for context_subdir in context_files:
            if str(file_path).startswith(context_subdir):
                # If it's a new file (.json or .md in CONTEXT), it's an append - OK
                if file_path.exists():
                    # Get git status to see if this is a modification (not new file)
                    try:
                        result = subprocess.run(
                            ["git", "diff", "--cached", "--name-status", changed_file],
                            capture_output=True, text=True, cwd=PROJECT_ROOT
                        )
                        status = result.stdout.strip().split('\t')[0] if result.stdout.strip() else ""
                        # M = modified, A = added, D = deleted
                        if status == "M":
                            violations.append(
                                f"CONTEXT record edited: {changed_file}. "
                                f"Editing existing CONTEXT requires explicit user instruction AND task intent (ADR-016)."
                            )
                    except Exception:
                        pass

    return violations


def check_output_roots(changed_files: List[str]) -> List[str]:
    """Check that artifacts are written only to allowed output roots (CONTRACT Rule 6).

    Scans changed files for hardcoded artifact paths outside allowed roots.
    """
    violations = []
    disallowed_patterns = [
        (r'["\']BUILD/', "References BUILD/ (reserved for user outputs, not system artifacts)"),
        (r'["\']\.\.?/BUILD/', "References BUILD/ via relative path"),
        (r'["\']LOGS/', "References LOGS/ (moved to CONTRACTS/_runs/ per ADR-015)"),
        (r'["\']MCP/logs', "References MCP/logs/ (moved to CONTRACTS/_runs/mcp_logs/ per ADR-015)"),
    ]

    python_files = [f for f in changed_files if f.endswith('.py')]
    for file_path in python_files:
        try:
            full_path = PROJECT_ROOT / file_path
            if not full_path.exists():
                continue
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            for pattern, msg in disallowed_patterns:
                if re.search(pattern, content):
                    violations.append(f"{file_path}: {msg}")
        except Exception:
            pass

    return violations


def main() -> int:
    quarantine_file = PROJECT_ROOT / ".quarantine"
    if quarantine_file.exists():
        print("[critic] QUARANTINE: System is in quarantine mode. No changes allowed.")
        return 1
    print("[critic] Running governance checks...")
    
    all_violations = []
    
    # Get changed files for context-aware checks
    changed_files = get_changed_files()
    
    # Run all checks
    all_violations.extend(check_canon_changelog(changed_files))
    all_violations.extend(check_skill_fixtures())
    all_violations.extend(check_raw_fs_access())
    all_violations.extend(check_skill_manifests())
    all_violations.extend(check_schema_validation())
    all_violations.extend(check_log_output_roots())  # Check ADR-015 compliance
    all_violations.extend(check_context_edits(changed_files))  # Check ADR-016 compliance
    all_violations.extend(check_output_roots(changed_files))  # Check CONTRACT Rule 6
    
    if all_violations:
        print(f"\n[critic] Found {len(all_violations)} violation(s):\n")
        for v in all_violations:
            print(f"  [FAIL] {v}")
        print()
        return 1
    
    print("[critic] All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
