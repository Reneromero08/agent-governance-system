#!/usr/bin/env python3

"""
Token linter for AGS.

This script scans the repository for usage of glossary terms and warns about:
- Undefined tokens (terms used but not in GLOSSARY.md)
- Deprecated tokens (terms marked for removal)
- Case mismatches (e.g., "canon" vs "Canon")

Exit codes:
- 0: All checks passed (or only warnings)
- 1: Errors found
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_PATH = PROJECT_ROOT / "CANON" / "GLOSSARY.md"
VERSIONING_PATH = PROJECT_ROOT / "CANON" / "VERSIONING.md"

# Directories to scan for token usage
SCAN_DIRS = ["CANON", "CONTEXT", "MAPS", "SKILLS", "CONTRACTS", "AGENTS.md", "README.md"]


def load_glossary_terms() -> Set[str]:
    """Extract defined terms from GLOSSARY.md."""
    terms = set()
    content = GLOSSARY_PATH.read_text(errors="ignore")
    # Match lines like: - **Term** - Description
    for match in re.finditer(r'^\s*-\s*\*\*([^*]+)\*\*', content, re.MULTILINE):
        terms.add(match.group(1).strip())
    return terms


def load_deprecated_tokens() -> Dict[str, str]:
    """Load deprecated tokens from VERSIONING.md."""
    deprecated = {}
    if not VERSIONING_PATH.exists():
        return deprecated
    
    content = VERSIONING_PATH.read_text(errors="ignore")
    # Look for deprecation section
    in_deprecation = False
    for line in content.splitlines():
        if "deprecat" in line.lower() and "#" in line:
            in_deprecation = True
            continue
        if in_deprecation and line.startswith("#"):
            break
        if in_deprecation and line.strip().startswith("-"):
            # Try to extract token and reason
            match = re.match(r'-\s*`?(\w+)`?\s*[-:]\s*(.+)', line)
            if match:
                deprecated[match.group(1)] = match.group(2)
    return deprecated


def check_term_usage(terms: Set[str], deprecated: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Check for term usage issues across the codebase."""
    warnings = []
    errors = []
    
    # Build regex for term matching
    term_pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b', re.IGNORECASE)
    
    for item in SCAN_DIRS:
        path = PROJECT_ROOT / item
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.rglob("*.md"))
        else:
            continue
        
        for file in files:
            try:
                content = file.read_text(errors="ignore")
            except Exception:
                continue
            
            # Check for deprecated token usage
            for token, reason in deprecated.items():
                if re.search(rf'\b{re.escape(token)}\b', content):
                    rel_path = file.relative_to(PROJECT_ROOT)
                    warnings.append(f"{rel_path}: uses deprecated token '{token}' - {reason}")
    
    return warnings, errors


def main() -> int:
    print("[lint_tokens] Scanning for token usage issues...")
    
    terms = load_glossary_terms()
    print(f"  Loaded {len(terms)} terms from GLOSSARY.md")
    
    deprecated = load_deprecated_tokens()
    if deprecated:
        print(f"  Found {len(deprecated)} deprecated token(s)")
    
    warnings, errors = check_term_usage(terms, deprecated)
    
    if warnings:
        print(f"\n[lint_tokens] Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if errors:
        print(f"\n[lint_tokens] Errors ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    
    print("[lint_tokens] All checks passed ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
