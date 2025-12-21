#!/usr/bin/env python3

"""
Artifact escape hatch validator.

This script checks that no files have been created outside the allowed
artifact output directories. It is run as a governance fixture.

Allowed output roots (per INV-006):
- CONTRACTS/_runs/
- CORTEX/_generated/
- MEMORY/LLM-PACKER/_packs/
- BUILD/ (user outputs only)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories that may contain generated artifacts
ALLOWED_ARTIFACT_ROOTS = {
    PROJECT_ROOT / "CONTRACTS" / "_runs",
    PROJECT_ROOT / "CORTEX" / "_generated",
    PROJECT_ROOT / "MEMORY" / "LLM-PACKER" / "_packs",
    PROJECT_ROOT / "BUILD",
}

# Patterns to always ignore (source files, not runtime artifacts)
IGNORE_PATTERNS = {
    ".git",
    "__pycache__",
    ".gitkeep",
    "node_modules",
    "fixtures",  # Skill fixture files are test data
    "schemas",   # Schema files are source, not generated
    "_packs",    # Pack directories are allowed
}

# Directories that contain source code, not runtime artifacts
SOURCE_DIRS = {
    "SKILLS",
    "CONTRACTS",
    "CORTEX",
    "CANON",
    "CONTEXT", 
    "MAPS",
    "TOOLS",
}


def is_allowed_path(path: Path) -> bool:
    """Check if a path is within an allowed artifact root."""
    for root in ALLOWED_ARTIFACT_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def should_ignore(path: Path) -> bool:
    """Check if path should be ignored."""
    return any(part in IGNORE_PATTERNS for part in path.parts)


def is_runtime_artifact(path: Path) -> bool:
    """Check if file looks like a runtime-generated artifact vs source code."""
    # Specific runtime artifact directories
    RUNTIME_DIRS = {"_runs", "_generated", "_packs"}
    if any(part in RUNTIME_DIRS for part in path.parts):
        return True
    # .log files are always runtime
    if path.suffix == ".log":
        return True
    return False


def find_escaped_artifacts(scan_roots: list[Path]) -> list[Path]:
    """Find any generated runtime files outside allowed roots."""
    escaped = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not should_ignore(path):
                # Only flag files that look like runtime artifacts
                if is_runtime_artifact(path) and not is_allowed_path(path):
                    escaped.append(path)
    return escaped


def main(input_path: Path, output_path: Path) -> int:
    """Run the artifact escape hatch check."""
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    # Scan the project for escaped artifacts
    scan_dirs = [
        PROJECT_ROOT / "CONTRACTS",
        PROJECT_ROOT / "CORTEX", 
        PROJECT_ROOT / "MEMORY",
        PROJECT_ROOT / "SKILLS",
    ]
    
    escaped = find_escaped_artifacts(scan_dirs)
    
    result = {
        **payload,
        "escaped_artifacts": [str(p.relative_to(PROJECT_ROOT)) for p in escaped],
        "escape_check_passed": len(escaped) == 0,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    
    if escaped:
        print(f"Found {len(escaped)} escaped artifact(s):")
        for p in escaped:
            print(f"  - {p.relative_to(PROJECT_ROOT)}")
        return 1
    
    print("Artifact escape hatch check passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
