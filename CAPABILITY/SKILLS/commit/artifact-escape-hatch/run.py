#!/usr/bin/env python3

"""
Artifact escape hatch validator.

This script checks that no files have been created outside the allowed
artifact output directories. It is run as a governance fixture.

Allowed output roots (per INV-006):
- CONTRACTS/_runs/
- CORTEX/_generated/
- MEMORY/LLM_PACKER/_packs/
- BUILD/ (user outputs only)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

PACKS_DIR = PROJECT_ROOT / "MEMORY" / "LLM_PACKER" / "_packs"

# Directories that may contain generated artifacts
ALLOWED_ARTIFACT_ROOTS = {
    PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs",
    PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "_generated",
    PACKS_DIR,
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
    "CAPABILITY",
    "LAW",
    "NAVIGATION",
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


def find_escaped_artifacts(scan_roots: List[Path]) -> List[Path]:
    """Find any generated runtime files outside allowed roots."""
    escaped = []
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                is_file = path.is_file()
            except OSError:
                is_file = False
            if is_file and not should_ignore(path):
                # Only flag files that look like runtime artifacts
                if is_runtime_artifact(path) and not is_allowed_path(path):
                    escaped.append(path)
    return escaped


def _git_untracked_files(project_root: Path) -> List[Path]:
    """
    Return untracked file paths (absolute) using git, or an empty list if git is unavailable.

    This is dramatically faster than scanning the entire repo tree, and matches the intent
    of the escape hatch: catching newly created runtime artifacts.
    """
    try:
        res = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(project_root),
            capture_output=True,
        )
        if res.returncode != 0:
            return []
        out: List[Path] = []
        for raw in res.stdout.split(b"\0"):
            if not raw:
                continue
            rel = os.fsdecode(raw)
            out.append((project_root / rel).resolve())
        return out
    except Exception:
        return []


def find_escaped_artifacts_fast(project_root: Path) -> List[Path]:
    """
    Fast-path escape hatch check.

    Only consider untracked files and flag those that look like runtime artifacts outside
    allowed artifact roots.
    """
    escaped: List[Path] = []
    candidates = _git_untracked_files(project_root)
    if not candidates:
        # Fallback: slow scan (git may be unavailable in some environments).
        scan_dirs = [
            project_root / "LAW",
            project_root / "NAVIGATION",
            project_root / "MEMORY",
            project_root / "CAPABILITY",
        ]
        return find_escaped_artifacts(scan_dirs)

    for path in candidates:
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if not is_file:
            continue
        if should_ignore(path):
            continue
        if is_runtime_artifact(path) and not is_allowed_path(path):
            escaped.append(path)
    return escaped


def main(input_path: Path, output_path: Path) -> int:
    """Run the artifact escape hatch check."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    escaped = find_escaped_artifacts_fast(PROJECT_ROOT)
    
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
