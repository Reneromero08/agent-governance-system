#!/usr/bin/env python3

"""
Pack validation skill.

Validates that a pack is complete, correctly structured, and navigable.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MEMORY.LLM_PACKER.Engine.packer import verify_manifest
from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat


def validate_structure(pack_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate pack structure."""
    errors = []
    warnings = []
    
    # Check required directories
    if not (pack_dir / "meta").exists():
        errors.append("Missing meta/ directory")
    if not (pack_dir / "repo").exists():
        errors.append("Missing repo/ directory")
    
    # Check required meta files
    required_meta = [
        "PACK_INFO.json",
        "REPO_STATE.json",
        "FILE_INDEX.json",
        "FILE_TREE.txt",
        "START_HERE.md",
        "ENTRYPOINTS.md",
        "BUILD_TREE.txt",
        "PROVENANCE.json",
    ]
    for f in required_meta:
        if not (pack_dir / "meta" / f).exists():
            errors.append(f"Missing meta/{f}")
    
    return errors, warnings


def validate_navigation(pack_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate pack is navigable."""
    errors = []
    warnings = []
    
    # Check for entry points
    if not (pack_dir / "meta" / "START_HERE.md").exists():
        if not (pack_dir / "meta" / "ENTRYPOINTS.md").exists():
            warnings.append("No START_HERE.md or ENTRYPOINTS.md found")
    
    # Check split files
    split_dir = pack_dir / "SPLIT"
    if split_dir.exists():
        index_files = sorted(p.name for p in split_dir.glob("*-00_INDEX.md") if p.is_file())
        if not index_files:
            warnings.append("SPLIT/ exists but no *-00_INDEX.md found")
        if (split_dir / "AGS-00_INDEX.md").exists():
            expected = ["AGS-01_LAW.md", "AGS-02_CAPABILITY.md", "AGS-03_NAVIGATION.md"]
            for name in expected:
                if not (split_dir / name).exists():
                    warnings.append(f"Missing expected AGS split file: {name}")
    
    return errors, warnings


def get_stats(pack_dir: Path) -> Dict[str, Any]:
    """Get pack statistics."""
    stats = {
        "files": 0,
        "bytes": 0,
    }
    
    for path in pack_dir.rglob("*"):
        if path.is_file():
            stats["files"] += 1
            stats["bytes"] += path.stat().st_size
    
    return stats


def main(input_path: Path, output_path: Path) -> int:
    """Run the validation skill."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input: {exc}")
        return 1
    
    pack_path_str = payload.get("pack_path", "")
    
    if not pack_path_str:
        result = {
            **payload,
            "valid": False,
            "errors": [f"Pack not found: "],
            "warnings": [],
            "stats": {},
        }
    elif not Path(pack_path_str).exists():
        result = {
            **payload,
            "valid": False,
            "errors": [f"Pack not found: {pack_path_str}"],
            "warnings": [],
            "stats": {},
        }
    else:
        pack_path = Path(pack_path_str)
        all_errors = []
        all_warnings = []
        
        # Structure validation
        errors, warnings = validate_structure(pack_path)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Manifest integrity
        is_valid, manifest_errors = verify_manifest(pack_path)
        all_errors.extend(manifest_errors)
        
        # Navigation validation
        errors, warnings = validate_navigation(pack_path)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Get stats
        stats = get_stats(pack_path)
        
        result = {
            **payload,
            "valid": len(all_errors) == 0,
            "errors": all_errors,
            "warnings": all_warnings,
            "stats": stats,
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    
    print("Pack validation completed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
