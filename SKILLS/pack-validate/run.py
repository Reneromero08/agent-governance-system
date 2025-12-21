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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MEMORY.LLM_PACKER.Engine.packer import verify_manifest


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
    required_meta = ["PACK_INFO.json", "REPO_STATE.json", "FILE_INDEX.json"]
    for f in required_meta:
        if not (pack_dir / "meta" / f).exists():
            errors.append(f"Missing meta/{f}")
    
    # Check CONTEXT.txt
    if not (pack_dir / "meta" / "CONTEXT.txt").exists():
        warnings.append("Missing meta/CONTEXT.txt - no token estimate available")
    
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
    split_dir = pack_dir / "COMBINED" / "SPLIT"
    if split_dir.exists():
        expected_splits = ["AGS-00_INDEX.md", "AGS-01_CANON.md"]
        for f in expected_splits:
            if not (split_dir / f).exists():
                warnings.append(f"Missing expected split file: {f}")
    
    return errors, warnings


def get_stats(pack_dir: Path) -> Dict[str, Any]:
    """Get pack statistics."""
    stats = {
        "files": 0,
        "bytes": 0,
        "tokens": None,
    }
    
    for path in pack_dir.rglob("*"):
        if path.is_file():
            stats["files"] += 1
            stats["bytes"] += path.stat().st_size
    
    # Try to get token count from CONTEXT.txt
    context_file = pack_dir / "meta" / "CONTEXT.txt"
    if context_file.exists():
        content = context_file.read_text(errors="ignore")
        for line in content.splitlines():
            if "Estimated tokens:" in line:
                try:
                    stats["tokens"] = int(line.split(":")[1].strip().replace(",", ""))
                except:
                    pass
    
    return stats


def main(input_path: Path, output_path: Path) -> int:
    """Run the validation skill."""
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
