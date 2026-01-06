#!/usr/bin/env python3

"""
Pack validation skill.

Validates that a pack is complete, correctly structured, and navigable.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
# Add repo root to path for imports
PROJECT_ROOT_GUESS = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT_GUESS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_GUESS))
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

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


def validate_pruned(pack_dir: Path) -> Tuple[List[str], List[str]]:
    """Validate PRUNED output if present."""
    errors = []
    warnings = []

    pruned_dir = pack_dir / "PRUNED"

    if not pruned_dir.exists():
        return errors, warnings

    manifest_path = pruned_dir / "PACK_MANIFEST_PRUNED.json"
    rules_path = pruned_dir / "meta" / "PRUNED_RULES.json"

    if not manifest_path.exists():
        errors.append("Missing PRUNED/PACK_MANIFEST_PRUNED.json")
        return errors, warnings

    if not rules_path.exists():
        errors.append("Missing PRUNED/meta/PRUNED_RULES.json")
        return errors, warnings

    try:
        manifest = json.loads(manifest_path.read_text())
        rules = json.loads(rules_path.read_text())
    except Exception as e:
        errors.append(f"Invalid PRUNED JSON: {e}")
        return errors, warnings

    version = manifest.get("version", "")
    if version != "PRUNED.1.0":
        errors.append(f"Invalid PRUNED manifest version: {version}")

    entries = manifest.get("entries", [])

    prev_path = None
    for entry in entries:
        path = entry.get("path")
        file_hash = entry.get("hash")
        size = entry.get("size")

        if not path:
            errors.append("PRUNED manifest entry missing 'path'")
            continue
        if not file_hash:
            errors.append(f"PRUNED manifest entry missing 'hash': {path}")
            continue
        if size is None:
            errors.append(f"PRUNED manifest entry missing 'size': {path}")
            continue

        file_path = pruned_dir / path
        if not file_path.exists():
            errors.append(f"PRUNED file missing: {path}")
            continue

        from MEMORY.LLM_PACKER.Engine.packer import hash_file
        computed_hash = hash_file(file_path)
        if computed_hash != file_hash:
            errors.append(f"PRUNED hash mismatch for {path}: expected {file_hash}, got {computed_hash}")
            continue

        actual_size = file_path.stat().st_size
        if actual_size != size:
            errors.append(f"PRUNED size mismatch for {path}: expected {size}, got {actual_size}")

        if prev_path is not None and path < prev_path:
            errors.append(f"PRUNED manifest not in canonical order: {path} after {prev_path}")
        prev_path = path

    version_rules = rules.get("version", "")
    if version_rules != "PRUNED.1.0":
        errors.append(f"Invalid PRUNED rules version: {version_rules}")

    for item in pruned_dir.iterdir():
        if item.is_dir() and item.name.startswith(".pruned_staging_"):
            errors.append(f"PRUNED staging directory not cleaned up: {item.name}")
        if item.name == "PRUNED._old":
            errors.append("PRUNED backup directory not cleaned up: PRUNED._old")

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

        # PRUNED validation (if present)
        errors, warnings = validate_pruned(pack_path)
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
    
    try:
        from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
        PROJECT_ROOT = Path(__file__).resolve().parents[4]
        writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
        
        rel_output_path = str(output_path.resolve().relative_to(PROJECT_ROOT))
        # Ensure parent exists
        writer.mkdir_tmp(str(Path(rel_output_path).parent))
        writer.write_tmp(rel_output_path, json.dumps(result, indent=2, sort_keys=True))
    except Exception as e:
        print(f"Failed to use GuardedWriter: {e}")
        # Fail closed
        return 1
    
    print("Pack validation completed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
