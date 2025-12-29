#!/usr/bin/env python3

"""
Canon migration skill.

This skill handles migrations when breaking changes occur to the canon.
It reads a pack, detects the source version, and applies necessary
transformations to update it to the target version.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow imports from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TOOLS.agents.skill_runtime import ensure_canon_compat

# Version migration functions
MIGRATIONS: Dict[str, callable] = {}


def register_migration(from_ver: str, to_ver: str):
    """Decorator to register a migration function."""
    def decorator(func):
        MIGRATIONS[f"{from_ver}:{to_ver}"] = func
        return func
    return decorator


@register_migration("0.1.0", "0.1.5")
def migrate_0_1_0_to_0_1_5(pack_info: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Migration from 0.1.0 to 0.1.5 - no structural changes."""
    log = []
    log.append("Updating canon_version field")
    pack_info["canon_version"] = "0.1.5"
    return pack_info, log


def detect_version(pack_dir: Path) -> Optional[str]:
    """Detect the canon version of a pack."""
    pack_info_path = pack_dir / "meta" / "PACK_INFO.json"
    if not pack_info_path.exists():
        return None
    try:
        data = json.loads(pack_info_path.read_text())
        return data.get("canon_version")
    except Exception:
        return None


def find_migration_path(from_ver: str, to_ver: str) -> List[str]:
    """Find the sequence of migrations needed."""
    # For now, simple direct migration
    key = f"{from_ver}:{to_ver}"
    if key in MIGRATIONS:
        return [key]
    return []


def apply_migrations(pack_dir: Path, target_version: str) -> Tuple[bool, List[str], List[str]]:
    """
    Apply migrations to bring a pack to the target version.
    
    Returns:
        (success, log, warnings)
    """
    log: List[str] = []
    warnings: List[str] = []
    
    source_version = detect_version(pack_dir)
    if source_version is None:
        return False, ["Could not detect source version"], []
    
    if source_version == target_version:
        log.append(f"Pack is already at version {target_version}")
        return True, log, warnings
    
    migration_path = find_migration_path(source_version, target_version)
    if not migration_path:
        warnings.append(f"No migration path from {source_version} to {target_version}")
        # For minor versions within same major, often no structural changes needed
        log.append(f"Attempting direct version update from {source_version} to {target_version}")
    
    # Update pack info
    pack_info_path = pack_dir / "meta" / "PACK_INFO.json"
    pack_info = json.loads(pack_info_path.read_text())
    pack_info["canon_version"] = target_version
    pack_info["migrated_from"] = source_version
    pack_info_path.write_text(json.dumps(pack_info, indent=2, sort_keys=True) + "\n")
    
    log.append(f"Updated pack version from {source_version} to {target_version}")
    return True, log, warnings


def main(input_path: Path, output_path: Path) -> int:
    """Run the migration skill."""
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1
    try:
        payload = json.loads(input_path.read_text())
    except Exception as exc:
        print(f"Error reading input: {exc}")
        return 1
    
    pack_path = Path(payload.get("pack_path", ""))
    target_version = payload.get("target_version", "0.1.5")
    
    if not pack_path.exists():
        result = {
            **payload,
            "success": False,
            "error": f"Pack not found: {pack_path}",
        }
    else:
        success, log, warnings = apply_migrations(pack_path, target_version)
        result = {
            **payload,
            "success": success,
            "migration_log": log,
            "warnings": warnings,
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    
    # Always return 0 if we produced valid output - fixtures validate correctness
    print("Migration skill completed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        sys.exit(1)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
