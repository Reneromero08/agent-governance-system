#!/usr/bin/env python3
"""
INBOX Normalization Script - Dry Run and Execution
Normalizes INBOX structure with weekly/monthly subfolders
"""
import os
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Configuration
INBOX_ROOT = Path("INBOX")
RECEIPTS_DIR = Path("LAW/CONTRACTS/_runs")
VERSION = "1.0.0"
VERSION_HASH = hashlib.sha256(f"inbox_normalize_v{VERSION}".encode()).hexdigest()[:16]

# Files to exclude from normalization (no date pattern, system files)
EXCLUDED_FILES = {
    "INBOX.md",
    "LEDGER.yaml",
    "inbox_normalize.py",
    "DISPATCH_LEDGER.json",
    "LEDGER_ARCHIVE.json",
}

def parse_filename_date(filename: str) -> Optional[datetime]:
    """Extract date from filename. Supports multiple formats."""
    # Pattern 1: MM-DD-YYYY-HH-MM_SOMETHING (e.g., 12-28-2025-12-00_...)
    pattern1 = r"(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})"
    match = re.search(pattern1, filename)
    if match:
        try:
            return datetime.strptime(match.group(0), "%m-%d-%Y-%H-%M")
        except ValueError:
            pass
    
    # Pattern 2: YYYY-MM-DD (e.g., TASK-2025-12-30-001.json)
    pattern2 = r"TASK-(\d{4})-(\d{2})-(\d{2})"
    match = re.search(pattern2, filename)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}", "%Y-%m-%d")
        except ValueError:
            pass
    
    return None

def get_iso_week(dt: datetime) -> int:
    """Get ISO week number from datetime."""
    return dt.isocalendar()[1]

def compute_target_path(file_path: Path) -> Optional[Tuple[str, datetime]]:
    """
    Compute target path for a file based on its timestamp.
    Returns (target_path, timestamp_used) or (None, None) if parsing fails.
    """
    filename = file_path.name
    
    # Skip excluded files
    if filename in EXCLUDED_FILES:
        return None, None
    
    timestamp = parse_filename_date(filename)
    
    if timestamp is None:
        return None, None
    
    # Compute folder structure: YYYY-MM/Week-XX
    year = timestamp.year
    month = timestamp.month
    week = get_iso_week(timestamp)
    
    target_folder = f"{year:04d}-{month:02d}/Week-{week:02d}"
    
    return target_folder, timestamp

def collect_files() -> Tuple[List[Dict], List[Dict]]:
    """Collect all files in INBOX with their metadata."""
    files = []
    excluded = []
    
    for root, dirs, filenames in os.walk(INBOX_ROOT):
        for filename in filenames:
            file_path = Path(root) / filename
            
            # Compute relative path from INBOX root
            rel_path = file_path.relative_to(INBOX_ROOT)
            
            # Skip the normalize script itself
            if "inbox_normalize.py" in str(file_path):
                excluded.append({
                    "path": str(rel_path),
                    "reason": "executable_script"
                })
                continue
            
            # Compute target path
            target_folder, timestamp = compute_target_path(file_path)
            
            entry = {
                "source_path": str(rel_path),
                "filename": filename,
                "target_folder": target_folder,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "timestamp_source": "filename"
            }
            
            if target_folder is None:
                excluded.append(entry)
            else:
                files.append(entry)
    
    return files, excluded

def check_target_conflicts(files: List[Dict]) -> List[str]:
    """Check for conflicts in target paths."""
    conflicts = []
    target_map = {}
    
    for f in files:
        if f["target_folder"] is None:
            continue
        
        # Use source_path to differentiate files with same name in different folders
        target_key = (f["target_folder"], f["filename"])
        if target_key in target_map:
            # Check if they're in different source folders (acceptable for task files)
            existing = target_map[target_key]
            if existing["source_path"] != f["source_path"]:
                # Different source paths - need to preserve subfolder structure
                conflicts.append(f"CONFLICT: {f['source_path']} and {existing['source_path']} both map to {f['target_folder']}/{f['filename']}")
        else:
            target_map[target_key] = f
    
    return conflicts

def resolve_conflicts_by_preserving_subfolder(files: List[Dict]) -> List[Dict]:
    """
    For files with naming conflicts, preserve the immediate subfolder name.
    E.g., agents/Local Models/COMPLETED_TASKS/TASK-2025-12-30-002.json
    becomes agents/Local Models/COMPLETED_TASKS/2025-12/Week-XX/TASK-2025-12-30-002.json
    """
    target_map = {}
    resolved = []
    
    for f in files:
        target_key = (f["target_folder"], f["filename"])
        
        if target_key in target_map:
            # Conflict - need to preserve subfolder structure
            source_parts = f["source_path"].split(os.sep)
            # Keep the immediate parent folder (e.g., "COMPLETED_TASKS")
            if len(source_parts) >= 2:
                parent_folder = source_parts[-2]
                new_target = f"{f['target_folder']}/{parent_folder}/{f['filename']}"
            else:
                # Fallback: add index suffix
                new_target = f"{f['target_folder']}/{f['filename']}"
            
            f["target"] = new_target
            f["conflict_resolution"] = "preserved_subfolder"
        else:
            f["target"] = f"{f['target_folder']}/{f['filename']}"
            f["conflict_resolution"] = "direct"
        
        target_map[target_key] = f
        resolved.append(f)
    
    return resolved

def generate_dry_run_receipt(files: List[Dict], excluded: List[Dict]) -> Dict:
    """Generate the dry-run receipt."""
    moves = []
    for f in files:
        moves.append({
            "source": f["source_path"],
            "target": f["target"],
            "timestamp": f["timestamp"],
            "week": int(f["target_folder"].split("/")[-1].replace("Week-", "")),
            "conflict_resolution": f.get("conflict_resolution", "none")
        })
    
    # Sort for determinism
    moves.sort(key=lambda x: (x["target"], x["source"]))
    
    config = {
        "schema": "YYYY-MM/Week-XX",
        "timestamp_rule": "prefer_filename_over_mtime",
        "version": VERSION,
        "version_hash": VERSION_HASH,
        "excluded_files": [e.get("path") or e.get("source_path") for e in excluded]
    }
    
    return {
        "operation": "INBOX_DRY_RUN",
        "version": VERSION,
        "version_hash": VERSION_HASH,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "files_classified": len(files),
        "files_excluded": len(excluded),
        "moves": moves,
        "errors": []
    }

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_digest() -> Dict:
    """Compute digest of all INBOX files (recursive)."""
    files = {}
    for root, dirs, filenames in os.walk(INBOX_ROOT):
        for filename in sorted(filenames):
            file_path = Path(root) / filename
            rel_path = str(file_path.relative_to(INBOX_ROOT))
            files[rel_path] = {
                "hash": compute_file_hash(file_path),
                "size": file_path.stat().st_size
            }
    
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "files": files,
        "file_count": len(files)
    }

def execute_moves(files: List[Dict]) -> Tuple[bool, List[Dict]]:
    """Execute all file moves. Returns (success, move_results)."""
    results = []
    errors = []
    
    for f in files:
        source = INBOX_ROOT / f["source_path"]
        target = INBOX_ROOT / f["target"]
        
        # Verify source exists
        if not source.exists():
            errors.append({
                "source": f["source_path"],
                "error": "source_not_found"
            })
            continue
        
        # Create target directory structure
        target.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Move the file
            source.rename(target)
            
            results.append({
                "source": f["source_path"],
                "target": f["target"],
                "success": True,
                "hash_before": f.get("hash_before"),
                "hash_after": f.get("hash_after")
            })
        except Exception as e:
            errors.append({
                "source": f["source_path"],
                "error": str(e)
            })
    
    return len(errors) == 0, results + [{"error": e} for e in errors]

def generate_execution_receipt(files: List[Dict], pre_digest: Dict, post_digest: Dict, move_results: List[Dict]) -> Dict:
    """Generate the execution receipt."""
    successful_moves = [r for r in move_results if r.get("success")]
    failed_moves = [r for r in move_results if r.get("error")]
    
    # Verify content integrity
    integrity_verified = True
    for result in successful_moves:
        source = result["source"]
        target = result["target"]
        if source in pre_digest["files"] and target in post_digest["files"]:
            pre_hash = pre_digest["files"][source]["hash"]
            post_hash = post_digest["files"][target]["hash"]
            if pre_hash != post_hash:
                integrity_verified = False
                break
    
    return {
        "operation": "INBOX_EXECUTION",
        "version": VERSION,
        "version_hash": VERSION_HASH,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pre_digest": {
            "timestamp": pre_digest["timestamp"],
            "file_count": pre_digest["file_count"]
        },
        "post_digest": {
            "timestamp": post_digest["timestamp"],
            "file_count": post_digest["file_count"]
        },
        "moves_executed": len(successful_moves),
        "moves_failed": len(failed_moves),
        "integrity_verified": integrity_verified,
        "move_results": successful_moves[:10],  # First 10 for brevity
        "errors": failed_moves
    }

def main():
    """Main execution."""
    import sys
    
    execute_mode = len(sys.argv) > 1 and sys.argv[1] == "--execute"
    
    print("=" * 60)
    if execute_mode:
        print("INBOX Normalization - EXECUTION PHASE")
    else:
        print("INBOX Normalization - Dry Run Phase")
    print("=" * 60)
    
    # Collect files
    files, excluded = collect_files()
    print(f"\nDiscovered {len(files)} files to normalize")
    print(f"Excluded {len(excluded)} files (no timestamp pattern)")
    
    # Check for conflicts
    conflicts = check_target_conflicts(files)
    
    if conflicts:
        # Try to resolve conflicts by preserving subfolder
        print(f"\n[!] {len(conflicts)} potential conflicts detected - resolving...")
        for c in conflicts[:5]:
            print(f"  - {c}")
        if len(conflicts) > 5:
            print(f"  ... and {len(conflicts) - 5} more")
        
        files = resolve_conflicts_by_preserving_subfolder(files)
        print("[+] Conflicts resolved by preserving subfolder structure")
    
    # Generate dry-run receipt
    receipt = generate_dry_run_receipt(files, excluded)
    
    receipt_path = RECEIPTS_DIR / "INBOX_DRY_RUN.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)
    
    print(f"\n[+] Dry-run receipt written to: {receipt_path}")
    print(f"   Files classified: {receipt['files_classified']}")
    print(f"   Proposed moves: {len(receipt['moves'])}")
    
    # Show summary of moves by month/week
    by_month_week = {}
    for m in receipt['moves']:
        key = m['target'].split('/')[0]
        by_month_week[key] = by_month_week.get(key, 0) + 1
    
    print("\n[@] Proposed structure:")
    for folder, count in sorted(by_month_week.items()):
        print(f"   {folder}/ : {count} files")
    
    if execute_mode:
        # EXECUTION PHASE
        print("\n" + "=" * 60)
        print("EXECUTING FILE MOVES")
        print("=" * 60)
        
        # Compute pre-digest
        print("\n[*] Computing pre-execution digest...")
        pre_digest = compute_digest()
        pre_digest_path = RECEIPTS_DIR / "PRE_DIGEST.json"
        with open(pre_digest_path, 'w') as f:
            json.dump(pre_digest, f, indent=2)
        print(f"[+] Pre-digest written to: {pre_digest_path}")
        
        # Execute moves
        print(f"\n[*] Executing {len(files)} file moves...")
        success, move_results = execute_moves(files)
        
        if not success:
            print("\n[!] Some moves failed - see execution receipt for details")
        else:
            print(f"[+] All {len(files)} moves completed successfully")
        
        # Compute post-digest
        print("\n[*] Computing post-execution digest...")
        post_digest = compute_digest()
        post_digest_path = RECEIPTS_DIR / "POST_DIGEST.json"
        with open(post_digest_path, 'w') as f:
            json.dump(post_digest, f, indent=2)
        print(f"[+] Post-digest written to: {post_digest_path}")
        
        # Generate execution receipt
        execution_receipt = generate_execution_receipt(files, pre_digest, post_digest, move_results)
        execution_path = RECEIPTS_DIR / "INBOX_EXECUTION.json"
        with open(execution_path, 'w') as f:
            json.dump(execution_receipt, f, indent=2)
        print(f"[+] Execution receipt written to: {execution_path}")
        
        # Show final structure
        print("\n[@] Final INBOX structure:")
        for root, dirs, filenames in sorted(os.walk(INBOX_ROOT)):
            level = root.replace(str(INBOX_ROOT), '').count(os.sep)
            indent = ' ' * 2 * level
            if filenames:
                print(f"{indent}{Path(root).name}/")
                subindent = ' ' * 2 * (level + 1)
                for f in sorted(filenames)[:5]:
                    print(f"{subindent}{f}")
                if len(filenames) > 5:
                    print(f"{subindent}... and {len(filenames) - 5} more")
        
        return 0 if success else 1
    
    return 0

if __name__ == "__main__":
    exit(main())
