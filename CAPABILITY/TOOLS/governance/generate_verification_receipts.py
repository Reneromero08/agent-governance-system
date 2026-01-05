#!/usr/bin/env python3
"""
Generate Purity Scan and Restore Proof receipts
"""
import json
import os
from datetime import datetime
from pathlib import Path
import hashlib

INBOX_ROOT = Path("INBOX")
RECEIPTS_DIR = Path("LAW/CONTRACTS/_runs")
VERSION = "1.0.0"
VERSION_HASH = hashlib.sha256(f"inbox_normalize_v{VERSION}".encode()).hexdigest()[:16]

def collect_all_files(base_path: Path, prefix: str = "") -> list:
    """Recursively collect all files with their paths."""
    files = []
    for item in sorted(base_path.iterdir()):
        rel_path = f"{prefix}/{item.name}" if prefix else item.name
        if item.is_file():
            files.append({
                "path": rel_path,
                "name": item.name,
                "size": item.stat().st_size
            })
        elif item.is_dir():
            files.extend(collect_all_files(item, rel_path))
    return files

def check_tmp_residue() -> list:
    """Check for temporary files or residue."""
    issues = []
    for item in INBOX_ROOT.rglob("*"):
        if item.is_file() and item.name.endswith((".tmp", ".bak", "~")):
            issues.append(f"Temp file found: {item.relative_to(INBOX_ROOT)}")
    return issues

def check_external_mutations() -> list:
    """Check if any non-INBOX paths were modified (simplified check)."""
    # This is a placeholder - in a real system, we'd compare against a baseline
    return []

def generate_purity_scan() -> dict:
    """Generate the purity scan receipt."""
    files = collect_all_files(INBOX_ROOT)
    
    tmp_residue = check_tmp_residue()
    
    # Check for any files outside the expected structure
    unexpected = []
    for f in files:
        path = f["path"]
        # Check if file is in expected monthly/weekly structure or excluded files
        if not any([
            path.startswith("2025-12/"),
            path.startswith("2026-01/"),
            path in ["INBOX.md", "LEDGER.yaml"],
            path.startswith("agents/Local Models/")
        ]):
            # These should be in the new structure
            if not any([
                path.startswith("2025-12/Week-"),
                path.startswith("2026-01/Week-")
            ]):
                unexpected.append(path)
    
    return {
        "operation": "PURITY_SCAN",
        "version": VERSION,
        "version_hash": VERSION_HASH,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_files": len(files),
        "tmp_residue": tmp_residue,
        "unexpected_files": unexpected,
        "status": "PASS" if len(tmp_residue) == 0 and len(unexpected) == 0 else "FAIL",
        "files_scanned": files
    }

def generate_restore_proof() -> dict:
    """Generate the restore proof receipt (reverse moves for rollback)."""
    # Read the dry run to get the move list
    dry_run_path = RECEIPTS_DIR / "INBOX_DRY_RUN.json"
    with open(dry_run_path) as f:
        dry_run = json.load(f)
    
    # Generate reverse moves
    reverse_moves = []
    for move in dry_run["moves"]:
        reverse_moves.append({
            "source": move["target"],
            "target": move["source"],
            "timestamp": move["timestamp"],
            "week": move["week"],
            "conflict_resolution": move.get("conflict_resolution", "none")
        })
    
    reverse_moves.sort(key=lambda x: (x["target"], x["source"]))
    
    return {
        "operation": "RESTORE_PROOF",
        "version": VERSION,
        "version_hash": VERSION_HASH,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "description": "Reverse moves to restore original structure if needed",
        "reverse_moves_count": len(reverse_moves),
        "reverse_moves": reverse_moves
    }

def main():
    print("=" * 60)
    print("Generating Verification Receipts")
    print("=" * 60)
    
    # Generate purity scan
    print("\n[*] Running purity scan...")
    purity = generate_purity_scan()
    purity_path = RECEIPTS_DIR / "PURITY_SCAN.json"
    with open(purity_path, 'w') as f:
        json.dump(purity, f, indent=2)
    print(f"[+] Purity scan written to: {purity_path}")
    print(f"   Status: {purity['status']}")
    print(f"   Files scanned: {purity['total_files']}")
    print(f"   Temp residue: {len(purity['tmp_residue'])} issues")
    
    # Generate restore proof
    print("\n[*] Generating restore proof...")
    restore = generate_restore_proof()
    restore_path = RECEIPTS_DIR / "RESTORE_PROOF.json"
    with open(restore_path, 'w') as f:
        json.dump(restore, f, indent=2)
    print(f"[+] Restore proof written to: {restore_path}")
    print(f"   Reverse moves: {restore['reverse_moves_count']}")
    
    return 0

if __name__ == "__main__":
    exit(main())
