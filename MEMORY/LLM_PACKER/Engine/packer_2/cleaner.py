#!/usr/bin/env python3
"""
Packer 2 -- Non-destructive codebase cleaner for CAT_CAS.

Copies source, strips to .py .md .rs .toml, removes empty dirs, zips.
Outputs to MEMORY/LLM_PACKER/_packs/.
"""

import argparse
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_EXTENSIONS = {".py", ".md", ".rs", ".toml"}

EXCLUDED_DIR_NAMES = {"__pycache__", ".git", "node_modules"}

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = REPO_ROOT / "THOUGHT" / "LAB" / "CAT_CAS"
DEFAULT_OUT_DIR = REPO_ROOT / "MEMORY" / "LLM_PACKER" / "_packs"
ARCHIVE_DIR = REPO_ROOT / "MEMORY" / "LLM_PACKER" / "_packs" / "_archive"


def strip_copy(source_dir: Path, dest_dir: Path, dry_run: bool = False) -> tuple:
    """Copy allowed files from source to dest. Returns (copied, skipped)."""
    if not dry_run:
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for root, dirs, files in os.walk(str(source_dir)):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]

        rel_root = Path(root).relative_to(source_dir)
        dest_root = dest_dir / rel_root

        for f in files:
            src_path = Path(root) / f
            ext = src_path.suffix.lower()

            if ext not in ALLOWED_EXTENSIONS:
                skipped += 1
                continue

            copied += 1
            if not dry_run:
                dest_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_path), str(dest_root / f))

    return copied, skipped


def remove_empty_dirs(target_dir: Path, dry_run: bool = False) -> int:
    """Remove empty directories bottom-up."""
    removed = 0
    for root, dirs, files in os.walk(str(target_dir), topdown=False):
        for d in dirs:
            dp = Path(root) / d
            try:
                if not any(dp.iterdir()):
                    if not dry_run:
                        dp.rmdir()
                    removed += 1
            except OSError:
                pass
    return removed


def make_zip(source_dir: Path, zip_path: Path) -> Path:
    """Create a zip archive of source_dir."""
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent).as_posix()
                zf.write(str(file_path), arcname)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Packer 2 -- Standalone CAT_CAS codebase cleaner")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source directory")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--name", default="", help="Output folder name (default: CAT_CAS_clean_<stamp>)")
    parser.add_argument("--no-zip", action="store_true", help="Skip zip creation")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists() or not source.is_dir():
        print(f"Error: source directory not found: {source}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    name = args.name or f"CAT_CAS_clean_{stamp}"

    clean_dir = out_dir / name
    zip_path = ARCHIVE_DIR / f"{name}.zip"

    print(f"Source:     {source}")
    print(f"Output:     {clean_dir}")
    print(f"Extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    if args.dry_run:
        copied, skipped = strip_copy(source, clean_dir, dry_run=True)
        print(f"\n[Dry run] Would copy {copied} files, skip {skipped} non-source files")
        print(f"[Dry run] Output dir:  {clean_dir}")
        if not args.no_zip:
            print(f"[Dry run] Zip:        {zip_path}")
        return 0

    copied, skipped = strip_copy(source, clean_dir)
    print(f"Copied:  {copied} files")
    print(f"Skipped: {skipped} non-source files (.bin, .json, .png, .csv, .exe, etc.)")

    removed = remove_empty_dirs(clean_dir)
    print(f"Removed: {removed} empty directories")

    if not args.no_zip:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        make_zip(clean_dir, zip_path)
        zip_size = zip_path.stat().st_size
        print(f"Zip:     {zip_path} ({zip_size:,} bytes)")

    total = sum(1 for _ in clean_dir.rglob("*") if _.is_file())
    print(f"\nDone. {total} source files in {clean_dir}")
    return 0
