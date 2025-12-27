"""
Archive generation (Phase 1).

Output structure:
- pack_dir/archive/pack.zip (contains ONLY meta/ and repo/)
- pack_dir/archive/{SCOPE}-FULL.txt (scope-prefixed siblings)
- pack_dir/archive/{SCOPE}-SPLIT-INDEX.txt
- etc.

FORBIDDEN: 
- Including FULL/, SPLIT/, LITE/ inside the zip
- Any non-scope-prefixed filenames in archive/
- Any reference to COMBINED/, FULL_COMBINED/, SPLIT_LITE/
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import List, Sequence

from .core import PackScope, SCOPE_AGS, read_text

def _iter_files_under(base: Path) -> List[Path]:
    if not base.exists():
        return []
    paths: List[Path] = []
    for p in base.rglob("*"):
        if p.is_file():
            paths.append(p)
    return sorted(paths, key=lambda p: p.as_posix())

def _write_zip(zip_path: Path, *, pack_dir: Path, roots: Sequence[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temp file first to avoid locking if process failed previously
    temp_zip = zip_path.parent / f"{zip_path.name}.tmp"
    if temp_zip.exists():
        temp_zip.unlink()
    
    if zip_path.exists():
        try:
            zip_path.unlink()
        except OSError:
            # If we can't delete it, it might be locked. 
            # We will try to overwrite it via move, or fail loudly if we can't.
            pass

    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root in roots:
            for file_path in _iter_files_under(root):
                # Always root at pack root relative path (e.g. repo/foo.txt)
                arcname = file_path.relative_to(pack_dir).as_posix()
                zf.write(file_path, arcname)
    
    # Atomic-ish move
    shutil.move(str(temp_zip), str(zip_path))

def write_pack_internal_archives(
    pack_dir: Path,
    *,
    scope: PackScope,
    system_archive_dir: Path,
) -> None:
    """
    Write per-pack archive zips under `<pack>/archive/` and generate sibling text files.
    """
    internal_archive_dir = pack_dir / "archive"
    internal_archive_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create canonical pack.zip (meta/ + repo/ only)
    pack_zip = internal_archive_dir / "pack.zip"
    _write_zip(pack_zip, pack_dir=pack_dir, roots=[pack_dir / "repo", pack_dir / "meta"])

    # 2. Copy to system archive
    system_archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pack_zip, system_archive_dir / f"{pack_dir.name}.zip")

    # 3. Generate sibling text files from FULL/ outputs
    # Must use scope prefix
    full_dir = pack_dir / "FULL"
    if full_dir.exists():
        for p in sorted(full_dir.glob("*")):
            if not p.is_file():
                continue
            
            # If filename already has scope prefix, keep it. 
            # If not, strictly enforce it.
            name = p.name
            if not name.startswith(f"{scope.file_prefix}-"):
                name = f"{scope.file_prefix}-{name}"
            
            # Ensure .txt extension for archive
            if name.lower().endswith(".md"):
                name = str(Path(name).with_suffix(".txt"))
            
            dest = internal_archive_dir / name
            shutil.copy2(p, dest)

    # 4. Generate sibling text files from SPLIT/ outputs
    split_dir = pack_dir / "SPLIT"
    if split_dir.exists():
        for p in sorted(split_dir.glob("*.md")):
            # Convert .md to .txt for archive sibling
            txt_name = p.stem + ".txt"
            
            # Ensure scope prefix
            # Ensure scope prefix and inject SPLIT
            stem = p.stem
            if stem.startswith(f"{scope.file_prefix}-"):
                if "SPLIT" not in stem:
                    # Inject SPLIT likely after prefix
                    rest = stem[len(scope.file_prefix)+1:] # skip prefix and dash
                    txt_name = f"{scope.file_prefix}-SPLIT-{rest}.txt"
                else:
                    txt_name = f"{stem}.txt"
            else:
                 txt_name = f"{scope.file_prefix}-SPLIT-{stem}.txt"
                
            dest = internal_archive_dir / txt_name
            dest.write_text(read_text(p), encoding="utf-8")

