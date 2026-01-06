#!/usr/bin/env python3
"""
Cortex Indexer (Lane C2)

Crawl CANON directory, parse markdown headings, and build the fast-retrieval index.
Generates:
- meta/FILE_INDEX.json
- meta/SECTION_INDEX.json
- Updates system1.db via System1DB
"""

import os
import json
import re
import hashlib
import sys
import io
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set
from NAVIGATION.CORTEX.db.system1_builder import System1DB

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None

# Fix Windows Unicode console encoding issue
if os.name == 'nt':  # Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        # For older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def safe_print(*args, **kwargs):
    """Safely print strings with Unicode characters."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace problematic characters
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode('utf-8', errors='replace').decode('utf-8'))
            else:
                safe_args.append(str(arg))
        print(*safe_args, **kwargs)

# Default Configuration
CANON_DIR = Path("CANON")
META_DIR = Path(__file__).resolve().parents[1] / "meta"
DB_PATH = Path(__file__).resolve().parents[1] / "db" / "system1.db"

class CortexIndexer:
    def __init__(self, db: System1DB, target_dir: Path = CANON_DIR, writer: Optional[GuardedWriter] = None):
        self.db = db
        self.target_dir = target_dir
        self.writer = writer
        self.file_index = {}
        self.section_index = []
        
        # Use GuardedWriter for mkdir if available, otherwise fallback
        if self.writer:
            self.writer.mkdir_tmp("NAVIGATION/CORTEX/meta")
        else:
            META_DIR.mkdir(parents=True, exist_ok=True)

    def _get_tracked_files(self, root: Path) -> Optional[Set[Path]]:
        """
        Return a set of absolute paths for git-tracked files under `root`.

        If git isn't available or the directory isn't a git repo, returns None (no filtering).
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=root,
                capture_output=True,
            )
            if result.returncode != 0:
                return None
            paths: Set[Path] = set()
            for p in result.stdout.split(b"\0"):
                if p:
                    paths.add((root / os.fsdecode(p)).resolve())
            return paths
        except Exception:
            return None

    def index_all(self):
        """Walk target directory and index all markdown files."""
        safe_print(f"[Indexer] Starting index of {self.target_dir}...")

        # Files and directories to ignore
        ignore_files = {".DS_Store"}
        ignore_dirs = {
            "node_modules", ".venv", ".git", "__pycache__",
            "_generated", "BUILD", "MEMORY", ".pytest_cache",
            "meta", "_runs"
        }

        target_abs = self.target_dir.resolve()
        cwd_abs = Path.cwd().resolve()

        tracked_files = self._get_tracked_files(cwd_abs)
        if tracked_files is not None:
            safe_print(f"[Indexer] Git filter active: {len(tracked_files)} tracked files found")

        for root, dirs, files in os.walk(target_abs):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                if file.endswith(".md") and file not in ignore_files:
                    full_path = (Path(root) / file).resolve()
                    try:
                        if tracked_files is not None and full_path not in tracked_files:
                            continue

                        # Try to make path relative to CWD if possible, else use absolute
                        try:
                            rel_path = full_path.relative_to(cwd_abs).as_posix()
                        except ValueError:
                            rel_path = full_path.as_posix()

                        self._index_file(full_path, rel_path)
                    except Exception as e:
                        safe_print(f"Skipping {file}: {e}")

        self._write_artifacts()
        safe_print("[Indexer] Indexing complete.")

    def _index_file(self, path: Path, rel_path: str):
        """Parse a markdown file into sections and index in DB."""
        content = path.read_text(encoding='utf-8')
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Update File Index
        self.file_index[rel_path] = {
            "path": rel_path,
            "hash": content_hash,
            "size": len(content),
            "sections": []
        }
        
        # Parse sections (by heading)
        sections = self._parse_sections(content)
        
        for sec in sections:
            anchor = f"{rel_path}#{sec['slug']}"
            section_data = {
                "anchor": anchor,
                "title": sec['title'],
                "level": sec['level'],
                "tokens": self.db._count_tokens(sec['content']),
                "hash": hashlib.sha256(sec['content'].encode()).hexdigest()
            }
            self.file_index[rel_path]["sections"].append(section_data)
            self.section_index.append(section_data)
            
            # Add to System 1 DB
            # We use System1DB.add_file but we can also add sections directly if we extend it.
            # For now, we index the whole file as chunks in DB, and track sections in JSON.
            
        # Add to System 1 DB (Full content for keyword search)
        self.db.add_file(rel_path, content)
        safe_print(f"[Indexer] Indexed: {rel_path} ({len(sections)} sections)")

    def _parse_sections(self, content: str) -> List[Dict]:
        """Split markdown by headings and return section metadata."""
        # Split by headings (Match #, ##, ### at start of line)
        pattern = r'^(#{1,6})\s+(.*)$'
        lines = content.splitlines()
        
        sections = []
        current_section = {
            "title": "Root",
            "slug": "root",
            "level": 0,
            "content": "",
            "start_line": 0
        }
        
        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section)
                
                level = len(match.group(1))
                title = match.group(2).strip()
                slug = self._slugify(title)
                
                current_section = {
                    "title": title,
                    "slug": slug,
                    "level": level,
                    "content": "",
                    "start_line": i + 1
                }
            else:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
            
        return sections

    def _slugify(self, text: str) -> str:
        """Convert heading to URL-safe slug."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', '', text)
        text = re.sub(r'[\s-]+', '-', text).strip('-')
        return text

    def _write_artifacts(self):
        """Write FILE_INDEX.json and SECTION_INDEX.json."""
        # Use GuardedWriter for writes if available, otherwise fallback
        if self.writer:
            self.writer.mkdir_tmp("NAVIGATION/CORTEX/meta")
            self.writer.write_tmp(
                "NAVIGATION/CORTEX/meta/FILE_INDEX.json", 
                json.dumps(self.file_index, indent=2, sort_keys=True)
            )
            self.writer.write_tmp(
                "NAVIGATION/CORTEX/meta/SECTION_INDEX.json", 
                json.dumps(self.section_index, indent=2, sort_keys=True)
            )
        else:
            META_DIR.mkdir(parents=True, exist_ok=True)
            with open(META_DIR / "FILE_INDEX.json", "w", encoding='utf-8') as f:
                json.dump(self.file_index, f, indent=2, sort_keys=True)

            with open(META_DIR / "SECTION_INDEX.json", "w", encoding='utf-8') as f:
                json.dump(self.section_index, f, indent=2, sort_keys=True)

        safe_print(f"[Indexer] Wrote artifacts to {META_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cortex Indexer CLI")
    parser.add_argument("--dir", type=str, help="Directory to index (default: CANON)")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Database path")
    parser.add_argument("--use-firewall", action="store_true", help="Use GuardedWriter for write firewall enforcement")
    args = parser.parse_args()

    db_path = Path(args.db)
    target_dir = Path(args.dir) if args.dir else CANON_DIR
    
    # Initialize GuardedWriter if requested
    writer = None
    if args.use_firewall and GuardedWriter:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        writer = GuardedWriter(project_root=repo_root)
    
    db = System1DB(db_path)
    indexer = CortexIndexer(db, target_dir=target_dir, writer=writer)
    indexer.index_all()
    db.close()
