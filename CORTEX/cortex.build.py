#!/usr/bin/env python3

"""
Build the cortex index.

This script scans the repository for Markdown files and other artifacts, extracting
basic metadata (id, type, title, tags) and writes the index to `CORTEX/_generated/cortex.db`.

The SQLite database provides O(1) lookups by ID, type, path, or tag.

Incremental Update Strategy (v1.1):
- Retains existing database.
- Checks file modification time (mtime) against DB `last_modified`.
- Only parses and updates changed files.
- Prunes entities for deleted files.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = Path(__file__).resolve().parent
GENERATED_DIR = CORTEX_DIR / "_generated"
SCHEMA_FILE = CORTEX_DIR / "schema.sql"
DB_FILE = GENERATED_DIR / "cortex.db"
VERSIONING_PATH = PROJECT_ROOT / "CANON" / "VERSIONING.md"


def get_canon_version() -> str:
    """Read canon_version from VERSIONING.md."""
    try:
        content = VERSIONING_PATH.read_text(errors="ignore")
        match = re.search(r'canon_version:\s*(\d+\.\d+\.\d+)', content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "0.1.0"  # Fallback


def extract_title(path: Path) -> str:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return path.stem


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema and handle migrations."""
    cursor = conn.cursor()
    
    # 1. ensure tables exist
    schema_sql = SCHEMA_FILE.read_text()
    cursor.executescript(schema_sql)
    
    # 2. Migration: Add last_modified if missing (for upgrades from v1.0)
    try:
        cursor.execute("SELECT last_modified FROM entities LIMIT 1")
    except sqlite3.OperationalError:
        # Check if error is due to missing column
        try:
            cursor.execute("ALTER TABLE entities ADD COLUMN last_modified REAL")
            print("[cortex] Migrated schema: added last_modified column")
        except sqlite3.OperationalError:
            pass # Already exists or other issue

    conn.commit()


def build_index(conn: sqlite3.Connection) -> int:
    """Scan the repository and populate the database incrementally. Returns updated entity count."""
    cursor = conn.cursor()

    # 1. Load snapshot of existing state
    cursor.execute("SELECT source_path, last_modified FROM entities")
    existing_state = {row[0]: row[1] for row in cursor.fetchall()}
    
    fs_paths = set()
    updates = 0
    
    # scan for deletions and updates
    for md_file in PROJECT_ROOT.rglob("*.md"):
         # Skip files under hidden directories and output artifacts.
        if any(part.startswith('.') for part in md_file.parts):
            continue
        if any(part in ("BUILD", "_runs", "_packs", "_generated") for part in md_file.parts):
             continue

        rel_path = str(md_file.relative_to(PROJECT_ROOT))
        fs_paths.add(rel_path)
        
        current_mtime = md_file.stat().st_mtime
        
        # Check if update needed
        if rel_path in existing_state:
            cached_mtime = existing_state[rel_path]
            # If cached_mtime is None (migration) or older, update
            if cached_mtime and cached_mtime >= current_mtime:
                continue

        # Needs update
        
        # Unique ID generation to avoid collisions (e.g. README.md -> page:context_readme)
        unique_suffix = rel_path.replace(os.sep, "_").replace(".", "_")
        entity_id = f"page:{unique_suffix}"
        entity_type = "page"
        title = extract_title(md_file)
        
        # Cleanup any existing entity for this path (handles ID changes or re-insertion)
        cursor.execute("DELETE FROM entities WHERE source_path = ?", (rel_path,))

        # Insert or Replace
        cursor.execute(
            "INSERT OR REPLACE INTO entities (id, type, title, source_path, last_modified) VALUES (?, ?, ?, ?, ?)",
            (entity_id, entity_type, title, rel_path, current_mtime)
        )
        
        # Refresh tags (naive: delete all tags for this entity and re-add if we extracted them, 
        # but currently extract_title doesn't get tags. if we did, we'd do it here)
        # For now, just ensuring the entity row is up to date.
        
        updates += 1

    # Pruning: Delete entities for files that no longer exist
    to_delete = existing_state.keys() - fs_paths
    if to_delete:
        cursor.executemany("DELETE FROM entities WHERE source_path = ?", [(p,) for p in to_delete])
        print(f"[cortex] Pruned {len(to_delete)} deleted entities")

    # Update global metadata
    canon_version = get_canon_version()
    generated_at = os.environ.get("CORTEX_BUILD_TIMESTAMP", datetime.now(timezone.utc).isoformat())
    
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("cortex_version", "1.1.0"))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("canon_version", canon_version))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("generated_at", generated_at))
    
    # Provenance
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header
        prov_header = generate_header(
            generator="CORTEX/cortex.build.py",
            inputs=["CANON/", "CONTEXT/", "MAPS/", "SKILLS/", "CONTRACTS/"]
        )
        prov_json = json.dumps(prov_header)
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("provenance", prov_json))
    except ImportError:
        pass

    conn.commit()
    return updates

def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Do NOT unlink DB file - we want persistence
    
    conn = sqlite3.connect(DB_FILE)
    try:
        init_db(conn)
        count = build_index(conn)
        print(f"Cortex index updated at {DB_FILE} ({count} updates)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
