#!/usr/bin/env python3

"""
Build the cortex index.

This script scans the repository for Markdown files and other artifacts, extracting
basic metadata (id, type, title, tags) and writes the index to `CORTEX/_generated/cortex.db`.

The SQLite database provides O(1) lookups by ID, type, path, or tag.
"""

import os
import re
import sqlite3
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
    """Initialize the database schema."""
    schema_sql = SCHEMA_FILE.read_text()
    conn.executescript(schema_sql)
    conn.commit()


def build_index(conn: sqlite3.Connection) -> int:
    """Scan the repository and populate the database. Returns entity count."""
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM tags")
    cursor.execute("DELETE FROM entities")
    cursor.execute("DELETE FROM metadata")

    entity_count = 0
    for md_file in PROJECT_ROOT.rglob("*.md"):
        # Skip files under hidden directories and output artifacts.
        if any(part.startswith('.') for part in md_file.parts):
            continue
        if any(part in ("BUILD", "_runs", "_packs", "_generated") for part in md_file.parts):
            continue

        entity_id = f"page:{md_file.stem}"
        entity_type = "page"
        title = extract_title(md_file)
        source_path = str(md_file.relative_to(PROJECT_ROOT))

        cursor.execute(
            "INSERT OR REPLACE INTO entities (id, type, title, source_path) VALUES (?, ?, ?, ?)",
            (entity_id, entity_type, title, source_path)
        )
        entity_count += 1

    # Insert metadata
    generated_at = os.environ.get("CORTEX_BUILD_TIMESTAMP", "1970-01-01T00:00:00Z")
    canon_version = get_canon_version()
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("cortex_version", canon_version))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("canon_version", canon_version))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", ("generated_at", generated_at))

    conn.commit()
    return entity_count


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove old database if exists for clean rebuild
    if DB_FILE.exists():
        DB_FILE.unlink()

    conn = sqlite3.connect(DB_FILE)
    try:
        init_db(conn)
        count = build_index(conn)
        print(f"Cortex index written to {DB_FILE} ({count} entities)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
