"""Reindex thought cassette — scans THOUGHT/LAB for new .md files and chunks them.

Usage: python reindex_thought.py
"""
import sqlite3, hashlib, os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
THOUGHT_LAB = PROJECT_ROOT / "THOUGHT" / "LAB"
DB_PATH = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "cassettes" / "thought.db"

def chunk_markdown(content, max_chunk=2000):
    """Split markdown by headings."""
    lines = content.split("\n")
    chunks = []
    current = []
    current_header = None
    current_depth = 0

    for line in lines:
        if line.startswith("#") and len(current) > 2:
            chunks.append((current_header, current_depth, "\n".join(current)))
            current = []
            current_header = line.lstrip("#").strip()
            current_depth = len(line) - len(line.lstrip("#"))
        else:
            if not current and line.startswith("#"):
                current_header = line.lstrip("#").strip()
                current_depth = len(line) - len(line.lstrip("#"))
            current.append(line)

    if current:
        chunks.append((current_header, current_depth, "\n".join(current)))
    return chunks


def reindex():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # Get existing file paths
    existing = set()
    for row in conn.execute("SELECT path FROM files"):
        existing.add(row[0])

    # Scan for new .md files
    new_files = []
    for md in sorted(THOUGHT_LAB.glob("**/*.md")):
        rel = str(md.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if rel not in existing and "node_modules" not in rel:
            new_files.append((rel, md))

    if not new_files:
        print("No new files found.")
        conn.close()
        return

    print("Indexing {} new files...".format(len(new_files)))
    indexed = 0
    chunked = 0

    for rel, path in new_files:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue

        if not content.strip():
            continue

        # Insert file
        try:
            size = path.stat().st_size
        except:
            size = 0
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO files (path, content_hash, size_bytes, indexed_at) VALUES (?, ?, ?, datetime('now'))",
            (rel, content_hash, size))

        # Get file_id
        cur = conn.execute("SELECT file_id FROM files WHERE path = ?", (rel,))
        file_id = cur.fetchone()[0]

        # Chunk and insert
        chunks = chunk_markdown(content)
        for i, (header, depth, text) in enumerate(chunks):
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()
            conn.execute(
                "INSERT OR IGNORE INTO chunks (file_id, chunk_index, chunk_hash, token_count, start_offset, end_offset, header_text, header_depth) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (file_id, i, chunk_hash, len(text.split()), 0, 0, header, depth if header else None))

            # Get chunk_id for FTS
            cur = conn.execute("SELECT chunk_id FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
            chunk_id = cur.fetchone()[0]

            # Insert into FTS
            conn.execute(
                "INSERT OR REPLACE INTO chunks_fts(chunk_id, content) VALUES (?, ?)",
                (chunk_id, text))
            chunked += 1

        indexed += 1
        if indexed % 20 == 0:
            print("  {}/{} files...".format(indexed, len(new_files)))

    conn.commit()
    # Rebuild FTS index
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    print("Done. {} files indexed, {} chunks created.".format(indexed, chunked))


if __name__ == "__main__":
    reindex()
