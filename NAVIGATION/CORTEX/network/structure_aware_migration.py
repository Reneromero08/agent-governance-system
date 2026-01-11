#!/usr/bin/env python3
"""
Structure-Aware Migration (Phase 1.5)

Re-indexes all cassettes using structure-aware markdown chunking.
Adds header_depth, header_text, and parent_chunk_id to the chunks table.

This is a BREAKING migration - all chunks are recreated with new hashes.
New Merkle roots are computed for each cassette.

Usage:
    python structure_aware_migration.py --dry-run    # Analyze only
    python structure_aware_migration.py --migrate    # Execute migration
    python structure_aware_migration.py --verify     # Verify migration
"""

import sqlite3
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Import the new chunker
sys.path.insert(0, str(Path(__file__).parent.parent / "db"))
from markdown_chunker import chunk_markdown, MarkdownChunk

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CASSETTES_DIR = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"
BACKUP_DIR = CASSETTES_DIR / "_phase_1_5_backup"

# Cassettes to migrate (from cassettes.json)
CASSETTES = [
    "canon", "governance", "capability", "navigation",
    "direction", "thought", "memory", "inbox", "resident"
]


@dataclass
class MigrationStats:
    """Statistics for a single cassette migration."""
    cassette_id: str
    files_before: int
    chunks_before: int
    files_after: int
    chunks_after: int
    chunks_with_headers: int
    max_depth: int
    merkle_root_before: str
    merkle_root_after: str


@dataclass
class MigrationReceipt:
    """Catalytic receipt for Phase 1.5 migration."""
    migration_id: str
    phase: str
    timestamp_utc: str
    cassettes: Dict[str, Dict]
    total_chunks_before: int
    total_chunks_after: int
    schema_version: str
    receipt_hash: str = ""


def compute_merkle_root(hashes: List[str]) -> str:
    """Compute Merkle root from sorted list of hashes."""
    if not hashes:
        return ""
    sorted_hashes = sorted(hashes)
    nodes = [bytes.fromhex(h) if len(h) == 64 else h.encode() for h in sorted_hashes]
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        next_level = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i] + nodes[i + 1]
            next_level.append(hashlib.sha256(combined).digest())
        nodes = next_level
    return nodes[0].hex()


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file."""
    if not path.exists():
        return ""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_receipt_hash(receipt: MigrationReceipt) -> str:
    """Compute content-addressed hash of receipt."""
    receipt_dict = asdict(receipt)
    receipt_dict.pop("receipt_hash", None)
    canonical = json.dumps(receipt_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def migrate_schema(conn: sqlite3.Connection) -> bool:
    """Add Phase 1.5 columns to chunks table if not present.

    Returns True if migration was needed, False if already migrated.
    """
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(chunks)")
    columns = {row[1] for row in cursor.fetchall()}

    needed_columns = {"header_depth", "header_text", "parent_chunk_id"}
    if needed_columns.issubset(columns):
        return False  # Already migrated

    # Add new columns
    conn.execute("ALTER TABLE chunks ADD COLUMN header_depth INTEGER")
    conn.execute("ALTER TABLE chunks ADD COLUMN header_text TEXT")
    conn.execute("ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER")
    conn.commit()

    # Create index for hierarchy navigation
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_depth ON chunks(header_depth)")
    conn.commit()

    return True


def get_file_content(file_path: str) -> Optional[str]:
    """Read file content from filesystem."""
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        return None
    try:
        return full_path.read_text(encoding='utf-8')
    except Exception:
        return None


def reindex_cassette(cassette_path: Path, dry_run: bool = True) -> Optional[MigrationStats]:
    """Re-index a single cassette with structure-aware chunking.

    Args:
        cassette_path: Path to cassette .db file
        dry_run: If True, only analyze without modifying

    Returns:
        MigrationStats or None if cassette is empty/doesn't exist
    """
    if not cassette_path.exists():
        return None

    cassette_id = cassette_path.stem
    conn = sqlite3.connect(str(cassette_path))
    conn.row_factory = sqlite3.Row

    # Get before stats
    cursor = conn.execute("SELECT COUNT(*) FROM files")
    files_before = cursor.fetchone()[0]

    cursor = conn.execute("SELECT COUNT(*) FROM chunks")
    chunks_before = cursor.fetchone()[0]

    if files_before == 0:
        conn.close()
        return MigrationStats(
            cassette_id=cassette_id,
            files_before=0, chunks_before=0,
            files_after=0, chunks_after=0,
            chunks_with_headers=0, max_depth=0,
            merkle_root_before="", merkle_root_after=""
        )

    # Get current Merkle root
    cursor = conn.execute("SELECT chunk_hash FROM chunks ORDER BY chunk_hash")
    old_hashes = [row[0] for row in cursor.fetchall()]
    merkle_root_before = compute_merkle_root(old_hashes)

    if dry_run:
        # Just analyze - read files and chunk them without modifying DB
        cursor = conn.execute("SELECT file_id, path FROM files")
        files = cursor.fetchall()

        total_new_chunks = 0
        chunks_with_headers = 0
        max_depth = 0

        for file_row in files:
            content = get_file_content(file_row['path'])
            if content is None:
                continue

            chunks = chunk_markdown(content)
            total_new_chunks += len(chunks)

            for chunk in chunks:
                if chunk.header_depth is not None:
                    chunks_with_headers += 1
                    max_depth = max(max_depth, chunk.header_depth)

        conn.close()
        return MigrationStats(
            cassette_id=cassette_id,
            files_before=files_before,
            chunks_before=chunks_before,
            files_after=files_before,  # Files don't change
            chunks_after=total_new_chunks,
            chunks_with_headers=chunks_with_headers,
            max_depth=max_depth,
            merkle_root_before=merkle_root_before,
            merkle_root_after="(dry-run)"
        )

    # ACTUAL MIGRATION
    print(f"\n  Migrating {cassette_id}...")

    # Add schema columns
    schema_changed = migrate_schema(conn)
    if schema_changed:
        print(f"    Schema updated with new columns")

    # Get all files
    cursor = conn.execute("SELECT file_id, path, content_hash FROM files")
    files = list(cursor.fetchall())

    total_new_chunks = 0
    chunks_with_headers = 0
    max_depth = 0
    new_hashes = []

    for file_row in files:
        file_id = file_row['file_id']
        file_path = file_row['path']

        # Read content
        content = get_file_content(file_path)
        if content is None:
            print(f"    WARNING: File not found: {file_path}")
            continue

        # Delete old chunks for this file
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

        # Delete old FTS entries
        # FTS5 uses rowid, we need to find and delete the right entries
        cursor = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE file_id = ?)",
            (file_id,)
        )
        # Actually, chunks are already deleted, so FTS entries are orphaned
        # We'll rebuild FTS at the end

        # Chunk with new structure-aware chunker
        chunks = chunk_markdown(content)

        # Build chunk_id -> index mapping for parent relationships
        chunk_id_map = {}

        # Insert new chunks
        for chunk in chunks:
            cursor = conn.execute(
                """INSERT INTO chunks
                   (file_id, chunk_index, chunk_hash, token_count, start_offset, end_offset,
                    header_depth, header_text, parent_chunk_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_id,
                    chunk.chunk_index,
                    chunk.chunk_hash,
                    chunk.token_count,
                    chunk.start_line,  # Using line numbers instead of byte offsets
                    chunk.end_line,
                    chunk.header_depth,
                    chunk.header_text,
                    None  # Parent set in second pass
                )
            )
            chunk_id = cursor.lastrowid
            chunk_id_map[chunk.chunk_index] = chunk_id
            new_hashes.append(chunk.chunk_hash)

            total_new_chunks += 1
            if chunk.header_depth is not None:
                chunks_with_headers += 1
                max_depth = max(max_depth, chunk.header_depth)

        # Second pass: set parent_chunk_id
        for chunk in chunks:
            if chunk.parent_index is not None:
                chunk_id = chunk_id_map[chunk.chunk_index]
                parent_chunk_id = chunk_id_map.get(chunk.parent_index)
                if parent_chunk_id:
                    conn.execute(
                        "UPDATE chunks SET parent_chunk_id = ? WHERE chunk_id = ?",
                        (parent_chunk_id, chunk_id)
                    )

        conn.commit()

    # Rebuild FTS index
    print(f"    Rebuilding FTS index...")
    conn.execute("DELETE FROM chunks_fts")

    cursor = conn.execute("""
        SELECT c.chunk_id, f.path, c.header_text, c.start_offset, c.end_offset
        FROM chunks c
        JOIN files f ON c.file_id = f.file_id
    """)

    for row in cursor.fetchall():
        chunk_id = row[0]
        file_path = row[1]
        header_text = row[2] or ""
        start_line = row[3]
        end_line = row[4]

        # Get content from file
        content = get_file_content(file_path)
        if content:
            lines = content.split('\n')
            # Extract chunk content by line numbers
            chunk_lines = lines[start_line-1:end_line]
            chunk_content = '\n'.join(chunk_lines)

            conn.execute(
                "INSERT INTO chunks_fts (rowid, content, chunk_id) VALUES (?, ?, ?)",
                (chunk_id, chunk_content, chunk_id)
            )

    conn.commit()

    # Compute new Merkle root
    merkle_root_after = compute_merkle_root(new_hashes)

    # Store in metadata
    conn.execute(
        "INSERT OR REPLACE INTO cassette_metadata (key, value) VALUES ('merkle_root', ?)",
        (merkle_root_after,)
    )
    conn.execute(
        "INSERT OR REPLACE INTO cassette_metadata (key, value) VALUES ('schema_version', ?)",
        ("1.5",)
    )
    conn.commit()
    conn.close()

    print(f"    {chunks_before} -> {total_new_chunks} chunks")
    print(f"    {chunks_with_headers} chunks with headers (max depth: {max_depth})")
    print(f"    Merkle: {merkle_root_after[:16]}...")

    return MigrationStats(
        cassette_id=cassette_id,
        files_before=files_before,
        chunks_before=chunks_before,
        files_after=files_before,
        chunks_after=total_new_chunks,
        chunks_with_headers=chunks_with_headers,
        max_depth=max_depth,
        merkle_root_before=merkle_root_before,
        merkle_root_after=merkle_root_after
    )


def run_migration(dry_run: bool = True) -> MigrationReceipt:
    """Run Phase 1.5 migration on all cassettes."""
    migration_id = hashlib.sha256(
        f"phase_1.5_{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    print(f"=== Phase 1.5: Structure-Aware Migration ===")
    print(f"Migration ID: {migration_id}")
    print(f"Mode: {'DRY RUN' if dry_run else 'MIGRATE'}")
    print()

    if not dry_run:
        # Create backup
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Backup directory: {BACKUP_DIR}")
        for cassette_id in CASSETTES:
            src = CASSETTES_DIR / f"{cassette_id}.db"
            if src.exists():
                dst = BACKUP_DIR / f"{cassette_id}.db"
                import shutil
                shutil.copy2(src, dst)
        print("Backups created\n")

    cassettes_stats = {}
    total_before = 0
    total_after = 0

    for cassette_id in CASSETTES:
        cassette_path = CASSETTES_DIR / f"{cassette_id}.db"
        stats = reindex_cassette(cassette_path, dry_run=dry_run)

        if stats:
            cassettes_stats[cassette_id] = asdict(stats)
            total_before += stats.chunks_before
            total_after += stats.chunks_after

            if dry_run:
                print(f"{cassette_id}:")
                print(f"  {stats.files_before} files, {stats.chunks_before} -> {stats.chunks_after} chunks")
                print(f"  {stats.chunks_with_headers} with headers (max depth: {stats.max_depth})")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total chunks: {total_before} -> {total_after}")
    print(f"Change: {total_after - total_before:+d} chunks")

    receipt = MigrationReceipt(
        migration_id=migration_id,
        phase="1.5",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        cassettes=cassettes_stats,
        total_chunks_before=total_before,
        total_chunks_after=total_after,
        schema_version="1.5"
    )

    if not dry_run:
        receipt.receipt_hash = compute_receipt_hash(receipt)
        receipt_path = CASSETTES_DIR / f"migration_receipt_phase1.5_{migration_id}.json"
        with open(receipt_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(receipt), f, indent=2)
        print(f"\nReceipt: {receipt_path}")
        print(f"Receipt hash: {receipt.receipt_hash[:16]}...")

    return receipt


def verify_migration() -> Dict:
    """Verify Phase 1.5 migration."""
    print("=== Verifying Phase 1.5 Migration ===\n")

    result = {
        "passed": True,
        "cassettes": {},
        "errors": []
    }

    for cassette_id in CASSETTES:
        cassette_path = CASSETTES_DIR / f"{cassette_id}.db"
        if not cassette_path.exists():
            continue

        cassette_result = {"passed": True, "errors": []}

        conn = sqlite3.connect(str(cassette_path))
        conn.row_factory = sqlite3.Row

        # Check if this cassette has any files (empty cassettes may not be migrated)
        cursor = conn.execute("SELECT COUNT(*) FROM files")
        file_count = cursor.fetchone()[0]

        if file_count == 0:
            # Empty cassette - just check it exists
            result["cassettes"][cassette_id] = {
                "passed": True,
                "total_chunks": 0,
                "chunks_with_headers": 0,
                "note": "Empty cassette"
            }
            print(f"  {cassette_id}: SKIP (empty)")
            conn.close()
            continue

        # Check schema
        cursor = conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}

        if "header_depth" not in columns:
            cassette_result["passed"] = False
            cassette_result["errors"].append("Missing header_depth column")

        if "header_text" not in columns:
            cassette_result["passed"] = False
            cassette_result["errors"].append("Missing header_text column")

        if "parent_chunk_id" not in columns:
            cassette_result["passed"] = False
            cassette_result["errors"].append("Missing parent_chunk_id column")

        # Check for chunks with headers
        cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE header_depth IS NOT NULL")
        header_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM chunks")
        total_count = cursor.fetchone()[0]

        # Verify Merkle root
        cursor = conn.execute("SELECT chunk_hash FROM chunks ORDER BY chunk_hash")
        hashes = [row[0] for row in cursor.fetchall()]
        computed_merkle = compute_merkle_root(hashes)

        cursor = conn.execute("SELECT value FROM cassette_metadata WHERE key = 'merkle_root'")
        stored_row = cursor.fetchone()
        stored_merkle = stored_row[0] if stored_row else ""

        if computed_merkle != stored_merkle:
            cassette_result["passed"] = False
            cassette_result["errors"].append(f"Merkle root mismatch")

        cassette_result["total_chunks"] = total_count
        cassette_result["chunks_with_headers"] = header_count
        cassette_result["merkle_root"] = computed_merkle[:16] + "..."

        conn.close()

        result["cassettes"][cassette_id] = cassette_result
        if not cassette_result["passed"]:
            result["passed"] = False

        status = "PASS" if cassette_result["passed"] else "FAIL"
        print(f"  {cassette_id}: {status}")
        print(f"    {total_count} chunks, {header_count} with headers")
        if cassette_result["errors"]:
            for e in cassette_result["errors"]:
                print(f"    ERROR: {e}")

    print()
    if result["passed"]:
        print("VERIFICATION PASSED - Phase 1.5 migration complete")
    else:
        print("VERIFICATION FAILED")

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        run_migration(dry_run=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--migrate":
        run_migration(dry_run=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_migration()
    else:
        print("Usage:")
        print("  python structure_aware_migration.py --dry-run   # Analyze only")
        print("  python structure_aware_migration.py --migrate   # Execute migration")
        print("  python structure_aware_migration.py --verify    # Verify migration")
