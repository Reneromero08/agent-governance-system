#!/usr/bin/env python3
"""
Phase 1.2: Migration Script - Partition system1.db into bucket-aligned cassettes.

This script:
1. Reads all data from the monolithic system1.db
2. Routes each file to the appropriate bucket-based cassette
3. Creates separate cassette databases with identical schema
4. Validates: total sections before = total sections after

Bucket Routing:
    LAW/* or CANON/*     → canon.db      (immutable constitutional docs)
    CONTEXT/*            → governance.db (decisions, preferences)
    CAPABILITY/*         → capability.db (code, skills, primitives)
    NAVIGATION/*         → navigation.db (maps, cortex metadata)
    DIRECTION/*          → direction.db  (roadmaps, strategy)
    THOUGHT/*            → thought.db    (research, lab, demos)
    MEMORY/*             → memory.db     (archive, reports)
    INBOX/*              → inbox.db      (staging, temporary)
    OTHER (root files)   → navigation.db (repo-level docs like README, AGENTS)
    resident.db          → empty (for AI memories, created fresh)
"""

import sqlite3
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# Configuration
SOURCE_DB = Path(__file__).parent.parent / "db" / "system1.db"
CASSETTES_DIR = Path(__file__).parent.parent / "cassettes"
BACKUP_DIR = Path(__file__).parent.parent / "db" / "_migration_backup"

# Bucket to cassette mapping
BUCKET_ROUTING = {
    "LAW": "canon",
    "CANON": "canon",  # Legacy path support
    "CONTEXT": "governance",
    "CAPABILITY": "capability",
    "NAVIGATION": "navigation",
    "DIRECTION": "direction",
    "THOUGHT": "thought",
    "MEMORY": "memory",
    "INBOX": "inbox",
}

# Files at root go to navigation (repo-level docs)
DEFAULT_CASSETTE = "navigation"


@dataclass
class CassetteManifest:
    """Catalytic manifest for a single cassette."""
    cassette_id: str
    path: str
    files: int
    chunks: int
    file_hash: str           # SHA-256 of .db file
    content_merkle_root: str # Merkle root of sorted chunk_hashes
    chunk_hashes: List[str]  # All chunk hashes for verification


@dataclass
class MigrationReceipt:
    """Catalytic receipt for migration verification."""
    migration_id: str
    source_db: str
    source_hash: str
    timestamp_utc: str
    total_files_source: int
    total_chunks_source: int
    cassettes_created: Dict[str, Dict]
    total_files_migrated: int
    total_chunks_migrated: int
    validation_passed: bool
    errors: List[str]
    # Catalytic fields
    receipt_hash: str = ""   # SHA-256 of receipt content (computed after creation)


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file."""
    if not path.exists():
        return ""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_merkle_root(hashes: List[str]) -> str:
    """Compute Merkle root from sorted list of hashes.

    Uses a simple binary Merkle tree. Empty list returns empty string.
    Single hash returns itself. Odd count duplicates last hash.
    """
    if not hashes:
        return ""

    # Sort for determinism
    sorted_hashes = sorted(hashes)

    # Convert to bytes
    nodes = [bytes.fromhex(h) if len(h) == 64 else h.encode() for h in sorted_hashes]

    # Build tree bottom-up
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])  # Duplicate last for odd count

        next_level = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i] + nodes[i + 1]
            next_level.append(hashlib.sha256(combined).digest())
        nodes = next_level

    return nodes[0].hex()


def compute_receipt_hash(receipt: MigrationReceipt) -> str:
    """Compute content-addressed hash of receipt (excluding receipt_hash field)."""
    # Create a copy without receipt_hash for deterministic hashing
    receipt_dict = asdict(receipt)
    receipt_dict.pop("receipt_hash", None)

    # Canonical JSON (sorted keys, no whitespace variance)
    canonical = json.dumps(receipt_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_bucket_from_path(file_path: str) -> str:
    """Determine which bucket a file belongs to based on its path."""
    # Normalize path separators
    normalized = file_path.replace("\\", "/")

    # Check each bucket prefix
    for prefix, cassette in BUCKET_ROUTING.items():
        if normalized.startswith(f"{prefix}/"):
            return cassette

    # Root-level files go to navigation
    return DEFAULT_CASSETTE


def create_cassette_schema(conn: sqlite3.Connection):
    """Create the standard cassette schema (matches system1.db)."""
    conn.executescript("""
        -- Files table: tracks all indexed files
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Chunks table: stores text chunks with metadata
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_hash TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            FOREIGN KEY (file_id) REFERENCES files(file_id),
            UNIQUE(file_id, chunk_index)
        );

        -- FTS5 virtual table for full-text search
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            chunk_id UNINDEXED,
            tokenize='porter unicode61'
        );

        -- Index for fast lookups
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);

        -- Cassette metadata
        CREATE TABLE IF NOT EXISTS cassette_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()


def migrate_file_to_cassette(
    source_conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    file_row: sqlite3.Row
) -> Tuple[int, int]:
    """
    Migrate a single file and its chunks to the target cassette.

    Returns: (files_migrated, chunks_migrated)
    """
    # Insert file record
    cursor = target_conn.execute("""
        INSERT INTO files (path, content_hash, size_bytes, indexed_at)
        VALUES (?, ?, ?, ?)
    """, (file_row['path'], file_row['content_hash'],
          file_row['size_bytes'], file_row['indexed_at']))
    new_file_id = cursor.lastrowid

    # Get chunks for this file
    chunk_cursor = source_conn.execute("""
        SELECT c.*, fts.content as fts_content
        FROM chunks c
        LEFT JOIN chunks_fts fts ON c.chunk_id = fts.chunk_id
        WHERE c.file_id = ?
        ORDER BY c.chunk_index
    """, (file_row['file_id'],))

    chunks_migrated = 0
    for chunk_row in chunk_cursor.fetchall():
        # Insert chunk
        cursor = target_conn.execute("""
            INSERT INTO chunks (file_id, chunk_index, chunk_hash, token_count,
                               start_offset, end_offset)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (new_file_id, chunk_row['chunk_index'], chunk_row['chunk_hash'],
              chunk_row['token_count'], chunk_row['start_offset'],
              chunk_row['end_offset']))
        new_chunk_id = cursor.lastrowid

        # Insert FTS content if available
        if chunk_row['fts_content']:
            target_conn.execute("""
                INSERT INTO chunks_fts (rowid, content, chunk_id)
                VALUES (?, ?, ?)
            """, (new_chunk_id, chunk_row['fts_content'], new_chunk_id))

        chunks_migrated += 1

    return 1, chunks_migrated


def run_migration(dry_run: bool = False) -> MigrationReceipt:
    """
    Execute the migration from system1.db to partitioned cassettes.

    Args:
        dry_run: If True, analyze but don't write files

    Returns:
        MigrationReceipt with results
    """
    errors = []
    migration_id = hashlib.sha256(
        f"migration-{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]

    print(f"=== Cassette Migration {migration_id} ===")
    print(f"Source: {SOURCE_DB}")
    print(f"Target: {CASSETTES_DIR}/")
    print(f"Dry run: {dry_run}")
    print()

    # Verify source exists
    if not SOURCE_DB.exists():
        errors.append(f"Source database not found: {SOURCE_DB}")
        return MigrationReceipt(
            migration_id=migration_id,
            source_db=str(SOURCE_DB),
            source_hash="",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            total_files_source=0,
            total_chunks_source=0,
            cassettes_created={},
            total_files_migrated=0,
            total_chunks_migrated=0,
            validation_passed=False,
            errors=errors
        )

    source_hash = compute_file_hash(SOURCE_DB)

    # Connect to source
    source_conn = sqlite3.connect(str(SOURCE_DB))
    source_conn.row_factory = sqlite3.Row

    # Get source counts
    cursor = source_conn.execute("SELECT COUNT(*) FROM files")
    total_files_source = cursor.fetchone()[0]
    cursor = source_conn.execute("SELECT COUNT(*) FROM chunks")
    total_chunks_source = cursor.fetchone()[0]

    print(f"Source contains: {total_files_source} files, {total_chunks_source} chunks")

    # Analyze distribution
    cursor = source_conn.execute("SELECT * FROM files")
    file_routing: Dict[str, List[sqlite3.Row]] = {}

    for file_row in cursor.fetchall():
        cassette = get_bucket_from_path(file_row['path'])
        if cassette not in file_routing:
            file_routing[cassette] = []
        file_routing[cassette].append(file_row)

    print("\nDistribution by cassette:")
    for cassette, files in sorted(file_routing.items()):
        print(f"  {cassette}: {len(files)} files")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        source_conn.close()
        return MigrationReceipt(
            migration_id=migration_id,
            source_db=str(SOURCE_DB),
            source_hash=source_hash,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            total_files_source=total_files_source,
            total_chunks_source=total_chunks_source,
            cassettes_created={cassette: {"files": len(files)}
                              for cassette, files in file_routing.items()},
            total_files_migrated=0,
            total_chunks_migrated=0,
            validation_passed=True,  # Dry run passes
            errors=errors
        )

    # Create backup
    print(f"\nCreating backup at {BACKUP_DIR}/")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"system1.db.{migration_id}"
    shutil.copy2(SOURCE_DB, backup_path)
    print(f"  Backed up to: {backup_path}")

    # Create cassettes directory
    CASSETTES_DIR.mkdir(parents=True, exist_ok=True)

    # Migrate each cassette
    cassettes_created = {}
    total_files_migrated = 0
    total_chunks_migrated = 0

    # All cassettes to create (including empty ones)
    all_cassettes = ["canon", "governance", "capability", "navigation",
                     "direction", "thought", "memory", "inbox", "resident"]

    for cassette_name in all_cassettes:
        cassette_path = CASSETTES_DIR / f"{cassette_name}.db"
        print(f"\nCreating cassette: {cassette_name}.db")

        # Remove existing if present
        if cassette_path.exists():
            cassette_path.unlink()

        # Create new cassette
        target_conn = sqlite3.connect(str(cassette_path))
        target_conn.row_factory = sqlite3.Row
        create_cassette_schema(target_conn)

        # Add metadata
        target_conn.execute("""
            INSERT INTO cassette_metadata (key, value) VALUES
            ('cassette_id', ?),
            ('created_at', ?),
            ('migration_id', ?),
            ('source_hash', ?)
        """, (cassette_name, datetime.now(timezone.utc).isoformat(),
              migration_id, source_hash))

        # Migrate files for this cassette
        files_in_cassette = file_routing.get(cassette_name, [])
        cassette_files = 0
        cassette_chunks = 0

        for file_row in files_in_cassette:
            try:
                f, c = migrate_file_to_cassette(source_conn, target_conn, file_row)
                cassette_files += f
                cassette_chunks += c
            except Exception as e:
                errors.append(f"Error migrating {file_row['path']}: {e}")

        target_conn.commit()

        # Get final stats
        cursor = target_conn.execute("SELECT COUNT(*) FROM files")
        final_files = cursor.fetchone()[0]
        cursor = target_conn.execute("SELECT COUNT(*) FROM chunks")
        final_chunks = cursor.fetchone()[0]

        # Collect chunk hashes for Merkle root (catalytic)
        cursor = target_conn.execute("SELECT chunk_hash FROM chunks ORDER BY chunk_hash")
        chunk_hashes = [row[0] for row in cursor.fetchall()]

        # Compute content Merkle root (catalytic)
        merkle_root = compute_merkle_root(chunk_hashes)

        # Store Merkle root in cassette metadata BEFORE closing
        if merkle_root:
            target_conn.execute(
                "INSERT OR REPLACE INTO cassette_metadata (key, value) VALUES ('merkle_root', ?)",
                (merkle_root,)
            )
            target_conn.commit()

        target_conn.close()

        # Compute file hash AFTER all writes complete
        file_hash = compute_file_hash(cassette_path)

        cassettes_created[cassette_name] = {
            "path": str(cassette_path),
            "files": final_files,
            "chunks": final_chunks,
            "file_hash": file_hash,
            "merkle_root": merkle_root,
            "chunk_hashes": chunk_hashes  # Full list for verification
        }

        total_files_migrated += cassette_files
        total_chunks_migrated += cassette_chunks

        print(f"  Migrated: {final_files} files, {final_chunks} chunks")
        if merkle_root:
            print(f"  Merkle root: {merkle_root[:16]}...")

    source_conn.close()

    # Validation
    validation_passed = (
        total_files_migrated == total_files_source and
        total_chunks_migrated == total_chunks_source and
        len(errors) == 0
    )

    print(f"\n=== Migration Summary ===")
    print(f"Files: {total_files_migrated}/{total_files_source}")
    print(f"Chunks: {total_chunks_migrated}/{total_chunks_source}")
    print(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:5]:
            print(f"  - {e}")

    receipt = MigrationReceipt(
        migration_id=migration_id,
        source_db=str(SOURCE_DB),
        source_hash=source_hash,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        total_files_source=total_files_source,
        total_chunks_source=total_chunks_source,
        cassettes_created=cassettes_created,
        total_files_migrated=total_files_migrated,
        total_chunks_migrated=total_chunks_migrated,
        validation_passed=validation_passed,
        errors=errors
    )

    # Compute content-addressed receipt hash (catalytic)
    receipt.receipt_hash = compute_receipt_hash(receipt)
    print(f"Receipt hash: {receipt.receipt_hash[:16]}...")

    # Write receipt
    receipt_path = CASSETTES_DIR / f"migration_receipt_{migration_id}.json"
    with open(receipt_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(receipt), f, indent=2)
    print(f"Receipt written: {receipt_path}")

    return receipt


def verify_migration(receipt_path: Path) -> Dict:
    """Verify a migration using its receipt.

    Checks:
    1. Receipt hash matches content
    2. Each cassette file hash matches
    3. Each cassette Merkle root matches computed value
    4. Chunk counts match

    Returns:
        {
            "passed": bool,
            "receipt_hash_valid": bool,
            "cassettes": {cassette_id: {"passed": bool, "errors": [...]}},
            "errors": [...]
        }
    """
    result = {
        "passed": True,
        "receipt_hash_valid": False,
        "cassettes": {},
        "errors": []
    }

    # Load receipt
    if not receipt_path.exists():
        result["passed"] = False
        result["errors"].append(f"Receipt not found: {receipt_path}")
        return result

    with open(receipt_path, 'r', encoding='utf-8') as f:
        receipt_data = json.load(f)

    # Verify receipt hash
    stored_hash = receipt_data.get("receipt_hash", "")
    receipt_data_copy = dict(receipt_data)
    receipt_data_copy.pop("receipt_hash", None)
    canonical = json.dumps(receipt_data_copy, sort_keys=True, separators=(',', ':'))
    computed_hash = hashlib.sha256(canonical.encode()).hexdigest()

    if stored_hash == computed_hash:
        result["receipt_hash_valid"] = True
    else:
        result["passed"] = False
        result["errors"].append(f"Receipt hash mismatch: stored={stored_hash[:16]}... computed={computed_hash[:16]}...")

    # Verify each cassette
    for cassette_id, manifest in receipt_data.get("cassettes_created", {}).items():
        cassette_result = {"passed": True, "errors": []}
        cassette_path = Path(manifest["path"])

        if not cassette_path.exists():
            cassette_result["passed"] = False
            cassette_result["errors"].append("File not found")
        else:
            # Verify file hash
            current_hash = compute_file_hash(cassette_path)
            if current_hash != manifest.get("file_hash"):
                cassette_result["passed"] = False
                cassette_result["errors"].append(f"File hash mismatch")

            # Verify Merkle root
            conn = sqlite3.connect(str(cassette_path))
            cursor = conn.execute("SELECT chunk_hash FROM chunks ORDER BY chunk_hash")
            current_hashes = [row[0] for row in cursor.fetchall()]
            conn.close()

            current_merkle = compute_merkle_root(current_hashes)
            if current_merkle != manifest.get("merkle_root"):
                cassette_result["passed"] = False
                cassette_result["errors"].append(f"Merkle root mismatch")

            # Verify chunk count
            if len(current_hashes) != manifest.get("chunks", 0):
                cassette_result["passed"] = False
                cassette_result["errors"].append(f"Chunk count mismatch: {len(current_hashes)} vs {manifest.get('chunks')}")

        result["cassettes"][cassette_id] = cassette_result
        if not cassette_result["passed"]:
            result["passed"] = False

    return result


def update_cassettes_json():
    """Update cassettes.json with the new partitioned cassettes."""
    config_path = Path(__file__).parent / "cassettes.json"

    new_config = {
        "config_version": "3.0",
        "description": "Phase 1: Bucket-aligned cassette partitions",
        "cassettes": [
            {
                "id": "canon",
                "name": "Canon (LAW)",
                "db_path": "NAVIGATION/CORTEX/cassettes/canon.db",
                "enabled": True,
                "description": "Constitutional documents - immutable (LAW/)",
                "capabilities": ["fts", "semantic_search"],
                "type": "generic",
                "mutability": "immutable"
            },
            {
                "id": "governance",
                "name": "Governance (CONTEXT)",
                "db_path": "NAVIGATION/CORTEX/cassettes/governance.db",
                "enabled": True,
                "description": "Decisions and preferences (CONTEXT/)",
                "capabilities": ["fts", "semantic_search"],
                "type": "generic",
                "mutability": "stable"
            },
            {
                "id": "capability",
                "name": "Capability",
                "db_path": "NAVIGATION/CORTEX/cassettes/capability.db",
                "enabled": True,
                "description": "Code, skills, primitives (CAPABILITY/)",
                "capabilities": ["fts", "semantic_search", "ast"],
                "type": "generic",
                "mutability": "mutable"
            },
            {
                "id": "navigation",
                "name": "Navigation",
                "db_path": "NAVIGATION/CORTEX/cassettes/navigation.db",
                "enabled": True,
                "description": "Maps, cortex metadata (NAVIGATION/)",
                "capabilities": ["fts", "semantic_search"],
                "type": "generic",
                "mutability": "mutable"
            },
            {
                "id": "direction",
                "name": "Direction",
                "db_path": "NAVIGATION/CORTEX/cassettes/direction.db",
                "enabled": True,
                "description": "Roadmaps, strategy (DIRECTION/)",
                "capabilities": ["fts", "semantic_search"],
                "type": "generic",
                "mutability": "mutable"
            },
            {
                "id": "thought",
                "name": "Thought",
                "db_path": "NAVIGATION/CORTEX/cassettes/thought.db",
                "enabled": True,
                "description": "Research, lab, demos (THOUGHT/)",
                "capabilities": ["fts", "semantic_search", "research"],
                "type": "generic",
                "mutability": "mutable"
            },
            {
                "id": "memory",
                "name": "Memory",
                "db_path": "NAVIGATION/CORTEX/cassettes/memory.db",
                "enabled": True,
                "description": "Archive, reports (MEMORY/)",
                "capabilities": ["fts", "semantic_search"],
                "type": "generic",
                "mutability": "append_only"
            },
            {
                "id": "inbox",
                "name": "Inbox",
                "db_path": "NAVIGATION/CORTEX/cassettes/inbox.db",
                "enabled": True,
                "description": "Staging, temporary (INBOX/)",
                "capabilities": ["fts", "semantic_search"],
                "type": "generic",
                "mutability": "ephemeral"
            },
            {
                "id": "resident",
                "name": "Resident (AI Memories)",
                "db_path": "NAVIGATION/CORTEX/cassettes/resident.db",
                "enabled": True,
                "description": "Per-agent AI memories",
                "capabilities": ["fts", "semantic_search", "agent_memory"],
                "type": "generic",
                "mutability": "read_write"
            }
        ],
        "legacy_cassettes": [
            {
                "id": "system1",
                "name": "Legacy System1 (deprecated)",
                "db_path": "NAVIGATION/CORTEX/db/system1.db",
                "enabled": False,
                "description": "Original monolithic database - kept for reference",
                "type": "generic"
            }
        ],
        "auto_discover": False,
        "default_type": "generic"
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=2)

    print(f"Updated: {config_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        receipt = run_migration(dry_run=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "--migrate":
        receipt = run_migration(dry_run=False)
        if receipt.validation_passed:
            print("\nUpdating cassettes.json...")
            update_cassettes_json()
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Find most recent receipt
        receipts = sorted(CASSETTES_DIR.glob("migration_receipt_*.json"))
        if not receipts:
            print("No migration receipts found")
            sys.exit(1)

        receipt_path = receipts[-1]
        if len(sys.argv) > 2:
            receipt_path = Path(sys.argv[2])

        print(f"=== Verifying Migration ===")
        print(f"Receipt: {receipt_path}")
        print()

        result = verify_migration(receipt_path)

        print(f"Receipt hash valid: {result['receipt_hash_valid']}")
        print()

        for cassette_id, cassette_result in result['cassettes'].items():
            status = "PASS" if cassette_result['passed'] else "FAIL"
            errors = ", ".join(cassette_result['errors']) if cassette_result['errors'] else ""
            print(f"  {cassette_id}: {status} {errors}")

        print()
        if result['passed']:
            print("VERIFICATION PASSED - Migration is catalytic")
        else:
            print("VERIFICATION FAILED")
            for e in result['errors']:
                print(f"  - {e}")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python migrate_to_cassettes.py --dry-run           # Analyze only")
        print("  python migrate_to_cassettes.py --migrate           # Execute migration")
        print("  python migrate_to_cassettes.py --verify [receipt]  # Verify migration")
