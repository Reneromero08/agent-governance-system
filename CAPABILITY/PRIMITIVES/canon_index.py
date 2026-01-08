#!/usr/bin/env python3
"""
Canon Index Primitive - Phase 5.1.1

Indexes all governance canon files (LAW/CANON/*) with:
- MemoryRecord instances for each file
- Vector embeddings via CORTEX EmbeddingEngine
- Deterministic manifest with content hashes
- Receipted operations

Usage:
    from CAPABILITY.PRIMITIVES.canon_index import (
        inventory_canon,
        embed_canon,
        search_canon,
        rebuild_index,
    )

    # Create inventory
    manifest = inventory_canon()

    # Embed all canon files
    result = embed_canon()

    # Search canon by semantic query
    results = search_canon("verification protocol", top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Lazy imports for dependencies
REPO_ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_content(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_canon_root() -> Path:
    """Get the LAW/CANON directory path."""
    return REPO_ROOT / "LAW" / "CANON"


def _get_index_db_path() -> Path:
    """Get the canon index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "canon_index.db"


def inventory_canon(
    canon_root: Optional[Path] = None,
    *,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    """
    Create inventory of all canon files with content hashes.

    Args:
        canon_root: Override default canon root path
        emit_receipt: Include receipt in result

    Returns:
        Manifest dict with:
        - files: {relative_path: {sha256, size, content_type}}
        - total_files: int
        - total_bytes: int
        - manifest_hash: sha256 of manifest
        - receipt: (if emit_receipt) creation receipt
    """
    root = canon_root or _get_canon_root()

    if not root.exists():
        raise FileNotFoundError(f"Canon root not found: {root}")

    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0

    # Enumerate all files
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel_path = str(path.relative_to(root)).replace("\\", "/")
            content = path.read_bytes()
            content_hash = _hash_content(content)
            size = len(content)

            # Determine content type
            suffix = path.suffix.lower()
            content_type = {
                ".md": "text/markdown",
                ".json": "application/json",
                ".yaml": "text/yaml",
                ".yml": "text/yaml",
                ".txt": "text/plain",
            }.get(suffix, "application/octet-stream")

            files[rel_path] = {
                "sha256": content_hash,
                "size": size,
                "content_type": content_type,
            }
            total_bytes += size

    # Compute manifest hash
    manifest_data = {
        "files": files,
        "total_files": len(files),
        "total_bytes": total_bytes,
    }
    manifest_json = json.dumps(manifest_data, sort_keys=True, separators=(",", ":"))
    manifest_hash = _hash_text(manifest_json)

    result = {
        "files": files,
        "total_files": len(files),
        "total_bytes": total_bytes,
        "manifest_hash": manifest_hash,
    }

    if emit_receipt:
        result["receipt"] = {
            "operation": "inventory_canon",
            "timestamp": _now_iso(),
            "canon_root": str(root),
            "total_files": len(files),
            "total_bytes": total_bytes,
            "manifest_hash": manifest_hash,
            "tool_version": "1.0.0",
        }

    return result


def _init_canon_db(db_path: Path) -> sqlite3.Connection:
    """Initialize canon index database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS canon_records (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding BLOB,
            model_id TEXT,
            dimensions INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            tags TEXT,
            trust REAL DEFAULT 1.0,
            UNIQUE(file_path)
        );

        CREATE INDEX IF NOT EXISTS idx_canon_path ON canon_records(file_path);
        CREATE INDEX IF NOT EXISTS idx_canon_hash ON canon_records(content_hash);
        CREATE INDEX IF NOT EXISTS idx_canon_model ON canon_records(model_id);

        CREATE TABLE IF NOT EXISTS canon_manifest (
            id INTEGER PRIMARY KEY,
            manifest_hash TEXT NOT NULL,
            total_files INTEGER NOT NULL,
            total_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS canon_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT NOT NULL,
            receipt_hash TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def embed_canon(
    canon_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    *,
    force: bool = False,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Embed all canon files and store as MemoryRecord-compatible entries.

    Args:
        canon_root: Override default canon root path
        db_path: Override default database path
        force: Re-embed even if embedding exists
        batch_size: Batch size for embedding generation
        verbose: Print progress

    Returns:
        Result dict with:
        - total_files: int
        - embedded: int
        - skipped: int
        - errors: int
        - receipt: operation receipt
    """
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine
    from CAPABILITY.PRIMITIVES.memory_record import create_record, hash_record

    root = canon_root or _get_canon_root()
    db = db_path or _get_index_db_path()

    # Get inventory first
    manifest = inventory_canon(root, emit_receipt=False)

    # Initialize database and embedding engine
    conn = _init_canon_db(db)
    engine = EmbeddingEngine()

    embedded = 0
    skipped = 0
    errors = 0
    now = _now_iso()

    # Collect files to embed
    to_embed: list[tuple[str, str, str]] = []  # (rel_path, text, content_hash)

    for rel_path, meta in manifest["files"].items():
        # Check if already embedded
        if not force:
            cursor = conn.execute(
                "SELECT embedding FROM canon_records WHERE file_path = ? AND content_hash = ?",
                (rel_path, meta["sha256"]),
            )
            existing = cursor.fetchone()
            if existing and existing["embedding"]:
                skipped += 1
                continue

        # Read file content
        file_path = root / rel_path
        try:
            text = file_path.read_text(encoding="utf-8")
            to_embed.append((rel_path, text, meta["sha256"]))
        except Exception as e:
            if verbose:
                print(f"Error reading {rel_path}: {e}")
            errors += 1

    if verbose:
        print(f"Embedding {len(to_embed)} canon files...")

    # Batch embed
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        texts = [item[1] for item in batch]

        try:
            embeddings = engine.embed_batch(texts, batch_size=batch_size)

            for j, (rel_path, text, content_hash) in enumerate(batch):
                try:
                    # Create MemoryRecord
                    record = create_record(
                        text,
                        doc_path=f"LAW/CANON/{rel_path}",
                        tags=["canon", "governance"],
                        trust=1.0,
                        created_by="canon_index.py",
                        tool_version="1.0.0",
                    )

                    # Serialize embedding
                    embedding_blob = engine.serialize(embeddings[j])

                    # Store in database
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO canon_records
                        (id, text, file_path, content_hash, embedding, model_id,
                         dimensions, created_at, updated_at, tags, trust)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["id"],
                            text,
                            rel_path,
                            content_hash,
                            embedding_blob,
                            engine.MODEL_ID,
                            engine.DIMENSIONS,
                            record["payload"]["created_at"],
                            now,
                            json.dumps(record["payload"].get("tags", [])),
                            record["scores"]["trust"],
                        ),
                    )
                    embedded += 1

                except Exception as e:
                    if verbose:
                        print(f"Error embedding {rel_path}: {e}")
                    errors += 1

            conn.commit()

            if verbose and (i + batch_size) % (batch_size * 5) == 0:
                print(f"  Progress: {i + batch_size}/{len(to_embed)}")

        except Exception as e:
            if verbose:
                print(f"Batch error at {i}: {e}")
            errors += len(batch)

    # Store manifest
    conn.execute(
        """
        INSERT INTO canon_manifest (manifest_hash, total_files, total_bytes, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (manifest["manifest_hash"], manifest["total_files"], manifest["total_bytes"], now),
    )

    # Create and store receipt
    receipt = {
        "operation": "embed_canon",
        "timestamp": now,
        "total_files": manifest["total_files"],
        "embedded": embedded,
        "skipped": skipped,
        "errors": errors,
        "manifest_hash": manifest["manifest_hash"],
        "model_id": engine.MODEL_ID,
        "dimensions": engine.DIMENSIONS,
        "tool_version": "1.0.0",
    }
    receipt_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt_hash = _hash_text(receipt_json)
    receipt["receipt_hash"] = receipt_hash

    conn.execute(
        "INSERT INTO canon_receipts (operation, timestamp, details, receipt_hash) VALUES (?, ?, ?, ?)",
        ("embed_canon", now, receipt_json, receipt_hash),
    )

    conn.commit()
    conn.close()

    if verbose:
        print(f"Complete: {embedded} embedded, {skipped} skipped, {errors} errors")

    return {
        "total_files": manifest["total_files"],
        "embedded": embedded,
        "skipped": skipped,
        "errors": errors,
        "receipt": receipt,
    }


def search_canon(
    query: str,
    *,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Search canon files by semantic similarity.

    Args:
        query: Search query text
        top_k: Maximum results to return
        db_path: Override default database path
        min_similarity: Minimum similarity threshold

    Returns:
        List of results with:
        - file_path: relative path in LAW/CANON
        - similarity: cosine similarity score
        - content_hash: SHA-256 of content
        - text: file content (truncated to 500 chars for display)
    """
    import numpy as np
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    db = db_path or _get_index_db_path()

    if not db.exists():
        raise FileNotFoundError(f"Canon index not found. Run embed_canon() first.")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    engine = EmbeddingEngine()

    # Embed query
    query_embedding = engine.embed(query)

    # Get all embeddings from database
    cursor = conn.execute(
        "SELECT id, file_path, content_hash, text, embedding FROM canon_records WHERE embedding IS NOT NULL"
    )
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return []

    # Compute similarities
    results = []
    for row in rows:
        try:
            stored_embedding = engine.deserialize(row["embedding"])
            similarity = engine.cosine_similarity(query_embedding, stored_embedding)

            if similarity >= min_similarity:
                results.append({
                    "file_path": row["file_path"],
                    "similarity": float(similarity),
                    "content_hash": row["content_hash"],
                    "text": row["text"][:500] + ("..." if len(row["text"]) > 500 else ""),
                    "id": row["id"],
                })
        except Exception:
            continue

    conn.close()

    # Sort by similarity and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def rebuild_index(
    canon_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Rebuild canon index from scratch.

    This deletes the existing index and re-embeds all files.
    Used to ensure deterministic rebuilds.

    Args:
        canon_root: Override default canon root path
        db_path: Override default database path
        verbose: Print progress

    Returns:
        Result from embed_canon with rebuild receipt
    """
    db = db_path or _get_index_db_path()

    # Delete existing database
    if db.exists():
        db.unlink()
        if verbose:
            print(f"Deleted existing index: {db}")

    # Re-embed all
    result = embed_canon(canon_root, db_path, force=True, verbose=verbose)

    result["receipt"]["operation"] = "rebuild_index"

    return result


def get_index_stats(db_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Get statistics about the canon index.

    Args:
        db_path: Override default database path

    Returns:
        Stats dict with counts, sizes, and model info
    """
    db = db_path or _get_index_db_path()

    if not db.exists():
        return {"exists": False, "error": "Index not found"}

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # Get record stats
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_records,
            COUNT(embedding) as embedded_records,
            model_id,
            dimensions,
            MIN(created_at) as first_indexed,
            MAX(updated_at) as last_updated
        FROM canon_records
        GROUP BY model_id
        """
    )
    models = [dict(row) for row in cursor.fetchall()]

    # Get manifest stats
    cursor = conn.execute(
        "SELECT manifest_hash, total_files, total_bytes, created_at FROM canon_manifest ORDER BY id DESC LIMIT 1"
    )
    manifest = cursor.fetchone()

    # Get receipt count
    cursor = conn.execute("SELECT COUNT(*) as count FROM canon_receipts")
    receipt_count = cursor.fetchone()["count"]

    conn.close()

    return {
        "exists": True,
        "models": models,
        "total_records": sum(m["total_records"] for m in models) if models else 0,
        "embedded_records": sum(m["embedded_records"] for m in models) if models else 0,
        "manifest": dict(manifest) if manifest else None,
        "receipt_count": receipt_count,
    }


def verify_index(db_path: Optional[Path] = None, canon_root: Optional[Path] = None) -> dict[str, Any]:
    """
    Verify index integrity against source files.

    Args:
        db_path: Override default database path
        canon_root: Override default canon root

    Returns:
        Verification result with any mismatches found
    """
    db = db_path or _get_index_db_path()
    root = canon_root or _get_canon_root()

    if not db.exists():
        return {"valid": False, "error": "Index not found"}

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    mismatches = []
    missing_files = []
    extra_files = []

    # Get current inventory
    current_manifest = inventory_canon(root, emit_receipt=False)

    # Get indexed files
    cursor = conn.execute("SELECT file_path, content_hash FROM canon_records")
    indexed_files = {row["file_path"]: row["content_hash"] for row in cursor.fetchall()}

    conn.close()

    # Compare
    current_files = {path: meta["sha256"] for path, meta in current_manifest["files"].items()}

    for path, hash_val in current_files.items():
        if path not in indexed_files:
            missing_files.append(path)
        elif indexed_files[path] != hash_val:
            mismatches.append({
                "file": path,
                "indexed_hash": indexed_files[path],
                "current_hash": hash_val,
            })

    for path in indexed_files:
        if path not in current_files:
            extra_files.append(path)

    return {
        "valid": len(mismatches) == 0 and len(missing_files) == 0 and len(extra_files) == 0,
        "mismatches": mismatches,
        "missing_files": missing_files,
        "extra_files": extra_files,
        "indexed_count": len(indexed_files),
        "current_count": len(current_files),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Canon Index Manager")
    parser.add_argument("--inventory", action="store_true", help="Show canon inventory")
    parser.add_argument("--embed", action="store_true", help="Embed all canon files")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--verify", action="store_true", help="Verify index integrity")
    parser.add_argument("--search", type=str, help="Search canon by query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results")

    args = parser.parse_args()

    if args.inventory:
        manifest = inventory_canon()
        print(f"\nCanon Inventory:")
        print(f"  Total files: {manifest['total_files']}")
        print(f"  Total bytes: {manifest['total_bytes']}")
        print(f"  Manifest hash: {manifest['manifest_hash'][:16]}...")
        print(f"\nFiles:")
        for path in sorted(manifest["files"].keys()):
            meta = manifest["files"][path]
            print(f"  {path}: {meta['sha256'][:16]}... ({meta['size']} bytes)")

    elif args.embed:
        result = embed_canon(verbose=True)
        print(f"\nReceipt hash: {result['receipt']['receipt_hash'][:16]}...")

    elif args.rebuild:
        result = rebuild_index(verbose=True)
        print(f"\nReceipt hash: {result['receipt']['receipt_hash'][:16]}...")

    elif args.stats:
        stats = get_index_stats()
        print(f"\nCanon Index Statistics:")
        if not stats["exists"]:
            print("  Index not found. Run --embed first.")
        else:
            print(f"  Total records: {stats['total_records']}")
            print(f"  Embedded records: {stats['embedded_records']}")
            print(f"  Receipt count: {stats['receipt_count']}")
            if stats["manifest"]:
                print(f"  Last manifest: {stats['manifest']['manifest_hash'][:16]}...")

    elif args.verify:
        result = verify_index()
        if result.get("error"):
            print(f"Error: {result['error']}")
        elif result["valid"]:
            print("Index is valid and up-to-date.")
        else:
            print("Index integrity issues found:")
            for m in result["mismatches"]:
                print(f"  Hash mismatch: {m['file']}")
            for f in result["missing_files"]:
                print(f"  Missing from index: {f}")
            for f in result["extra_files"]:
                print(f"  Extra in index: {f}")

    elif args.search:
        results = search_canon(args.search, top_k=args.top_k)
        print(f"\nSearch results for: '{args.search}'")
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['file_path']} (similarity: {r['similarity']:.4f})")
            print(f"   {r['text'][:100]}...")

    else:
        parser.print_help()
