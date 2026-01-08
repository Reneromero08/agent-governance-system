#!/usr/bin/env python3
"""
ADR Index Primitive - Phase 5.1.2

Indexes all Architecture Decision Records (LAW/CONTEXT/decisions/*) with:
- MemoryRecord instances for each ADR
- Parsed YAML frontmatter metadata
- Vector embeddings via CORTEX EmbeddingEngine
- Deterministic manifest with content hashes
- Cross-references to related canon files
- Receipted operations

Usage:
    from CAPABILITY.PRIMITIVES.adr_index import (
        inventory_adrs,
        embed_adrs,
        search_adrs,
        rebuild_adr_index,
    )

    # Create inventory with metadata
    manifest = inventory_adrs()

    # Embed all ADRs
    result = embed_adrs()

    # Search ADRs by semantic query
    results = search_adrs("governance protocol", top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

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


def _get_adr_root() -> Path:
    """Get the LAW/CONTEXT/decisions directory path."""
    return REPO_ROOT / "LAW" / "CONTEXT" / "decisions"


def _get_adr_db_path() -> Path:
    """Get the ADR index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "adr_index.db"


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: Full markdown content

    Returns:
        Tuple of (metadata dict, body content without frontmatter)
    """
    # Match YAML frontmatter between --- delimiters
    pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    try:
        metadata = yaml.safe_load(match.group(1))
        body = content[match.end():]
        return metadata or {}, body
    except yaml.YAMLError:
        return {}, content


def inventory_adrs(
    adr_root: Optional[Path] = None,
    *,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    """
    Create inventory of all ADR files with metadata.

    Args:
        adr_root: Override default ADR root path
        emit_receipt: Include receipt in result

    Returns:
        Manifest dict with:
        - files: {relative_path: {sha256, size, metadata}}
        - total_files: int
        - total_bytes: int
        - manifest_hash: sha256 of manifest
        - by_status: {status: [adr_ids]}
        - receipt: (if emit_receipt) creation receipt
    """
    root = adr_root or _get_adr_root()

    if not root.exists():
        raise FileNotFoundError(f"ADR root not found: {root}")

    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    by_status: dict[str, list[str]] = {}

    # Enumerate all markdown files
    for path in sorted(root.glob("*.md")):
        if path.is_file():
            rel_path = path.name
            content = path.read_bytes()
            content_hash = _hash_content(content)
            size = len(content)

            # Parse frontmatter for metadata
            text_content = content.decode("utf-8", errors="replace")
            metadata, _ = _parse_frontmatter(text_content)

            # Extract key fields
            adr_id = metadata.get("id", rel_path.replace(".md", ""))
            title = metadata.get("title", "")
            status = metadata.get("status", "Unknown")
            date = metadata.get("date", "")
            confidence = metadata.get("confidence", "")
            impact = metadata.get("impact", "")
            tags = metadata.get("tags", [])

            files[rel_path] = {
                "sha256": content_hash,
                "size": size,
                "content_type": "text/markdown",
                "metadata": {
                    "id": adr_id,
                    "title": title,
                    "status": status,
                    "date": date,
                    "confidence": confidence,
                    "impact": impact,
                    "tags": tags,
                },
            }
            total_bytes += size

            # Group by status
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(adr_id)

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
        "by_status": by_status,
    }

    if emit_receipt:
        result["receipt"] = {
            "operation": "inventory_adrs",
            "timestamp": _now_iso(),
            "adr_root": str(root),
            "total_files": len(files),
            "total_bytes": total_bytes,
            "manifest_hash": manifest_hash,
            "status_counts": {s: len(ids) for s, ids in by_status.items()},
            "tool_version": "1.0.0",
        }

    return result


def _init_adr_db(db_path: Path) -> sqlite3.Connection:
    """Initialize ADR index database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS adr_records (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            adr_id TEXT,
            title TEXT,
            status TEXT,
            adr_date TEXT,
            confidence TEXT,
            impact TEXT,
            tags TEXT,
            embedding BLOB,
            model_id TEXT,
            dimensions INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            trust REAL DEFAULT 1.0,
            UNIQUE(file_path)
        );

        CREATE INDEX IF NOT EXISTS idx_adr_path ON adr_records(file_path);
        CREATE INDEX IF NOT EXISTS idx_adr_hash ON adr_records(content_hash);
        CREATE INDEX IF NOT EXISTS idx_adr_id ON adr_records(adr_id);
        CREATE INDEX IF NOT EXISTS idx_adr_status ON adr_records(status);
        CREATE INDEX IF NOT EXISTS idx_adr_model ON adr_records(model_id);

        CREATE TABLE IF NOT EXISTS adr_manifest (
            id INTEGER PRIMARY KEY,
            manifest_hash TEXT NOT NULL,
            total_files INTEGER NOT NULL,
            total_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adr_receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT NOT NULL,
            receipt_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adr_canon_refs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            adr_file TEXT NOT NULL,
            canon_file TEXT NOT NULL,
            similarity REAL,
            created_at TEXT NOT NULL,
            UNIQUE(adr_file, canon_file)
        );
    """)
    conn.commit()
    return conn


def embed_adrs(
    adr_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    *,
    force: bool = False,
    batch_size: int = 32,
    verbose: bool = True,
    cross_ref_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Embed all ADR files and store as MemoryRecord-compatible entries.

    Args:
        adr_root: Override default ADR root path
        db_path: Override default database path
        force: Re-embed even if embedding exists
        batch_size: Batch size for embedding generation
        verbose: Print progress
        cross_ref_threshold: Min similarity for canon cross-references

    Returns:
        Result dict with:
        - total_files: int
        - embedded: int
        - skipped: int
        - errors: int
        - cross_refs_created: int
        - receipt: operation receipt
    """
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine
    from CAPABILITY.PRIMITIVES.memory_record import create_record

    root = adr_root or _get_adr_root()
    db = db_path or _get_adr_db_path()

    # Get inventory first
    manifest = inventory_adrs(root, emit_receipt=False)

    # Initialize database and embedding engine
    conn = _init_adr_db(db)
    engine = EmbeddingEngine()

    embedded = 0
    skipped = 0
    errors = 0
    cross_refs_created = 0
    now = _now_iso()

    # Collect files to embed
    to_embed: list[tuple[str, str, str, dict]] = []  # (rel_path, text, content_hash, metadata)

    for rel_path, meta in manifest["files"].items():
        # Check if already embedded
        if not force:
            cursor = conn.execute(
                "SELECT embedding FROM adr_records WHERE file_path = ? AND content_hash = ?",
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
            to_embed.append((rel_path, text, meta["sha256"], meta["metadata"]))
        except Exception as e:
            if verbose:
                print(f"Error reading {rel_path}: {e}")
            errors += 1

    if verbose:
        print(f"Embedding {len(to_embed)} ADR files...")

    # Batch embed
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i : i + batch_size]
        texts = [item[1] for item in batch]

        try:
            embeddings = engine.embed_batch(texts, batch_size=batch_size)

            for j, (rel_path, text, content_hash, metadata) in enumerate(batch):
                try:
                    # Create MemoryRecord with ADR metadata in payload
                    record = create_record(
                        text,
                        doc_path=f"LAW/CONTEXT/decisions/{rel_path}",
                        tags=["adr", "architecture"] + metadata.get("tags", []),
                        trust=1.0,
                        created_by="adr_index.py",
                        tool_version="1.0.0",
                    )

                    # Serialize embedding
                    embedding_blob = engine.serialize(embeddings[j])

                    # Store in database with metadata
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO adr_records
                        (id, text, file_path, content_hash, adr_id, title, status,
                         adr_date, confidence, impact, tags, embedding, model_id,
                         dimensions, created_at, updated_at, trust)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record["id"],
                            text,
                            rel_path,
                            content_hash,
                            metadata.get("id", ""),
                            metadata.get("title", ""),
                            metadata.get("status", ""),
                            metadata.get("date", ""),
                            metadata.get("confidence", ""),
                            metadata.get("impact", ""),
                            json.dumps(metadata.get("tags", [])),
                            embedding_blob,
                            engine.MODEL_ID,
                            engine.DIMENSIONS,
                            record["payload"]["created_at"],
                            now,
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
        INSERT INTO adr_manifest (manifest_hash, total_files, total_bytes, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (manifest["manifest_hash"], manifest["total_files"], manifest["total_bytes"], now),
    )

    # Create cross-references to canon files if canon index exists
    canon_db = REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "canon_index.db"
    if canon_db.exists():
        cross_refs_created = _create_canon_cross_refs(conn, engine, canon_db, cross_ref_threshold, verbose)

    # Create and store receipt
    receipt = {
        "operation": "embed_adrs",
        "timestamp": now,
        "total_files": manifest["total_files"],
        "embedded": embedded,
        "skipped": skipped,
        "errors": errors,
        "cross_refs_created": cross_refs_created,
        "manifest_hash": manifest["manifest_hash"],
        "model_id": engine.MODEL_ID,
        "dimensions": engine.DIMENSIONS,
        "tool_version": "1.0.0",
    }
    receipt_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt_hash = _hash_text(receipt_json)
    receipt["receipt_hash"] = receipt_hash

    conn.execute(
        "INSERT INTO adr_receipts (operation, timestamp, details, receipt_hash) VALUES (?, ?, ?, ?)",
        ("embed_adrs", now, receipt_json, receipt_hash),
    )

    conn.commit()
    conn.close()

    if verbose:
        print(f"Complete: {embedded} embedded, {skipped} skipped, {errors} errors")
        if cross_refs_created > 0:
            print(f"Cross-references to canon: {cross_refs_created}")

    return {
        "total_files": manifest["total_files"],
        "embedded": embedded,
        "skipped": skipped,
        "errors": errors,
        "cross_refs_created": cross_refs_created,
        "receipt": receipt,
    }


def _create_canon_cross_refs(
    adr_conn: sqlite3.Connection,
    engine: Any,
    canon_db_path: Path,
    threshold: float,
    verbose: bool,
) -> int:
    """
    Create cross-references between ADRs and related canon files.

    For each ADR, finds the most similar canon files above threshold.
    """
    import numpy as np

    canon_conn = sqlite3.connect(str(canon_db_path))
    canon_conn.row_factory = sqlite3.Row

    # Get all ADR embeddings
    adr_cursor = adr_conn.execute(
        "SELECT file_path, embedding FROM adr_records WHERE embedding IS NOT NULL"
    )
    adr_rows = adr_cursor.fetchall()

    # Get all canon embeddings
    canon_cursor = canon_conn.execute(
        "SELECT file_path, embedding FROM canon_records WHERE embedding IS NOT NULL"
    )
    canon_rows = canon_cursor.fetchall()

    if not adr_rows or not canon_rows:
        canon_conn.close()
        return 0

    # Deserialize embeddings
    adr_embeddings = []
    adr_paths = []
    for row in adr_rows:
        adr_embeddings.append(engine.deserialize(row["embedding"]))
        adr_paths.append(row["file_path"])

    canon_embeddings = []
    canon_paths = []
    for row in canon_rows:
        canon_embeddings.append(engine.deserialize(row["embedding"]))
        canon_paths.append(row["file_path"])

    adr_matrix = np.array(adr_embeddings)
    canon_matrix = np.array(canon_embeddings)

    # Compute similarities
    refs_created = 0
    now = _now_iso()

    for i, adr_path in enumerate(adr_paths):
        similarities = engine.batch_similarity(adr_matrix[i], canon_matrix)

        for j, similarity in enumerate(similarities):
            if similarity >= threshold:
                try:
                    adr_conn.execute(
                        """
                        INSERT OR REPLACE INTO adr_canon_refs
                        (adr_file, canon_file, similarity, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (adr_path, canon_paths[j], float(similarity), now),
                    )
                    refs_created += 1
                except Exception:
                    pass

    adr_conn.commit()
    canon_conn.close()

    return refs_created


def search_adrs(
    query: str,
    *,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    min_similarity: float = 0.0,
    status_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Search ADR files by semantic similarity.

    Args:
        query: Search query text
        top_k: Maximum results to return
        db_path: Override default database path
        min_similarity: Minimum similarity threshold
        status_filter: Optional filter by ADR status (e.g., "Accepted")

    Returns:
        List of results with:
        - file_path: relative path
        - similarity: cosine similarity score
        - metadata: ADR metadata (id, title, status, etc.)
        - text: ADR content (truncated)
    """
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    db = db_path or _get_adr_db_path()

    if not db.exists():
        raise FileNotFoundError(f"ADR index not found. Run embed_adrs() first.")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    engine = EmbeddingEngine()

    # Embed query
    query_embedding = engine.embed(query)

    # Get all embeddings from database
    if status_filter:
        cursor = conn.execute(
            """SELECT id, file_path, content_hash, text, embedding,
                      adr_id, title, status, adr_date, confidence, impact, tags
               FROM adr_records
               WHERE embedding IS NOT NULL AND status = ?""",
            (status_filter,),
        )
    else:
        cursor = conn.execute(
            """SELECT id, file_path, content_hash, text, embedding,
                      adr_id, title, status, adr_date, confidence, impact, tags
               FROM adr_records WHERE embedding IS NOT NULL"""
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
                tags = []
                try:
                    tags = json.loads(row["tags"]) if row["tags"] else []
                except Exception:
                    pass

                results.append({
                    "file_path": row["file_path"],
                    "similarity": float(similarity),
                    "content_hash": row["content_hash"],
                    "text": row["text"][:500] + ("..." if len(row["text"]) > 500 else ""),
                    "id": row["id"],
                    "metadata": {
                        "adr_id": row["adr_id"],
                        "title": row["title"],
                        "status": row["status"],
                        "date": row["adr_date"],
                        "confidence": row["confidence"],
                        "impact": row["impact"],
                        "tags": tags,
                    },
                })
        except Exception:
            continue

    conn.close()

    # Sort by similarity and return top_k
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def get_related_canon(
    adr_file: str,
    *,
    db_path: Optional[Path] = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Get canon files related to a specific ADR.

    Args:
        adr_file: ADR filename (e.g., "ADR-001-build-and-artifacts.md")
        db_path: Override default database path
        top_k: Maximum results to return

    Returns:
        List of related canon files with similarity scores
    """
    db = db_path or _get_adr_db_path()

    if not db.exists():
        return []

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        """SELECT canon_file, similarity
           FROM adr_canon_refs
           WHERE adr_file = ?
           ORDER BY similarity DESC
           LIMIT ?""",
        (adr_file, top_k),
    )
    results = [{"canon_file": row["canon_file"], "similarity": row["similarity"]} for row in cursor.fetchall()]
    conn.close()

    return results


def rebuild_adr_index(
    adr_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Rebuild ADR index from scratch.

    This deletes the existing index and re-embeds all files.
    Used to ensure deterministic rebuilds.

    Args:
        adr_root: Override default ADR root path
        db_path: Override default database path
        verbose: Print progress

    Returns:
        Result from embed_adrs with rebuild receipt
    """
    db = db_path or _get_adr_db_path()

    # Delete existing database
    if db.exists():
        db.unlink()
        if verbose:
            print(f"Deleted existing index: {db}")

    # Re-embed all
    result = embed_adrs(adr_root, db_path, force=True, verbose=verbose)

    result["receipt"]["operation"] = "rebuild_adr_index"

    return result


def get_adr_stats(db_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Get statistics about the ADR index.

    Args:
        db_path: Override default database path

    Returns:
        Stats dict with counts, sizes, and status breakdown
    """
    db = db_path or _get_adr_db_path()

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
        FROM adr_records
        GROUP BY model_id
        """
    )
    models = [dict(row) for row in cursor.fetchall()]

    # Get status breakdown
    cursor = conn.execute(
        "SELECT status, COUNT(*) as count FROM adr_records GROUP BY status"
    )
    status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

    # Get manifest stats
    cursor = conn.execute(
        "SELECT manifest_hash, total_files, total_bytes, created_at FROM adr_manifest ORDER BY id DESC LIMIT 1"
    )
    manifest = cursor.fetchone()

    # Get receipt count
    cursor = conn.execute("SELECT COUNT(*) as count FROM adr_receipts")
    receipt_count = cursor.fetchone()["count"]

    # Get cross-reference count
    cursor = conn.execute("SELECT COUNT(*) as count FROM adr_canon_refs")
    cross_ref_count = cursor.fetchone()["count"]

    conn.close()

    return {
        "exists": True,
        "models": models,
        "total_records": sum(m["total_records"] for m in models) if models else 0,
        "embedded_records": sum(m["embedded_records"] for m in models) if models else 0,
        "status_counts": status_counts,
        "manifest": dict(manifest) if manifest else None,
        "receipt_count": receipt_count,
        "cross_ref_count": cross_ref_count,
    }


def verify_adr_index(db_path: Optional[Path] = None, adr_root: Optional[Path] = None) -> dict[str, Any]:
    """
    Verify ADR index integrity against source files.

    Args:
        db_path: Override default database path
        adr_root: Override default ADR root

    Returns:
        Verification result with any mismatches found
    """
    db = db_path or _get_adr_db_path()
    root = adr_root or _get_adr_root()

    if not db.exists():
        return {"valid": False, "error": "Index not found"}

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    mismatches = []
    missing_files = []
    extra_files = []

    # Get current inventory
    current_manifest = inventory_adrs(root, emit_receipt=False)

    # Get indexed files
    cursor = conn.execute("SELECT file_path, content_hash FROM adr_records")
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

    parser = argparse.ArgumentParser(description="ADR Index Manager")
    parser.add_argument("--inventory", action="store_true", help="Show ADR inventory")
    parser.add_argument("--embed", action="store_true", help="Embed all ADR files")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--verify", action="store_true", help="Verify index integrity")
    parser.add_argument("--search", type=str, help="Search ADRs by query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search results")
    parser.add_argument("--status", type=str, help="Filter by status")

    args = parser.parse_args()

    if args.inventory:
        manifest = inventory_adrs()
        print(f"\nADR Inventory:")
        print(f"  Total files: {manifest['total_files']}")
        print(f"  Total bytes: {manifest['total_bytes']}")
        print(f"  Manifest hash: {manifest['manifest_hash'][:16]}...")
        print(f"\nBy Status:")
        for status, ids in manifest["by_status"].items():
            print(f"  {status}: {len(ids)} ADRs")
        print(f"\nFiles:")
        for path in sorted(manifest["files"].keys()):
            meta = manifest["files"][path]
            adr_id = meta["metadata"]["id"]
            title = meta["metadata"]["title"][:50]
            print(f"  {adr_id}: {title}")

    elif args.embed:
        result = embed_adrs(verbose=True)
        print(f"\nReceipt hash: {result['receipt']['receipt_hash'][:16]}...")

    elif args.rebuild:
        result = rebuild_adr_index(verbose=True)
        print(f"\nReceipt hash: {result['receipt']['receipt_hash'][:16]}...")

    elif args.stats:
        stats = get_adr_stats()
        print(f"\nADR Index Statistics:")
        if not stats["exists"]:
            print("  Index not found. Run --embed first.")
        else:
            print(f"  Total records: {stats['total_records']}")
            print(f"  Embedded records: {stats['embedded_records']}")
            print(f"  Cross-references: {stats['cross_ref_count']}")
            print(f"  Receipt count: {stats['receipt_count']}")
            print(f"\nBy Status:")
            for status, count in stats["status_counts"].items():
                print(f"  {status}: {count}")
            if stats["manifest"]:
                print(f"\nLast manifest: {stats['manifest']['manifest_hash'][:16]}...")

    elif args.verify:
        result = verify_adr_index()
        if result.get("error"):
            print(f"Error: {result['error']}")
        elif result["valid"]:
            print("ADR index is valid and up-to-date.")
        else:
            print("Index integrity issues found:")
            for m in result["mismatches"]:
                print(f"  Hash mismatch: {m['file']}")
            for f in result["missing_files"]:
                print(f"  Missing from index: {f}")
            for f in result["extra_files"]:
                print(f"  Extra in index: {f}")

    elif args.search:
        results = search_adrs(args.search, top_k=args.top_k, status_filter=args.status)
        print(f"\nSearch results for: '{args.search}'")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n{i}. [{meta['adr_id']}] {meta['title']}")
            print(f"   Status: {meta['status']} | Similarity: {r['similarity']:.4f}")
            print(f"   {r['text'][:100]}...")

    else:
        parser.print_help()
