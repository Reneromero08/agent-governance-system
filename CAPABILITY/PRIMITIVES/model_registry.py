#!/usr/bin/env python3
"""
Model Weight Registry - Phase 5.1.3

Content-addressable registry for ML model weights with semantic search.

This registry enables:
- Storing model metadata with CAS-referenced weights
- Semantic search by model description
- Version tracking and deduplication
- Deterministic model resolution

Schema follows MemoryRecord contract:
- id: deterministic hash (name + version or weights hash)
- text: model description (canonical for embedding)
- embedding: vector for semantic search
- payload: model-specific metadata
- receipts: provenance tracking

Usage:
    from CAPABILITY.PRIMITIVES.model_registry import (
        register_model,
        get_model,
        search_models,
        get_registry_stats
    )

    # Register a model
    record = register_model(
        name="all-MiniLM-L6-v2",
        version="2.0.0",
        description="Sentence transformer for semantic similarity",
        weights_path=Path("models/minilm/pytorch_model.bin"),
        format="pytorch"
    )

    # Search by description
    results = search_models("sentence embedding similarity")
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Import embedding engine
try:
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine
except ImportError:
    EmbeddingEngine = None


# Default paths
DEFAULT_DB_PATH = REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "model_registry.db"
DEFAULT_CAS_ROOT = REPO_ROOT / "CAPABILITY" / "CAS" / "models"


# ============================================================================
# Schema Types
# ============================================================================

class ModelMetadata(TypedDict, total=False):
    """Model-specific metadata."""
    architecture: str
    parameters: int
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_seq_length: int
    license: str
    author: str
    paper_url: str
    repo_url: str
    training_data: str
    task: str
    language: str
    tags: list[str]
    extra: dict[str, Any]


class ModelRecord(TypedDict):
    """
    Model weight registry record.

    Follows MemoryRecord contract:
    - id: deterministic identifier
    - text: description (canonical, embeddable)
    - embedding: vector for semantic search
    - payload: model metadata
    - receipts: provenance
    """
    id: str
    name: str
    version: str
    description: str
    format: str
    weights_hash: Optional[str]
    size_bytes: Optional[int]
    embedding: Optional[bytes]
    dimensions: int
    metadata: ModelMetadata
    source: str
    created_at: str
    updated_at: str


# ============================================================================
# Hash Functions
# ============================================================================

def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of binary data."""
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _model_id(name: str, version: str) -> str:
    """Generate deterministic model ID from name and version."""
    canonical = f"{name}@{version}"
    return _hash_text(canonical)


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# Database Schema
# ============================================================================

MODEL_REGISTRY_SCHEMA = """
-- Model weight registry schema
-- Phase 5.1.3: Vector-indexed CAS for model weights

CREATE TABLE IF NOT EXISTS models (
    -- Core identity (MemoryRecord compatible)
    id TEXT PRIMARY KEY,           -- SHA-256(name@version)
    name TEXT NOT NULL,            -- Model name (e.g., "all-MiniLM-L6-v2")
    version TEXT NOT NULL,         -- Version string
    description TEXT NOT NULL,     -- Model description (text for embedding)

    -- Model format and storage
    format TEXT NOT NULL,          -- pytorch, safetensors, onnx, etc.
    weights_hash TEXT,             -- SHA-256 of weights (CAS reference)
    size_bytes INTEGER,            -- Size of model weights

    -- Vector index
    embedding BLOB,                -- Description embedding for semantic search
    dimensions INTEGER DEFAULT 384,
    model_id TEXT DEFAULT 'all-MiniLM-L6-v2',

    -- Metadata (JSON)
    metadata TEXT,                 -- JSON: architecture, params, license, etc.

    -- Source and provenance
    source TEXT DEFAULT 'local',   -- huggingface, local, custom, etc.
    source_url TEXT,               -- Original download URL

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    -- Uniqueness constraint
    UNIQUE(name, version)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_format ON models(format);
CREATE INDEX IF NOT EXISTS idx_models_weights_hash ON models(weights_hash);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at);

-- Registry receipts table (audit trail)
CREATE TABLE IF NOT EXISTS registry_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,       -- register, update, embed, delete
    model_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    details TEXT,                  -- JSON: operation details
    receipt_hash TEXT NOT NULL,    -- Hash of receipt for verification

    FOREIGN KEY (model_id) REFERENCES models(id)
);

CREATE INDEX IF NOT EXISTS idx_receipts_model ON registry_receipts(model_id);
CREATE INDEX IF NOT EXISTS idx_receipts_operation ON registry_receipts(operation);
"""


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(MODEL_REGISTRY_SCHEMA)
    conn.commit()
    return conn


def _emit_receipt(
    conn: sqlite3.Connection,
    operation: str,
    model_id: str,
    details: dict
) -> str:
    """Emit an audit receipt for an operation."""
    timestamp = _now_iso()
    details_json = json.dumps(details, sort_keys=True)
    receipt_content = f"{operation}:{model_id}:{timestamp}:{details_json}"
    receipt_hash = _hash_text(receipt_content)

    conn.execute("""
        INSERT INTO registry_receipts (operation, model_id, timestamp, details, receipt_hash)
        VALUES (?, ?, ?, ?, ?)
    """, (operation, model_id, timestamp, details_json, receipt_hash))

    return receipt_hash


# ============================================================================
# Registry Functions
# ============================================================================

def create_model_record(
    name: str,
    version: str,
    description: str,
    format: str,
    *,
    weights_hash: Optional[str] = None,
    size_bytes: Optional[int] = None,
    metadata: Optional[ModelMetadata] = None,
    source: str = "local",
) -> ModelRecord:
    """
    Create a ModelRecord without persisting it.

    Args:
        name: Model name (e.g., "all-MiniLM-L6-v2")
        version: Version string (e.g., "2.0.0")
        description: What the model does (for semantic search)
        format: Model format (pytorch, safetensors, onnx, etc.)
        weights_hash: SHA-256 of model weights (CAS reference)
        size_bytes: Size of model weights in bytes
        metadata: Additional model metadata
        source: Model source (huggingface, local, custom)

    Returns:
        ModelRecord dict ready for persistence
    """
    now = _now_iso()
    model_id = _model_id(name, version)

    record: ModelRecord = {
        "id": model_id,
        "name": name,
        "version": version,
        "description": description,
        "format": format,
        "weights_hash": weights_hash,
        "size_bytes": size_bytes,
        "embedding": None,
        "dimensions": 384,
        "metadata": metadata or {},
        "source": source,
        "created_at": now,
        "updated_at": now,
    }

    return record


def register_model(
    name: str,
    version: str,
    description: str,
    format: str,
    *,
    weights_path: Optional[Path] = None,
    weights_hash: Optional[str] = None,
    size_bytes: Optional[int] = None,
    metadata: Optional[ModelMetadata] = None,
    source: str = "local",
    source_url: Optional[str] = None,
    embed_description: bool = True,
    db_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Register a model in the registry.

    Args:
        name: Model name
        version: Version string
        description: Model description (will be embedded)
        format: Model format (pytorch, safetensors, onnx, etc.)
        weights_path: Path to model weights (for hashing)
        weights_hash: Pre-computed weights hash (alternative to weights_path)
        size_bytes: Pre-computed size (alternative to weights_path)
        metadata: Additional model metadata
        source: Model source (huggingface, local, custom)
        source_url: Original download URL
        embed_description: Whether to generate embedding for description
        db_path: Path to registry database
        verbose: Print progress messages

    Returns:
        Result dict with model_id, status, receipt
    """
    db_path = db_path or DEFAULT_DB_PATH
    conn = _init_db(db_path)

    try:
        # Compute weights hash if path provided
        if weights_path and weights_path.exists():
            weights_hash = _hash_file(weights_path)
            size_bytes = weights_path.stat().st_size
            if verbose:
                print(f"Computed weights hash: {weights_hash[:16]}...")

        # Create record
        record = create_model_record(
            name=name,
            version=version,
            description=description,
            format=format,
            weights_hash=weights_hash,
            size_bytes=size_bytes,
            metadata=metadata,
            source=source,
        )

        # Generate embedding
        embedding_blob = None
        if embed_description and EmbeddingEngine:
            engine = EmbeddingEngine()
            embedding = engine.embed(description)
            embedding_blob = engine.serialize(embedding)
            record["embedding"] = embedding_blob
            record["dimensions"] = len(embedding)
            if verbose:
                print(f"Generated description embedding: {len(embedding)} dims")

        # Insert into database
        cursor = conn.execute("""
            INSERT INTO models (
                id, name, version, description, format,
                weights_hash, size_bytes, embedding, dimensions, model_id,
                metadata, source, source_url, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, version) DO UPDATE SET
                description = excluded.description,
                format = excluded.format,
                weights_hash = excluded.weights_hash,
                size_bytes = excluded.size_bytes,
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                metadata = excluded.metadata,
                source = excluded.source,
                source_url = excluded.source_url,
                updated_at = excluded.updated_at
        """, (
            record["id"],
            record["name"],
            record["version"],
            record["description"],
            record["format"],
            record["weights_hash"],
            record["size_bytes"],
            embedding_blob,
            record["dimensions"],
            "all-MiniLM-L6-v2",
            json.dumps(record["metadata"]),
            record["source"],
            source_url,
            record["created_at"],
            record["updated_at"],
        ))

        is_update = cursor.rowcount == 0  # ON CONFLICT UPDATE doesn't increment
        operation = "update" if is_update else "register"

        # Emit receipt
        receipt_hash = _emit_receipt(conn, operation, record["id"], {
            "name": name,
            "version": version,
            "weights_hash": weights_hash,
            "embedded": embed_description and EmbeddingEngine is not None,
        })

        conn.commit()

        if verbose:
            print(f"{'Updated' if is_update else 'Registered'} model: {name}@{version}")

        return {
            "model_id": record["id"],
            "name": name,
            "version": version,
            "status": "updated" if is_update else "registered",
            "weights_hash": weights_hash,
            "embedded": embed_description and EmbeddingEngine is not None,
            "receipt": {
                "operation": operation,
                "model_id": record["id"],
                "receipt_hash": receipt_hash,
            }
        }

    finally:
        conn.close()


def get_model(
    name: str,
    version: Optional[str] = None,
    *,
    db_path: Optional[Path] = None,
) -> Optional[ModelRecord]:
    """
    Retrieve a model record by name and version.

    Args:
        name: Model name
        version: Version string (if None, returns latest)
        db_path: Path to registry database

    Returns:
        ModelRecord or None if not found
    """
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if version:
            cursor = conn.execute("""
                SELECT * FROM models WHERE name = ? AND version = ?
            """, (name, version))
        else:
            # Get latest version (by created_at)
            cursor = conn.execute("""
                SELECT * FROM models WHERE name = ?
                ORDER BY created_at DESC LIMIT 1
            """, (name,))

        row = cursor.fetchone()
        if not row:
            return None

        return _row_to_record(row)

    finally:
        conn.close()


def get_model_by_id(
    model_id: str,
    *,
    db_path: Optional[Path] = None,
) -> Optional[ModelRecord]:
    """
    Retrieve a model record by ID.

    Args:
        model_id: Model ID (hash)
        db_path: Path to registry database

    Returns:
        ModelRecord or None if not found
    """
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return _row_to_record(row)

    finally:
        conn.close()


def get_model_by_weights_hash(
    weights_hash: str,
    *,
    db_path: Optional[Path] = None,
) -> Optional[ModelRecord]:
    """
    Retrieve a model record by weights hash.

    This enables deduplication - same weights = same model.

    Args:
        weights_hash: SHA-256 of model weights
        db_path: Path to registry database

    Returns:
        ModelRecord or None if not found
    """
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(
            "SELECT * FROM models WHERE weights_hash = ?",
            (weights_hash,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        return _row_to_record(row)

    finally:
        conn.close()


def _row_to_record(row: sqlite3.Row) -> ModelRecord:
    """Convert database row to ModelRecord."""
    metadata = {}
    if row["metadata"]:
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            pass

    return {
        "id": row["id"],
        "name": row["name"],
        "version": row["version"],
        "description": row["description"],
        "format": row["format"],
        "weights_hash": row["weights_hash"],
        "size_bytes": row["size_bytes"],
        "embedding": row["embedding"],
        "dimensions": row["dimensions"],
        "metadata": metadata,
        "source": row["source"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def search_models(
    query: str,
    *,
    top_k: int = 5,
    min_similarity: float = 0.0,
    format_filter: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Search models by description similarity.

    Args:
        query: Search query (will be embedded)
        top_k: Maximum results to return
        min_similarity: Minimum similarity threshold
        format_filter: Filter by model format
        db_path: Path to registry database

    Returns:
        List of results with similarity scores
    """
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        return []

    if not EmbeddingEngine:
        raise ImportError("EmbeddingEngine not available for search")

    engine = EmbeddingEngine()
    query_embedding = engine.embed(query)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Get all models with embeddings
        if format_filter:
            cursor = conn.execute("""
                SELECT * FROM models WHERE embedding IS NOT NULL AND format = ?
            """, (format_filter,))
        else:
            cursor = conn.execute("""
                SELECT * FROM models WHERE embedding IS NOT NULL
            """)

        rows = cursor.fetchall()

        if not rows:
            return []

        # Compute similarities
        results = []
        for row in rows:
            if row["embedding"]:
                db_embedding = engine.deserialize(row["embedding"])
                similarity = float(engine.cosine_similarity(query_embedding, db_embedding))

                if similarity >= min_similarity:
                    results.append({
                        "model_id": row["id"],
                        "name": row["name"],
                        "version": row["version"],
                        "description": row["description"],
                        "format": row["format"],
                        "similarity": similarity,
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    })

        # Sort by similarity descending, then by name for tie-breaking
        results.sort(key=lambda x: (-x["similarity"], x["name"]))

        return results[:top_k]

    finally:
        conn.close()


def list_models(
    *,
    format_filter: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    List all registered models.

    Args:
        format_filter: Filter by model format
        db_path: Path to registry database

    Returns:
        List of model summaries
    """
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        if format_filter:
            cursor = conn.execute("""
                SELECT id, name, version, format, source, created_at
                FROM models WHERE format = ?
                ORDER BY name, version
            """, (format_filter,))
        else:
            cursor = conn.execute("""
                SELECT id, name, version, format, source, created_at
                FROM models ORDER BY name, version
            """)

        return [dict(row) for row in cursor.fetchall()]

    finally:
        conn.close()


def get_registry_stats(db_path: Optional[Path] = None) -> dict:
    """
    Get registry statistics.

    Args:
        db_path: Path to registry database

    Returns:
        Statistics dict
    """
    db_path = db_path or DEFAULT_DB_PATH

    if not db_path.exists():
        return {
            "exists": False,
            "total_models": 0,
            "embedded_models": 0,
            "formats": {},
            "sources": {},
        }

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Total models
        cursor = conn.execute("SELECT COUNT(*) as count FROM models")
        total = cursor.fetchone()["count"]

        # Embedded models
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM models WHERE embedding IS NOT NULL"
        )
        embedded = cursor.fetchone()["count"]

        # By format
        cursor = conn.execute("""
            SELECT format, COUNT(*) as count FROM models GROUP BY format
        """)
        formats = {row["format"]: row["count"] for row in cursor.fetchall()}

        # By source
        cursor = conn.execute("""
            SELECT source, COUNT(*) as count FROM models GROUP BY source
        """)
        sources = {row["source"]: row["count"] for row in cursor.fetchall()}

        # Total size
        cursor = conn.execute("""
            SELECT COALESCE(SUM(size_bytes), 0) as total_size FROM models
        """)
        total_size = cursor.fetchone()["total_size"]

        # Receipt count
        cursor = conn.execute("SELECT COUNT(*) as count FROM registry_receipts")
        receipts = cursor.fetchone()["count"]

        return {
            "exists": True,
            "total_models": total,
            "embedded_models": embedded,
            "formats": formats,
            "sources": sources,
            "total_size_bytes": total_size,
            "receipt_count": receipts,
        }

    finally:
        conn.close()


def verify_registry(
    db_path: Optional[Path] = None,
    cas_root: Optional[Path] = None,
) -> dict:
    """
    Verify registry integrity.

    Checks:
    - All embeddings are valid
    - Referenced weights exist in CAS (if CAS root provided)
    - IDs match computed values

    Args:
        db_path: Path to registry database
        cas_root: Path to CAS storage root

    Returns:
        Verification result dict
    """
    db_path = db_path or DEFAULT_DB_PATH
    cas_root = cas_root or DEFAULT_CAS_ROOT

    if not db_path.exists():
        return {"valid": False, "errors": ["Database does not exist"]}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    errors = []
    warnings = []

    try:
        engine = EmbeddingEngine() if EmbeddingEngine else None

        cursor = conn.execute("SELECT * FROM models")
        for row in cursor.fetchall():
            # Check ID consistency
            expected_id = _model_id(row["name"], row["version"])
            if row["id"] != expected_id:
                errors.append(
                    f"ID mismatch for {row['name']}@{row['version']}: "
                    f"stored={row['id'][:16]}, expected={expected_id[:16]}"
                )

            # Check embedding validity
            if row["embedding"] and engine:
                try:
                    emb = engine.deserialize(row["embedding"])
                    if len(emb) != row["dimensions"]:
                        warnings.append(
                            f"Dimension mismatch for {row['name']}: "
                            f"stored={row['dimensions']}, actual={len(emb)}"
                        )
                except Exception as e:
                    errors.append(
                        f"Invalid embedding for {row['name']}: {e}"
                    )

            # Check CAS reference
            if row["weights_hash"] and cas_root.exists():
                weights_path = cas_root / row["weights_hash"][:2] / row["weights_hash"]
                if not weights_path.exists():
                    warnings.append(
                        f"Missing CAS blob for {row['name']}: {row['weights_hash'][:16]}"
                    )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checked_models": cursor.rowcount,
        }

    finally:
        conn.close()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Weight Registry")
    sub = parser.add_subparsers(dest="cmd")

    # Register command
    reg_p = sub.add_parser("register", help="Register a model")
    reg_p.add_argument("--name", required=True, help="Model name")
    reg_p.add_argument("--version", required=True, help="Version string")
    reg_p.add_argument("--description", required=True, help="Model description")
    reg_p.add_argument("--format", required=True, help="Model format")
    reg_p.add_argument("--weights", type=Path, help="Path to weights file")
    reg_p.add_argument("--source", default="local", help="Model source")

    # Search command
    search_p = sub.add_parser("search", help="Search models by description")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--top-k", type=int, default=5, help="Max results")

    # List command
    list_p = sub.add_parser("list", help="List all models")
    list_p.add_argument("--format", dest="format_filter", help="Filter by format")

    # Stats command
    sub.add_parser("stats", help="Show registry statistics")

    # Verify command
    sub.add_parser("verify", help="Verify registry integrity")

    args = parser.parse_args()

    if args.cmd == "register":
        result = register_model(
            name=args.name,
            version=args.version,
            description=args.description,
            format=args.format,
            weights_path=args.weights,
            source=args.source,
        )
        print(json.dumps(result, indent=2))

    elif args.cmd == "search":
        results = search_models(args.query, top_k=args.top_k)
        for r in results:
            print(f"{r['name']}@{r['version']}: {r['similarity']:.4f}")
            print(f"  {r['description'][:60]}...")

    elif args.cmd == "list":
        models = list_models(format_filter=args.format_filter)
        for m in models:
            print(f"{m['name']}@{m['version']} ({m['format']}) - {m['source']}")

    elif args.cmd == "stats":
        stats = get_registry_stats()
        print(json.dumps(stats, indent=2))

    elif args.cmd == "verify":
        result = verify_registry()
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
