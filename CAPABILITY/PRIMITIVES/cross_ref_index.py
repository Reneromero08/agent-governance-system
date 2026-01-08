#!/usr/bin/env python3
"""
Cross-Reference Index Primitive - Phase 5.1.5.1

Provides unified cross-reference graph for all vector-indexed artifacts:
- Canon files (LAW/CANON/*)
- ADRs (THOUGHT/LEDGER/ADR/*)
- Skills (CAPABILITY/SKILLS/*/SKILL.md)
- Model weights (future: NAVIGATION/CORTEX/db/model_registry.db)

Creates a semantic relationship graph where edges represent embedding similarity.

Usage:
    from CAPABILITY.PRIMITIVES.cross_ref_index import (
        build_cross_refs,
        find_related,
        get_cross_ref_stats,
    )

    # Build cross-reference graph across all artifact types
    result = build_cross_refs(threshold=0.3, top_k_per_artifact=10)

    # Find related artifacts for a specific item
    related = find_related("LAW/CANON/GOVERNANCE/IMMUTABILITY.md", top_k=5)

    # Get graph statistics
    stats = get_cross_ref_stats()
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


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_index_db_path() -> Path:
    """Get the cross-reference index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "cross_ref_index.db"


def _get_canon_db_path() -> Path:
    """Get the canon index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "canon_index.db"


def _get_adr_db_path() -> Path:
    """Get the ADR index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "adr_index.db"


def _get_skill_db_path() -> Path:
    """Get the skill index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "skill_index.db"


def _init_db(db_path: Path) -> sqlite3.Connection:
    """
    Initialize cross-reference database with schema.

    Schema:
    - artifacts: Registry of all indexed artifacts (id, type, path, embedding_blob)
    - cross_refs: Graph edges (source_id, target_id, similarity, created_at)
    - build_history: Tracking rebuilds (timestamp, artifacts_count, refs_count)
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            artifact_type TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            embedding_blob BLOB,
            model_name TEXT NOT NULL,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(artifact_type, artifact_path)
        );

        CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
        CREATE INDEX IF NOT EXISTS idx_artifacts_path ON artifacts(artifact_path);

        CREATE TABLE IF NOT EXISTS cross_refs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            similarity REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(source_id, target_id),
            FOREIGN KEY (source_id) REFERENCES artifacts(artifact_id),
            FOREIGN KEY (target_id) REFERENCES artifacts(artifact_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cross_refs_source ON cross_refs(source_id);
        CREATE INDEX IF NOT EXISTS idx_cross_refs_target ON cross_refs(target_id);
        CREATE INDEX IF NOT EXISTS idx_cross_refs_similarity ON cross_refs(similarity);

        CREATE TABLE IF NOT EXISTS build_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            build_timestamp TEXT NOT NULL,
            artifacts_count INTEGER NOT NULL,
            refs_count INTEGER NOT NULL,
            threshold REAL NOT NULL,
            top_k_per_artifact INTEGER NOT NULL,
            duration_seconds REAL,
            receipt_hash TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


def _load_artifacts_from_index(
    db_path: Path,
    artifact_type: str,
    id_col: str,
    path_col: str,
    embedding_col: str,
    model_col: str,
    table_name: str,
    metadata_cols: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Load artifacts and embeddings from a specific index database.

    Args:
        db_path: Path to the index database
        artifact_type: Type label (e.g., "canon", "adr", "skill")
        id_col: Column name for artifact ID
        path_col: Column name for file path
        embedding_col: Column name for embedding blob
        model_col: Column name for model identifier
        table_name: Table name to query
        metadata_cols: Optional list of additional metadata columns

    Returns:
        List of artifact dicts with id, type, path, embedding, model, metadata
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Build query
    cols = [id_col, path_col, embedding_col, model_col]
    if metadata_cols:
        cols.extend(metadata_cols)

    query = f"""
        SELECT {', '.join(cols)}
        FROM {table_name}
        WHERE {embedding_col} IS NOT NULL
    """

    cursor = conn.execute(query)
    rows = cursor.fetchall()
    conn.close()

    artifacts = []
    for row in rows:
        artifact_id = f"{artifact_type}:{row[path_col]}"
        metadata = {}
        if metadata_cols:
            for col in metadata_cols:
                metadata[col] = row[col]

        artifacts.append({
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "artifact_path": row[path_col],
            "embedding_blob": row[embedding_col],
            "model_name": row[model_col] if row[model_col] else "unknown",
            "metadata": metadata,
        })

    return artifacts


def _load_all_artifacts() -> list[dict[str, Any]]:
    """
    Load all artifacts from all index databases.

    Returns:
        List of artifact dicts from canon, adr, and skill indices
    """
    artifacts = []

    # Load canon files
    canon_artifacts = _load_artifacts_from_index(
        _get_canon_db_path(),
        artifact_type="canon",
        id_col="id",
        path_col="file_path",
        embedding_col="embedding",
        model_col="model_id",
        table_name="canon_records",
        metadata_cols=["tags", "trust"],
    )
    artifacts.extend(canon_artifacts)

    # Load ADRs
    adr_artifacts = _load_artifacts_from_index(
        _get_adr_db_path(),
        artifact_type="adr",
        id_col="id",
        path_col="file_path",
        embedding_col="embedding",
        model_col="model_id",
        table_name="adr_records",
        metadata_cols=["adr_id", "title", "status", "confidence", "impact"],
    )
    artifacts.extend(adr_artifacts)

    # Load skills (requires JOIN between skills and embeddings tables)
    skill_db = _get_skill_db_path()
    if skill_db.exists():
        conn = sqlite3.connect(str(skill_db))
        conn.row_factory = sqlite3.Row

        query = """
            SELECT s.skill_id, s.skill_path, e.embedding_blob, e.model_name, s.metadata_json
            FROM embeddings e
            JOIN skills s ON e.skill_id = s.skill_id
            WHERE e.embedding_blob IS NOT NULL
        """

        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            artifact_id = f"skill:{row['skill_path']}"
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

            artifacts.append({
                "artifact_id": artifact_id,
                "artifact_type": "skill",
                "artifact_path": row["skill_path"],
                "embedding_blob": row["embedding_blob"],
                "model_name": row["model_name"] if row["model_name"] else "unknown",
                "metadata": metadata,
            })

    else:
        # No skill index available
        pass

    return artifacts


def _compute_pairwise_similarities(
    artifacts: list[dict[str, Any]],
    threshold: float,
    top_k_per_artifact: int,
) -> list[tuple[str, str, float]]:
    """
    Compute pairwise similarities between all artifacts.

    Args:
        artifacts: List of artifact dicts with embedding_blob
        threshold: Minimum similarity to store
        top_k_per_artifact: Maximum edges per source artifact

    Returns:
        List of (source_id, target_id, similarity) tuples
    """
    import numpy as np
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    if not artifacts:
        return []

    engine = EmbeddingEngine()

    # Deserialize all embeddings
    embeddings = []
    artifact_ids = []
    for artifact in artifacts:
        try:
            embedding = np.frombuffer(artifact["embedding_blob"], dtype=np.float32)
            embeddings.append(embedding)
            artifact_ids.append(artifact["artifact_id"])
        except Exception:
            continue

    if not embeddings:
        return []

    embeddings_matrix = np.array(embeddings)

    # Compute all pairwise similarities
    cross_refs = []
    for i, source_id in enumerate(artifact_ids):
        # Compute similarities to all other artifacts
        similarities = engine.batch_similarity(embeddings_matrix[i], embeddings_matrix)

        # Create (similarity, target_id) pairs, excluding self
        candidates = [
            (sim, artifact_ids[j])
            for j, sim in enumerate(similarities)
            if i != j and sim >= threshold
        ]

        # Sort by similarity descending, then by target_id ascending for determinism
        candidates.sort(key=lambda x: (-x[0], x[1]))

        # Take top_k
        top_candidates = candidates[:top_k_per_artifact]

        # Add to cross_refs
        for similarity, target_id in top_candidates:
            cross_refs.append((source_id, target_id, float(similarity)))

    return cross_refs


def build_cross_refs(
    db_path: Optional[Path] = None,
    *,
    threshold: float = 0.3,
    top_k_per_artifact: int = 10,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    """
    Build cross-reference graph across all indexed artifacts.

    Args:
        db_path: Override default database path
        threshold: Minimum similarity to store (default: 0.3)
        top_k_per_artifact: Maximum edges per source (default: 10)
        emit_receipt: Include receipt in result

    Returns:
        Result dict with:
        - artifacts_count: int
        - refs_count: int
        - threshold: float
        - top_k_per_artifact: int
        - duration_seconds: float
        - receipt: (if emit_receipt) operation receipt
    """
    import time

    start_time = time.time()

    db = db_path or _get_index_db_path()
    conn = _init_db(db)

    # Load all artifacts from all indices
    artifacts = _load_all_artifacts()

    if not artifacts:
        conn.close()
        raise ValueError("No artifacts found in any index. Run embedding first.")

    # Clear existing artifacts and cross_refs
    conn.execute("DELETE FROM artifacts")
    conn.execute("DELETE FROM cross_refs")

    # Insert artifacts
    now = _now_iso()
    for artifact in artifacts:
        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts
            (artifact_id, artifact_type, artifact_path, embedding_blob, model_name,
             metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact["artifact_id"],
                artifact["artifact_type"],
                artifact["artifact_path"],
                artifact["embedding_blob"],
                artifact["model_name"],
                json.dumps(artifact["metadata"]),
                now,
                now,
            ),
        )

    conn.commit()

    # Compute pairwise similarities
    cross_refs = _compute_pairwise_similarities(artifacts, threshold, top_k_per_artifact)

    # Insert cross-references
    for source_id, target_id, similarity in cross_refs:
        conn.execute(
            """
            INSERT OR REPLACE INTO cross_refs
            (source_id, target_id, similarity, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_id, target_id, similarity, now),
        )

    conn.commit()

    duration = time.time() - start_time

    # Record build history
    receipt = {
        "operation": "build_cross_refs",
        "timestamp": now,
        "artifacts_count": len(artifacts),
        "refs_count": len(cross_refs),
        "threshold": threshold,
        "top_k_per_artifact": top_k_per_artifact,
        "duration_seconds": round(duration, 3),
    }
    receipt_hash = _hash_text(json.dumps(receipt, sort_keys=True))
    receipt["receipt_hash"] = receipt_hash

    conn.execute(
        """
        INSERT INTO build_history
        (build_timestamp, artifacts_count, refs_count, threshold, top_k_per_artifact,
         duration_seconds, receipt_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (now, len(artifacts), len(cross_refs), threshold, top_k_per_artifact, duration, receipt_hash),
    )

    conn.commit()
    conn.close()

    result = {
        "artifacts_count": len(artifacts),
        "refs_count": len(cross_refs),
        "threshold": threshold,
        "top_k_per_artifact": top_k_per_artifact,
        "duration_seconds": round(duration, 3),
    }

    if emit_receipt:
        result["receipt"] = receipt

    return result


def find_related(
    artifact_id: str,
    db_path: Optional[Path] = None,
    *,
    top_k: int = 5,
    threshold: Optional[float] = None,
) -> dict[str, Any]:
    """
    Find related artifacts by embedding similarity.

    Args:
        artifact_id: Artifact identifier (format: "type:path")
        db_path: Override default database path
        top_k: Maximum results to return (default: 5)
        threshold: Optional minimum similarity filter

    Returns:
        Result dict with:
        - artifact_id: str
        - related: [{artifact_id, artifact_type, artifact_path, similarity, metadata}]
        - total_candidates: int
    """
    db = db_path or _get_index_db_path()

    if not db.exists():
        raise FileNotFoundError(f"Cross-reference index not found: {db}")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # Build query with optional threshold
    query = """
        SELECT
            cr.target_id,
            cr.similarity,
            a.artifact_type,
            a.artifact_path,
            a.metadata_json
        FROM cross_refs cr
        JOIN artifacts a ON cr.target_id = a.artifact_id
        WHERE cr.source_id = ?
    """
    params = [artifact_id]

    if threshold is not None:
        query += " AND cr.similarity >= ?"
        params.append(threshold)

    query += " ORDER BY cr.similarity DESC, cr.target_id ASC LIMIT ?"
    params.append(top_k)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    # Get total candidate count
    count_query = "SELECT COUNT(*) as count FROM cross_refs WHERE source_id = ?"
    count_params = [artifact_id]
    if threshold is not None:
        count_query += " AND similarity >= ?"
        count_params.append(threshold)

    count_cursor = conn.execute(count_query, count_params)
    total_candidates = count_cursor.fetchone()["count"]

    conn.close()

    # Format results
    related = []
    for row in rows:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        related.append({
            "artifact_id": row["target_id"],
            "artifact_type": row["artifact_type"],
            "artifact_path": row["artifact_path"],
            "similarity": row["similarity"],
            "metadata": metadata,
        })

    return {
        "artifact_id": artifact_id,
        "related": related,
        "total_candidates": total_candidates,
    }


def get_cross_ref_stats(db_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Get statistics about the cross-reference index.

    Args:
        db_path: Override default database path

    Returns:
        Stats dict with:
        - exists: bool
        - artifacts_by_type: {type: count}
        - total_artifacts: int
        - total_refs: int
        - avg_refs_per_artifact: float
        - last_build: {timestamp, artifacts_count, refs_count, threshold, duration}
    """
    db = db_path or _get_index_db_path()

    if not db.exists():
        return {"exists": False}

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # Count artifacts by type
    cursor = conn.execute("""
        SELECT artifact_type, COUNT(*) as count
        FROM artifacts
        GROUP BY artifact_type
    """)
    artifacts_by_type = {row["artifact_type"]: row["count"] for row in cursor.fetchall()}

    # Total artifacts
    cursor = conn.execute("SELECT COUNT(*) as count FROM artifacts")
    total_artifacts = cursor.fetchone()["count"]

    # Total refs
    cursor = conn.execute("SELECT COUNT(*) as count FROM cross_refs")
    total_refs = cursor.fetchone()["count"]

    # Average refs per artifact
    avg_refs = total_refs / total_artifacts if total_artifacts > 0 else 0.0

    # Last build info
    cursor = conn.execute("""
        SELECT build_timestamp, artifacts_count, refs_count, threshold,
               top_k_per_artifact, duration_seconds
        FROM build_history
        ORDER BY id DESC
        LIMIT 1
    """)
    last_build_row = cursor.fetchone()
    last_build = dict(last_build_row) if last_build_row else None

    conn.close()

    return {
        "exists": True,
        "artifacts_by_type": artifacts_by_type,
        "total_artifacts": total_artifacts,
        "total_refs": total_refs,
        "avg_refs_per_artifact": round(avg_refs, 2),
        "last_build": last_build,
    }


def rebuild_cross_ref_index(
    db_path: Optional[Path] = None,
    *,
    threshold: float = 0.3,
    top_k_per_artifact: int = 10,
) -> dict[str, Any]:
    """
    Rebuild the entire cross-reference index from scratch.

    This is equivalent to build_cross_refs but explicitly named for clarity.

    Args:
        db_path: Override default database path
        threshold: Minimum similarity to store
        top_k_per_artifact: Maximum edges per source

    Returns:
        Result dict from build_cross_refs
    """
    return build_cross_refs(
        db_path=db_path,
        threshold=threshold,
        top_k_per_artifact=top_k_per_artifact,
        emit_receipt=True,
    )
