#!/usr/bin/env python3
"""
Skill Index Primitive - Phase 5.1.4

Indexes all skill files (CAPABILITY/SKILLS/*/SKILL.md) with:
- MemoryRecord instances for each skill
- Vector embeddings via CORTEX EmbeddingEngine
- Deterministic manifest with content hashes
- Semantic search by skill intent/purpose
- Receipted operations

Usage:
    from CAPABILITY.PRIMITIVES.skill_index import (
        inventory_skills,
        embed_skills,
        search_skills,
        find_skills_by_intent,
        rebuild_index,
    )

    # Create inventory
    manifest = inventory_skills()

    # Embed all skills
    result = embed_skills()

    # Search skills by semantic query
    results = find_skills_by_intent("verify canon changes", top_k=5)
"""

from __future__ import annotations

import hashlib
import json
import re
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


def _get_skills_root() -> Path:
    """Get the CAPABILITY/SKILLS directory path."""
    return REPO_ROOT / "CAPABILITY" / "SKILLS"


def _get_index_db_path() -> Path:
    """Get the skill index database path."""
    return REPO_ROOT / "NAVIGATION" / "CORTEX" / "db" / "skill_index.db"


def _parse_frontmatter(content: str) -> dict[str, Any]:
    """
    Parse YAML frontmatter from skill markdown.

    Returns:
        dict with frontmatter fields, or empty dict if no frontmatter
    """
    # Match YAML frontmatter (between --- markers)
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return {}

    yaml_text = match.group(1)
    result = {}

    # Simple YAML parser for key: value pairs
    for line in yaml_text.split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            result[key] = value

    return result


def _parse_skill_metadata(content: str) -> dict[str, Any]:
    """
    Parse skill metadata from SKILL.md content.

    Extracts:
    - Frontmatter (name, version, description, compatibility)
    - Title from first # heading
    - Purpose/Trigger/Inputs/Outputs sections

    Returns:
        dict with parsed metadata
    """
    metadata: dict[str, Any] = {}

    # Parse frontmatter
    frontmatter = _parse_frontmatter(content)
    metadata['frontmatter'] = frontmatter

    # Extract name from frontmatter or heading
    if 'name' in frontmatter:
        metadata['name'] = frontmatter['name']
    else:
        # Try to extract from "# Skill: skill_name" heading
        title_match = re.search(r'^#\s+(?:Skill:\s*)?(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['name'] = title_match.group(1).strip()

    # Extract version
    if 'version' in frontmatter:
        metadata['version'] = frontmatter['version']
    else:
        version_match = re.search(r'\*\*Version:\*\*\s*(.+)', content)
        if version_match:
            metadata['version'] = version_match.group(1).strip()

    # Extract description
    if 'description' in frontmatter:
        metadata['description'] = frontmatter['description']

    # Extract purpose section
    purpose_match = re.search(
        r'##\s+(?:Purpose|What it (?:does|checks))\s*\n(.+?)(?=\n##|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    if purpose_match:
        metadata['purpose'] = purpose_match.group(1).strip()

    # Extract trigger section
    trigger_match = re.search(
        r'##\s+Trigger\s*\n(.+?)(?=\n##|\Z)',
        content,
        re.DOTALL
    )
    if trigger_match:
        metadata['trigger'] = trigger_match.group(1).strip()

    # Extract inputs section
    inputs_match = re.search(
        r'##\s+(?:Inputs?|Input Schema)\s*\n(.+?)(?=\n##|\Z)',
        content,
        re.DOTALL
    )
    if inputs_match:
        metadata['inputs'] = inputs_match.group(1).strip()

    # Extract outputs section
    outputs_match = re.search(
        r'##\s+Outputs?\s*\n(.+?)(?=\n##|\Z)',
        content,
        re.DOTALL
    )
    if outputs_match:
        metadata['outputs'] = outputs_match.group(1).strip()

    # Extract usage section
    usage_match = re.search(
        r'##\s+Usage\s*\n(.+?)(?=\n##|\Z)',
        content,
        re.DOTALL
    )
    if usage_match:
        metadata['usage'] = usage_match.group(1).strip()

    return metadata


def _build_searchable_text(metadata: dict[str, Any], skill_path: str) -> str:
    """
    Build searchable text from skill metadata for embedding.

    Combines name, description, purpose, trigger, and other metadata
    into a coherent text for semantic search.
    """
    parts = []

    # Add skill name/title
    if 'name' in metadata:
        parts.append(f"Skill: {metadata['name']}")

    # Add skill path for context
    parts.append(f"Path: {skill_path}")

    # Add description
    if 'description' in metadata:
        parts.append(f"Description: {metadata['description']}")

    # Add purpose
    if 'purpose' in metadata:
        parts.append(f"Purpose: {metadata['purpose']}")

    # Add trigger
    if 'trigger' in metadata:
        parts.append(f"Trigger: {metadata['trigger']}")

    # Add inputs summary
    if 'inputs' in metadata:
        # Extract first 200 chars of inputs for context
        inputs_text = metadata['inputs'][:200]
        parts.append(f"Inputs: {inputs_text}")

    # Add outputs summary
    if 'outputs' in metadata:
        # Extract first 200 chars of outputs for context
        outputs_text = metadata['outputs'][:200]
        parts.append(f"Outputs: {outputs_text}")

    return "\n\n".join(parts)


def inventory_skills(
    skills_root: Optional[Path] = None,
    *,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    """
    Create inventory of all skill files with content hashes.

    Args:
        skills_root: Override default skills root path
        emit_receipt: Include receipt in result

    Returns:
        Manifest dict with:
        - skills: {skill_id: {path, sha256, size, metadata}}
        - total_skills: int
        - total_bytes: int
        - manifest_hash: sha256 of manifest
        - receipt: (if emit_receipt) creation receipt
    """
    root = skills_root or _get_skills_root()

    if not root.exists():
        raise FileNotFoundError(f"Skills root not found: {root}")

    skills: dict[str, dict[str, Any]] = {}
    total_bytes = 0

    # Enumerate all SKILL.md files
    for skill_md in sorted(root.rglob("SKILL.md")):
        # Get skill directory name
        skill_dir = skill_md.parent
        skill_id = str(skill_dir.relative_to(root)).replace("\\", "/")

        # Skip template
        if "_TEMPLATE" in skill_id:
            continue

        # Read and hash content
        content = skill_md.read_bytes()
        content_hash = _hash_content(content)

        # Parse metadata
        content_str = content.decode("utf-8", errors="ignore")
        metadata = _parse_skill_metadata(content_str)

        # Use relative path from skills root for portability
        skills[skill_id] = {
            "path": str(skill_md.relative_to(root)).replace("\\", "/"),
            "sha256": content_hash,
            "size": len(content),
            "metadata": metadata,
        }

        total_bytes += len(content)

    # Create manifest
    manifest = {
        "skills": skills,
        "total_skills": len(skills),
        "total_bytes": total_bytes,
    }

    # Hash the manifest itself
    manifest_json = json.dumps(manifest, sort_keys=True, indent=2)
    manifest["manifest_hash"] = _hash_text(manifest_json)

    # Add receipt
    if emit_receipt:
        manifest["receipt"] = {
            "timestamp": _now_iso(),
            "operation": "inventory_skills",
            "total_skills": len(skills),
            "manifest_hash": manifest["manifest_hash"],
        }

    return manifest


def _init_db(db_path: Path) -> None:
    """Initialize the skill index database schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Skills table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            skill_id TEXT PRIMARY KEY,
            skill_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            content_size INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            searchable_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Embeddings table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            skill_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            embedding_blob BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (skill_id, model_name),
            FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
        )
    """)

    # Create indices for fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_skills_hash ON skills(content_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name)")

    conn.commit()
    conn.close()


def embed_skills(
    skills_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    force_rebuild: bool = False,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    """
    Embed all skills and store in index database.

    Args:
        skills_root: Override default skills root path
        db_path: Override default database path
        model_name: Embedding model to use
        force_rebuild: Rebuild index even if it exists
        emit_receipt: Include receipt in result

    Returns:
        Result dict with:
        - embedded_count: int
        - skipped_count: int
        - total_skills: int
        - model_name: str
        - db_path: str
        - receipt: (if emit_receipt) operation receipt
    """
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    db = db_path or _get_index_db_path()

    # Initialize database
    _init_db(db)

    # Create inventory
    manifest = inventory_skills(skills_root, emit_receipt=False)
    skills = manifest["skills"]

    # Initialize embedding engine
    engine = EmbeddingEngine()

    # Connect to database
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    embedded_count = 0
    skipped_count = 0

    for skill_id, skill_info in skills.items():
        # Check if already embedded (unless force_rebuild)
        if not force_rebuild:
            cur.execute(
                "SELECT 1 FROM embeddings WHERE skill_id = ? AND model_name = ?",
                (skill_id, model_name)
            )
            if cur.fetchone():
                skipped_count += 1
                continue

        # Build searchable text
        searchable_text = _build_searchable_text(
            skill_info["metadata"],
            skill_info["path"]
        )

        # Generate embedding
        embedding = engine.embed(searchable_text)
        embedding_bytes = embedding.tobytes()

        timestamp = _now_iso()

        # Insert or update skill record
        cur.execute("""
            INSERT OR REPLACE INTO skills
            (skill_id, skill_path, content_hash, content_size, metadata_json,
             searchable_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            skill_id,
            skill_info["path"],
            skill_info["sha256"],
            skill_info["size"],
            json.dumps(skill_info["metadata"], sort_keys=True),
            searchable_text,
            timestamp,
            timestamp,
        ))

        # Insert or update embedding
        cur.execute("""
            INSERT OR REPLACE INTO embeddings
            (skill_id, model_name, embedding_blob, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            skill_id,
            model_name,
            embedding_bytes,
            timestamp,
        ))

        embedded_count += 1

    conn.commit()
    conn.close()

    # Build result
    result = {
        "embedded_count": embedded_count,
        "skipped_count": skipped_count,
        "total_skills": len(skills),
        "model_name": model_name,
        "db_path": str(db),
    }

    # Add receipt
    if emit_receipt:
        result["receipt"] = {
            "timestamp": _now_iso(),
            "operation": "embed_skills",
            "embedded_count": embedded_count,
            "skipped_count": skipped_count,
            "model_name": model_name,
        }

    return result


def find_skills_by_intent(
    query: str,
    top_k: int = 5,
    db_path: Optional[Path] = None,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    threshold: Optional[float] = None,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    """
    Find skills by semantic similarity to intent query.

    Args:
        query: Natural language intent/query
        top_k: Number of results to return
        db_path: Override default database path
        model_name: Embedding model to use
        threshold: Minimum similarity score (0-1)
        emit_receipt: Include receipt in result

    Returns:
        Result dict with:
        - query: str
        - results: [{skill_id, score, metadata, path}]
        - total_candidates: int
        - receipt: (if emit_receipt) operation receipt
    """
    import numpy as np
    from NAVIGATION.CORTEX.semantic.embeddings import EmbeddingEngine

    db = db_path or _get_index_db_path()

    if not db.exists():
        raise FileNotFoundError(f"Skill index not found: {db}")

    # Generate query embedding
    engine = EmbeddingEngine()
    query_embedding = engine.embed(query)

    # Connect to database
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    # Fetch all embeddings for the specified model
    cur.execute("""
        SELECT e.skill_id, e.embedding_blob, s.metadata_json, s.skill_path
        FROM embeddings e
        JOIN skills s ON e.skill_id = s.skill_id
        WHERE e.model_name = ?
    """, (model_name,))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {
            "query": query,
            "results": [],
            "total_candidates": 0,
        }

    # Compute similarities
    similarities = []
    for skill_id, embedding_bytes, metadata_json, skill_path in rows:
        # Reconstruct embedding
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        # Compute cosine similarity
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )

        similarities.append({
            "skill_id": skill_id,
            "score": float(similarity),
            "metadata": json.loads(metadata_json),
            "path": skill_path,
        })

    # Sort by similarity (descending), then by skill_id (ascending) for determinism
    similarities.sort(key=lambda x: (-x["score"], x["skill_id"]))

    # Apply threshold filter if specified
    if threshold is not None:
        similarities = [s for s in similarities if s["score"] >= threshold]

    # Take top-K
    results = similarities[:top_k]

    # Build result
    result = {
        "query": query,
        "results": results,
        "total_candidates": len(rows),
    }

    # Add receipt
    if emit_receipt:
        result["receipt"] = {
            "timestamp": _now_iso(),
            "operation": "find_skills_by_intent",
            "query": query,
            "top_k": top_k,
            "results_count": len(results),
            "model_name": model_name,
        }

    return result


def search_skills(
    query: str,
    top_k: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias for find_skills_by_intent for consistency with other indexes."""
    return find_skills_by_intent(query, top_k, **kwargs)


def rebuild_index(
    skills_root: Optional[Path] = None,
    db_path: Optional[Path] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """
    Rebuild the skill index from scratch.

    Args:
        skills_root: Override default skills root path
        db_path: Override default database path
        model_name: Embedding model to use

    Returns:
        Result dict from embed_skills
    """
    db = db_path or _get_index_db_path()

    # Remove existing database
    if db.exists():
        db.unlink()

    # Rebuild
    return embed_skills(
        skills_root=skills_root,
        db_path=db,
        model_name=model_name,
        force_rebuild=True,
        emit_receipt=True,
    )


def get_skill_by_id(
    skill_id: str,
    db_path: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """
    Get skill metadata by skill ID.

    Args:
        skill_id: Skill identifier (e.g., "governance/canon-governance-check")
        db_path: Override default database path

    Returns:
        Skill dict with metadata, or None if not found
    """
    db = db_path or _get_index_db_path()

    if not db.exists():
        return None

    conn = sqlite3.connect(db)
    cur = conn.cursor()

    cur.execute("""
        SELECT skill_id, skill_path, content_hash, content_size,
               metadata_json, searchable_text
        FROM skills
        WHERE skill_id = ?
    """, (skill_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "skill_id": row[0],
        "path": row[1],
        "content_hash": row[2],
        "size": row[3],
        "metadata": json.loads(row[4]),
        "searchable_text": row[5],
    }


def list_all_skills(
    db_path: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """
    List all indexed skills.

    Args:
        db_path: Override default database path

    Returns:
        List of skill dicts with metadata
    """
    db = db_path or _get_index_db_path()

    if not db.exists():
        return []

    conn = sqlite3.connect(db)
    cur = conn.cursor()

    cur.execute("""
        SELECT skill_id, skill_path, metadata_json
        FROM skills
        ORDER BY skill_id
    """)

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "skill_id": row[0],
            "path": row[1],
            "metadata": json.loads(row[2]),
        }
        for row in rows
    ]


if __name__ == "__main__":
    # CLI for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python skill_index.py inventory")
        print("  python skill_index.py embed")
        print("  python skill_index.py search <query>")
        print("  python skill_index.py rebuild")
        sys.exit(1)

    command = sys.argv[1]

    if command == "inventory":
        manifest = inventory_skills()
        print(json.dumps(manifest, indent=2))

    elif command == "embed":
        result = embed_skills()
        print(json.dumps(result, indent=2))

    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: search requires a query")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        result = find_skills_by_intent(query)
        print(json.dumps(result, indent=2))

    elif command == "rebuild":
        result = rebuild_index()
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
