#!/usr/bin/env python3
"""
MemoryRecord Primitive - Phase 5.0 Foundation

Canonical data structure for all vector-indexed content.

Contract rules:
- Text is canonical (source of truth)
- Vectors are derived (rebuildable from text)
- All exports are receipted and hashed

Usage:
    from CAPABILITY.PRIMITIVES.memory_record import (
        create_record, validate_record, hash_record, MemoryRecord
    )

    # Create a record
    record = create_record(
        text="Some governance content",
        doc_path="LAW/CANON/CONSTITUTION/CONTRACT.md",
        tags=["canon", "constitution"]
    )

    # Validate a record
    verdict = validate_record(record)
    if verdict["valid"]:
        print("Record is valid")

    # Hash a record (deterministic)
    content_hash = hash_record(record)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict


# Schema path for validation
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "LAW" / "SCHEMAS" / "memory_record.schema.json"


class EmbeddingEntry(TypedDict, total=False):
    """Single embedding with metadata."""
    vector: list[float]
    model: str
    dimensions: int
    quantization: str
    generated_at: str


class Payload(TypedDict, total=False):
    """Metadata: tags, timestamps, roles, doc_ids."""
    tags: list[str]
    created_at: str
    updated_at: str
    roles: list[str]
    doc_id: str
    doc_path: str
    chunk_index: int
    chunk_count: int
    content_type: str
    language: str
    extra: dict[str, Any]


class Scores(TypedDict, total=False):
    """Ranking scores: ELO, recency, trust, decay."""
    elo: float
    recency: float
    trust: float
    decay: float
    relevance: float
    citation_count: int


class Lineage(TypedDict, total=False):
    """Derivation chain and summarization history."""
    derived_from: list[str]
    summarized_from: list[str]
    version: int
    supersedes: str
    superseded_by: str
    merge_sources: list[str]


class EmbeddingReceipt(TypedDict, total=False):
    """Per-model embedding generation receipt."""
    model_version: str
    generated_at: str
    input_hash: str
    output_hash: str


class ValidationReceipt(TypedDict, total=False):
    """Validation receipt."""
    validator: str
    version: str
    validated_at: str
    verdict: str


class Receipts(TypedDict, total=False):
    """Provenance hashes and tool version refs."""
    content_hash: str
    created_at: str
    created_by: str
    tool_version: str
    embedding_receipts: dict[str, EmbeddingReceipt]
    validation_receipts: list[ValidationReceipt]


class MemoryRecord(TypedDict):
    """Canonical vector memory record."""
    id: str
    text: str
    embeddings: dict[str, EmbeddingEntry]
    payload: Payload
    scores: Scores
    lineage: Lineage
    receipts: Receipts


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_record(
    text: str,
    *,
    doc_path: Optional[str] = None,
    doc_id: Optional[str] = None,
    tags: Optional[list[str]] = None,
    roles: Optional[list[str]] = None,
    chunk_index: Optional[int] = None,
    chunk_count: Optional[int] = None,
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    elo: Optional[float] = None,
    trust: float = 1.0,
    created_by: str = "memory_record.py",
    tool_version: str = "0.1.0",
) -> MemoryRecord:
    """
    Create a new MemoryRecord from text content.

    The id is derived deterministically from the text content (SHA-256).
    Embeddings start empty and are populated by embedding functions.

    Args:
        text: The canonical text content (required)
        doc_path: Source file path (repo-relative)
        doc_id: Parent document identifier
        tags: Classification tags
        roles: Access control roles
        chunk_index: Position within parent document
        chunk_count: Total chunks in parent document
        content_type: MIME type or content classification
        language: ISO 639-1 language code
        extra: Extension metadata
        elo: Initial ELO score
        trust: Trust score (0-1, default 1.0)
        created_by: Tool/agent identifier
        tool_version: Version of creation tool

    Returns:
        A valid MemoryRecord ready for embedding and storage
    """
    now = _now_iso()
    content_hash = _hash_text(text)

    # Build payload
    payload: Payload = {
        "created_at": now,
        "updated_at": now,
    }
    if tags:
        payload["tags"] = tags
    if roles:
        payload["roles"] = roles
    if doc_id:
        payload["doc_id"] = doc_id
    if doc_path:
        payload["doc_path"] = doc_path
    if chunk_index is not None:
        payload["chunk_index"] = chunk_index
    if chunk_count is not None:
        payload["chunk_count"] = chunk_count
    if content_type:
        payload["content_type"] = content_type
    if language:
        payload["language"] = language
    if extra:
        payload["extra"] = extra

    # Build scores
    scores: Scores = {
        "recency": 1.0,
        "trust": trust,
        "decay": 1.0,
        "citation_count": 0,
    }
    if elo is not None:
        scores["elo"] = elo

    # Build lineage
    lineage: Lineage = {
        "version": 1,
    }

    # Build receipts
    receipts: Receipts = {
        "content_hash": content_hash,
        "created_at": now,
        "created_by": created_by,
        "tool_version": tool_version,
    }

    record: MemoryRecord = {
        "id": content_hash,
        "text": text,
        "embeddings": {},
        "payload": payload,
        "scores": scores,
        "lineage": lineage,
        "receipts": receipts,
    }

    return record


def hash_record(record: MemoryRecord) -> str:
    """
    Compute the content hash of a record.

    Hash is derived from the text field only (text is canonical).
    This ensures deterministic hashing regardless of other fields.

    Args:
        record: The MemoryRecord to hash

    Returns:
        SHA-256 hex digest of the text content
    """
    return _hash_text(record["text"])


def validate_record(
    record: dict[str, Any],
    *,
    check_hash: bool = True,
    check_schema: bool = True,
) -> dict[str, Any]:
    """
    Validate a MemoryRecord.

    Checks:
    1. Required fields present
    2. Hash consistency (id matches content hash)
    3. Schema validation (if jsonschema available)

    Args:
        record: The record to validate
        check_hash: Verify id matches content hash
        check_schema: Run JSON schema validation

    Returns:
        Verdict dict with: valid (bool), errors (list), warnings (list)
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required fields
    required = ["id", "text", "embeddings", "payload", "scores", "lineage", "receipts"]
    for field in required:
        if field not in record:
            errors.append(f"Missing required field: {field}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Check text is non-empty
    if not record.get("text"):
        errors.append("Text field must be non-empty")

    # Check hash consistency
    if check_hash and "text" in record:
        computed_hash = _hash_text(record["text"])
        if record.get("id") != computed_hash:
            errors.append(f"Hash mismatch: id={record.get('id')}, computed={computed_hash}")

        # Also check receipts.content_hash
        receipts = record.get("receipts", {})
        if receipts.get("content_hash") and receipts["content_hash"] != computed_hash:
            errors.append(f"Receipt hash mismatch: content_hash={receipts['content_hash']}, computed={computed_hash}")

    # Check embeddings structure
    embeddings = record.get("embeddings", {})
    if not isinstance(embeddings, dict):
        errors.append("Embeddings must be an object")
    else:
        for name, entry in embeddings.items():
            if not isinstance(entry, dict):
                errors.append(f"Embedding '{name}' must be an object")
                continue
            if "vector" not in entry:
                errors.append(f"Embedding '{name}' missing vector")
            elif not isinstance(entry.get("vector"), list):
                errors.append(f"Embedding '{name}' vector must be an array")
            if "model" not in entry:
                errors.append(f"Embedding '{name}' missing model")
            if "dimensions" not in entry:
                warnings.append(f"Embedding '{name}' missing dimensions")

    # Check scores
    scores = record.get("scores", {})
    for score_name in ["recency", "trust", "decay"]:
        if score_name in scores:
            val = scores[score_name]
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"Score '{score_name}' must be between 0 and 1")

    # Schema validation (optional, requires jsonschema)
    if check_schema:
        try:
            import jsonschema
            if SCHEMA_PATH.exists():
                with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                try:
                    jsonschema.validate(record, schema)
                except jsonschema.ValidationError as e:
                    errors.append(f"Schema validation failed: {e.message}")
        except ImportError:
            warnings.append("jsonschema not installed, skipping schema validation")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def add_embedding(
    record: MemoryRecord,
    name: str,
    vector: list[float],
    model: str,
    *,
    quantization: str = "float32",
) -> MemoryRecord:
    """
    Add an embedding to a record.

    Embeddings are derived from text. This function adds a named embedding
    with proper metadata and receipts.

    Args:
        record: The record to update (mutated in place)
        name: Embedding name (e.g., "default", "text-embedding-3-small")
        vector: The embedding vector
        model: Model identifier
        quantization: Vector format (float32, float16, int8)

    Returns:
        The updated record (same reference, mutated)
    """
    now = _now_iso()

    record["embeddings"][name] = {
        "vector": vector,
        "model": model,
        "dimensions": len(vector),
        "quantization": quantization,
        "generated_at": now,
    }

    # Add embedding receipt
    if "embedding_receipts" not in record["receipts"]:
        record["receipts"]["embedding_receipts"] = {}

    record["receipts"]["embedding_receipts"][name] = {
        "model_version": model,
        "generated_at": now,
        "input_hash": record["id"],
    }

    # Update updated_at
    record["payload"]["updated_at"] = now

    return record


def to_json(record: MemoryRecord, *, indent: Optional[int] = None) -> str:
    """
    Serialize a MemoryRecord to canonical JSON.

    Uses sorted keys and no extra whitespace for deterministic output.

    Args:
        record: The record to serialize
        indent: Optional indentation for pretty printing

    Returns:
        JSON string
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":") if indent is None else (", ", ": "), indent=indent)


def from_json(json_str: str) -> MemoryRecord:
    """
    Deserialize a MemoryRecord from JSON.

    Args:
        json_str: JSON string

    Returns:
        MemoryRecord dict
    """
    return json.loads(json_str)


def canonical_bytes(record: MemoryRecord) -> bytes:
    """
    Get canonical byte representation for hashing.

    Uses sorted keys and minimal whitespace for determinism.

    Args:
        record: The record to serialize

    Returns:
        UTF-8 encoded bytes
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def full_hash(record: MemoryRecord) -> str:
    """
    Compute hash of the entire record (not just text).

    Useful for detecting any changes to the record.

    Args:
        record: The record to hash

    Returns:
        SHA-256 hex digest of canonical JSON
    """
    return hashlib.sha256(canonical_bytes(record)).hexdigest()
