"""
Canonical JSON serialization primitives.

Single source of truth for deterministic JSON encoding used across
all receipt, proof, and manifest modules in the governance system.

Rules:
- Keys sorted lexicographically
- No extra whitespace (separators are ',' and ':')
- UTF-8 safe (ensure_ascii=False)
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Convert object to canonical JSON bytes (UTF-8)."""
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Compute SHA-256 hex digest (64 lowercase chars)."""
    return hashlib.sha256(data).hexdigest()
