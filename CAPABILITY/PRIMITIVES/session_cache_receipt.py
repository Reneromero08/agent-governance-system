#!/usr/bin/env python3
"""
SessionCacheReceipt Primitive (L4 Session Cache)

Receipt for session cache operations. Every cache operation emits a receipt
for provenance tracking and compression validation.

Operations:
- HIT: Cache hit - expansion served from cache
- MISS: Cache miss - full resolution required
- PUT: Expansion stored in cache
- INVALIDATE: Cache entry invalidated

Usage:
    from CAPABILITY.PRIMITIVES.session_cache_receipt import (
        SessionCacheReceipt, create_cache_receipt, VALID_CACHE_OPERATIONS
    )

    # Create a hit receipt
    receipt = create_cache_receipt(
        session_id="agent-001-20260118",
        operation="HIT",
        pointer="C3",
        expansion_hash="abc123...",
        tokens_cold=52,
        codebook_hash="def456...",
    )

    print(f"Tokens saved: {receipt.tokens_saved}")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Schema version
SCHEMA_VERSION = "1.0.0"

# Valid cache operation types
VALID_CACHE_OPERATIONS = frozenset([
    "HIT",         # Cache hit - expansion served from cache
    "MISS",        # Cache miss - full resolution required
    "PUT",         # Expansion stored in cache
    "INVALIDATE",  # Cache entry invalidated
])

# Token cost for warm query (hash confirmation only)
TOKENS_WARM = 1


# ==============================================================================
# CANONICAL JSON UTILITIES
# ==============================================================================


def canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hex digest."""
    return hashlib.sha256(data).hexdigest()


# ==============================================================================
# SESSION CACHE RECEIPT
# ==============================================================================


@dataclass
class SessionCacheReceipt:
    """Receipt for session cache operation.

    Every cache lookup/store emits a receipt for:
    - Provenance tracking
    - Token savings calculation
    - Compression validation

    Attributes:
        session_id: Session this operation belongs to
        operation: HIT | MISS | PUT | INVALIDATE
        pointer: SPC pointer (e.g., "C3")
        expansion_hash: SHA256 of expansion (if applicable)
        tokens_cold: Token cost of full expansion
        tokens_warm: Token cost of cache confirmation (always 1)
        codebook_hash: Codebook version for this operation
        timestamp_utc: ISO8601 timestamp
        receipt_hash: Computed hash of receipt content
    """

    session_id: str
    operation: str
    pointer: str
    expansion_hash: str
    tokens_cold: int
    tokens_warm: int
    codebook_hash: str
    timestamp_utc: str
    receipt_hash: str = ""

    def __post_init__(self):
        """Validate and compute receipt hash."""
        if self.operation not in VALID_CACHE_OPERATIONS:
            raise ValueError(f"Invalid operation: {self.operation}. Must be one of {VALID_CACHE_OPERATIONS}")

        if not self.receipt_hash:
            self.receipt_hash = self._compute_hash()

    @property
    def tokens_saved(self) -> int:
        """Tokens saved by this operation (HIT only)."""
        if self.operation == "HIT":
            return max(0, self.tokens_cold - self.tokens_warm)
        return 0

    @property
    def savings_pct(self) -> float:
        """Savings percentage (HIT only)."""
        if self.operation == "HIT" and self.tokens_cold > 0:
            return (self.tokens_saved / self.tokens_cold) * 100
        return 0.0

    def _compute_hash(self) -> str:
        """Compute deterministic hash of receipt content."""
        content = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "operation": self.operation,
            "pointer": self.pointer,
            "expansion_hash": self.expansion_hash,
            "tokens_cold": self.tokens_cold,
            "tokens_warm": self.tokens_warm,
            "codebook_hash": self.codebook_hash,
            "timestamp_utc": self.timestamp_utc,
        }
        canonical = canonical_json(content)
        return sha256_hex(canonical.encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dict."""
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "operation": self.operation,
            "pointer": self.pointer,
            "expansion_hash": self.expansion_hash,
            "tokens_cold": self.tokens_cold,
            "tokens_warm": self.tokens_warm,
            "tokens_saved": self.tokens_saved,
            "savings_pct": round(self.savings_pct, 2),
            "codebook_hash": self.codebook_hash,
            "timestamp_utc": self.timestamp_utc,
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionCacheReceipt":
        """Deserialize receipt from dict."""
        return cls(
            session_id=data["session_id"],
            operation=data["operation"],
            pointer=data["pointer"],
            expansion_hash=data["expansion_hash"],
            tokens_cold=data["tokens_cold"],
            tokens_warm=data["tokens_warm"],
            codebook_hash=data["codebook_hash"],
            timestamp_utc=data["timestamp_utc"],
            receipt_hash=data.get("receipt_hash", ""),
        )

    def verify(self) -> bool:
        """Verify receipt hash integrity."""
        expected = self._compute_hash()
        return self.receipt_hash == expected


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def create_cache_receipt(
    session_id: str,
    operation: str,
    pointer: str,
    expansion_hash: str,
    tokens_cold: int,
    codebook_hash: str,
    tokens_warm: int = TOKENS_WARM,
    timestamp_utc: Optional[str] = None,
) -> SessionCacheReceipt:
    """Create a session cache receipt.

    Args:
        session_id: Session identifier
        operation: HIT | MISS | PUT | INVALIDATE
        pointer: SPC pointer
        expansion_hash: SHA256 of expansion
        tokens_cold: Full expansion token count
        codebook_hash: Codebook version
        tokens_warm: Cache confirmation token count (default 1)
        timestamp_utc: Optional timestamp (default: now)

    Returns:
        SessionCacheReceipt with computed hash
    """
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()

    return SessionCacheReceipt(
        session_id=session_id,
        operation=operation,
        pointer=pointer,
        expansion_hash=expansion_hash,
        tokens_cold=tokens_cold,
        tokens_warm=tokens_warm,
        codebook_hash=codebook_hash,
        timestamp_utc=timestamp_utc,
    )


def verify_receipt(receipt: SessionCacheReceipt) -> bool:
    """Verify receipt hash integrity.

    Args:
        receipt: Receipt to verify

    Returns:
        True if valid, False if tampered
    """
    return receipt.verify()
