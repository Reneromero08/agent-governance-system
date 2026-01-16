#!/usr/bin/env python3
"""
CassetteReceipt Primitive (Phase 6.0/6.2)

Mandatory receipt for cassette write operations. Every cassette write
MUST emit a CassetteReceipt for provenance and restore capability.

Contract rules:
- Every write emits a receipt with deterministic hash
- Receipts form a chain via parent_receipt_hash
- Session Merkle roots computed from receipt hashes
- Restore-from-receipts requires receipt chain + source records

Usage:
    from CAPABILITY.PRIMITIVES.cassette_receipt import (
        CassetteReceipt, create_receipt, verify_receipt,
        compute_session_merkle_root, verify_receipt_chain
    )

    # Create a receipt
    receipt = create_receipt(
        cassette_id="resident",
        operation="SAVE",
        record_id="abc123...",
        record_hash="def456...",
    )

    # Verify a receipt
    is_valid = verify_receipt(receipt)

    # Compute session Merkle root
    merkle_root = compute_session_merkle_root([r.receipt_hash for r in receipts])
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Schema version - must match cassette_receipt.schema.json
SCHEMA_VERSION = "1.0.0"

# Valid operation types
VALID_OPERATIONS = frozenset([
    "SAVE",      # New memory saved
    "UPDATE",    # Existing memory updated
    "DELETE",    # Memory deleted
    "MIGRATE",   # Schema migration
    "COMPACT",   # Database compaction
    "RESTORE",   # Restored from receipts
])


# ==============================================================================
# CANONICAL JSON UTILITIES
# ==============================================================================


def canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string.

    Args:
        obj: Object to serialize

    Returns:
        Canonical JSON with sorted keys and minimal separators
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_bytes(obj: Any) -> bytes:
    """Convert object to canonical JSON bytes with trailing newline.

    Args:
        obj: Object to serialize

    Returns:
        UTF-8 encoded canonical JSON with exactly one trailing newline
    """
    return (canonical_json(obj) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hex digest.

    Args:
        data: Bytes to hash

    Returns:
        SHA256 hex digest (64 lowercase hex chars)
    """
    return hashlib.sha256(data).hexdigest()


# ==============================================================================
# CASSETTE RECEIPT DATACLASS
# ==============================================================================


@dataclass
class CassetteReceipt:
    """
    Receipt for a cassette write operation.

    Required fields:
        cassette_id: Identifier of the cassette (e.g., "resident")
        operation: Operation type (SAVE, UPDATE, DELETE, MIGRATE, COMPACT, RESTORE)
        record_id: SHA-256 hash of record text (MemoryRecord.id)
        record_hash: Full hash of entire MemoryRecord

    Chain fields:
        parent_receipt_hash: Hash of previous receipt (None for first)
        receipt_index: 0-based index in chain (None if not using indices)

    Auto-computed fields:
        timestamp_utc: ISO8601 timestamp (auto-set if not provided)
        receipt_hash: SHA-256 of receipt (computed in __post_init__)
    """
    # Required
    cassette_id: str
    operation: str
    record_id: str
    record_hash: str

    # Chain linkage
    parent_receipt_hash: Optional[str] = None
    receipt_index: Optional[int] = None

    # Metadata
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    text_length: Optional[int] = None
    embedding_model: Optional[str] = None

    # Timestamp (auto-set if None)
    timestamp_utc: Optional[str] = None

    # Auto-computed (set in __post_init__)
    receipt_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Validate operation
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operation: '{self.operation}'. "
                f"Must be one of: {sorted(VALID_OPERATIONS)}"
            )

        # Validate record_id format (should be 64 hex chars)
        if not self.record_id or len(self.record_id) != 64:
            raise ValueError(
                f"record_id must be 64 hex chars (SHA-256), got: {len(self.record_id) if self.record_id else 0}"
            )

        # Validate record_hash format
        if not self.record_hash or len(self.record_hash) != 64:
            raise ValueError(
                f"record_hash must be 64 hex chars (SHA-256), got: {len(self.record_hash) if self.record_hash else 0}"
            )

        # Validate parent_receipt_hash format if provided
        if self.parent_receipt_hash is not None and len(self.parent_receipt_hash) != 64:
            raise ValueError(
                f"parent_receipt_hash must be 64 hex chars, got: {len(self.parent_receipt_hash)}"
            )

        # Set timestamp if not provided
        if self.timestamp_utc is None:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Compute receipt_hash (must be last)
        self.receipt_hash = self._compute_receipt_hash()

    def _compute_receipt_hash(self) -> str:
        """
        Compute deterministic receipt hash.

        Excludes receipt_hash and timestamp_utc for determinism.
        This ensures identical inputs produce identical hashes.
        """
        data = self._to_hashable_dict()
        canonical_bytes = canonical_json_bytes(data)
        return sha256_hex(canonical_bytes)

    def _to_hashable_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing (excludes receipt_hash and timestamp)."""
        result = {
            "schema_version": SCHEMA_VERSION,
            "cassette_id": self.cassette_id,
            "operation": self.operation,
            "record_id": self.record_id,
            "record_hash": self.record_hash,
            "parent_receipt_hash": self.parent_receipt_hash,
            "receipt_index": self.receipt_index,
        }

        # Include optional fields if present
        if self.agent_id is not None:
            result["agent_id"] = self.agent_id
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.text_length is not None:
            result["text_length"] = self.text_length
        if self.embedding_model is not None:
            result["embedding_model"] = self.embedding_model

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary for JSON serialization."""
        result = self._to_hashable_dict()
        result["timestamp_utc"] = self.timestamp_utc
        result["receipt_hash"] = self.receipt_hash
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def compact(self) -> str:
        """
        Compact format for CLI display.

        Format: [RECEIPT] op cassette:record_id[:8] -> receipt_hash[:8]
        Example: [RECEIPT] SAVE resident:abc12345 -> def45678
        """
        return (
            f"[RECEIPT] {self.operation} {self.cassette_id}:{self.record_id[:8]} "
            f"-> {self.receipt_hash[:8]}"
        )

    def verbose(self) -> str:
        """
        Verbose format for reports.

        Multi-line format with all details.
        """
        lines = [
            "CASSETTE RECEIPT",
            "-" * 50,
            f"Cassette:      {self.cassette_id}",
            f"Operation:     {self.operation}",
            f"Record ID:     {self.record_id}",
            f"Record Hash:   {self.record_hash}",
            f"Receipt Hash:  {self.receipt_hash}",
        ]

        if self.parent_receipt_hash:
            lines.append(f"Parent Hash:   {self.parent_receipt_hash}")

        if self.receipt_index is not None:
            lines.append(f"Receipt Index: {self.receipt_index}")

        if self.agent_id:
            lines.append(f"Agent ID:      {self.agent_id}")

        if self.session_id:
            lines.append(f"Session ID:    {self.session_id}")

        if self.text_length is not None:
            lines.append(f"Text Length:   {self.text_length:,} bytes")

        if self.embedding_model:
            lines.append(f"Embedding:     {self.embedding_model}")

        lines.append(f"Timestamp:     {self.timestamp_utc}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Default string representation uses compact format."""
        return self.compact()


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def create_receipt(
    cassette_id: str,
    operation: str,
    record_id: str,
    record_hash: str,
    *,
    parent_receipt_hash: Optional[str] = None,
    receipt_index: Optional[int] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    text_length: Optional[int] = None,
    embedding_model: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
) -> CassetteReceipt:
    """
    Create a new CassetteReceipt.

    Args:
        cassette_id: Identifier of the cassette
        operation: Operation type (SAVE, UPDATE, DELETE, etc.)
        record_id: SHA-256 hash of record text
        record_hash: Full hash of entire MemoryRecord
        parent_receipt_hash: Hash of previous receipt in chain
        receipt_index: 0-based index in chain
        agent_id: Agent that performed the operation
        session_id: Session during which operation occurred
        text_length: Length of text in bytes
        embedding_model: Embedding model used
        timestamp_utc: ISO8601 timestamp (auto-set if None)

    Returns:
        CassetteReceipt with computed receipt_hash
    """
    return CassetteReceipt(
        cassette_id=cassette_id,
        operation=operation,
        record_id=record_id,
        record_hash=record_hash,
        parent_receipt_hash=parent_receipt_hash,
        receipt_index=receipt_index,
        agent_id=agent_id,
        session_id=session_id,
        text_length=text_length,
        embedding_model=embedding_model,
        timestamp_utc=timestamp_utc,
    )


def receipt_from_dict(data: Dict[str, Any]) -> CassetteReceipt:
    """
    Create CassetteReceipt from dictionary.

    Args:
        data: Dictionary with receipt fields

    Returns:
        CassetteReceipt instance

    Raises:
        ValueError: If required fields missing or invalid
    """
    return CassetteReceipt(
        cassette_id=data["cassette_id"],
        operation=data["operation"],
        record_id=data["record_id"],
        record_hash=data["record_hash"],
        parent_receipt_hash=data.get("parent_receipt_hash"),
        receipt_index=data.get("receipt_index"),
        agent_id=data.get("agent_id"),
        session_id=data.get("session_id"),
        text_length=data.get("text_length"),
        embedding_model=data.get("embedding_model"),
        timestamp_utc=data.get("timestamp_utc"),
    )


# ==============================================================================
# VERIFICATION FUNCTIONS
# ==============================================================================


def verify_receipt(receipt: CassetteReceipt) -> bool:
    """
    Verify receipt hash integrity.

    Args:
        receipt: CassetteReceipt to verify

    Returns:
        True if receipt_hash matches computed hash
    """
    computed = receipt._compute_receipt_hash()
    return computed == receipt.receipt_hash


def verify_receipt_dict(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Verify receipt dictionary integrity.

    Args:
        data: Receipt dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        receipt = receipt_from_dict(data)
        stored_hash = data.get("receipt_hash")

        if stored_hash != receipt.receipt_hash:
            return False, f"Hash mismatch: stored={stored_hash}, computed={receipt.receipt_hash}"

        return True, None
    except Exception as e:
        return False, str(e)


# ==============================================================================
# MERKLE ROOT COMPUTATION
# ==============================================================================


def compute_session_merkle_root(receipt_hashes: List[str]) -> str:
    """
    Compute Merkle root from list of receipt hashes.

    Uses binary tree pairing. For odd counts, duplicate last leaf.
    Leaves must be in execution order (same order as receipt chain).

    Args:
        receipt_hashes: List of receipt_hash hex strings in execution order

    Returns:
        Merkle root as SHA256 hex string

    Raises:
        ValueError: If receipt_hashes is empty
    """
    if not receipt_hashes:
        raise ValueError("Cannot compute Merkle root from empty list")

    # Convert hex strings to bytes
    level = [bytes.fromhex(h) for h in receipt_hashes]

    # Build tree bottom-up
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                # Pair two nodes
                combined = level[i] + level[i + 1]
            else:
                # Odd node: duplicate self
                combined = level[i] + level[i]
            next_level.append(hashlib.sha256(combined).digest())
        level = next_level

    return level[0].hex()


# ==============================================================================
# RECEIPT CHAIN VERIFICATION
# ==============================================================================


def verify_receipt_chain(
    receipts: List[CassetteReceipt],
    *,
    verify_hashes: bool = True,
) -> Dict[str, Any]:
    """
    Verify receipt chain integrity.

    Validates:
    - First receipt has parent_receipt_hash = None
    - Each subsequent receipt's parent_receipt_hash matches previous receipt_hash
    - Receipt indices are contiguous if present (0, 1, 2, ...)
    - All receipt_hashes are unique
    - All receipt_hashes are correct (if verify_hashes=True)

    Args:
        receipts: List of receipts in execution order
        verify_hashes: Whether to verify receipt_hash correctness

    Returns:
        Dict with keys:
            valid: bool
            errors: List[str]
            merkle_root: str (if valid)
            chain_length: int
    """
    errors = []
    receipt_hashes = []
    seen_hashes = set()

    if not receipts:
        return {
            "valid": False,
            "errors": ["Receipt chain cannot be empty"],
            "merkle_root": None,
            "chain_length": 0,
        }

    has_receipt_index = receipts[0].receipt_index is not None

    for i, receipt in enumerate(receipts):
        # Check for duplicate hashes
        if receipt.receipt_hash in seen_hashes:
            errors.append(f"Duplicate receipt_hash at index {i}: {receipt.receipt_hash}")
        seen_hashes.add(receipt.receipt_hash)
        receipt_hashes.append(receipt.receipt_hash)

        # Verify hash integrity
        if verify_hashes and not verify_receipt(receipt):
            computed = receipt._compute_receipt_hash()
            errors.append(
                f"Receipt {i} hash mismatch: stored={receipt.receipt_hash}, "
                f"computed={computed}"
            )

        # Check chain linkage
        if i == 0:
            if receipt.parent_receipt_hash is not None:
                errors.append(
                    f"First receipt must have parent_receipt_hash=None, "
                    f"got: {receipt.parent_receipt_hash}"
                )
            if has_receipt_index and receipt.receipt_index != 0:
                errors.append(
                    f"First receipt_index must be 0, got: {receipt.receipt_index}"
                )
        else:
            prev_receipt = receipts[i - 1]

            # Check parent hash linkage
            if receipt.parent_receipt_hash != prev_receipt.receipt_hash:
                errors.append(
                    f"Receipt {i} parent_receipt_hash mismatch: "
                    f"expected={prev_receipt.receipt_hash}, "
                    f"got={receipt.parent_receipt_hash}"
                )

            # Check receipt_index consistency
            current_has_index = receipt.receipt_index is not None
            if current_has_index != has_receipt_index:
                errors.append(
                    f"Receipt {i} receipt_index consistency error: "
                    f"all receipts must have receipt_index set or all must be None"
                )

            # Check receipt_index is contiguous
            if receipt.receipt_index is not None:
                expected_index = prev_receipt.receipt_index + 1
                if receipt.receipt_index != expected_index:
                    errors.append(
                        f"Receipt {i} receipt_index not contiguous: "
                        f"expected={expected_index}, got={receipt.receipt_index}"
                    )

    # Compute Merkle root if valid
    merkle_root = None
    if not errors and receipt_hashes:
        merkle_root = compute_session_merkle_root(receipt_hashes)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "merkle_root": merkle_root,
        "chain_length": len(receipts),
    }


# ==============================================================================
# FILE I/O
# ==============================================================================


def write_receipt_to_file(receipt: CassetteReceipt, path: Path) -> None:
    """
    Write receipt to file as canonical JSON bytes.

    Args:
        receipt: Receipt to write
        path: Output file path
    """
    receipt_bytes = canonical_json_bytes(receipt.to_dict())
    path.write_bytes(receipt_bytes)


def load_receipt_from_file(path: Path) -> Optional[CassetteReceipt]:
    """
    Load receipt from file.

    Args:
        path: Path to receipt file

    Returns:
        CassetteReceipt or None if file doesn't exist
    """
    if not path.exists():
        return None

    receipt_bytes = path.read_bytes()
    receipt_text = receipt_bytes.decode("utf-8").rstrip("\n")
    data = json.loads(receipt_text)
    return receipt_from_dict(data)


# ==============================================================================
# SCHEMA PATH
# ==============================================================================


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "LAW" / "SCHEMAS" / "cassette_receipt.schema.json"


def validate_receipt_schema(receipt: Dict[str, Any]) -> None:
    """
    Validate receipt against JSON schema.

    Args:
        receipt: Receipt dictionary to validate

    Raises:
        ValueError: If validation fails
        ImportError: If jsonschema not available
    """
    try:
        import jsonschema
    except ImportError:
        raise ImportError("jsonschema package required for schema validation")

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Receipt schema not found: {SCHEMA_PATH}")

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    jsonschema.validate(instance=receipt, schema=schema)


# ==============================================================================
# CLI SELF-TEST
# ==============================================================================


if __name__ == "__main__":
    print("CassetteReceipt Primitive Self-Test")
    print("=" * 50)

    # Create sample receipts
    receipt1 = create_receipt(
        cassette_id="resident",
        operation="SAVE",
        record_id="a" * 64,
        record_hash="b" * 64,
        agent_id="test_agent",
        session_id="test_session",
        text_length=256,
        embedding_model="all-MiniLM-L6-v2",
    )

    print("\nReceipt 1 (compact):")
    print(receipt1.compact())

    print("\nReceipt 1 (verbose):")
    print(receipt1.verbose())

    # Create chained receipt
    receipt2 = create_receipt(
        cassette_id="resident",
        operation="SAVE",
        record_id="c" * 64,
        record_hash="d" * 64,
        parent_receipt_hash=receipt1.receipt_hash,
        receipt_index=1,
    )

    # Update receipt1 with index for chain
    receipt1_indexed = create_receipt(
        cassette_id="resident",
        operation="SAVE",
        record_id="a" * 64,
        record_hash="b" * 64,
        receipt_index=0,
    )
    receipt2_indexed = create_receipt(
        cassette_id="resident",
        operation="SAVE",
        record_id="c" * 64,
        record_hash="d" * 64,
        parent_receipt_hash=receipt1_indexed.receipt_hash,
        receipt_index=1,
    )

    print("\nChained receipts:")
    print(f"  Receipt 0: {receipt1_indexed.receipt_hash[:16]}...")
    print(f"  Receipt 1: {receipt2_indexed.receipt_hash[:16]}... (parent: {receipt2_indexed.parent_receipt_hash[:16]}...)")

    # Verify chain
    chain_result = verify_receipt_chain([receipt1_indexed, receipt2_indexed])
    print(f"\nChain verification: valid={chain_result['valid']}")
    print(f"Merkle root: {chain_result['merkle_root'][:16]}...")

    # Verify individual receipt
    print(f"\nReceipt 1 hash verification: {verify_receipt(receipt1)}")

    print("\n" + "=" * 50)
    print("Self-test complete!")
