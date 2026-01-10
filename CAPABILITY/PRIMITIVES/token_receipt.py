#!/usr/bin/env python3
"""
TokenReceipt Primitive (Phase 5.2.7.2)

Mandatory receipt for token-consuming operations. Every operation that
consumes or saves tokens MUST emit a TokenReceipt.

Usage:
    from CAPABILITY.PRIMITIVES.token_receipt import TokenReceipt, TokenizerInfo

    receipt = TokenReceipt(
        operation="semantic_query",
        tokens_out=834,
        tokenizer=TokenizerInfo(library="tiktoken", encoding="o200k_base"),
        baseline_equiv=624170,
    )
    print(receipt.compact())  # [TOKEN] semantic_query: 834 tokens (saved 623,336 / 99.87%)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Schema version - must match token_receipt.schema.json
SCHEMA_VERSION = "1.0.0"

# Valid operation types
VALID_OPERATIONS = frozenset([
    "semantic_query",
    "scl_decode",
    "scl_encode",
    "cas_get",
    "cas_put",
    "skill_invoke",
    "session_load",
    "expand_hash",
])

# Valid baseline methods
VALID_BASELINE_METHODS = frozenset([
    "sum_corpus_tokens",
    "paste_scan",
    "full_context",
    "expanded_output",
    "manual",
])


@dataclass
class TokenizerInfo:
    """Tokenizer configuration for receipts."""
    library: str          # "tiktoken" or "word-count-proxy"
    encoding: str         # "o200k_base", "cl100k_base"
    version: Optional[str] = None
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "library": self.library,
            "encoding": self.encoding,
            "version": self.version,
            "fallback_used": self.fallback_used,
        }


@dataclass
class DeterminismProof:
    """Optional reproducibility proof for auditing."""
    methodology_hash: Optional[str] = None  # SHA-256 of script/method
    git_head: Optional[str] = None          # Git commit hash
    git_clean: Optional[bool] = None        # Whether working tree was clean

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.methodology_hash is not None:
            result["methodology_hash"] = self.methodology_hash
        if self.git_head is not None:
            result["git_head"] = self.git_head
        if self.git_clean is not None:
            result["git_clean"] = self.git_clean
        return result


@dataclass
class QueryMetadata:
    """Operation-specific metadata for semantic queries."""
    query_hash: Optional[str] = None        # SHA-256 of query text
    results_count: Optional[int] = None     # Number of results returned
    threshold_used: Optional[float] = None  # Similarity threshold
    top_k: Optional[int] = None             # Top-K parameter
    index_sections_count: Optional[int] = None  # Total sections in index

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.query_hash is not None:
            result["query_hash"] = self.query_hash
        if self.results_count is not None:
            result["results_count"] = self.results_count
        if self.threshold_used is not None:
            result["threshold_used"] = self.threshold_used
        if self.top_k is not None:
            result["top_k"] = self.top_k
        if self.index_sections_count is not None:
            result["index_sections_count"] = self.index_sections_count
        return result


@dataclass
class TokenReceipt:
    """
    Mandatory receipt for token-consuming operations.

    Required fields:
        operation: Operation type (semantic_query, scl_decode, etc.)
        tokens_out: Output tokens
        tokenizer: TokenizerInfo object

    Auto-computed fields:
        tokens_saved: baseline_equiv - tokens_out
        savings_pct: (tokens_saved / baseline_equiv) * 100
        operation_id: Unique hash for this operation
        receipt_hash: SHA-256 of receipt for integrity
    """
    # Required
    operation: str
    tokens_out: int
    tokenizer: TokenizerInfo

    # Optional - core
    tokens_in: int = 0
    baseline_equiv: int = 0
    timestamp_utc: Optional[str] = None
    corpus_anchor: Optional[str] = None

    # Optional - hardening
    session_id: Optional[str] = None
    parent_receipt_hash: Optional[str] = None
    baseline_method: Optional[str] = None
    determinism_proof: Optional[DeterminismProof] = None
    query_metadata: Optional[QueryMetadata] = None

    # Auto-computed (set in __post_init__)
    tokens_saved: int = field(default=0, init=False)
    savings_pct: float = field(default=0.0, init=False)
    operation_id: Optional[str] = field(default=None, init=False)
    receipt_hash: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Validate operation
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operation: '{self.operation}'. "
                f"Must be one of: {sorted(VALID_OPERATIONS)}"
            )

        # Validate baseline_method if provided
        if self.baseline_method is not None and self.baseline_method not in VALID_BASELINE_METHODS:
            raise ValueError(
                f"Invalid baseline_method: '{self.baseline_method}'. "
                f"Must be one of: {sorted(VALID_BASELINE_METHODS)}"
            )

        # Set timestamp if not provided
        if self.timestamp_utc is None:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Compute savings
        if self.baseline_equiv > 0:
            self.tokens_saved = self.baseline_equiv - self.tokens_out
            self.savings_pct = (self.tokens_saved / self.baseline_equiv) * 100
        else:
            self.tokens_saved = 0
            self.savings_pct = 0.0

        # Generate operation_id
        self.operation_id = self._compute_operation_id()

        # Compute receipt_hash (must be last)
        self.receipt_hash = self._compute_receipt_hash()

    def _compute_operation_id(self) -> str:
        """Generate unique operation identifier."""
        # Use timestamp + operation + tokens_out for uniqueness
        data = f"{self.timestamp_utc}:{self.operation}:{self.tokens_out}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

    def _compute_receipt_hash(self) -> str:
        """
        Compute deterministic receipt hash.

        Excludes receipt_hash and timestamp_utc for determinism.
        """
        data = self.to_dict()
        # Remove non-deterministic fields
        data.pop("receipt_hash", None)
        data.pop("timestamp_utc", None)

        # Canonical JSON
        canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "schema_version": SCHEMA_VERSION,
            "operation": self.operation,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "baseline_equiv": self.baseline_equiv,
            "tokens_saved": self.tokens_saved,
            "savings_pct": round(self.savings_pct, 2),
            "tokenizer": self.tokenizer.to_dict(),
            "timestamp_utc": self.timestamp_utc,
            "operation_id": self.operation_id,
            "receipt_hash": self.receipt_hash,
        }

        # Add optional fields if present
        if self.corpus_anchor is not None:
            result["corpus_anchor"] = self.corpus_anchor
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.parent_receipt_hash is not None:
            result["parent_receipt_hash"] = self.parent_receipt_hash
        if self.baseline_method is not None:
            result["baseline_method"] = self.baseline_method
        if self.determinism_proof is not None:
            proof_dict = self.determinism_proof.to_dict()
            if proof_dict:  # Only add if non-empty
                result["determinism_proof"] = proof_dict
        if self.query_metadata is not None:
            meta_dict = self.query_metadata.to_dict()
            if meta_dict:  # Only add if non-empty
                result["query_metadata"] = meta_dict

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════════════
    # DISPLAY FORMATS (Phase 5.2.7.7)
    # ═══════════════════════════════════════════════════════════════════════

    def compact(self) -> str:
        """
        Compact format for CLI display.

        Format: [TOKEN] op: N tokens (saved M / P%)
        Example: [TOKEN] semantic_query: 834 tokens (saved 623,336 / 99.87%)
        """
        if self.baseline_equiv > 0:
            return (
                f"[TOKEN] {self.operation}: {self.tokens_out:,} tokens "
                f"(saved {self.tokens_saved:,} / {self.savings_pct:.2f}%)"
            )
        else:
            return f"[TOKEN] {self.operation}: {self.tokens_out:,} tokens"

    def verbose(self) -> str:
        """
        Verbose format for reports.

        Multi-line format with all details.
        """
        lines = [
            "TOKEN RECEIPT",
            "-" * 40,
            f"Operation:     {self.operation}",
            f"Tokens In:     {self.tokens_in:,}",
            f"Tokens Out:    {self.tokens_out:,}",
        ]

        if self.baseline_equiv > 0:
            lines.extend([
                f"Baseline:      {self.baseline_equiv:,}",
                f"Saved:         {self.tokens_saved:,} ({self.savings_pct:.2f}%)",
            ])

        lines.extend([
            f"Tokenizer:     {self.tokenizer.library}/{self.tokenizer.encoding}",
            f"Operation ID:  {self.operation_id}",
        ])

        if self.corpus_anchor:
            lines.append(f"Corpus:        {self.corpus_anchor[:16]}...")

        if self.session_id:
            lines.append(f"Session:       {self.session_id}")

        if self.receipt_hash:
            lines.append(f"Receipt Hash:  {self.receipt_hash[:16]}...")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Default string representation uses compact format."""
        return self.compact()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def get_default_tokenizer() -> TokenizerInfo:
    """
    Get default tokenizer info, detecting tiktoken availability.

    Returns:
        TokenizerInfo with tiktoken if available, word-count-proxy otherwise
    """
    try:
        import tiktoken
        return TokenizerInfo(
            library="tiktoken",
            encoding="o200k_base",
            version=tiktoken.__version__,
            fallback_used=False,
        )
    except ImportError:
        return TokenizerInfo(
            library="word-count-proxy",
            encoding="o200k_base",
            version=None,
            fallback_used=True,
        )


def count_tokens(text: str, tokenizer: Optional[TokenizerInfo] = None) -> int:
    """
    Count tokens in text using specified tokenizer.

    Args:
        text: Text to tokenize
        tokenizer: TokenizerInfo (uses default if None)

    Returns:
        Token count
    """
    if tokenizer is None:
        tokenizer = get_default_tokenizer()

    if tokenizer.library == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer.encoding)
            return len(enc.encode(text))
        except Exception:
            pass

    # Fallback: word count proxy (approx 0.75 tokens per word)
    return max(1, int(len(text.split()) / 0.75))


def hash_query(query: str) -> str:
    """Hash query text for privacy in receipts."""
    return hashlib.sha256(query.encode('utf-8')).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_receipt(receipt: Dict[str, Any]) -> bool:
    """
    Validate TokenReceipt dictionary against schema.

    Args:
        receipt: Receipt dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    required = ["schema_version", "operation", "tokens_out", "tokenizer"]

    for field in required:
        if field not in receipt:
            raise ValueError(f"Missing required field: {field}")

    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {SCHEMA_VERSION}, "
            f"got {receipt.get('schema_version')}"
        )

    if receipt.get("operation") not in VALID_OPERATIONS:
        raise ValueError(f"Invalid operation: {receipt.get('operation')}")

    # Validate computed fields if baseline provided
    baseline = receipt.get("baseline_equiv", 0)
    if baseline > 0:
        expected_saved = baseline - receipt.get("tokens_out", 0)
        actual_saved = receipt.get("tokens_saved", 0)
        if actual_saved != expected_saved:
            raise ValueError(
                f"tokens_saved mismatch: expected {expected_saved}, got {actual_saved}"
            )

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # Self-test
    print("TokenReceipt Primitive Self-Test")
    print("=" * 50)

    # Create a sample receipt
    tokenizer = get_default_tokenizer()
    print(f"Tokenizer: {tokenizer.library}/{tokenizer.encoding}")

    receipt = TokenReceipt(
        operation="semantic_query",
        tokens_out=834,
        tokenizer=tokenizer,
        tokens_in=12,
        baseline_equiv=624170,
        baseline_method="sum_corpus_tokens",
        query_metadata=QueryMetadata(
            results_count=10,
            top_k=10,
            threshold_used=0.4,
        ),
    )

    print("\nCompact format:")
    print(receipt.compact())

    print("\nVerbose format:")
    print(receipt.verbose())

    print("\nJSON format:")
    print(receipt.to_json())

    # Validate
    print("\nValidation:", validate_receipt(receipt.to_dict()))

    print("\n" + "=" * 50)
    print("Self-test complete!")
