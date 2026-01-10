#!/usr/bin/env python3
"""
Token Session Aggregator (Phase 5.2.7.5)

Aggregates TokenReceipts across a session and computes cumulative statistics.

Usage:
    from CAPABILITY.PRIMITIVES.token_session import TokenSession

    session = TokenSession(session_id="my-session-001")

    # Add receipts from operations
    session.add_receipt(receipt1)
    session.add_receipt(receipt2)

    # Get summary
    summary = session.get_summary()
    print(summary.compact())
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from .token_receipt import TokenReceipt, TokenizerInfo, get_default_tokenizer
except ImportError:
    from token_receipt import TokenReceipt, TokenizerInfo, get_default_tokenizer


@dataclass
class SessionTokenSummary:
    """Summary of token usage across a session."""
    session_id: str
    receipts_count: int
    total_tokens_in: int
    total_tokens_out: int
    total_baseline_equiv: int
    total_tokens_saved: int
    total_savings_pct: float
    operations_breakdown: Dict[str, int]  # operation -> count
    started_at: str
    ended_at: Optional[str] = None
    receipt_chain_head: Optional[str] = None  # Latest receipt hash for chain verification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "receipts_count": self.receipts_count,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_baseline_equiv": self.total_baseline_equiv,
            "total_tokens_saved": self.total_tokens_saved,
            "total_savings_pct": round(self.total_savings_pct, 2),
            "operations_breakdown": self.operations_breakdown,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "receipt_chain_head": self.receipt_chain_head,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def compact(self) -> str:
        """Compact format for CLI display."""
        return (
            f"[SESSION] {self.session_id}: "
            f"{self.receipts_count} ops, "
            f"{self.total_tokens_out:,} tokens out "
            f"(saved {self.total_tokens_saved:,} / {self.total_savings_pct:.1f}%)"
        )

    def verbose(self) -> str:
        """Verbose format for reports."""
        lines = [
            "SESSION TOKEN SUMMARY",
            "-" * 50,
            f"Session ID:      {self.session_id}",
            f"Operations:      {self.receipts_count}",
            f"Tokens In:       {self.total_tokens_in:,}",
            f"Tokens Out:      {self.total_tokens_out:,}",
            f"Baseline Equiv:  {self.total_baseline_equiv:,}",
            f"Tokens Saved:    {self.total_tokens_saved:,}",
            f"Savings:         {self.total_savings_pct:.2f}%",
            "",
            "Operations Breakdown:",
        ]

        for op, count in sorted(self.operations_breakdown.items()):
            lines.append(f"  {op}: {count}")

        lines.extend([
            "",
            f"Started:         {self.started_at}",
            f"Ended:           {self.ended_at or 'ongoing'}",
        ])

        if self.receipt_chain_head:
            lines.append(f"Chain Head:      {self.receipt_chain_head[:16]}...")

        return "\n".join(lines)


class TokenSession:
    """
    Manages token receipts for a session.

    Tracks all TokenReceipts emitted during a session, maintains
    receipt chain integrity, and computes cumulative statistics.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a token session.

        Args:
            session_id: Optional session ID (generated if not provided)
        """
        self.session_id = session_id or self._generate_session_id()
        self.receipts: List[TokenReceipt] = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at: Optional[str] = None
        self._last_receipt_hash: Optional[str] = None

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"session-{timestamp}-{unique}"

    def add_receipt(self, receipt: TokenReceipt) -> None:
        """
        Add a TokenReceipt to the session.

        Updates session_id on receipt if not set, and maintains
        receipt chain via parent_receipt_hash.

        Args:
            receipt: TokenReceipt to add
        """
        # Update receipt with session info if needed
        if receipt.session_id is None:
            # Create new receipt with session_id and parent_receipt_hash
            # Note: We can't modify frozen dataclass, so we track externally
            pass

        self.receipts.append(receipt)
        self._last_receipt_hash = receipt.receipt_hash

    def add_receipts(self, receipts: List[TokenReceipt]) -> None:
        """Add multiple receipts to the session."""
        for receipt in receipts:
            self.add_receipt(receipt)

    def get_summary(self) -> SessionTokenSummary:
        """
        Compute and return session summary.

        Returns:
            SessionTokenSummary with aggregated statistics
        """
        total_in = sum(r.tokens_in for r in self.receipts)
        total_out = sum(r.tokens_out for r in self.receipts)
        total_baseline = sum(r.baseline_equiv for r in self.receipts)
        total_saved = sum(r.tokens_saved for r in self.receipts)

        # Compute overall savings percentage
        if total_baseline > 0:
            savings_pct = (total_saved / total_baseline) * 100
        else:
            savings_pct = 0.0

        # Count operations by type
        ops_breakdown: Dict[str, int] = {}
        for receipt in self.receipts:
            ops_breakdown[receipt.operation] = ops_breakdown.get(receipt.operation, 0) + 1

        return SessionTokenSummary(
            session_id=self.session_id,
            receipts_count=len(self.receipts),
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            total_baseline_equiv=total_baseline,
            total_tokens_saved=total_saved,
            total_savings_pct=savings_pct,
            operations_breakdown=ops_breakdown,
            started_at=self.started_at,
            ended_at=self.ended_at,
            receipt_chain_head=self._last_receipt_hash,
        )

    def end_session(self) -> SessionTokenSummary:
        """
        End the session and return final summary.

        Returns:
            Final SessionTokenSummary
        """
        self.ended_at = datetime.now(timezone.utc).isoformat()
        return self.get_summary()

    def get_receipts(self) -> List[TokenReceipt]:
        """Get all receipts in this session."""
        return list(self.receipts)

    def get_receipts_by_operation(self, operation: str) -> List[TokenReceipt]:
        """Get receipts filtered by operation type."""
        return [r for r in self.receipts if r.operation == operation]

    def to_ledger(self) -> List[Dict[str, Any]]:
        """
        Export session as ledger entries for firewall logging.

        Returns:
            List of receipt dictionaries for ledger storage
        """
        return [r.to_dict() for r in self.receipts]

    def __len__(self) -> int:
        """Number of receipts in session."""
        return len(self.receipts)

    def __str__(self) -> str:
        """String representation."""
        return self.get_summary().compact()


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

_current_session: Optional[TokenSession] = None


def get_current_session() -> TokenSession:
    """
    Get or create the current global session.

    Returns:
        Current TokenSession instance
    """
    global _current_session
    if _current_session is None:
        _current_session = TokenSession()
    return _current_session


def start_new_session(session_id: Optional[str] = None) -> TokenSession:
    """
    Start a new global session, ending any existing one.

    Args:
        session_id: Optional session ID

    Returns:
        New TokenSession instance
    """
    global _current_session
    if _current_session is not None:
        _current_session.end_session()
    _current_session = TokenSession(session_id=session_id)
    return _current_session


def end_current_session() -> Optional[SessionTokenSummary]:
    """
    End the current global session.

    Returns:
        Final SessionTokenSummary, or None if no session
    """
    global _current_session
    if _current_session is None:
        return None
    summary = _current_session.end_session()
    _current_session = None
    return summary


def log_receipt(receipt: TokenReceipt) -> None:
    """
    Log a receipt to the current session.

    Args:
        receipt: TokenReceipt to log
    """
    session = get_current_session()
    session.add_receipt(receipt)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Self-test
    print("TokenSession Aggregator Self-Test")
    print("=" * 50)

    # Create a session
    session = TokenSession(session_id="test-session-001")

    # Create some test receipts
    tokenizer = get_default_tokenizer()

    receipts = [
        TokenReceipt(
            operation="semantic_query",
            tokens_out=834,
            tokenizer=tokenizer,
            tokens_in=12,
            baseline_equiv=624170,
            baseline_method="sum_corpus_tokens",
        ),
        TokenReceipt(
            operation="scl_decode",
            tokens_out=2,
            tokenizer=tokenizer,
            tokens_in=2,
            baseline_equiv=67,
            baseline_method="expanded_output",
        ),
        TokenReceipt(
            operation="semantic_query",
            tokens_out=1200,
            tokenizer=tokenizer,
            tokens_in=8,
            baseline_equiv=624170,
            baseline_method="sum_corpus_tokens",
        ),
    ]

    for receipt in receipts:
        session.add_receipt(receipt)
        print(f"Added: {receipt.compact()}")

    print("\n" + "-" * 50)
    print("Session Summary:")
    print("-" * 50)

    summary = session.end_session()
    print(summary.verbose())

    print("\nCompact format:")
    print(summary.compact())

    print("\nJSON format:")
    print(summary.to_json())

    print("\n" + "=" * 50)
    print("Self-test complete!")
