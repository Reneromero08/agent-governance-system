#!/usr/bin/env python3
"""
Token Firewall (Phase 5.2.7.6)

Enforces token accountability rules:
- REJECT: Outputs > 1000 tokens without TokenReceipt
- WARN: semantic_query with savings_pct < 50%
- LOG: All receipts to session ledger

Usage:
    from CAPABILITY.PRIMITIVES.token_firewall import TokenFirewall

    firewall = TokenFirewall()

    # Validate an output with receipt
    result = firewall.validate_output(output_tokens=1500, receipt=my_receipt)
    if not result.allowed:
        raise ValueError(result.message)

    # Validate without receipt (will fail for large outputs)
    result = firewall.validate_output(output_tokens=1500, receipt=None)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from .token_receipt import TokenReceipt
    from .token_session import TokenSession, get_current_session, log_receipt
except ImportError:
    from token_receipt import TokenReceipt
    from token_session import TokenSession, get_current_session, log_receipt


class FirewallVerdict(Enum):
    """Firewall validation verdict."""
    ALLOW = "ALLOW"
    WARN = "WARN"
    REJECT = "REJECT"


@dataclass
class FirewallResult:
    """Result of firewall validation."""
    verdict: FirewallVerdict
    allowed: bool
    rule_id: str
    message: str
    warnings: List[str] = field(default_factory=list)
    receipt_logged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "allowed": self.allowed,
            "rule_id": self.rule_id,
            "message": self.message,
            "warnings": self.warnings,
            "receipt_logged": self.receipt_logged,
        }


# Firewall rule thresholds
LARGE_OUTPUT_THRESHOLD = 1000  # Tokens requiring receipt
SAVINGS_WARNING_THRESHOLD = 50.0  # % savings below which to warn


class TokenFirewall:
    """
    Enforces token accountability rules.

    Rules:
        REJECT-001: Outputs > 1000 tokens without TokenReceipt
        WARN-001: semantic_query with savings_pct < 50%
        LOG-001: All receipts logged to session ledger
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        large_output_threshold: int = LARGE_OUTPUT_THRESHOLD,
        savings_warning_threshold: float = SAVINGS_WARNING_THRESHOLD,
        auto_log_receipts: bool = True,
    ):
        """
        Initialize token firewall.

        Args:
            large_output_threshold: Token count above which receipt is required
            savings_warning_threshold: savings_pct below which to warn
            auto_log_receipts: Whether to automatically log receipts to session
        """
        self.large_output_threshold = large_output_threshold
        self.savings_warning_threshold = savings_warning_threshold
        self.auto_log_receipts = auto_log_receipts
        self._violation_log: List[Dict[str, Any]] = []

    def validate_output(
        self,
        output_tokens: int,
        receipt: Optional[TokenReceipt] = None,
        operation: Optional[str] = None,
    ) -> FirewallResult:
        """
        Validate an output against firewall rules.

        Args:
            output_tokens: Number of output tokens
            receipt: Optional TokenReceipt for the operation
            operation: Optional operation type (overrides receipt.operation)

        Returns:
            FirewallResult with verdict and details
        """
        warnings = []
        receipt_logged = False

        # Determine operation
        op = operation or (receipt.operation if receipt else "unknown")

        # Rule REJECT-001: Large outputs require receipt
        if output_tokens > self.large_output_threshold and receipt is None:
            result = FirewallResult(
                verdict=FirewallVerdict.REJECT,
                allowed=False,
                rule_id="REJECT-001",
                message=(
                    f"Output of {output_tokens:,} tokens exceeds threshold "
                    f"({self.large_output_threshold:,}) without TokenReceipt"
                ),
                warnings=warnings,
                receipt_logged=False,
            )
            self._log_violation(result, output_tokens, receipt)
            return result

        # If we have a receipt, check savings rules
        if receipt is not None:
            # Rule WARN-001: Low savings for semantic_query
            if receipt.operation == "semantic_query":
                if receipt.baseline_equiv > 0 and receipt.savings_pct < self.savings_warning_threshold:
                    warnings.append(
                        f"WARN-001: semantic_query savings ({receipt.savings_pct:.1f}%) "
                        f"below threshold ({self.savings_warning_threshold}%)"
                    )

            # Rule LOG-001: Log receipt to session
            if self.auto_log_receipts:
                log_receipt(receipt)
                receipt_logged = True

        # Determine final verdict
        if warnings:
            verdict = FirewallVerdict.WARN
        else:
            verdict = FirewallVerdict.ALLOW

        return FirewallResult(
            verdict=verdict,
            allowed=True,
            rule_id="ALLOW" if not warnings else "WARN-001",
            message="Output validated" + (f" with {len(warnings)} warning(s)" if warnings else ""),
            warnings=warnings,
            receipt_logged=receipt_logged,
        )

    def require_receipt(
        self,
        output_tokens: int,
        receipt: Optional[TokenReceipt],
        context: str = "",
    ) -> None:
        """
        Require a receipt for an output, raising exception if invalid.

        Args:
            output_tokens: Number of output tokens
            receipt: TokenReceipt (required if over threshold)
            context: Context string for error messages

        Raises:
            ValueError: If receipt is required but not provided
        """
        result = self.validate_output(output_tokens, receipt)
        if not result.allowed:
            ctx = f" ({context})" if context else ""
            raise ValueError(f"TokenFirewall{ctx}: {result.message}")

    def check_savings(
        self,
        receipt: TokenReceipt,
        min_savings_pct: Optional[float] = None,
    ) -> FirewallResult:
        """
        Check if a receipt meets savings requirements.

        Args:
            receipt: TokenReceipt to check
            min_savings_pct: Override minimum savings percentage

        Returns:
            FirewallResult with verdict
        """
        threshold = min_savings_pct or self.savings_warning_threshold
        warnings = []

        if receipt.baseline_equiv > 0 and receipt.savings_pct < threshold:
            warnings.append(
                f"Savings ({receipt.savings_pct:.1f}%) below threshold ({threshold}%)"
            )

        return FirewallResult(
            verdict=FirewallVerdict.WARN if warnings else FirewallVerdict.ALLOW,
            allowed=True,  # Savings check is warning-only
            rule_id="WARN-001" if warnings else "ALLOW",
            message="Savings check " + ("warning" if warnings else "passed"),
            warnings=warnings,
            receipt_logged=False,
        )

    def _log_violation(
        self,
        result: FirewallResult,
        output_tokens: int,
        receipt: Optional[TokenReceipt],
    ) -> None:
        """Log a firewall violation for auditing."""
        violation = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rule_id": result.rule_id,
            "verdict": result.verdict.value,
            "output_tokens": output_tokens,
            "has_receipt": receipt is not None,
            "message": result.message,
        }
        self._violation_log.append(violation)

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get list of all violations logged."""
        return list(self._violation_log)

    def clear_violations(self) -> None:
        """Clear the violation log."""
        self._violation_log.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_default_firewall: Optional[TokenFirewall] = None


def get_token_firewall() -> TokenFirewall:
    """Get the default token firewall instance."""
    global _default_firewall
    if _default_firewall is None:
        _default_firewall = TokenFirewall()
    return _default_firewall


def validate_token_output(
    output_tokens: int,
    receipt: Optional[TokenReceipt] = None,
) -> FirewallResult:
    """Convenience function to validate output with default firewall."""
    return get_token_firewall().validate_output(output_tokens, receipt)


def require_token_receipt(
    output_tokens: int,
    receipt: Optional[TokenReceipt],
    context: str = "",
) -> None:
    """Convenience function to require receipt with default firewall."""
    get_token_firewall().require_receipt(output_tokens, receipt, context)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("TokenFirewall Self-Test")
    print("=" * 50)

    firewall = TokenFirewall()

    # Test 1: Small output without receipt (should pass)
    print("\nTest 1: Small output without receipt")
    result = firewall.validate_output(500, receipt=None)
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Allowed: {result.allowed}")
    print(f"  Message: {result.message}")

    # Test 2: Large output without receipt (should reject)
    print("\nTest 2: Large output without receipt")
    result = firewall.validate_output(1500, receipt=None)
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Allowed: {result.allowed}")
    print(f"  Message: {result.message}")

    # Test 3: Large output with receipt (should pass)
    print("\nTest 3: Large output with receipt")
    try:
        from token_receipt import TokenReceipt, get_default_tokenizer
        tokenizer = get_default_tokenizer()
        receipt = TokenReceipt(
            operation="semantic_query",
            tokens_out=1500,
            tokenizer=tokenizer,
            baseline_equiv=624170,
            baseline_method="sum_corpus_tokens",
        )
        result = firewall.validate_output(1500, receipt=receipt)
        print(f"  Verdict: {result.verdict.value}")
        print(f"  Allowed: {result.allowed}")
        print(f"  Receipt logged: {result.receipt_logged}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 4: Low savings warning
    print("\nTest 4: Low savings warning")
    try:
        receipt_low = TokenReceipt(
            operation="semantic_query",
            tokens_out=400,
            tokenizer=tokenizer,
            baseline_equiv=500,  # Only 20% savings
            baseline_method="paste_scan",
        )
        result = firewall.validate_output(400, receipt=receipt_low)
        print(f"  Verdict: {result.verdict.value}")
        print(f"  Allowed: {result.allowed}")
        print(f"  Warnings: {result.warnings}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 5: Require receipt (exception test)
    print("\nTest 5: Require receipt (should raise)")
    try:
        firewall.require_receipt(2000, None, "test context")
        print("  ERROR: Should have raised!")
    except ValueError as e:
        print(f"  Correctly raised: {e}")

    print("\n" + "=" * 50)
    print(f"Violations logged: {len(firewall.get_violations())}")
    print("Self-test complete!")
