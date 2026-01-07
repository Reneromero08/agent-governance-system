"""
Timing-Safe Comparison Primitives

Provides constant-time comparison functions to prevent timing attacks.
Uses hmac.compare_digest() which is guaranteed constant-time in CPython.

Usage:
    from CAPABILITY.PRIMITIVES.timing_safe import compare_hash

    if compare_hash(computed_hash, expected_hash):
        print("Match")
"""

from __future__ import annotations

import hmac


def compare_hash(a: str, b: str) -> bool:
    """
    Compare two hash strings in constant time.

    Uses hmac.compare_digest() which compares all bytes regardless
    of where the first difference occurs.

    Args:
        a: First hash string (hex)
        b: Second hash string (hex)

    Returns:
        True if hashes are equal, False otherwise

    Note:
        Both inputs must be strings. For bytes comparison,
        use compare_bytes() instead.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return False

    # hmac.compare_digest handles strings directly
    return hmac.compare_digest(a, b)


def compare_bytes(a: bytes, b: bytes) -> bool:
    """
    Compare two byte sequences in constant time.

    Args:
        a: First byte sequence
        b: Second byte sequence

    Returns:
        True if sequences are equal, False otherwise
    """
    if not isinstance(a, bytes) or not isinstance(b, bytes):
        return False

    return hmac.compare_digest(a, b)


def compare_signature(sig_a: str, sig_b: str) -> bool:
    """
    Compare two signature hex strings in constant time.

    Alias for compare_hash() with clearer semantics for signatures.
    """
    return compare_hash(sig_a, sig_b)
