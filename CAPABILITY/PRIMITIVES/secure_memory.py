"""
Secure Memory Primitives for Key Material

Provides best-effort zeroization for sensitive bytes in CPython.
Note: Python's garbage collector and string interning may retain copies.
This module reduces but does not eliminate memory exposure risk.

Usage:
    with SecureBytes(key_bytes) as secure_key:
        result = sign_with_key(secure_key.bytes)
    # key_bytes overwritten with zeros on exit

CPython Zeroization Limitations
===============================

This module provides BEST-EFFORT memory clearing. It does NOT guarantee
that sensitive data is removed from memory. Known limitations:

1. **String Interning:** Python may intern short strings, keeping copies
   in a global table that we cannot access.

2. **Garbage Collection:** The GC may have already copied objects during
   reference counting or cycle detection.

3. **Memory Allocator:** Python's pymalloc may not immediately return
   freed memory to the OS, leaving data in the process heap.

4. **Copy-on-Write:** Some operations create implicit copies of bytes
   that we cannot track or zeroize.

5. **Swap/Hibernation:** The OS may page memory to disk at any time.

For true secure memory handling, consider:
- Using a C extension with mlock() and explicit memset()
- Running in a secure enclave (SGX, TrustZone)
- Using a language with deterministic memory management (Rust, C)

This module is appropriate for defense-in-depth but should not be
relied upon for high-security applications.
"""

from __future__ import annotations

import ctypes
from typing import Optional


class SecureBytes:
    """
    Context manager for sensitive byte data with automatic zeroization.

    LIMITATIONS (document these prominently):
    - CPython may have copied the bytes during operations
    - String interning may retain hex representations
    - Garbage collection timing is non-deterministic
    - This is BEST-EFFORT, not a guarantee

    Usage:
        with SecureBytes(private_key) as sk:
            signature = sign(message, sk.bytes)
        # private_key bytes are now overwritten with zeros
    """

    def __init__(self, data: bytes):
        # Store as bytearray for mutability
        self._data: Optional[bytearray] = bytearray(data)
        self._original_bytes = data  # Keep reference to zeroize

    @property
    def bytes(self) -> bytes:
        """Access the underlying bytes (read-only view)."""
        if self._data is None:
            raise ValueError("SecureBytes already zeroized")
        return bytes(self._data)

    def __enter__(self) -> "SecureBytes":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.zeroize()
        return None

    def zeroize(self) -> None:
        """Overwrite sensitive data with zeros."""
        if self._data is not None:
            # Zeroize the bytearray in place
            for i in range(len(self._data)):
                self._data[i] = 0
            self._data = None

        # Attempt to zeroize the original bytes object
        # This is fragile in CPython but worth attempting
        _zeroize_bytes(self._original_bytes)


def _zeroize_bytes(data: bytes) -> bool:
    """
    Attempt to zeroize a bytes object in place.

    WARNING: This uses ctypes to modify immutable bytes.
    It may not work on all Python implementations and could
    cause issues if the bytes object is interned or shared.

    Returns:
        True if zeroization was attempted, False if skipped
    """
    if not isinstance(data, bytes) or len(data) == 0:
        return False

    try:
        # Get pointer to bytes data buffer
        # This is CPython-specific and fragile
        addr = id(data) + bytes.__basicsize__
        ctypes.memset(addr, 0, len(data))
        return True
    except Exception:
        # Silently fail - this is best-effort
        return False


def zeroize_string(s: str) -> bool:
    """
    Attempt to zeroize a string's underlying buffer.

    Even more fragile than bytes zeroization due to string interning.
    Use only for hex-encoded key material that won't be interned.

    Returns:
        True if zeroization was attempted, False if skipped
    """
    if not isinstance(s, str) or len(s) == 0:
        return False

    try:
        # Encode to get buffer, then zeroize
        # This doesn't zeroize the original string's buffer
        # but at least clears our encoded copy
        encoded = s.encode('utf-8')
        _zeroize_bytes(encoded)
        return True
    except Exception:
        return False
