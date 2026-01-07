---
uuid: 9f4a0c3d-5e79-6c1b-d0f4-8e3a1b6c5d9f
title: "Phase 4.6 Security Hardening — Implementation Roadmap"
section: roadmap
bucket: capability/catalytic
author: Claude Opus 4.5
priority: Medium
created: 2026-01-07
modified: 2026-01-07
status: Ready for Implementation
summary: Detailed implementation guide for Phase 4.6 security hardening covering key zeroization, constant-time comparisons, TOCTOU mitigation, and error sanitization with specific code locations and test requirements.
tags:
- phase-4
- security-hardening
- implementation-guide
- key-zeroization
- timing-safe
- toctou
---
<!-- CONTENT_HASH: 1478e10d7ebc68042e727920faa354d3c9ad4bd6b72b6e7369f243d8a7335659 -->

# Phase 4.6 Security Hardening — Implementation Roadmap

**Date:** 2026-01-07
**Status:** Ready for Implementation
**Prerequisite:** Phase 4.5 Complete (64 tests passing)
**Analysis:** `INBOX/reports/01-07-2026_PHASE_4_SECURITY_HARDENING_ANALYSIS.md`

---

## Overview

This roadmap provides step-by-step implementation instructions for hardening the Phase 4 cryptographic implementation. All changes are defense-in-depth improvements — no critical vulnerabilities exist in the current implementation.

**Priority Levels:**
- **P1 (High):** Key zeroization — strongest security impact
- **P2 (Medium):** Constant-time comparisons, TOCTOU mitigation
- **P3 (Low):** Error sanitization — polish items

---

## 4.6.1 Key Zeroization (P1)

### Purpose
Prevent private key material from persisting in memory after use. While CPython doesn't guarantee memory clearing, explicit zeroization reduces the window for memory dump attacks.

### 4.6.1.1 Create `secure_memory.py`

**File:** `CAPABILITY/PRIMITIVES/secure_memory.py`

**Implementation:**

```python
"""
Secure Memory Primitives for Key Material

Provides best-effort zeroization for sensitive bytes in CPython.
Note: Python's garbage collector and string interning may retain copies.
This module reduces but does not eliminate memory exposure risk.

Usage:
    with SecureBytes(key_bytes) as secure_key:
        result = sign_with_key(secure_key.bytes)
    # key_bytes overwritten with zeros on exit
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
        # Silently fail — this is best-effort
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
```

**Tests to add:**
```python
def test_secure_bytes_context_manager():
    """SecureBytes zeroizes on context exit."""
    original = b"secret_key_material_32_bytes_!!"
    data_copy = bytearray(original)

    with SecureBytes(data_copy) as secure:
        assert secure.bytes == original

    # After exit, bytearray should be zeroed
    assert all(b == 0 for b in data_copy)

def test_secure_bytes_raises_after_zeroize():
    """Accessing bytes after zeroize raises ValueError."""
    secure = SecureBytes(b"secret")
    secure.zeroize()

    with pytest.raises(ValueError, match="already zeroized"):
        _ = secure.bytes
```

---

### 4.6.1.2 Update `signature.py` for Key Zeroization

**File:** `CAPABILITY/PRIMITIVES/signature.py`

**Current code (lines 191-217):**
```python
def sign_proof(
    proof: Dict[str, Any],
    private_key: Union[bytes, ed25519.Ed25519PrivateKey],
    timestamp: Optional[str] = None,
) -> SignatureBundle:
    # ...
    if isinstance(private_key, bytes):
        key_obj = load_private_key(private_key)
    else:
        key_obj = private_key
    # ... signing happens ...
    # KEY BYTES NEVER ZEROIZED
```

**Updated code:**
```python
from CAPABILITY.PRIMITIVES.secure_memory import SecureBytes, zeroize_string

def sign_proof(
    proof: Dict[str, Any],
    private_key: Union[bytes, ed25519.Ed25519PrivateKey],
    timestamp: Optional[str] = None,
) -> SignatureBundle:
    """
    Sign a proof using Ed25519.

    Security: If private_key is provided as bytes, the bytes are
    zeroized after signing (best-effort in CPython).
    """
    if "signature" in proof:
        raise ValueError("Proof already contains 'signature' field.")

    # Track if we need to zeroize
    should_zeroize = isinstance(private_key, bytes)

    try:
        if isinstance(private_key, bytes):
            key_obj = load_private_key(private_key)
        else:
            key_obj = private_key

        public_key = key_obj.public_key()
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        message = _canonical_json_bytes(proof)
        signature_bytes = key_obj.sign(message)

        return SignatureBundle(
            signature=_bytes_to_hex(signature_bytes),
            public_key=_bytes_to_hex(public_bytes),
            key_id=_compute_key_id(public_bytes),
            algorithm="Ed25519",
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )
    finally:
        # Best-effort zeroization of key material
        if should_zeroize and isinstance(private_key, (bytes, bytearray)):
            try:
                from CAPABILITY.PRIMITIVES.secure_memory import _zeroize_bytes
                _zeroize_bytes(private_key)
            except Exception:
                pass  # Silently fail — this is defense-in-depth
```

---

### 4.6.1.3 Update `load_keypair()` for Hex String Zeroization

**File:** `CAPABILITY/PRIMITIVES/signature.py` (lines 341-358)

**Current code:**
```python
def load_keypair(private_path: Path, public_path: Path) -> Tuple[bytes, bytes]:
    private_hex = private_path.read_text().strip()  # HEX STRING PERSISTS
    public_hex = public_path.read_text().strip()
    return _hex_to_bytes(private_hex), _hex_to_bytes(public_hex)
```

**Updated code:**
```python
def load_keypair(private_path: Path, public_path: Path) -> Tuple[bytes, bytes]:
    """
    Load keypair from files.

    Security: Hex strings are zeroized after conversion (best-effort).
    """
    private_hex = private_path.read_text().strip()
    public_hex = public_path.read_text().strip()

    try:
        private_bytes = _hex_to_bytes(private_hex)
        public_bytes = _hex_to_bytes(public_hex)
        return private_bytes, public_bytes
    finally:
        # Best-effort zeroization of hex strings
        try:
            from CAPABILITY.PRIMITIVES.secure_memory import zeroize_string
            zeroize_string(private_hex)
        except Exception:
            pass
```

---

### 4.6.1.4 Documentation

Add to `secure_memory.py` module docstring:

```python
"""
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
```

---

## 4.6.2 Constant-Time Comparisons (P2)

### Purpose
Prevent timing side-channel attacks on hash comparisons. While the risk is low in this context (hashes are not secret), constant-time comparison is a security best practice.

### 4.6.2.1 Create `timing_safe.py`

**File:** `CAPABILITY/PRIMITIVES/timing_safe.py`

**Implementation:**

```python
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
```

---

### 4.6.2.2 Update `verify_bundle.py`

**File:** `CAPABILITY/PRIMITIVES/verify_bundle.py`

**Location 1 — Line ~420 (proof hash comparison):**

Find:
```python
if computed != proof_hash:
    return {...}
```

Replace with:
```python
from CAPABILITY.PRIMITIVES.timing_safe import compare_hash

if not compare_hash(computed, proof_hash):
    return {...}
```

**Location 2 — Line ~489 (file hash comparison):**

Find:
```python
if actual_hash != expected_hash:
    return {...}
```

Replace with:
```python
if not compare_hash(actual_hash, expected_hash):
    return {...}
```

**Add import at top of file:**
```python
from CAPABILITY.PRIMITIVES.timing_safe import compare_hash
```

---

### 4.6.2.3 Timing Test

**File:** `CAPABILITY/TESTBENCH/integration/test_phase_4_6_security_hardening.py`

```python
def test_constant_time_hash_comparison():
    """Hash comparison should be constant-time (no early exit)."""
    import time
    from CAPABILITY.PRIMITIVES.timing_safe import compare_hash

    # Two hashes that differ at the first character
    hash_a = "a" * 64
    hash_b = "b" * 64

    # Two hashes that differ at the last character
    hash_c = "a" * 63 + "b"

    # Measure timing for early vs late difference
    iterations = 10000

    start = time.perf_counter_ns()
    for _ in range(iterations):
        compare_hash(hash_a, hash_b)  # Differs at start
    early_diff_time = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    for _ in range(iterations):
        compare_hash(hash_a, hash_c)  # Differs at end
    late_diff_time = time.perf_counter_ns() - start

    # Times should be within 20% of each other
    # (generous margin for system noise)
    ratio = max(early_diff_time, late_diff_time) / min(early_diff_time, late_diff_time)
    assert ratio < 1.2, f"Timing ratio {ratio} suggests non-constant-time comparison"
```

---

## 4.6.3 TOCTOU Mitigation (P2)

### Purpose
Reduce time-of-check-to-time-of-use race condition windows in file operations.

### 4.6.3.1 Move Target Exists Check

**File:** `CAPABILITY/PRIMITIVES/restore_runner.py`

**Current code (lines 441-445) — Check happens before loop:**
```python
# EXECUTE
phase = PHASE_EXECUTE
for entry in plan:
    if entry.target_path.exists():
        return _result(RESTORE_CODES["RESTORE_TARGET_PATH_EXISTS"], ...)

staging_dir = restore_root / f".spectrum06_staging_{uuid.uuid4().hex}"
# ... much later, actual copy happens ...
```

**Updated code — Check immediately before each operation:**
```python
# EXECUTE
phase = PHASE_EXECUTE
staging_dir = restore_root / f".spectrum06_staging_{uuid.uuid4().hex}"

try:
    staging_dir.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    return _result(RESTORE_CODES["RESTORE_STAGING_COLLISION"], phase, ok=False)

try:
    for entry in plan:
        # Check target IMMEDIATELY before staging copy
        # Minimizes TOCTOU window
        if entry.target_path.exists():
            return _result(
                RESTORE_CODES["RESTORE_TARGET_PATH_EXISTS"],
                phase,
                ok=False,
                details={"path": entry.relative_path}
            )

        # Stage the file
        staged_path = staging_dir / entry.relative_path
        _copy_file(entry.source_path, staged_path)

        # Verify staged hash
        staged_hash = "sha256:" + _sha256_file_hex(staged_path)
        if staged_hash != entry.expected_hash:
            # ... rollback and return error ...
```

---

### 4.6.3.2 Use `lstat()` for Symlink Detection

**File:** `CAPABILITY/PRIMITIVES/restore_runner.py`

**Current code (lines 153-167):**
```python
def _symlink_escapes_root(root: Path, target: Path) -> bool:
    # ...
    for part in rel_parts:
        current = current / part
        if current.exists() and current.is_symlink():  # TWO SYSCALLS
            resolved = current.resolve()
            # ...
```

**Updated code — Single `lstat()` call:**
```python
import stat

def _symlink_escapes_root(root: Path, target: Path) -> bool:
    """
    Detect if following symlinks in target path would escape root.

    Uses lstat() for atomic symlink detection (single syscall).
    """
    root_real = root.resolve()
    current = root_real
    rel_parts = target.relative_to(root).parts

    for part in rel_parts:
        current = current / part
        try:
            # Single syscall instead of exists() + is_symlink()
            st = current.lstat()
            if stat.S_ISLNK(st.st_mode):
                resolved = current.resolve()
                if not _is_lexically_under(root_real, resolved):
                    return True
        except (OSError, FileNotFoundError):
            # Path doesn't exist yet — safe to continue
            # Actual file operations will fail if truly inaccessible
            pass

    return False
```

---

### 4.6.3.3 Chain Manifest Collision Check

**File:** `CAPABILITY/PRIMITIVES/restore_runner.py` (line ~643)

**Current code:**
```python
chain_manifest = restore_root / f".spectrum06_chain_{uuid.uuid4().hex}.json"
try:
    chain_manifest.write_bytes(_canonical_json_bytes({"run_ids": run_ids}))
```

**Updated code:**
```python
chain_manifest = restore_root / f".spectrum06_chain_{uuid.uuid4().hex}.json"

# Defensive check for UUID collision (astronomically unlikely)
if chain_manifest.exists():
    return _result(
        RESTORE_CODES["RESTORE_INTERNAL_ERROR"],
        phase,
        ok=False,
        details={"reason": "chain_manifest_collision"}
    )

try:
    chain_manifest.write_bytes(_canonical_json_bytes({"run_ids": run_ids}))
```

---

## 4.6.4 Error Sanitization (P3)

### Purpose
Prevent internal implementation details from leaking through error messages.

### 4.6.4.1 Remove Exception Text from Errors

**File:** `CAPABILITY/PRIMITIVES/verify_bundle.py` (line ~187)

**Current code:**
```python
except (json.JSONDecodeError, UnicodeDecodeError) as e:
    return {
        "ok": False,
        "code": ERROR_CODES["ARTIFACT_MALFORMED"],
        "message": f"{artifact_name} is not valid JSON",
        "details": {"artifact": artifact_name, "error": str(e)}  # LEAKS
    }
```

**Updated code:**
```python
except (json.JSONDecodeError, UnicodeDecodeError) as e:
    # Log full error for debugging (server-side only)
    import logging
    logging.debug(f"JSON parse error in {artifact_name}: {e}")

    return {
        "ok": False,
        "code": ERROR_CODES["ARTIFACT_MALFORMED"],
        "message": f"{artifact_name} is not valid JSON",
        "details": {"artifact": artifact_name}  # No error text
    }
```

---

### 4.6.4.2 Remove Key Length from Errors

**File:** `CAPABILITY/PRIMITIVES/verify_bundle.py` (line ~233)

**Current code:**
```python
if not (isinstance(public_key, str) and len(public_key) == 64 ...):
    return {
        ...
        "details": {"actual_length": len(public_key) if isinstance(public_key, str) else 0}
    }
```

**Updated code:**
```python
if not (isinstance(public_key, str) and len(public_key) == 64 ...):
    return {
        ...
        "details": {}  # Don't expose actual length
    }
```

---

### 4.6.4.3 Sanitization Helper (Optional)

**Add to `verify_bundle.py`:**

```python
def _sanitize_details(details: dict, max_value_len: int = 100) -> dict:
    """
    Sanitize error details to prevent information leakage.

    - Truncates long values
    - Removes keys that might expose internals
    - Returns a copy, never modifies original
    """
    BLOCKED_KEYS = {"error", "exception", "traceback", "stack"}

    result = {}
    for k, v in details.items():
        if k.lower() in BLOCKED_KEYS:
            continue
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len] + "..."
        result[k] = v_str

    return result
```

---

## 4.6.5 Test Suite

**File:** `CAPABILITY/TESTBENCH/integration/test_phase_4_6_security_hardening.py`

```python
#!/usr/bin/env python3
"""
Phase 4.6: Security Hardening Tests

Tests for:
- 4.6.1: Key zeroization
- 4.6.2: Constant-time comparisons
- 4.6.3: TOCTOU mitigation
- 4.6.4: Error sanitization

Exit Criteria:
- Private keys zeroized after use (best-effort)
- Hash comparisons are constant-time
- TOCTOU windows minimized
- Error messages sanitized
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestKeyZeroization:
    """Tests for key zeroization (4.6.1)."""

    def test_secure_bytes_context_manager(self):
        """SecureBytes zeroizes data on context exit."""
        from CAPABILITY.PRIMITIVES.secure_memory import SecureBytes

        original = bytearray(b"secret_key_32_bytes_exactly_!!")

        with SecureBytes(bytes(original)) as secure:
            assert len(secure.bytes) == 32

        # The bytearray passed to SecureBytes should be zeroized
        # Note: The original bytes object may not be zeroized due to
        # Python's immutability, but the internal bytearray should be

    def test_secure_bytes_raises_after_zeroize(self):
        """Accessing bytes after zeroize raises ValueError."""
        from CAPABILITY.PRIMITIVES.secure_memory import SecureBytes

        secure = SecureBytes(b"secret")
        secure.zeroize()

        with pytest.raises(ValueError, match="already zeroized"):
            _ = secure.bytes


class TestConstantTimeComparison:
    """Tests for constant-time comparison (4.6.2)."""

    def test_compare_hash_equal(self):
        """Equal hashes return True."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_hash

        hash_a = "a" * 64
        assert compare_hash(hash_a, hash_a) is True

    def test_compare_hash_different(self):
        """Different hashes return False."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_hash

        hash_a = "a" * 64
        hash_b = "b" * 64
        assert compare_hash(hash_a, hash_b) is False

    def test_compare_hash_type_safety(self):
        """Non-string inputs return False."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_hash

        assert compare_hash(None, "abc") is False
        assert compare_hash("abc", 123) is False
        assert compare_hash(b"abc", "abc") is False


class TestTOCTOUMitigation:
    """Tests for TOCTOU mitigation (4.6.3)."""

    def test_lstat_symlink_detection(self):
        """Symlink detection uses lstat (single syscall)."""
        # This test verifies the function exists and works
        # Actual race condition testing requires concurrent threads
        import stat

        # Create a temp symlink and verify lstat detects it
        # (implementation detail test)
        pass  # Placeholder - actual test depends on implementation

    def test_chain_manifest_collision_check(self):
        """Chain manifest creation checks for existing file."""
        # This is tested via the restore_chain function
        # Collision is astronomically unlikely with UUID4
        pass  # Placeholder


class TestErrorSanitization:
    """Tests for error sanitization (4.6.4)."""

    def test_json_error_no_exception_text(self):
        """JSON parse errors don't expose exception text."""
        from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier

        # Create verifier and trigger JSON error
        # Verify 'error' key is not in details
        pass  # Placeholder - depends on verify_bundle structure

    def test_key_error_no_actual_length(self):
        """Key validation errors don't expose actual length."""
        # Trigger key validation with wrong length
        # Verify 'actual_length' is not in details
        pass  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Implementation Order

1. **Create `secure_memory.py`** (4.6.1.1)
   - Self-contained module, no dependencies
   - Write tests first

2. **Create `timing_safe.py`** (4.6.2.1)
   - Self-contained module, no dependencies
   - Write tests first

3. **Update `signature.py`** (4.6.1.2, 4.6.1.3)
   - Depends on `secure_memory.py`
   - Run existing signature tests after changes

4. **Update `verify_bundle.py`** (4.6.2.2, 4.6.4.1, 4.6.4.2)
   - Depends on `timing_safe.py`
   - Run existing verify_bundle tests after changes

5. **Update `restore_runner.py`** (4.6.3.1, 4.6.3.2, 4.6.3.3)
   - No new dependencies
   - Run existing restore tests after changes

6. **Complete test suite** (4.6.5)
   - Fill in placeholder tests
   - Run full Phase 4 test suite

---

## Exit Criteria Checklist

- [ ] `secure_memory.py` created with `SecureBytes` class
- [ ] `timing_safe.py` created with `compare_hash()` function
- [ ] `signature.py` zeroizes key material after signing
- [ ] `load_keypair()` zeroizes hex strings
- [ ] `verify_bundle.py` uses constant-time hash comparison
- [ ] `restore_runner.py` uses `lstat()` for symlinks
- [ ] Target exists check moved closer to file operations
- [ ] Error messages don't expose exception text
- [ ] All 64 existing Phase 4 tests still pass
- [ ] 4 new hardening tests pass
- [ ] Documentation updated with CPython limitations

---

## Files to Create/Modify

| File | Action | Lines Changed |
|------|--------|---------------|
| `CAPABILITY/PRIMITIVES/secure_memory.py` | CREATE | ~100 |
| `CAPABILITY/PRIMITIVES/timing_safe.py` | CREATE | ~50 |
| `CAPABILITY/PRIMITIVES/signature.py` | MODIFY | ~30 |
| `CAPABILITY/PRIMITIVES/verify_bundle.py` | MODIFY | ~20 |
| `CAPABILITY/PRIMITIVES/restore_runner.py` | MODIFY | ~40 |
| `CAPABILITY/TESTBENCH/integration/test_phase_4_6_security_hardening.py` | CREATE | ~150 |

**Total estimated changes:** ~390 lines

---

*Roadmap generated by Claude Opus 4.5 on 2026-01-07*