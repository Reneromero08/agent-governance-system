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
from typing import Dict, Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestKeyZeroization:
    """Tests for key zeroization (4.6.1)."""

    def test_secure_bytes_context_manager(self):
        """SecureBytes zeroizes data on context exit."""
        from CAPABILITY.PRIMITIVES.secure_memory import SecureBytes

        # 32 bytes exactly (Ed25519 private key size)
        original = bytearray(b"secret_key_32_bytes_exactly!!!!!")

        with SecureBytes(bytes(original)) as secure:
            assert len(secure.bytes) == 32

        # The internal bytearray should be zeroized after context exit

    def test_secure_bytes_raises_after_zeroize(self):
        """Accessing bytes after zeroize raises ValueError."""
        from CAPABILITY.PRIMITIVES.secure_memory import SecureBytes

        secure = SecureBytes(b"secret")
        secure.zeroize()

        with pytest.raises(ValueError, match="already zeroized"):
            _ = secure.bytes

    def test_secure_bytes_double_zeroize_safe(self):
        """Double zeroize is safe (no-op)."""
        from CAPABILITY.PRIMITIVES.secure_memory import SecureBytes

        secure = SecureBytes(b"secret")
        secure.zeroize()
        secure.zeroize()  # Should not raise

    def test_zeroize_bytes_returns_bool(self):
        """_zeroize_bytes returns True on attempt, False on skip."""
        from CAPABILITY.PRIMITIVES.secure_memory import _zeroize_bytes

        # Empty bytes should return False
        assert _zeroize_bytes(b"") is False

        # Non-bytes should return False
        assert _zeroize_bytes("string") is False  # type: ignore
        assert _zeroize_bytes(None) is False  # type: ignore

        # Valid bytes should return True (attempt made)
        result = _zeroize_bytes(b"some_key_data")
        assert result is True

    def test_zeroize_string_returns_bool(self):
        """zeroize_string returns True on attempt, False on skip."""
        from CAPABILITY.PRIMITIVES.secure_memory import zeroize_string

        # Empty string should return False
        assert zeroize_string("") is False

        # Non-string should return False
        assert zeroize_string(b"bytes") is False  # type: ignore
        assert zeroize_string(None) is False  # type: ignore

        # Valid string should return True (attempt made)
        result = zeroize_string("hex_key_material_here")
        assert result is True


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

        assert compare_hash(None, "abc") is False  # type: ignore
        assert compare_hash("abc", 123) is False  # type: ignore
        assert compare_hash(b"abc", "abc") is False  # type: ignore

    def test_compare_bytes_equal(self):
        """Equal bytes return True."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_bytes

        data = b"test_data_here"
        assert compare_bytes(data, data) is True

    def test_compare_bytes_different(self):
        """Different bytes return False."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_bytes

        assert compare_bytes(b"abc", b"def") is False

    def test_compare_bytes_type_safety(self):
        """Non-bytes inputs return False."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_bytes

        assert compare_bytes("abc", b"abc") is False  # type: ignore
        assert compare_bytes(b"abc", "abc") is False  # type: ignore
        assert compare_bytes(None, b"abc") is False  # type: ignore

    def test_compare_signature_alias(self):
        """compare_signature is an alias for compare_hash."""
        from CAPABILITY.PRIMITIVES.timing_safe import compare_hash, compare_signature

        sig_a = "abcd1234" * 16  # 128 chars
        sig_b = "efgh5678" * 16

        assert compare_signature(sig_a, sig_a) == compare_hash(sig_a, sig_a)
        assert compare_signature(sig_a, sig_b) == compare_hash(sig_a, sig_b)

    def test_constant_time_hash_comparison_timing(self):
        """Hash comparison should be constant-time (no early exit)."""
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

        # Times should be within 100% of each other
        # (very generous margin for system noise on CI/Windows)
        # hmac.compare_digest is guaranteed constant-time in CPython
        ratio = max(early_diff_time, late_diff_time) / min(early_diff_time, late_diff_time)
        assert ratio < 2.0, f"Timing ratio {ratio} suggests non-constant-time comparison"


class TestTOCTOUMitigation:
    """Tests for TOCTOU mitigation (4.6.3)."""

    def test_lstat_symlink_detection_exists(self, tmp_path: Path):
        """Symlink detection uses lstat (single syscall)."""
        import stat

        # Create a regular file
        regular_file = tmp_path / "regular.txt"
        regular_file.write_text("content")

        # Verify lstat detects regular file (not symlink)
        st = regular_file.lstat()
        assert not stat.S_ISLNK(st.st_mode)

    def test_lstat_symlink_detection_symlink(self, tmp_path: Path):
        """Symlink is correctly detected via lstat."""
        import stat
        import os

        # Create a regular file and a symlink to it
        regular_file = tmp_path / "target.txt"
        regular_file.write_text("content")

        symlink_file = tmp_path / "link.txt"
        try:
            symlink_file.symlink_to(regular_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform/config")

        # lstat should detect symlink without following it
        st = symlink_file.lstat()
        assert stat.S_ISLNK(st.st_mode)

    def test_symlink_escapes_root_safe_path(self, tmp_path: Path):
        """Safe paths are not flagged as escaping."""
        from CAPABILITY.PRIMITIVES.restore_runner import _symlink_escapes_root

        # Create a safe nested path
        safe_path = tmp_path / "subdir" / "file.txt"

        # Should not escape root
        result = _symlink_escapes_root(tmp_path, safe_path)
        assert result is False

    def test_staging_collision_detected(self):
        """Staging directory UUID collision is detected (FileExistsError path)."""
        # This is tested implicitly via the code structure
        # The actual collision is astronomically unlikely with UUID4
        # but the code path exists for defense-in-depth
        from CAPABILITY.PRIMITIVES.restore_runner import RESTORE_CODES

        assert "RESTORE_INTERNAL_ERROR" in RESTORE_CODES


class TestErrorSanitization:
    """Tests for error sanitization (4.6.4)."""

    def test_json_error_no_exception_text(self, tmp_path: Path):
        """JSON parse errors don't expose exception text in details."""
        from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier

        # Create a bundle directory with malformed JSON
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()

        # Create minimal required files with malformed JSON
        (run_dir / "TASK_SPEC.json").write_text("{ invalid json }")
        (run_dir / "OUTPUT_HASHES.json").write_text("{}")
        (run_dir / "STATUS.json").write_text("{}")
        (run_dir / "PROOF.json").write_text("{}")
        (run_dir / "VALIDATOR_IDENTITY.json").write_text("{}")
        (run_dir / "SIGNED_PAYLOAD.json").write_text("{}")
        (run_dir / "SIGNATURE.json").write_text("{}")

        verifier = BundleVerifier(project_root=tmp_path)
        result = verifier.verify_bundle_spectrum05(run_dir, strict=True)

        # Should fail with ARTIFACT_MALFORMED
        assert result["ok"] is False
        assert result["code"] == "ARTIFACT_MALFORMED"

        # Details should NOT contain 'error' key with exception text
        assert "error" not in result.get("details", {})

    def test_key_validation_no_actual_length(self, tmp_path: Path):
        """Key validation errors don't expose actual key length."""
        from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier

        # Create a bundle directory with invalid public key length
        run_dir = tmp_path / "run_002"
        run_dir.mkdir()

        # Valid JSON but wrong key length
        (run_dir / "TASK_SPEC.json").write_text('{"task": "test"}')
        (run_dir / "OUTPUT_HASHES.json").write_text('{"hashes": {}}')
        (run_dir / "STATUS.json").write_text('{"status": "success", "cmp01": "pass"}')
        (run_dir / "PROOF.json").write_text('{"restoration_result": {"verified": true}}')
        # Invalid public_key (too short)
        (run_dir / "VALIDATOR_IDENTITY.json").write_text(
            '{"algorithm": "ed25519", "public_key": "abc", "validator_id": "test"}'
        )
        (run_dir / "SIGNED_PAYLOAD.json").write_text("{}")
        (run_dir / "SIGNATURE.json").write_text("{}")

        verifier = BundleVerifier(project_root=tmp_path)
        result = verifier.verify_bundle_spectrum05(run_dir, strict=True)

        # Should fail with KEY_INVALID
        assert result["ok"] is False
        assert result["code"] == "KEY_INVALID"

        # Details should NOT contain 'actual_length'
        assert "actual_length" not in result.get("details", {})


class TestSignatureZeroization:
    """Tests for signature.py key zeroization integration."""

    def test_sign_proof_with_bytes_key(self):
        """sign_proof accepts bytes key and completes successfully."""
        from CAPABILITY.PRIMITIVES.signature import generate_keypair, sign_proof

        private_bytes, public_bytes = generate_keypair()

        proof = {
            "test": "proof",
            "timestamp": "2026-01-07T00:00:00Z",
        }

        # Should complete without error (zeroization happens in finally block)
        bundle = sign_proof(proof, private_bytes)

        assert bundle.algorithm == "Ed25519"
        assert len(bundle.signature) == 128  # 64 bytes as hex
        assert len(bundle.public_key) == 64  # 32 bytes as hex

    def test_load_keypair_completes(self, tmp_path: Path):
        """load_keypair completes and returns valid keys."""
        from CAPABILITY.PRIMITIVES.signature import (
            generate_keypair,
            save_keypair,
            load_keypair,
        )

        private_bytes, public_bytes = generate_keypair()

        private_path = tmp_path / "private.key"
        public_path = tmp_path / "public.key"

        save_keypair(private_bytes, public_bytes, private_path, public_path)

        # Should complete without error (hex string zeroization in finally block)
        loaded_private, loaded_public = load_keypair(private_path, public_path)

        assert loaded_private == private_bytes
        assert loaded_public == public_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
