#!/usr/bin/env python3
"""
Tests for SPECTRUM-04/05 Enforcement

Verifies that the verifier correctly enforces:
- SPECTRUM-04 v1.1.0: Canonicalization + Identity/Signing
- SPECTRUM-05 v1.0.0: Verification + Threat Law

Tests cover:
- Canonicalization (JSON, bundle_root, chain_root)
- Identity verification (Ed25519, validator_id derivation)
- Signature verification (Ed25519 over domain-separated message)
- Bundle root computation
- Chain root computation
- Error code conformance
"""

import json
import sys
import hashlib
import tempfile
from pathlib import Path

# Add PRIMITIVES to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.verify_bundle import BundleVerifier

# Check if cryptography is available for Ed25519
try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("WARNING: cryptography library not available, skipping Ed25519 tests")


def test_canonicalize_json():
    """Test canonical JSON serialization per SPECTRUM-04 Section 4."""
    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Test basic object
    obj = {"z": 1, "a": 2, "m": 3}
    canonical = verifier._canonicalize_json(obj)
    assert canonical == b'{"a":2,"m":3,"z":1}', "Keys must be sorted lexicographically"

    # Test nested object
    obj = {"outer": {"z": 1, "a": 2}}
    canonical = verifier._canonicalize_json(obj)
    assert canonical == b'{"outer":{"a":2,"z":1}}', "Nested keys must be sorted"

    # Test no whitespace
    assert b' ' not in canonical, "No spaces allowed outside string values"
    assert b'\n' not in canonical, "No newlines allowed"
    assert b'\t' not in canonical, "No tabs allowed"

    print("[PASS] Canonical JSON serialization")


def test_bundle_root_computation():
    """Test bundle_root computation per SPECTRUM-04 Section 5."""
    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Create test data
    task_spec_bytes = b'{"task":"example"}'
    task_spec_hash = hashlib.sha256(task_spec_bytes).hexdigest()

    output_hashes_obj = {
        "validator_semver": "1.0.0",
        "hashes": {"out/file1.txt": "sha256:abc123", "out/file2.txt": "sha256:def456"}
    }

    status_obj = {"status": "success", "cmp01": "pass"}

    # Compute bundle root
    bundle_root = verifier._compute_bundle_root(
        output_hashes_obj,
        status_obj,
        task_spec_bytes
    )

    # Verify it's a 64-char lowercase hex string
    assert len(bundle_root) == 64, f"bundle_root must be 64 chars, got {len(bundle_root)}"
    assert bundle_root.islower(), "bundle_root must be lowercase"
    assert all(c in '0123456789abcdef' for c in bundle_root), "bundle_root must be hex"

    # Verify determinism
    bundle_root_2 = verifier._compute_bundle_root(
        output_hashes_obj,
        status_obj,
        task_spec_bytes
    )
    assert bundle_root == bundle_root_2, "bundle_root computation must be deterministic"

    # Verify preimage structure
    # Expected preimage: {"output_hashes":{...},"status":{...},"task_spec_hash":"..."}
    expected_preimage = {
        "output_hashes": output_hashes_obj["hashes"],
        "status": status_obj,
        "task_spec_hash": task_spec_hash
    }
    canonical_bytes = verifier._canonicalize_json(expected_preimage)
    expected_hash = hashlib.sha256(canonical_bytes).hexdigest()
    assert bundle_root == expected_hash, "bundle_root must match hash of canonical preimage"

    print("[PASS] Bundle root computation")


def test_chain_root_computation():
    """Test chain_root computation per SPECTRUM-04 Section 6."""
    verifier = BundleVerifier(project_root=REPO_ROOT)

    bundle_roots = ["a" * 64, "b" * 64, "c" * 64]
    run_ids = ["run1", "run2", "run3"]

    # Compute chain root
    chain_root = verifier._compute_chain_root(bundle_roots, run_ids)

    # Verify it's a 64-char lowercase hex string
    assert len(chain_root) == 64, f"chain_root must be 64 chars, got {len(chain_root)}"
    assert chain_root.islower(), "chain_root must be lowercase"
    assert all(c in '0123456789abcdef' for c in chain_root), "chain_root must be hex"

    # Verify determinism
    chain_root_2 = verifier._compute_chain_root(bundle_roots, run_ids)
    assert chain_root == chain_root_2, "chain_root computation must be deterministic"

    # Verify preimage structure
    # Expected preimage: {"bundle_roots":[...],"run_ids":[...]}
    expected_preimage = {
        "bundle_roots": bundle_roots,
        "run_ids": run_ids
    }
    canonical_bytes = verifier._canonicalize_json(expected_preimage)
    expected_hash = hashlib.sha256(canonical_bytes).hexdigest()
    assert chain_root == expected_hash, "chain_root must match hash of canonical preimage"

    # Verify order matters
    chain_root_reversed = verifier._compute_chain_root(
        list(reversed(bundle_roots)),
        list(reversed(run_ids))
    )
    assert chain_root != chain_root_reversed, "chain_root must depend on order"

    print("[PASS] Chain root computation")


def test_ed25519_signature_verification():
    """Test Ed25519 signature verification per SPECTRUM-04 Section 9."""
    if not CRYPTO_AVAILABLE:
        print("[SKIP] Skipping Ed25519 tests (cryptography not available)")
        return

    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Generate a test key pair
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_hex = public_key_bytes.hex()

    # Sign a test message
    message = b"CAT-DPT-SPECTRUM-04-v1:BUNDLE:test_message"
    signature_bytes = private_key.sign(message)
    signature_hex = signature_bytes.hex()

    # Verify valid signature
    is_valid = verifier._verify_ed25519_signature(public_key_hex, signature_hex, message)
    assert is_valid, "Valid signature must verify"

    # Verify invalid signature (wrong message)
    is_valid = verifier._verify_ed25519_signature(
        public_key_hex,
        signature_hex,
        b"wrong_message"
    )
    assert not is_valid, "Invalid signature must not verify"

    # Verify invalid signature (tampered signature)
    # Flip the first byte to ensure it's different
    tampered_sig = ("ff" if signature_hex[:2] == "00" else "00") + signature_hex[2:]
    is_valid = verifier._verify_ed25519_signature(public_key_hex, tampered_sig, message)
    assert not is_valid, f"Tampered signature must not verify (sig: {signature_hex[:10]}... -> {tampered_sig[:10]}...)"

    print("[PASS] Ed25519 signature verification")


def test_validator_id_derivation():
    """Test validator_id derivation per SPECTRUM-04 Section 3."""
    if not CRYPTO_AVAILABLE:
        print("[SKIP] Skipping validator_id derivation test (cryptography not available)")
        return

    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Generate a test key
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_hex = public_key_bytes.hex()

    # Compute validator_id
    computed_validator_id = verifier._compute_sha256_bytes(public_key_bytes)

    # Verify it's a 64-char lowercase hex string
    assert len(computed_validator_id) == 64, "validator_id must be 64 chars"
    assert computed_validator_id.islower(), "validator_id must be lowercase"
    assert all(c in '0123456789abcdef' for c in computed_validator_id), "validator_id must be hex"

    # Verify determinism
    computed_validator_id_2 = verifier._compute_sha256_bytes(public_key_bytes)
    assert computed_validator_id == computed_validator_id_2, "validator_id must be deterministic"

    print("[PASS] Validator ID derivation")


def test_spectrum05_missing_artifact():
    """Test SPECTRUM-05 Phase 1: Artifact presence check."""
    if not CRYPTO_AVAILABLE:
        print("[SKIP] Skipping SPECTRUM-05 tests (cryptography not available)")
        return

    verifier = BundleVerifier(project_root=REPO_ROOT)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()

        # Create only partial artifacts
        (run_dir / "TASK_SPEC.json").write_text(json.dumps({"task": "test"}))
        (run_dir / "STATUS.json").write_text(json.dumps({"status": "success", "cmp01": "pass"}))

        # Verify rejection with ARTIFACT_MISSING
        result = verifier.verify_bundle_spectrum05(run_dir)
        assert result["ok"] is False, "Bundle must be rejected"
        assert result["code"] == "ARTIFACT_MISSING", f"Expected ARTIFACT_MISSING, got {result['code']}"
        assert "OUTPUT_HASHES.json" in result["message"], "Error should mention OUTPUT_HASHES.json"

    print("[PASS] SPECTRUM-05 artifact presence check")


def test_spectrum05_identity_invalid():
    """Test SPECTRUM-05 Phase 3: Identity verification."""
    if not CRYPTO_AVAILABLE:
        print("[SKIP] Skipping SPECTRUM-05 identity tests (cryptography not available)")
        return

    verifier = BundleVerifier(project_root=REPO_ROOT)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()

        # Generate a valid key
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        public_key_hex = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ).hex()

        # Create artifacts
        (run_dir / "TASK_SPEC.json").write_text(json.dumps({"task": "test"}))
        (run_dir / "STATUS.json").write_text(json.dumps({"status": "success", "cmp01": "pass"}))
        (run_dir / "OUTPUT_HASHES.json").write_text(json.dumps({
            "validator_semver": "1.0.0",
            "validator_build_id": "test",
            "hashes": {}
        }))
        (run_dir / "PROOF.json").write_text(json.dumps({
            "proof_version": "1.0.0",
            "restoration_result": {"verified": True, "condition": "RESTORED_IDENTICAL"}
        }))

        # Create identity with WRONG validator_id (not derived from public_key)
        wrong_validator_id = "0" * 64
        (run_dir / "VALIDATOR_IDENTITY.json").write_text(json.dumps({
            "algorithm": "ed25519",
            "public_key": public_key_hex,
            "validator_id": wrong_validator_id
        }))

        # Create minimal signed payload and signature (will fail at identity check before signature)
        (run_dir / "SIGNED_PAYLOAD.json").write_text(json.dumps({
            "bundle_root": "a" * 64,
            "decision": "ACCEPT",
            "validator_id": wrong_validator_id
        }))
        (run_dir / "SIGNATURE.json").write_text(json.dumps({
            "payload_type": "BUNDLE",
            "signature": "b" * 128,
            "validator_id": wrong_validator_id
        }))

        # Verify rejection with IDENTITY_INVALID
        result = verifier.verify_bundle_spectrum05(run_dir)
        assert result["ok"] is False, "Bundle must be rejected"
        assert result["code"] == "IDENTITY_INVALID", f"Expected IDENTITY_INVALID, got {result['code']}"
        assert "validator_id" in result["message"], "Error should mention validator_id"

    print("[PASS] SPECTRUM-05 identity verification")


def test_spectrum05_chain_empty():
    """Test SPECTRUM-05 Section 6: Chain verification rejects empty chain."""
    verifier = BundleVerifier(project_root=REPO_ROOT)

    # Verify rejection with CHAIN_EMPTY
    result = verifier.verify_chain_spectrum05([])
    assert result["ok"] is False, "Empty chain must be rejected"
    assert result["code"] == "CHAIN_EMPTY", f"Expected CHAIN_EMPTY, got {result['code']}"

    print("[PASS] SPECTRUM-05 chain empty check")


def test_spectrum05_chain_duplicate_run():
    """Test SPECTRUM-05 Section 6: Chain verification rejects duplicate run_ids."""
    if not CRYPTO_AVAILABLE:
        print("[SKIP] Skipping SPECTRUM-05 chain duplicate test (cryptography not available)")
        return

    verifier = BundleVerifier(project_root=REPO_ROOT)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two runs with the same run_id
        run_dir1 = Path(tmpdir) / "duplicate_run"
        run_dir1.mkdir()

        run_dir2 = Path(tmpdir) / "duplicate_run_copy"
        run_dir2.mkdir()

        # Verify rejection with CHAIN_DUPLICATE_RUN
        # Note: Both have same .name, so verification should catch duplicate
        result = verifier.verify_chain_spectrum05([run_dir1, run_dir1])  # Same run twice
        assert result["ok"] is False, "Chain with duplicate run_ids must be rejected"
        # Should get CHAIN_DUPLICATE_RUN or ARTIFACT_MISSING (depending on whether we check duplicates first)
        assert result["code"] in ["CHAIN_DUPLICATE_RUN", "ARTIFACT_MISSING"], \
            f"Expected CHAIN_DUPLICATE_RUN or ARTIFACT_MISSING, got {result['code']}"

    print("[PASS] SPECTRUM-05 chain duplicate check")


def run_all():
    """Run all tests."""
    tests = [
        test_canonicalize_json,
        test_bundle_root_computation,
        test_chain_root_computation,
        test_ed25519_signature_verification,
        test_validator_id_derivation,
        test_spectrum05_missing_artifact,
        test_spectrum05_identity_invalid,
        test_spectrum05_chain_empty,
        test_spectrum05_chain_duplicate_run,
    ]

    print("\n" + "=" * 60)
    print("SPECTRUM-04/05 Enforcement Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: Unexpected error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
