#!/usr/bin/env python3
"""
Phase 4.3: Ed25519 Signature Tests (SPECTRUM-04)

Tests for:
- 4.3.1: Ed25519 signing primitives
- 4.3.2: Schema validation with signatures
- 4.3.3: sign_proof.py CLI
- 4.3.4: End-to-end signing workflow

Exit Criteria:
- Signatures generated correctly
- Verification passes for valid signatures
- Verification fails for tampered data
- CLI tools work end-to-end
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.signature import (
    generate_keypair,
    sign_proof,
    verify_signature,
    SignatureBundle,
    load_private_key,
    load_public_key,
    verify_key_id,
    _bytes_to_hex,
    _compute_key_id,
)


def _sha256_hex(data: bytes) -> str:
    """Compute SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# 4.3.1 Tests: Ed25519 Signing Primitives
# =============================================================================


class TestKeypairGeneration:
    """Tests for keypair generation."""

    def test_keypair_length(self):
        """Keypair bytes have correct length."""
        private_key, public_key = generate_keypair()
        assert len(private_key) == 32, "Private key should be 32 bytes (seed)"
        assert len(public_key) == 32, "Public key should be 32 bytes"

    def test_keypairs_are_unique(self):
        """Each call generates unique keypairs."""
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert kp1[0] != kp2[0], "Private keys should be unique"
        assert kp1[1] != kp2[1], "Public keys should be unique"

    def test_key_id_derivation(self):
        """Key ID is first 8 hex chars of sha256(public_key)."""
        _, public_key = generate_keypair()
        expected_id = hashlib.sha256(public_key).hexdigest()[:8]
        actual_id = _compute_key_id(public_key)
        assert actual_id == expected_id

    def test_load_private_key(self):
        """Private key can be loaded from bytes."""
        private_bytes, _ = generate_keypair()
        key_obj = load_private_key(private_bytes)
        assert key_obj is not None

    def test_load_public_key(self):
        """Public key can be loaded from bytes."""
        _, public_bytes = generate_keypair()
        key_obj = load_public_key(public_bytes)
        assert key_obj is not None


class TestSignature:
    """Tests for signing and verification."""

    @pytest.fixture
    def sample_proof(self) -> dict:
        """Sample proof for testing."""
        return {
            "proof_version": "1.0.0",
            "run_id": "test-signature-001",
            "timestamp": "2025-01-01T00:00:00Z",
            "catalytic_domains": ["test_domain"],
            "pre_state": {
                "domain_root_hash": _sha256_hex(b"pre"),
                "file_manifest": {"a.txt": _sha256_hex(b"a")},
            },
            "post_state": {
                "domain_root_hash": _sha256_hex(b"post"),
                "file_manifest": {"a.txt": _sha256_hex(b"a")},
            },
            "restoration_result": {
                "verified": True,
                "condition": "RESTORED_IDENTICAL",
            },
            "proof_hash": _sha256_hex(b"hash"),
        }

    def test_sign_returns_bundle(self, sample_proof: dict):
        """Signing returns a SignatureBundle."""
        private_key, _ = generate_keypair()
        bundle = sign_proof(sample_proof, private_key)

        assert isinstance(bundle, SignatureBundle)
        assert len(bundle.signature) == 128  # 64 bytes as hex
        assert len(bundle.public_key) == 64  # 32 bytes as hex
        assert len(bundle.key_id) == 8
        assert bundle.algorithm == "Ed25519"

    def test_sign_reject_already_signed(self, sample_proof: dict):
        """Signing rejects proof with existing signature."""
        private_key, _ = generate_keypair()
        sample_proof["signature"] = {"dummy": "value"}

        with pytest.raises(ValueError, match="already contains 'signature'"):
            sign_proof(sample_proof, private_key)

    def test_verify_valid_signature(self, sample_proof: dict):
        """Valid signature verifies correctly."""
        private_key, public_key = generate_keypair()
        bundle = sign_proof(sample_proof, private_key)

        assert verify_signature(sample_proof, bundle, public_key)

    def test_verify_without_explicit_key(self, sample_proof: dict):
        """Verification uses embedded public key when none provided."""
        private_key, _ = generate_keypair()
        bundle = sign_proof(sample_proof, private_key)

        # No public key provided - uses bundle's key
        assert verify_signature(sample_proof, bundle)

    def test_verify_wrong_key_rejected(self, sample_proof: dict):
        """Wrong public key rejects signature."""
        private_key1, _ = generate_keypair()
        _, public_key2 = generate_keypair()

        bundle = sign_proof(sample_proof, private_key1)

        # Different key should fail
        assert not verify_signature(sample_proof, bundle, public_key2)

    def test_verify_tampered_proof_rejected(self, sample_proof: dict):
        """Tampered proof rejects signature."""
        private_key, public_key = generate_keypair()
        bundle = sign_proof(sample_proof, private_key)

        # Tamper with proof
        sample_proof["run_id"] = "tampered-id"

        assert not verify_signature(sample_proof, bundle, public_key)

    def test_verify_wrong_algorithm_rejected(self, sample_proof: dict):
        """Wrong algorithm rejects signature."""
        private_key, public_key = generate_keypair()
        bundle = sign_proof(sample_proof, private_key)

        # Tamper with algorithm
        bundle_dict = bundle.to_dict()
        bundle_dict["algorithm"] = "RSA-256"
        tampered_bundle = SignatureBundle.from_dict(bundle_dict)

        assert not verify_signature(sample_proof, tampered_bundle, public_key)

    def test_signature_deterministic(self, sample_proof: dict):
        """Same proof + key = same signature (Ed25519 is deterministic)."""
        private_key, _ = generate_keypair()

        bundle1 = sign_proof(sample_proof, private_key, timestamp="2025-01-01T00:00:00Z")
        bundle2 = sign_proof(sample_proof, private_key, timestamp="2025-01-01T00:00:00Z")

        # Signatures should be identical for same input
        assert bundle1.signature == bundle2.signature

    def test_verify_key_id(self):
        """verify_key_id correctly validates key IDs."""
        _, public_key = generate_keypair()
        expected_id = _compute_key_id(public_key)

        assert verify_key_id(public_key, expected_id)
        assert not verify_key_id(public_key, "00000000")


class TestSignatureBundleSerialization:
    """Tests for SignatureBundle serialization."""

    def test_to_dict_roundtrip(self):
        """SignatureBundle survives dict serialization."""
        private_key, _ = generate_keypair()
        proof = {"test": "data"}
        bundle = sign_proof(proof, private_key)

        bundle_dict = bundle.to_dict()
        restored = SignatureBundle.from_dict(bundle_dict)

        assert restored.signature == bundle.signature
        assert restored.public_key == bundle.public_key
        assert restored.key_id == bundle.key_id
        assert restored.algorithm == bundle.algorithm

    def test_json_roundtrip(self):
        """SignatureBundle survives JSON serialization."""
        private_key, _ = generate_keypair()
        proof = {"test": "data"}
        bundle = sign_proof(proof, private_key)

        json_str = json.dumps(bundle.to_dict())
        parsed = json.loads(json_str)
        restored = SignatureBundle.from_dict(parsed)

        assert restored.signature == bundle.signature


# =============================================================================
# 4.3.2 Tests: Schema Validation
# =============================================================================


class TestSchemaWithSignature:
    """Tests for proof schema with signature field."""

    @pytest.fixture
    def proof_schema_path(self) -> Path:
        return REPO_ROOT / "LAW" / "SCHEMAS" / "proof.schema.json"

    def test_proof_with_signature_validates(self, proof_schema_path: Path):
        """Proof with valid signature field passes schema validation."""
        from jsonschema import Draft7Validator

        schema = json.loads(proof_schema_path.read_text(encoding="utf-8"))
        validator = Draft7Validator(schema)

        proof = {
            "proof_version": "1.0.0",
            "run_id": "test-sig",
            "timestamp": "2025-01-01T00:00:00Z",
            "catalytic_domains": ["domain"],
            "pre_state": {
                "domain_root_hash": "a" * 64,
                "file_manifest": {"f.txt": "b" * 64},
            },
            "post_state": {
                "domain_root_hash": "a" * 64,
                "file_manifest": {"f.txt": "b" * 64},
            },
            "restoration_result": {
                "verified": True,
                "condition": "RESTORED_IDENTICAL",
            },
            "proof_hash": "c" * 64,
            "signature": {
                "signature": "d" * 128,
                "public_key": "e" * 64,
                "key_id": "f" * 8,
                "algorithm": "Ed25519",
                "timestamp": "2025-01-01T00:00:00Z",
            },
        }

        errors = list(validator.iter_errors(proof))
        assert len(errors) == 0, f"Validation failed: {errors[0].message if errors else ''}"


# =============================================================================
# 4.3.3 Tests: CLI
# =============================================================================


class TestSignProofCLI:
    """Tests for sign_proof.py CLI."""

    @pytest.fixture
    def temp_keys(self, tmp_path: Path) -> tuple[Path, Path]:
        """Generate temp keypair files."""
        private_path = tmp_path / "test.key"
        public_path = tmp_path / "test.pub"

        private_key, public_key = generate_keypair()
        private_path.write_text(_bytes_to_hex(private_key))
        public_path.write_text(_bytes_to_hex(public_key))

        return private_path, public_path

    @pytest.fixture
    def temp_proof(self, tmp_path: Path) -> Path:
        """Create a temp proof file."""
        proof = {
            "proof_version": "1.0.0",
            "run_id": "cli-test",
            "timestamp": "2025-01-01T00:00:00Z",
            "catalytic_domains": ["domain"],
            "pre_state": {
                "domain_root_hash": "a" * 64,
                "file_manifest": {},
            },
            "post_state": {
                "domain_root_hash": "a" * 64,
                "file_manifest": {},
            },
            "restoration_result": {
                "verified": True,
                "condition": "RESTORED_IDENTICAL",
            },
            "proof_hash": "b" * 64,
        }

        proof_path = tmp_path / "PROOF.json"
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
        return proof_path

    def test_cli_keygen(self, tmp_path: Path):
        """CLI keygen generates valid keypair."""
        private_path = tmp_path / "new.key"
        public_path = tmp_path / "new.pub"

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "sign_proof.py"),
                "--json",
                "keygen",
                "--private-key", str(private_path),
                "--public-key", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"keygen failed: {result.stderr}"
        assert private_path.exists()
        assert public_path.exists()

        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert "key_id" in output

    def test_cli_sign_and_verify(
        self, temp_keys: tuple[Path, Path], temp_proof: Path
    ):
        """CLI sign and verify workflow."""
        private_path, public_path = temp_keys

        # Sign
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "sign_proof.py"),
                "--json",
                "sign",
                "--proof-file", str(temp_proof),
                "--private-key", str(private_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"sign failed: {result.stderr}"

        # Verify proof now has signature
        proof = json.loads(temp_proof.read_text())
        assert "signature" in proof

        # Verify
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "sign_proof.py"),
                "--json",
                "verify",
                "--proof-file", str(temp_proof),
                "--public-key", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"verify failed: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert output["code"] == "VERIFIED"

    def test_cli_keyinfo(self, temp_keys: tuple[Path, Path]):
        """CLI keyinfo shows key information."""
        _, public_path = temp_keys

        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic" / "sign_proof.py"),
                "--json",
                "keyinfo",
                "--public-key", str(public_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["ok"] is True
        assert "key_id" in output
        assert output["algorithm"] == "Ed25519"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
