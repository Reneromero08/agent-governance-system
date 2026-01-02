#!/usr/bin/env python3
"""
Tests for identity pinning (Phase 6.7)

Ensures identity fields are cryptographically bound and validator_id is used for trust lookup.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from catalytic_chat.receipt import receipt_canonical_bytes, compute_receipt_hash, receipt_signed_bytes
from catalytic_chat.attestation import sign_receipt, verify_receipt_attestation, AttestationError
from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation_with_trust, MerkleAttestationError


def test_receipt_attestation_fails_on_build_id_mismatch_strict():
    """Test B: Receipt attestation fails on build_id mismatch in strict mode."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    public_key_hex = vk.encode().hex()

    receipt = {
        "receipt_version": "1.0.0",
        "run_id": "test_run",
        "job_id": "test_job",
        "bundle_id": "test_bundle",
        "plan_hash": "test_plan_hash",
        "executor_version": "1.0.0",
        "outcome": "SUCCESS",
        "error": None,
        "steps": [],
        "artifacts": [],
        "root_hash": "test_root_hash",
        "parent_receipt_hash": None,
        "receipt_hash": None,
        "attestation": None
    }

    receipt["receipt_hash"] = compute_receipt_hash(receipt)

    signed_receipt = sign_receipt(
        receipt,
        key_bytes,
        validator_id="test_validator",
        build_id="git:wrongsha"
    )

    trust_index = {
        "by_public_key": {
            public_key_hex.lower(): {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": "git:correctsha",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "test_validator": {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": "git:correctsha",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        }
    }

    with pytest.raises(AttestationError, match="BUILD_ID_MISMATCH"):
        verify_receipt_attestation(signed_receipt, trust_index, strict=True, strict_identity=True)


def test_merkle_attestation_fails_on_build_id_mismatch_strict():
    """Test C: Merkle attestation fails on build_id mismatch in strict mode."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    public_key_hex = vk.encode().hex()
    signing_key_hex = key_bytes.hex()

    merkle_root = "a" * 64

    att = sign_merkle_root(
        merkle_root,
        signing_key_hex,
        validator_id="test_validator",
        build_id="git:wrongsha"
    )

    trust_index = {
        "by_public_key": {
            public_key_hex.lower(): {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": "git:correctsha",
                "schemes": ["ed25519"],
                "scope": ["MERKLE"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "test_validator": {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": "git:correctsha",
                "schemes": ["ed25519"],
                "scope": ["MERKLE"],
                "enabled": True
            }
        }
    }

    with pytest.raises(MerkleAttestationError, match="BUILD_ID_MISMATCH"):
        verify_merkle_attestation_with_trust(att, merkle_root, trust_index, strict=True, strict_identity=True)


def test_attestation_passes_without_build_id_when_not_pinned():
    """Test D: Attestation passes without build_id when not pinned in policy."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    public_key_hex = vk.encode().hex()

    receipt = {
        "receipt_version": "1.0.0",
        "run_id": "test_run",
        "job_id": "test_job",
        "bundle_id": "test_bundle",
        "plan_hash": "test_plan_hash",
        "executor_version": "1.0.0",
        "outcome": "SUCCESS",
        "error": None,
        "steps": [],
        "artifacts": [],
        "root_hash": "test_root_hash",
        "parent_receipt_hash": None,
        "receipt_hash": None,
        "attestation": None
    }

    receipt["receipt_hash"] = compute_receipt_hash(receipt)

    signed_receipt = sign_receipt(
        receipt,
        key_bytes,
        validator_id="test_validator",
        build_id="git:abcdef"
    )

    trust_index = {
        "by_public_key": {
            public_key_hex.lower(): {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": None,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "test_validator": {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": None,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        }
    }

    verify_receipt_attestation(signed_receipt, trust_index, strict=True, strict_identity=True)


def test_attestation_passes_without_build_id_in_attestation():
    """Test D variant: Attestation passes when build_id not in attestation."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    public_key_hex = vk.encode().hex()

    receipt = {
        "receipt_version": "1.0.0",
        "run_id": "test_run",
        "job_id": "test_job",
        "bundle_id": "test_bundle",
        "plan_hash": "test_plan_hash",
        "executor_version": "1.0.0",
        "outcome": "SUCCESS",
        "error": None,
        "steps": [],
        "artifacts": [],
        "root_hash": "test_root_hash",
        "parent_receipt_hash": None,
        "receipt_hash": None,
        "attestation": None
    }

    receipt["receipt_hash"] = compute_receipt_hash(receipt)

    signed_receipt = sign_receipt(
        receipt,
        key_bytes,
        validator_id="test_validator",
        build_id="git:abcdef"
    )

    trust_index = {
        "by_public_key": {
            public_key_hex.lower(): {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "test_validator": {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        }
    }

    verify_receipt_attestation(signed_receipt, trust_index, strict=True, strict_identity=True)


def test_trust_policy_with_pinned_build_id_enforced():
    """Test E: Trust policy with pinned build_id is enforced."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    public_key_hex = vk.encode().hex()

    receipt = {
        "receipt_version": "1.0.0",
        "run_id": "test_run",
        "job_id": "test_job",
        "bundle_id": "test_bundle",
        "plan_hash": "test_plan_hash",
        "executor_version": "1.0.0",
        "outcome": "SUCCESS",
        "error": None,
        "steps": [],
        "artifacts": [],
        "root_hash": "test_root_hash",
        "parent_receipt_hash": None,
        "receipt_hash": None,
        "attestation": None
    }

    receipt["receipt_hash"] = compute_receipt_hash(receipt)

    correct_build_id = "git:correctsha123"
    signed_receipt = sign_receipt(
        receipt,
        key_bytes,
        validator_id="test_validator",
        build_id=correct_build_id
    )

    trust_index = {
        "by_public_key": {
            public_key_hex.lower(): {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": correct_build_id,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "test_validator": {
                "validator_id": "test_validator",
                "public_key": public_key_hex,
                "build_id": correct_build_id,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        }
    }

    verify_receipt_attestation(signed_receipt, trust_index, strict=True, strict_identity=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
