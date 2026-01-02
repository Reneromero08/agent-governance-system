#!/usr/bin/env python3
"""
Tests for Phase 6.7 PATCH: Trust identity pinning verification

Ensures identity fields are cryptographically bound and validator_id is used for trust lookup.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from catalytic_chat.receipt import receipt_canonical_bytes, compute_receipt_hash, receipt_signed_bytes
from catalytic_chat.attestation import sign_receipt, verify_receipt_attestation, AttestationError
from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation_with_trust, MerkleAttestationError


def test_receipt_attestation_identity_fields_are_signed():
    """Test: Identity fields (validator_id, build_id, public_key) are in signed bytes.

    Any modification to these fields must break signature verification.
    """
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]

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
        build_id="git:abc1234"
    )

    attestation = signed_receipt["attestation"]

    assert attestation["validator_id"] == "test_validator"
    assert attestation["build_id"] == "git:abc1234"

    verify_receipt_attestation(signed_receipt, None, strict=False)

    mutated_receipt = dict(signed_receipt)
    mutated_receipt["attestation"] = dict(attestation)
    mutated_receipt["attestation"]["validator_id"] = "different_validator"

    with pytest.raises(AttestationError, match="bad signature"):
        verify_receipt_attestation(mutated_receipt, None, strict=False)

    mutated_receipt2 = dict(signed_receipt)
    mutated_receipt2["attestation"] = dict(attestation)
    mutated_receipt2["attestation"]["build_id"] = "git:differentsha"

    with pytest.raises(AttestationError, match="bad signature"):
        verify_receipt_attestation(mutated_receipt2, None, strict=False)

    mutated_receipt3 = dict(signed_receipt)
    vk = signing_key.verify_key
    different_key = SigningKey.generate()
    different_vk = different_key.verify_key
    mutated_receipt3["attestation"] = dict(attestation)
    mutated_receipt3["attestation"]["public_key"] = different_vk.encode().hex()

    with pytest.raises(AttestationError, match="bad signature"):
        verify_receipt_attestation(mutated_receipt3, None, strict=False)


def test_trust_lookup_requires_validator_id_match_when_present():
    """Test: When validator_id is present in attestation, it MUST match trust policy entry.

    Verification should fail if:
    - Receipt signed with key_A
    - validator_id in attestation says "validator_B"
    - Trust policy has validator_B with different key

    Even if signature is cryptographically valid, validator_id mismatch should fail.
    """
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    key_A = SigningKey.generate()
    key_B = SigningKey.generate()

    vk_A = key_A.verify_key
    vk_B = key_B.verify_key
    pub_A_hex = vk_A.encode().hex()
    pub_B_hex = vk_B.encode().hex()

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
        key_A.encode()[:32],
        validator_id="validator_B",
        build_id="git:abc1234"
    )

    verify_receipt_attestation(signed_receipt, None, strict=False)

    trust_index = {
        "by_public_key": {
            pub_A_hex.lower(): {
                "validator_id": "validator_A",
                "public_key": pub_A_hex,
                "build_id": "git:abc1234",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            },
            pub_B_hex.lower(): {
                "validator_id": "validator_B",
                "public_key": pub_B_hex,
                "build_id": "git:def5678",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "validator_A": {
                "validator_id": "validator_A",
                "public_key": pub_A_hex,
                "build_id": "git:abc1234",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            },
            "validator_B": {
                "validator_id": "validator_B",
                "public_key": pub_B_hex,
                "build_id": "git:def5678",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        }
    }

    with pytest.raises(AttestationError, match="PUBLIC_KEY_MISMATCH"):
        verify_receipt_attestation(signed_receipt, trust_index, strict=True, strict_identity=False)


def test_trust_lookup_fails_on_unknown_validator_id():
    """Test: Verification fails when validator_id is not in trust policy (strict mode)."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    pub_hex = vk.encode().hex()

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
        validator_id="unknown_validator",
        build_id="git:abc1234"
    )

    trust_index = {
        "by_public_key": {
            pub_hex.lower(): {
                "validator_id": "known_validator",
                "public_key": pub_hex,
                "build_id": "git:abc1234",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        },
        "by_validator_id": {
            "known_validator": {
                "validator_id": "known_validator",
                "public_key": pub_hex,
                "build_id": "git:abc1234",
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        }
    }

    with pytest.raises(AttestationError, match="VALIDATOR_ID_NOT_FOUND"):
        verify_receipt_attestation(signed_receipt, trust_index, strict=True, strict_identity=False)


def test_merkle_attestation_identity_is_signed():
    """Test: Merkle attestation identity fields are included in signed bytes.

    The signing message format must be:
    b"CAT_CHAT_MERKLE_V1:" + merkle_root + b"|VID:" + validator_id + b"|BID:" + build_id + b"|PK:" + public_key

    Any modification to validator_id, build_id, or public_key must break verification.
    """
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()
    vk = signing_key.verify_key
    pub_hex = vk.encode().hex()
    signing_key_hex = key_bytes.hex()

    merkle_root = "a" * 64

    att = sign_merkle_root(
        merkle_root,
        signing_key_hex,
        validator_id="test_validator",
        build_id="git:abc1234"
    )

    assert att["validator_id"] == "test_validator"
    assert att["build_id"] == "git:abc1234"

    verify_merkle_attestation_with_trust(att, merkle_root, None, strict=False, strict_identity=False)

    mutated_att = dict(att)
    mutated_att["validator_id"] = "different_validator"

    with pytest.raises(MerkleAttestationError, match="bad signature"):
        verify_merkle_attestation_with_trust(mutated_att, merkle_root, None, strict=False, strict_identity=False)

    mutated_att2 = dict(att)
    mutated_att2["build_id"] = "git:differentsha"

    with pytest.raises(MerkleAttestationError, match="bad signature"):
        verify_merkle_attestation_with_trust(mutated_att2, merkle_root, None, strict=False, strict_identity=False)

    mutated_att3 = dict(att)
    different_key = SigningKey.generate()
    different_vk = different_key.verify_key
    mutated_att3["public_key"] = different_vk.encode().hex()

    with pytest.raises(MerkleAttestationError, match="bad signature"):
        verify_merkle_attestation_with_trust(mutated_att3, merkle_root, None, strict=False, strict_identity=False)


def test_receipt_signed_bytes_includes_identity_fields():
    """Test: receipt_signed_bytes correctly includes identity fields in canonicalization.

    The SIGNING_STUB must contain scheme, public_key, validator_id, build_id, signature="".
    """
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    vk = signing_key.verify_key
    pub_hex = vk.encode().hex()

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
        "receipt_hash": "test_receipt_hash",
        "attestation": {
            "scheme": "ed25519",
            "public_key": pub_hex,
            "validator_id": "test_validator",
            "build_id": "git:abc1234",
            "signature": "dummy_signature_for_testing"
        }
    }

    signed_bytes = receipt_signed_bytes(receipt)

    signed_text = signed_bytes.decode('utf-8')

    assert '"validator_id":"test_validator"' in signed_text or '"validator_id": "test_validator"' in signed_text
    assert '"build_id":"git:abc1234"' in signed_text or '"build_id": "git:abc1234"' in signed_text
    assert '"public_key":"' + pub_hex.lower() + '"' in signed_text or '"public_key": "' + pub_hex.lower() + '"' in signed_text
    assert '"signature":""' in signed_text or '"signature": ""' in signed_text

    receipt_without_identity = {
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
        "receipt_hash": "test_receipt_hash",
        "attestation": {
            "scheme": "ed25519",
            "public_key": pub_hex,
            "signature": "dummy_signature_for_testing"
        }
    }

    signed_bytes2 = receipt_signed_bytes(receipt_without_identity)

    signed_text2 = signed_bytes2.decode('utf-8')

    assert "test_validator" not in signed_text2
    assert "git:abc1234" not in signed_text2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
