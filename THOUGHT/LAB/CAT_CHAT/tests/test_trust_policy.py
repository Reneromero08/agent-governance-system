#!/usr/bin/env python3
"""
Trust Policy Module Tests (Phase 6.6)
"""

import json
import hashlib
import subprocess
import sys
import os
from pathlib import Path

import pytest


def test_trust_policy_schema_and_uniqueness():
    """Test trust policy schema validation and uniqueness rules."""
    from catalytic_chat.trust_policy import (
        parse_trust_policy,
        build_trust_index,
        TrustPolicyError
    )

    policy_bytes_empty = json.dumps({
        "policy_version": "1.0.0",
        "allow": []
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes_empty)
    index = build_trust_index(policy)
    assert index == {"by_public_key": {}, "by_validator_id": {}}

    policy_bytes_duplicate_pubkey = json.dumps({
        "policy_version": "1.0.0",
        "allow": [
            {
                "validator_id": "validator1",
                "public_key": "ab" * 32,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            },
            {
                "validator_id": "validator2",
                "public_key": "AB" * 32,
                "schemes": ["ed25519"],
                "scope": ["MERKLE"],
                "enabled": True
            }
        ]
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes_duplicate_pubkey)
    with pytest.raises(TrustPolicyError) as exc_info:
        build_trust_index(policy)
    assert "duplicate public_key" in str(exc_info.value).lower()

    policy_bytes_duplicate_validator_id = json.dumps({
        "policy_version": "1.0.0",
        "allow": [
            {
                "validator_id": "validator1",
                "public_key": "ab" * 32,
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            },
            {
                "validator_id": "validator1",
                "public_key": "cd" * 32,
                "schemes": ["ed25519"],
                "scope": ["MERKLE"],
                "enabled": True
            }
        ]
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes_duplicate_validator_id)
    with pytest.raises(TrustPolicyError) as exc_info:
        build_trust_index(policy)
    assert "duplicate validator_id" in str(exc_info.value).lower()


def test_receipt_attestation_strict_trust_blocks_unknown_key():
    """Test strict trust blocks unknown keys for receipt attestation."""
    from catalytic_chat.attestation import (
        sign_receipt,
        verify_receipt_attestation,
        AttestationError
    )
    from catalytic_chat.trust_policy import parse_trust_policy, build_trust_index

    receipt_dict = {
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
        "attestation": None
    }
    signing_key = b'a' * 32

    signed_receipt = sign_receipt(receipt_dict, signing_key)

    policy_bytes = json.dumps({
        "policy_version": "1.0.0",
        "allow": []
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes)
    trust_index = build_trust_index(policy)

    verify_receipt_attestation(signed_receipt, trust_index, strict=False)

    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_attestation(signed_receipt, trust_index, strict=True)
    assert "UNTRUSTED_VALIDATOR_KEY" in str(exc_info.value)

    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_attestation(signed_receipt, None, strict=True)
    assert "UNTRUSTED_VALIDATOR_KEY" in str(exc_info.value)


def test_receipt_attestation_strict_trust_allows_pinned_key():
    """Test strict trust allows pinned keys for receipt attestation."""
    from catalytic_chat.attestation import (
        sign_receipt,
        verify_receipt_attestation
    )
    from catalytic_chat.trust_policy import parse_trust_policy, build_trust_index

    receipt_dict = {
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
        "attestation": None
    }
    signing_key = b'a' * 32

    signed_receipt = sign_receipt(receipt_dict, signing_key)

    policy_bytes = json.dumps({
        "policy_version": "1.0.0",
        "allow": [
            {
                "validator_id": "validator1",
                "public_key": signed_receipt["attestation"]["public_key"],
                "schemes": ["ed25519"],
                "scope": ["RECEIPT"],
                "enabled": True
            }
        ]
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes)
    trust_index = build_trust_index(policy)

    verify_receipt_attestation(signed_receipt, trust_index, strict=True)


def test_merkle_attestation_strict_trust_blocks_unknown_key_and_allows_pinned():
    """Test strict trust blocks unknown keys and allows pinned keys for Merkle attestation."""
    from catalytic_chat.merkle_attestation import (
        sign_merkle_root,
        verify_merkle_attestation_with_trust,
        MerkleAttestationError
    )
    from catalytic_chat.trust_policy import parse_trust_policy, build_trust_index

    merkle_root = "ab" * 32
    signing_key_bytes = b'c' * 32
    from nacl.signing import SigningKey
    sk = SigningKey(signing_key_bytes)
    signing_key_hex = sk.encode().hex()

    attestation = sign_merkle_root(merkle_root, signing_key_hex)

    policy_bytes = json.dumps({
        "policy_version": "1.0.0",
        "allow": []
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes)
    trust_index = build_trust_index(policy)

    with pytest.raises(MerkleAttestationError) as exc_info:
        verify_merkle_attestation_with_trust(attestation, merkle_root, trust_index, strict=True)
    assert "UNTRUSTED_VALIDATOR_KEY" in str(exc_info.value)

    policy_bytes = json.dumps({
        "policy_version": "1.0.0",
        "allow": [
            {
                "validator_id": "validator1",
                "public_key": attestation["public_key"],
                "schemes": ["ed25519"],
                "scope": ["MERKLE"],
                "enabled": True
            }
        ]
    }).encode('utf-8')

    policy = parse_trust_policy(policy_bytes)
    trust_index = build_trust_index(policy)

    verify_merkle_attestation_with_trust(attestation, merkle_root, trust_index, strict=True)


def test_cli_trust_verify():
    """CLI smoke test for trust verify command."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "trust_policy.json"
        policy_path.write_text(json.dumps({
            "policy_version": "1.0.0",
            "allow": [
                {
                    "validator_id": "validator1",
                    "public_key": "ab" * 32,
                    "schemes": ["ed25519"],
                    "scope": ["RECEIPT", "MERKLE"],
                    "enabled": True
                }
            ]
        }, sort_keys=True))

        env = {**os.environ, "PYTHONPATH": "THOUGHT/LAB/CAT_CHAT"}
        result = subprocess.run(
            [sys.executable, "-m", "catalytic_chat.cli", "trust", "verify", "--trust-policy", str(policy_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"


def test_cli_trust_show():
    """CLI smoke test for trust show command."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        policy_path = Path(tmpdir) / "trust_policy.json"
        policy_path.write_text(json.dumps({
            "policy_version": "1.0.0",
            "allow": [
                {
                    "validator_id": "validator1",
                    "public_key": "ab" * 32,
                    "schemes": ["ed25519"],
                    "scope": ["RECEIPT"],
                    "enabled": True
                },
                {
                    "validator_id": "validator2",
                    "public_key": "cd" * 32,
                    "schemes": ["ed25519"],
                    "scope": ["MERKLE"],
                    "enabled": True
                }
            ]
        }, sort_keys=True))

        env = {**os.environ, "PYTHONPATH": "THOUGHT/LAB/CAT_CHAT"}
        result = subprocess.run(
            [sys.executable, "-m", "catalytic_chat.cli", "trust", "show", "--trust-policy", str(policy_path)],
            env=env,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"

        output = json.loads(result.stdout)
        assert output["policy_version"] == "1.0.0"
        assert output["enabled"] == 2
        assert output["scopes"]["RECEIPT"] == 1
        assert output["scopes"]["MERKLE"] == 1
