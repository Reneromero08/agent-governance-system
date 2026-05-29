#!/usr/bin/env python3
"""
Multi-Validator Attestation Tests (Phase 6.13)
"""

import json
import hashlib
import tempfile
from pathlib import Path

import pytest


def create_minimal_bundle(bundle_dir):
    """Create a minimal bundle for testing."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    artifact_id = "test_artifact_001"
    artifact_content = "Test content for artifact\n"
    artifact_path = artifacts_dir / f"{artifact_id}.txt"
    artifact_path.write_text(artifact_content)

    content_hash = hashlib.sha256(artifact_content.encode('utf-8')).hexdigest()

    step_id = "step_001"
    steps = [{
        "step_id": step_id,
        "ordinal": 1,
        "op": "READ_SECTION",
        "refs": {"section_id": "test_section"},
        "constraints": {"slice": None},
        "expected_outputs": {}
    }]

    artifacts = [{
        "artifact_id": artifact_id,
        "kind": "SECTION_SLICE",
        "ref": "test_section",
        "slice": None,
        "path": f"artifacts/{artifact_id}.txt",
        "sha256": content_hash,
        "bytes": len(artifact_content.encode('utf-8'))
    }]

    plan_hash = hashlib.sha256(
        json.dumps({
            "run_id": "test_run",
            "steps": steps
        }, sort_keys=True).encode('utf-8')
    ).hexdigest()

    manifest = {
        "bundle_version": "5.0.0",
        "bundle_id": "",
        "run_id": "test_run",
        "job_id": "test_job",
        "message_id": "test_msg",
        "plan_hash": plan_hash,
        "steps": steps,
        "inputs": {"symbols": [], "files": [], "slices": []},
        "artifacts": artifacts,
        "hashes": {"root_hash": ""},
        "provenance": {}
    }

    pre_manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    bundle_id = hashlib.sha256(pre_manifest_json.encode('utf-8')).hexdigest()

    hash_strings = [f"{artifacts[0]['artifact_id']}:{content_hash}"]
    root_hash = hashlib.sha256(("\n".join(hash_strings) + "\n").encode('utf-8')).hexdigest()

    manifest["bundle_id"] = bundle_id
    manifest["hashes"]["root_hash"] = root_hash

    bundle_json = bundle_dir / "bundle.json"

    with open(bundle_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    return bundle_dir


def test_receipt_attestations_order_rejected_if_unsorted():
    """Reject unsorted attestations array."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.attestation import verify_receipt_attestations_with_quorum, sign_receipt, AttestationError

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        executor.execute()

        receipt_data = json.loads(receipt_out.read_text())

        signing_key_1 = b'a' * 32
        att1 = sign_receipt(receipt_data, signing_key_1, validator_id="validator_a")["attestation"]

        signing_key_2 = b'b' * 32
        att2 = {
            "scheme": "ed25519",
            "public_key": "b" * 64,
            "signature": "c" * 128,
            "validator_id": "validator_b"
        }

        unsorted_attestations = [att2, att1]
        receipt_data["attestations"] = unsorted_attestations
        del receipt_data["attestation"]

        policy = {"receipt_attestation_quorum": {"required": 2, "scope": "RECEIPT"}}

        with pytest.raises(AttestationError) as exc_info:
            verify_receipt_attestations_with_quorum(receipt_data, policy, None, False)
        assert "sorted" in str(exc_info.value).lower()


def test_receipt_quorum_passes_with_two_valid_of_two():
    """Pass when quorum requires 2 and we have 2 valid attestations."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.attestation import verify_receipt_attestations_with_quorum, sign_receipt

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        executor.execute()

        receipt_data = json.loads(receipt_out.read_text())

        signing_key_1 = b'a' * 32
        att1 = sign_receipt(receipt_data, signing_key_1, validator_id="validator_a")["attestation"]

        signing_key_2 = b'b' * 32
        att2 = sign_receipt(receipt_data, signing_key_2, validator_id="validator_b")["attestation"]

        sorted_attestations = [att1, att2]
        receipt_data["attestations"] = sorted_attestations
        del receipt_data["attestation"]

        policy = {"receipt_attestation_quorum": {"required": 2, "scope": "RECEIPT"}}

        valid_count = verify_receipt_attestations_with_quorum(receipt_data, policy, None, False)
        assert valid_count == 2


def test_receipt_quorum_fails_with_one_valid_of_two_when_required_two():
    """Fail when quorum requires 2 but only 1 attestation is valid."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.attestation import verify_receipt_attestations_with_quorum, sign_receipt, AttestationError

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        executor.execute()

        receipt_data = json.loads(receipt_out.read_text())

        signing_key_1 = b'a' * 32
        att1 = sign_receipt(receipt_data, signing_key_1, validator_id="validator_a")["attestation"]

        att2 = {
            "scheme": "ed25519",
            "public_key": "b" * 64,
            "signature": "c" * 128,
            "validator_id": "validator_b"
        }

        sorted_attestations = [att1, att2]
        receipt_data["attestations"] = sorted_attestations
        del receipt_data["attestation"]

        policy = {"receipt_attestation_quorum": {"required": 2, "scope": "RECEIPT"}}

        with pytest.raises(AttestationError) as exc_info:
            verify_receipt_attestations_with_quorum(receipt_data, policy, None, False)
        assert "quorum" in str(exc_info.value).lower() or "not met" in str(exc_info.value).lower()


def test_merkle_quorum_passes_and_fails():
    """Test merkle attestation quorum pass and fail cases."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation, MerkleAttestationError

    merkle_root_hex = "0" * 64
    signing_key_1_hex = "a" * 64
    signing_key_2_hex = "b" * 64

    att1 = sign_merkle_root(merkle_root_hex, signing_key_1_hex, validator_id="validator_a")
    att2 = sign_merkle_root(merkle_root_hex, signing_key_2_hex, validator_id="validator_b")

    verify_merkle_attestation(att1)
    verify_merkle_attestation(att2)

    from catalytic_chat.merkle_attestation import verify_merkle_attestations_with_quorum

    sorted_attestations = [att1, att2]
    policy = {"merkle_attestation_quorum": {"required": 2, "scope": "MERKLE"}}

    valid_count = verify_merkle_attestations_with_quorum(sorted_attestations, merkle_root_hex, policy, None, False)
    assert valid_count == 2

    att3 = {
        "scheme": "ed25519",
        "merkle_root": merkle_root_hex,
        "public_key": "c" * 64,
        "signature": "d" * 128,
        "validator_id": "validator_c"
    }

    unsorted_attestations = [att1, att3, att2]
    with pytest.raises(MerkleAttestationError) as exc_info:
        verify_merkle_attestations_with_quorum(unsorted_attestations, merkle_root_hex, policy, None, False)
    assert "sorted" in str(exc_info.value).lower()

    valid_attestations = [att1, att3]
    with pytest.raises(MerkleAttestationError) as exc_info:
        verify_merkle_attestations_with_quorum(valid_attestations, merkle_root_hex, policy, None, False)
    assert "quorum" in str(exc_info.value).lower() or "not met" in str(exc_info.value).lower()


def test_single_attestation_backward_compatible():
    """Old single attestation receipts and merkle attestations still verify."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.attestation import verify_receipt_attestations_with_quorum, sign_receipt
    from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        executor.execute()

        receipt_data = json.loads(receipt_out.read_text())

        signing_key = b'a' * 32
        receipt_signed = sign_receipt(receipt_data, signing_key, validator_id="validator_test")
        receipt_data = receipt_signed

        policy = {"receipt_attestation_quorum": {"required": 1, "scope": "RECEIPT"}}
        valid_count = verify_receipt_attestations_with_quorum(receipt_data, policy, None, False)
        assert valid_count == 1

        merkle_root_hex = "0" * 64
        signing_key_hex = "a" * 64
        merkle_att = sign_merkle_root(merkle_root_hex, signing_key_hex)
        verify_merkle_attestation(merkle_att)

        from catalytic_chat.merkle_attestation import verify_merkle_attestations_with_quorum
        merkle_policy = {"merkle_attestation_quorum": {"required": 1, "scope": "MERKLE"}}
        valid_count = verify_merkle_attestations_with_quorum([merkle_att], merkle_root_hex, merkle_policy, None, False)
        assert valid_count == 1
