#!/usr/bin/env python3
"""
Merkle Attestation Module Tests (Phase 6.5)
"""

import json
import hashlib
import tempfile
import shutil
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


def test_merkle_attestation_sign_verify_roundtrip():
    """Roundtrip: sign merkle root and verify succeeds."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation, MerkleAttestationError

    merkle_root_hex = "0" * 64
    signing_key_hex = "a" * 64

    att = sign_merkle_root(merkle_root_hex, signing_key_hex)

    assert att["scheme"] == "ed25519"
    assert att["merkle_root"] == merkle_root_hex
    assert len(att["public_key"]) == 64
    assert len(att["signature"]) == 128

    verify_merkle_attestation(att)


def test_merkle_attestation_rejects_modified_root():
    """Modify merkle root by 1 nibble; verify fails."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation, MerkleAttestationError

    merkle_root_hex = "0" * 64
    signing_key_hex = "a" * 64

    att = sign_merkle_root(merkle_root_hex, signing_key_hex)

    att["merkle_root"] = "1" + att["merkle_root"][1:]

    with pytest.raises(MerkleAttestationError) as exc_info:
        verify_merkle_attestation(att)
    assert "bad signature" in str(exc_info.value).lower()


def test_merkle_attestation_rejects_invalid_merkle_root_length():
    """Reject merkle_root with wrong length."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, MerkleAttestationError

    signing_key_hex = "a" * 64

    with pytest.raises(MerkleAttestationError) as exc_info:
        sign_merkle_root("0" * 63, signing_key_hex)
    assert "merkle_root" in str(exc_info.value).lower()

    with pytest.raises(MerkleAttestationError) as exc_info:
        sign_merkle_root("0" * 65, signing_key_hex)
    assert "merkle_root" in str(exc_info.value).lower()


def test_merkle_attestation_rejects_invalid_signing_key_length():
    """Reject signing_key with wrong length."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, MerkleAttestationError

    merkle_root_hex = "0" * 64

    with pytest.raises(MerkleAttestationError) as exc_info:
        sign_merkle_root(merkle_root_hex, "a" * 63)
    assert "signing_key" in str(exc_info.value).lower()

    with pytest.raises(MerkleAttestationError) as exc_info:
        sign_merkle_root(merkle_root_hex, "a" * 65)
    assert "signing_key" in str(exc_info.value).lower()


def test_merkle_attestation_rejects_invalid_hex():
    """Reject non-hex strings."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, MerkleAttestationError

    merkle_root_hex = "0" * 64
    signing_key_hex = "a" * 64

    with pytest.raises(MerkleAttestationError) as exc_info:
        sign_merkle_root("g" * 64, signing_key_hex)
    assert "hex" in str(exc_info.value).lower()


def test_merkle_attestation_verify_rejects_wrong_scheme():
    """Reject non-ed25519 scheme."""
    from catalytic_chat.merkle_attestation import verify_merkle_attestation, MerkleAttestationError

    att = {
        "scheme": "rsa",
        "merkle_root": "0" * 64,
        "public_key": "a" * 64,
        "signature": "b" * 128
    }

    with pytest.raises(MerkleAttestationError) as exc_info:
        verify_merkle_attestation(att)
    assert "unsupported scheme" in str(exc_info.value).lower()


def test_merkle_attestation_verify_rejects_wrong_key_length():
    """Reject public_key with wrong length."""
    from catalytic_chat.merkle_attestation import verify_merkle_attestation, MerkleAttestationError

    att = {
        "scheme": "ed25519",
        "merkle_root": "0" * 64,
        "public_key": "a" * 31,
        "signature": "b" * 128
    }

    with pytest.raises(MerkleAttestationError):
        verify_merkle_attestation(att)


def test_merkle_attestation_verify_rejects_wrong_signature_length():
    """Reject signature with wrong length."""
    from catalytic_chat.merkle_attestation import verify_merkle_attestation, MerkleAttestationError

    att = {
        "scheme": "ed25519",
        "merkle_root": "0" * 64,
        "public_key": "a" * 64,
        "signature": "b" * 127
    }

    with pytest.raises(MerkleAttestationError):
        verify_merkle_attestation(att)


def test_bundle_run_attest_merkle_outputs_deterministic_bytes():
    """Run bundle twice with --attest-merkle; outputs identical bytes."""
    from catalytic_chat.executor import BundleExecutor

    run_id = "test_deterministic_merkle"
    merkle_key_hex = "b" * 64

    outputs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i in range(2):
            bundle_dir = create_minimal_bundle(tmpdir / f"bundle_{i}")
            manifest_path = bundle_dir / "bundle.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = run_id
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            signing_key_path = bundle_dir / "signing_key.bin"
            signing_key_path.write_bytes(b'a' * 32)

            receipt_out = tmpdir / f"{run_id}_{i}.json"

            executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path)
            executor.execute()

            receipt_data = json.loads(receipt_out.read_text())
            receipt_hashes = [receipt_data.get("receipt_hash")]

            from catalytic_chat.receipt import compute_merkle_root

            merkle_root = compute_merkle_root(receipt_hashes)

            from catalytic_chat.merkle_attestation import sign_merkle_root, write_merkle_attestation
            from catalytic_chat.receipt import canonical_json_bytes

            att = sign_merkle_root(merkle_root, merkle_key_hex)
            att["receipt_count"] = 1
            att["receipt_chain_head_hash"] = receipt_data.get("receipt_hash")
            att["run_id"] = run_id
            att["job_id"] = manifest.get("job_id")
            att["bundle_id"] = manifest.get("bundle_id")

            att_bytes = canonical_json_bytes(att)
            outputs.append(att_bytes)

            receipt_out.unlink()

    assert outputs[0] == outputs[1], "Merkle attestation bytes should be identical"


def test_bundle_run_verify_merkle_attestation_fails_on_mismatch():
    """Produce attestation for one merkle root; verify against different root fails."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, verify_merkle_attestation, MerkleAttestationError

    merkle_root_hex_1 = "0" * 64
    merkle_root_hex_2 = "1" * 64
    signing_key_hex = "a" * 64

    att = sign_merkle_root(merkle_root_hex_1, signing_key_hex)

    att2 = att.copy()
    att2["merkle_root"] = merkle_root_hex_2

    with pytest.raises(MerkleAttestationError) as exc_info:
        verify_merkle_attestation(att2)
    assert "bad signature" in str(exc_info.value).lower()


def test_merkle_attestation_write_load_roundtrip():
    """Write and load merkle attestation from file."""
    from catalytic_chat.merkle_attestation import sign_merkle_root, write_merkle_attestation, load_merkle_attestation

    merkle_root_hex = "c" * 64
    signing_key_hex = "d" * 64

    att = sign_merkle_root(merkle_root_hex, signing_key_hex)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        att_path = tmpdir / "attestation.json"

        write_merkle_attestation(att_path, att)

        loaded_att = load_merkle_attestation(att_path)

        assert loaded_att is not None
        assert loaded_att["scheme"] == att["scheme"]
        assert loaded_att["merkle_root"] == att["merkle_root"]
        assert loaded_att["public_key"] == att["public_key"]
        assert loaded_att["signature"] == att["signature"]


def test_merkle_attestation_load_nonexistent_file():
    """Load from nonexistent file returns None."""
    from catalytic_chat.merkle_attestation import load_merkle_attestation

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        att_path = tmpdir / "nonexistent.json"

        loaded_att = load_merkle_attestation(att_path)

        assert loaded_att is None

