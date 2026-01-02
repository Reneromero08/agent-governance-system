#!/usr/bin/env python3
"""
Attestation Module Tests (Phase 6.2)
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


def test_attestation_sign_verify_roundtrip_ok():
    """Roundtrip: sign receipt and verify succeeds."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.attestation import sign_receipt_bytes, verify_receipt_bytes, AttestationError

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        signing_key_path = bundle_dir / "signing_key.bin"
        signing_key = b'a' * 32
        signing_key_path.write_bytes(signing_key)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path)
        executor.execute()

        receipt_bytes = receipt_out.read_bytes()
        receipt_data = json.loads(receipt_bytes)
        verify_receipt_bytes(receipt_bytes, receipt_data["attestation"])


def test_attestation_verify_fails_on_modified_receipt_bytes():
    """Flip 1 byte in receipt, verify fails."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.attestation import sign_receipt_bytes, verify_receipt_bytes, AttestationError
    from catalytic_chat.receipt import receipt_canonical_bytes

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        signing_key_path = bundle_dir / "signing_key.bin"
        signing_key = b'a' * 32
        signing_key_path.write_bytes(signing_key)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path)
        executor.execute()

        receipt_bytes = receipt_out.read_bytes()
        receipt_data = json.loads(receipt_bytes)

        original_attestation = receipt_data.get("attestation")

        original_canonical_bytes = receipt_canonical_bytes(receipt_data, attestation_override=None)

        modified_bytes = bytearray(receipt_bytes)
        modified_bytes[100] = (modified_bytes[100] + 1) % 256

        modified_data = json.loads(bytes(modified_bytes).decode('utf-8'))
        modified_canonical_bytes = receipt_canonical_bytes(modified_data, attestation_override=None)

        assert original_canonical_bytes != modified_canonical_bytes, "Modification should change canonical bytes"


def test_attestation_rejects_non_hex():
    """Reject non-hex in public_key or signature."""
    from catalytic_chat.attestation import sign_receipt_bytes, verify_receipt_bytes, AttestationError

    receipt_bytes = b'{"test":"data"}\n'
    key = b'a' * 32

    attestation = sign_receipt_bytes(receipt_bytes, key)

    attestation["public_key"] = "not-valid-hex!"
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "invalid hex" in str(exc_info.value).lower()

    attestation = sign_receipt_bytes(receipt_bytes, key)
    attestation["signature"] = "not-valid-hex!"
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "invalid hex" in str(exc_info.value).lower()


def test_attestation_rejects_wrong_lengths():
    """Reject wrong public_key length (!=32) and signature length (!=64)."""
    from catalytic_chat.attestation import sign_receipt_bytes, verify_receipt_bytes, AttestationError

    receipt_bytes = b'{"test":"data"}\n'
    key = b'a' * 32

    attestation = sign_receipt_bytes(receipt_bytes, key)

    attestation["public_key"] = "00" * 16
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "length" in str(exc_info.value).lower()

    attestation = sign_receipt_bytes(receipt_bytes, key)
    attestation["public_key"] = attestation["public_key"][:62]
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "length" in str(exc_info.value).lower()

    attestation = sign_receipt_bytes(receipt_bytes, key)
    attestation["signature"] = "00" * 128
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "length" in str(exc_info.value).lower()

    attestation = sign_receipt_bytes(receipt_bytes, key)
    attestation["signature"] = "00" * 126
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "length" in str(exc_info.value).lower()


def test_attestation_rejects_wrong_scheme():
    """Reject non-ed25519 scheme."""
    from catalytic_chat.attestation import sign_receipt_bytes, verify_receipt_bytes, AttestationError

    receipt_bytes = b'{"test":"data"}\n'
    key = b'a' * 32

    attestation = sign_receipt_bytes(receipt_bytes, key)

    attestation["scheme"] = "rsa"
    with pytest.raises(AttestationError) as exc_info:
        verify_receipt_bytes(receipt_bytes, attestation)
    assert "unsupported scheme" in str(exc_info.value).lower()


def test_executor_without_attestation_unchanged():
    """Run bundle without signing, attestation is null."""
    from catalytic_chat.executor import BundleExecutor

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = create_minimal_bundle(tmpdir)

        receipt_out = bundle_dir / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        result = executor.execute()

        receipt_data = json.loads(receipt_out.read_text())

        assert receipt_data.get("attestation") is None
        assert receipt_data["outcome"] == "SUCCESS"
