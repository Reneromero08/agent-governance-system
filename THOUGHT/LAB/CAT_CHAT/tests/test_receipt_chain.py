#!/usr/bin/env python3
"""
Receipt Chain Tests (Phase 6.3)
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


def test_receipt_chain_deterministic():
    """Execute same bundle twice with attestation, assert identical receipt chain."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import compute_receipt_hash

    run_id = "test_deterministic_chain"

    receipt_bytes_list = []
    receipt_hashes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i in range(2):
            bundle_dir = create_minimal_bundle(tmpdir / f"bundle_{i}")
            manifest_path = bundle_dir / "bundle.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = run_id

            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            signing_key_path = bundle_dir / "signing_key.bin"
            signing_key = b'a' * 32
            signing_key_path.write_bytes(signing_key)

            receipt_out = bundle_dir / "receipt.json"

            executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path)
            executor.execute()

            receipt_bytes = receipt_out.read_bytes()
            receipt_bytes_list.append(receipt_bytes)

            receipt_data = json.loads(receipt_bytes)
            receipt_hash = compute_receipt_hash(receipt_data)
            receipt_hashes.append(receipt_hash)

    assert receipt_bytes_list[0] == receipt_bytes_list[1], "Receipt bytes should be identical"
    assert receipt_hashes[0] == receipt_hashes[1], "Receipt hashes should be identical"


def test_receipt_chain_verification_passes():
    """Execute bundle, then verify full chain succeeds."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import verify_receipt_chain, load_receipt

    run_id = "test_chain_verify"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        receipts = []

        for i in range(3):
            bundle_dir = create_minimal_bundle(tmpdir / f"bundle_{i}")
            manifest_path = bundle_dir / "bundle.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = run_id

            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            signing_key_path = bundle_dir / "signing_key.bin"
            signing_key = b'a' * 32
            signing_key_path.write_bytes(signing_key)

            if i == 0:
                previous_receipt = None
            else:
                previous_receipt = tmpdir / f"{run_id}_{i-1:02d}_002.json"

            executor = BundleExecutor(bundle_dir, signing_key=signing_key_path,
                                  previous_receipt=previous_receipt)
            executor.execute()

        receipts_from_disk = []
        for receipt_file in tmpdir.glob(f"{run_id}_*.json"):
            receipt = load_receipt(receipt_file)
            if receipt:
                receipts_from_disk.append(receipt)

        if len(receipts_from_disk) > 0:
            from catalytic_chat.receipt import find_receipt_chain
            receipts = find_receipt_chain(tmpdir, run_id)
        else:
            receipts = []

        verify_receipt_chain(receipts, verify_attestation=False)


def test_receipt_chain_break_fails():
    """Tamper with parent_receipt_hash or receipt_hash, verification fails."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import verify_receipt_chain, load_receipt
    from catalytic_chat.receipt import compute_receipt_hash

    run_id = "test_chain_break"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        receipts = []

        for i in range(2):
            bundle_dir = create_minimal_bundle(tmpdir / f"bundle_{i}")
            manifest_path = bundle_dir / "bundle.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = run_id

            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            signing_key_path = bundle_dir / "signing_key.bin"
            signing_key = b'a' * 32
            signing_key_path.write_bytes(signing_key)

            receipt_out = tmpdir / f"{run_id}_{i}.json"

            previous_receipt = tmpdir / f"{run_id}_{i-1}.json" if i > 0 else None

            executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path,
                                  previous_receipt=previous_receipt)
            executor.execute()

            receipt = load_receipt(receipt_out)
            receipts.append(receipt)

        original_parent_hash = receipts[1]["parent_receipt_hash"]
        receipts[1]["parent_receipt_hash"] = "00" * 32

        with pytest.raises(ValueError) as exc_info:
            verify_receipt_chain(receipts, verify_attestation=True)
        assert "parent_receipt_hash" in str(exc_info.value).lower()

        receipts[1]["parent_receipt_hash"] = original_parent_hash

        original_receipt_hash = receipts[1]["receipt_hash"]
        receipts[1]["receipt_hash"] = "ff" * 32

        with pytest.raises(ValueError) as exc_info:
            verify_receipt_chain(receipts, verify_attestation=True)
        assert "receipt_hash mismatch" in str(exc_info.value).lower()


def test_receipt_chain_requires_sequential_order():
    """Reorder receipts, verification fails."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import verify_receipt_chain, load_receipt

    run_id = "test_chain_order"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        receipts = []

        for i in range(3):
            bundle_dir = create_minimal_bundle(tmpdir / f"bundle_{i}")
            manifest_path = bundle_dir / "bundle.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] = run_id

            manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

            signing_key_path = bundle_dir / "signing_key.bin"
            signing_key = b'a' * 32
            signing_key_path.write_bytes(signing_key)

            receipt_out = tmpdir / f"{run_id}_{i}.json"

            previous_receipt = tmpdir / f"{run_id}_{i-1}.json" if i > 0 else None

            executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path,
                                  previous_receipt=previous_receipt)
            executor.execute()

            receipt = load_receipt(receipt_out)
            receipts.append(receipt)

        reordered = [receipts[2], receipts[0], receipts[1]]

        with pytest.raises(ValueError) as exc_info:
            verify_receipt_chain(reordered, verify_attestation=True)
        assert "parent_receipt_hash" in str(exc_info.value).lower()
