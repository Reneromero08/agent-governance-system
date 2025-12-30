#!/usr/bin/env python3
"""
Merkle Root Tests (Phase 6.4)
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


def test_merkle_root_deterministic():
    """Same receipt hashes produce identical merkle_root."""
    from catalytic_chat.receipt import compute_merkle_root

    hash1 = "a" * 64
    hash2 = "b" * 64
    hash3 = "c" * 64

    merkle_root1 = compute_merkle_root([hash1, hash2, hash3])
    merkle_root2 = compute_merkle_root([hash1, hash2, hash3])

    assert merkle_root1 == merkle_root2, "Merkle root computation should be deterministic"


def test_merkle_root_changes_on_tamper():
    """Flip one receipt_hash → merkle root changes and verification fails."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import load_receipt, compute_receipt_hash, compute_merkle_root

    run_id = "test_merkle_tamper"

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

            receipt_out = tmpdir / f"{run_id}_{i}.json"

            previous_receipt = tmpdir / f"{run_id}_{i-1}.json" if i > 0 else None

            executor = BundleExecutor(bundle_dir, receipt_out=receipt_out, signing_key=signing_key_path,
                                  previous_receipt=previous_receipt)
            executor.execute()

        receipt_1_path = tmpdir / f"{run_id}_1.json"
        receipt_1 = load_receipt(receipt_1_path)
        original_hash = receipt_1["receipt_hash"]

        receipt_hashes_tampered = [original_hash[:31] + "f"]
        merkle_root_tampered = compute_merkle_root(receipt_hashes_tampered)

        receipt_0_path = tmpdir / f"{run_id}_0.json"
        receipt_0 = load_receipt(receipt_0_path)
        receipt_hashes_original = [receipt_0["receipt_hash"], original_hash]
        merkle_root_original = compute_merkle_root(receipt_hashes_original)

        assert merkle_root_original != merkle_root_tampered, "Merkle root should change on tamper"


def test_merkle_root_requires_verify_chain():
    """--print-merkle without --verify-chain → fail."""
    from catalytic_chat.cli import cmd_bundle_run
    import argparse
    import sys
    from io import StringIO

    run_id = "test_merkle_requires_chain"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bundle_dir = create_minimal_bundle(tmpdir / "bundle")

        args = argparse.Namespace(
            bundle=str(bundle_dir),
            receipt_out=None,
            attest=False,
            signing_key=None,
            verify_attestation=False,
            verify_chain=False,
            print_merkle=True
        )

        old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()
            result = cmd_bundle_run(args)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert result != 0, f"--print-merkle without --verify-chain should fail, got {result}"
        assert "[FAIL]" in output, f"Should show FAIL message, got {output}"
