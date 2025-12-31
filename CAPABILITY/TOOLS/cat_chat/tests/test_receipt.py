#!/usr/bin/env python3
"""
Receipt Module Tests (Phase 6.1)
"""

import json
import hashlib
import tempfile
import shutil
import os
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


def test_receipt_bytes_deterministic_for_same_inputs():
    """Create bundle, run executor twice, assert receipt bytes identical."""
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()

    try:
        from catalytic_chat.executor import BundleExecutor
        from catalytic_chat.receipt import (
            build_receipt_from_bundle_run,
            write_receipt,
            RECEIPT_VERSION
        )

        create_minimal_bundle(tmpdir1)
        create_minimal_bundle(tmpdir2)

        receipt_out1 = Path(tmpdir1) / "receipt1.json"
        receipt_out2 = Path(tmpdir2) / "receipt2.json"

        executor1 = BundleExecutor(Path(tmpdir1), receipt_out=receipt_out1)
        result1 = executor1.execute()

        executor2 = BundleExecutor(Path(tmpdir2), receipt_out=receipt_out2)
        result2 = executor2.execute()

        receipt1_bytes = receipt_out1.read_bytes()
        receipt2_bytes = receipt_out2.read_bytes()

        assert receipt1_bytes == receipt2_bytes, "Receipt bytes should be identical"
    finally:
        shutil.rmtree(tmpdir1)
        shutil.rmtree(tmpdir2)


@pytest.mark.skip(reason="Artifact verification before execution not implemented")
def test_receipt_failure_on_verify_failed_is_deterministic():
    """Create bundle, tamper to make verify fail, run twice, assert receipts identical."""
    tmpdir1 = tempfile.mkdtemp()
    tmpdir2 = tempfile.mkdtemp()

    try:
        from catalytic_chat.executor import BundleExecutor
        from catalytic_chat.receipt import (
            build_receipt_from_bundle_run,
            write_receipt,
            RECEIPT_VERSION,
            EXECUTOR_VERSION
        )

        bundle_dir1 = Path(tmpdir1)
        create_minimal_bundle(bundle_dir1)
        bundle_dir2 = Path(tmpdir2)
        create_minimal_bundle(bundle_dir2)

        artifact_file = bundle_dir1 / "artifacts" / "test_artifact_001.txt"
        with open(artifact_file, 'r') as f:
            original_content = f.read()
        
        with open(artifact_file, 'w') as f:
            f.write(original_content[:-1])

        receipt_out1 = Path(tmpdir1) / "receipt1.json"
        receipt_out2 = Path(tmpdir2) / "receipt2.json"

        executor1 = BundleExecutor(bundle_dir1, receipt_out=receipt_out1)
        result1 = executor1.execute()

        executor2 = BundleExecutor(bundle_dir2, receipt_out=receipt_out2)
        result2 = executor2.execute()

        assert receipt_out1.exists(), "Receipt should be written even on verify failure"
        assert receipt_out2.exists(), "Receipt should be written even on verify failure"

        receipt1_data = json.loads(receipt_out1.read_text())
        receipt2_data = json.loads(receipt_out2.read_text())

        assert receipt1_data == receipt2_data, "Receipts should be identical"
        assert receipt1_data["outcome"] == "FAILURE"
    finally:
        shutil.rmtree(tmpdir1)
        shutil.rmtree(tmpdir2)


@pytest.mark.skip(reason="Step operation type validation not implemented")
def test_receipt_failure_on_unsupported_step():
    """Create bundle with unsupported op, run executor, assert FAILURE receipt."""
    tmpdir = tempfile.mkdtemp()

    try:
        from catalytic_chat.executor import BundleExecutor
        from catalytic_chat.receipt import (
            build_receipt_from_bundle_run,
            write_receipt,
            RECEIPT_VERSION,
            EXECUTOR_VERSION
        )

        bundle_dir = Path(tmpdir)
        create_minimal_bundle(bundle_dir)

        bundle_json = bundle_dir / "bundle.json"
        with open(bundle_json, 'r') as f:
            bundle_data = json.load(f)

        bundle_data["steps"][0]["op"] = "UNSUPPORTED_OP"

        with open(bundle_json, 'w') as f:
            f.write(json.dumps(bundle_data, sort_keys=True, separators=(",", ":")))

        receipt_out = Path(tmpdir) / "receipt.json"

        with pytest.raises(Exception) as exc_info:
            executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
            executor.execute()

        assert "UNSUPPORTED_STEP" in str(exc_info.value), f"Expected UNSUPPORTED_STEP error, got: {exc_info.value}"
        assert "step_001" in str(exc_info.value), f"Expected step_id in error, got: {exc_info.value}"

        receipt_data = json.loads(receipt_out.read_text())

        assert receipt_data["outcome"] == "FAILURE", f"Expected FAILURE outcome, got: {receipt_data['outcome']}"
        assert receipt_data["error"]["code"] == "UNSUPPORTED_STEP", f"Expected UNSUPPORTED_STEP code, got: {receipt_data['error']['code']}"
        assert receipt_data["error"]["step_id"] == "step_001", f"Expected step_001, got: {receipt_data['error']['step_id']}"
    finally:
        shutil.rmtree(tmpdir)


def test_receipt_schema_validation():
    """Load produced receipt and validate against schema."""
    tmpdir = tempfile.mkdtemp()

    try:
        from catalytic_chat.executor import BundleExecutor
        from catalytic_chat.receipt import validate_receipt_schema

        bundle_dir = Path(tmpdir)
        create_minimal_bundle(bundle_dir)

        receipt_out = Path(tmpdir) / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        executor.execute()

        receipt_data = json.loads(receipt_out.read_text())
        validate_receipt_schema(receipt_data)
    finally:
        shutil.rmtree(tmpdir)


def test_receipt_has_no_absolute_paths_or_timestamps():
    """Ensure receipt JSON text does not contain absolute paths or timestamps."""
    tmpdir = tempfile.mkdtemp()

    try:
        from catalytic_chat.executor import BundleExecutor

        bundle_dir = Path(tmpdir)
        create_minimal_bundle(bundle_dir)

        receipt_out = Path(tmpdir) / "receipt.json"

        executor = BundleExecutor(bundle_dir, receipt_out=receipt_out)
        executor.execute()

        receipt_text = receipt_out.read_text()

        assert ":\\" not in receipt_text, "Receipt should not contain Windows absolute paths"
        assert "/root" not in receipt_text, "Receipt should not contain Unix absolute paths"
        assert "timestamp" not in receipt_text, "Receipt should not contain timestamp field"
        assert "created_at" not in receipt_text, "Receipt should not contain created_at field"
        assert "cwd" not in receipt_text, "Receipt should not contain cwd field"
        assert "now" not in receipt_text.lower(), "Receipt should not contain now field"
    finally:
        shutil.rmtree(tmpdir)
