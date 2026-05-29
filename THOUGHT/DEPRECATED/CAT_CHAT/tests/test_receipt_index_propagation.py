#!/usr/bin/env python3
"""
Receipt Index Propagation Tests (Phase 6.12)
"""

import json
import hashlib
import tempfile
from pathlib import Path
import pytest


def create_minimal_bundle(bundle_dir, run_id="test_run", num_steps=3):
    """Create a minimal bundle for testing executor."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    steps = []
    for i in range(num_steps):
        step_id = f"step_{i+1:03d}"
        steps.append({
            "step_id": step_id,
            "ordinal": i + 1,
            "op": "READ_SYMBOL" if i % 2 == 0 else "READ_SECTION",
            "refs": {"section_id": f"test_section_{i+1}"},
            "constraints": {"slice": None},
            "expected_outputs": {}
        })

    artifacts = []
    for i in range(num_steps):
        artifact_id = f"test_artifact_{i+1:03d}"
        artifact_content = f"Test content for artifact {i+1}\n"
        artifact_path = artifacts_dir / f"{artifact_id}.txt"
        artifact_path.write_text(artifact_content)

        content_hash = hashlib.sha256(artifact_content.encode('utf-8')).hexdigest()

        artifacts.append({
            "artifact_id": artifact_id,
            "kind": "SECTION_SLICE",
            "ref": f"test_section_{i+1}",
            "slice": None,
            "path": f"artifacts/{artifact_id}.txt",
            "sha256": content_hash,
            "bytes": len(artifact_content.encode('utf-8'))
        })

    plan_hash = hashlib.sha256(
        json.dumps({
            "run_id": run_id,
            "steps": steps
        }, sort_keys=True).encode('utf-8')
    ).hexdigest()

    manifest = {
        "bundle_version": "5.0.0",
        "bundle_id": "",
        "run_id": run_id,
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

    hash_strings = [f"{art['artifact_id']}:{art['sha256']}" for art in artifacts]
    root_hash = hashlib.sha256(("\n".join(hash_strings) + "\n").encode('utf-8')).hexdigest()

    manifest["bundle_id"] = bundle_id
    manifest["hashes"]["root_hash"] = root_hash

    bundle_json = bundle_dir / "bundle.json"
    with open(bundle_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

    return bundle_dir


def create_minimal_receipt(receipt_index, run_id="test_run"):
    """Create a minimal receipt for testing."""
    receipt = {
        "receipt_version": "1.0.0",
        "run_id": run_id,
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
        "attestation": None,
        "receipt_index": receipt_index
    }

    return receipt


def receipt_canonical_bytes(receipt):
    """Convert dict to canonical JSON bytes."""
    json_str = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return (json_str + "\n").encode('utf-8')


def compute_receipt_hash(receipt):
    """Compute receipt hash."""
    from catalytic_chat.receipt import sha256_hex
    receipt_copy = dict(receipt)

    if "receipt_hash" in receipt_copy:
        del receipt_copy["receipt_hash"]

    canonical_bytes = receipt_canonical_bytes(receipt_copy)
    return sha256_hex(canonical_bytes)


def link_receipt_chain(receipts):
    """Link a chain of receipts by setting parent_receipt_hash and recomputing receipt_hash."""
    for i, receipt in enumerate(receipts):
        if i > 0:
            receipt["parent_receipt_hash"] = receipts[i-1]["receipt_hash"]
        receipt["receipt_hash"] = compute_receipt_hash(receipt)
    return receipts


def test_executor_emits_contiguous_receipt_index():
    """Executor emits receipt_index=0 deterministically (no caller control)."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import load_receipt

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        bundle_dir = create_minimal_bundle(tmpdir / "bundle", run_id="test_run", num_steps=3)

        executor = BundleExecutor(bundle_dir)
        result = executor.execute()

        receipt_path = result.get("receipt_path")
        assert receipt_path is not None, "Executor should return receipt_path"

        receipt = load_receipt(Path(receipt_path))
        assert receipt["receipt_index"] == 0, f"Receipt should have index 0 (deterministic, no caller control), got {receipt['receipt_index']}"


def test_multiple_runs_do_not_affect_each_other_indices():
    """Multiple runs with same run_id each produce receipt_index=0 (independent, no caller control)."""
    from catalytic_chat.executor import BundleExecutor
    from catalytic_chat.receipt import load_receipt

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        bundle1_dir = create_minimal_bundle(tmpdir / "bundle1", run_id="test_run", num_steps=3)

        executor1 = BundleExecutor(bundle1_dir)
        result1 = executor1.execute()

        receipt1 = load_receipt(Path(result1.get("receipt_path")))
        assert receipt1["receipt_index"] == 0, f"First run receipt should have index 0 (no caller control), got {receipt1['receipt_index']}"

        bundle2_dir = create_minimal_bundle(tmpdir / "bundle2", run_id="test_run", num_steps=3)

        executor2 = BundleExecutor(bundle2_dir)
        result2 = executor2.execute()

        receipt2 = load_receipt(Path(result2.get("receipt_path")))
        assert receipt2["receipt_index"] == 0, f"Second run receipt should have index 0 (independent, no caller control), got {receipt2['receipt_index']}"


def test_verify_chain_fails_on_gap():
    """verify_receipt_chain fails when receipt_index has gaps."""
    from catalytic_chat.receipt import verify_receipt_chain

    receipt1 = create_minimal_receipt(0)
    receipt2 = create_minimal_receipt(2)
    receipt3 = create_minimal_receipt(3)

    receipts = link_receipt_chain([receipt1, receipt2, receipt3])

    with pytest.raises(ValueError, match="gap|contiguous"):
        verify_receipt_chain(receipts, verify_attestation=False)


def test_verify_chain_fails_on_nonzero_start():
    """verify_receipt_chain fails when receipt_index doesn't start at 0."""
    from catalytic_chat.receipt import verify_receipt_chain

    receipt1 = create_minimal_receipt(1)
    receipt2 = create_minimal_receipt(2)
    receipt3 = create_minimal_receipt(3)

    receipts = link_receipt_chain([receipt1, receipt2, receipt3])

    with pytest.raises(ValueError, match="start at 0"):
        verify_receipt_chain(receipts, verify_attestation=False)


def test_verify_chain_fails_on_mixed_null_and_int():
    """verify_receipt_chain fails when some receipts have receipt_index and others don't."""
    from catalytic_chat.receipt import verify_receipt_chain

    receipt1 = create_minimal_receipt(0)
    receipt2 = create_minimal_receipt(None)
    receipt3 = create_minimal_receipt(2)

    receipts = link_receipt_chain([receipt1, receipt2, receipt3])

    with pytest.raises(ValueError, match="All receipts must have receipt_index set or all must be null"):
        verify_receipt_chain(receipts, verify_attestation=False)


def test_verify_chain_passes_contiguous_indices():
    """verify_receipt_chain passes with contiguous receipt_index sequence."""
    from catalytic_chat.receipt import verify_receipt_chain

    receipt1 = create_minimal_receipt(0)
    receipt2 = create_minimal_receipt(1)
    receipt3 = create_minimal_receipt(2)

    receipts = link_receipt_chain([receipt1, receipt2, receipt3])

    merkle_root = verify_receipt_chain(receipts, verify_attestation=False)
    assert merkle_root is not None
    assert len(merkle_root) == 64
