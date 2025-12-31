#!/usr/bin/env python3
"""
Compression Validator Tests (Phase 7)

Deterministic tests for compression protocol validation.
"""

import json
import hashlib
import tempfile
import shutil
from pathlib import Path

import pytest

from catalytic_chat.compression_validator import (
    CompressionValidator,
    CompressionValidationError,
    validate_compression_claim,
    _canonical_json_bytes,
    _estimate_tokens,
    _load_json
)
from catalytic_chat.bundle import BundleBuilder, BundleError, _sha256
from catalytic_chat.message_cassette import MessageCassette, _generate_id
from catalytic_chat.section_indexer import SectionIndexer, build_index


@pytest.fixture
def repo_root(tmp_path):
    """Create a minimal test repository structure."""
    src_dir = tmp_path / "THOUGHT" / "LAB" / "CAT_CHAT"
    src_dir.mkdir(parents=True)

    law_dir = tmp_path / "LAW" / "CANON"
    law_dir.mkdir(parents=True)

    cortex_dir = tmp_path / "CORTEX" / "_generated"
    cortex_dir.mkdir(parents=True)

    (law_dir / "CONTRACT.md").write_text("""
# Test Document

This is a test document for symbol extraction.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
""")

    return tmp_path


@pytest.fixture
def indexed_repo(repo_root):
    """Build index for test repository."""
    build_index(repo_root=repo_root, substrate_mode="sqlite")
    return repo_root


def create_completed_job(repo_root, run_id, request_id, intent):
    """Create a completed job for testing."""
    cassette = MessageCassette(repo_root=repo_root)
    section_indexer = SectionIndexer(repo_root=repo_root)

    sections = section_indexer.build_full_index()
    if not sections:
        raise ValueError("No sections found in index")

    section_id = sections[0].section_id

    request = {
        "run_id": run_id,
        "request_id": request_id,
        "intent": intent,
        "inputs": {
            "files": [],
            "notes": []
        },
        "budgets": {
            "max_steps": 2,
            "max_bytes": 10000000,
            "max_symbols": 10
        }
    }

    try:
        conn = cassette._get_conn()

        message_id = _generate_id("msg", run_id, request_id)
        job_id = _generate_id("job", message_id)

        conn.execute("""
            INSERT INTO cassette_messages
            (message_id, run_id, source, idempotency_key, payload_json)
            VALUES (?, ?, 'USER', ?, ?)
        """, (message_id, run_id, request_id, json.dumps(request)))

        conn.execute("""
            INSERT INTO cassette_jobs
            (job_id, message_id, intent, ordinal)
            VALUES (?, ?, ?, 1)
        """, (job_id, message_id, intent))

        conn.execute("""
            INSERT INTO cassette_job_budgets
            (job_id, bytes_consumed, symbols_consumed)
            VALUES (?, 0, 0)
        """, (job_id,))

        step1_id = _generate_id("step", job_id, "1")
        step1_payload = {
            "step_id": step1_id,
            "ordinal": 1,
            "op": "READ_SECTION",
            "refs": {"section_id": section_id},
            "constraints": {"slice": "head(10)"}
        }

        conn.execute("""
            INSERT INTO cassette_steps
            (step_id, job_id, ordinal, status, payload_json)
            VALUES (?, ?, 1, 'COMMITTED', ?)
        """, (step1_id, job_id, json.dumps(step1_payload)))

        step1_receipt = {
            "outcome": "SUCCESS",
            "result": {"content": "test"}
        }

        receipt1_id = _generate_id("receipt", step1_id, "1")
        conn.execute("""
            INSERT INTO cassette_receipts
            (receipt_id, step_id, job_id, worker_id, fencing_token, outcome, receipt_json)
            VALUES (?, ?, ?, 'test_worker', 0, 'SUCCESS', ?)
        """, (receipt1_id, step1_id, job_id, json.dumps(step1_receipt)))

        conn.commit()

    finally:
        cassette.close()

    return run_id, job_id, message_id


def create_bundle_and_receipts(repo_root, tmp_path, run_id, job_id):
    """Create bundle and receipts for testing."""
    bundle_dir = tmp_path / "test_bundle"
    receipts_dir = tmp_path / "test_receipts"
    bundle_dir.mkdir(parents=True)
    receipts_dir.mkdir(parents=True)

    bundle_builder = BundleBuilder(repo_root=repo_root)
    try:
        result = bundle_builder.build(run_id, job_id, bundle_dir)

        # Create a minimal receipt
        receipt = {
            "receipt_version": "1.0.0",
            "run_id": run_id,
            "job_id": job_id,
            "bundle_id": result["bundle_id"],
            "plan_hash": "dummy_plan_hash",
            "executor_version": "1.0.0",
            "outcome": "SUCCESS",
            "error": None,
            "steps": [{
                "ordinal": 1,
                "step_id": "test_step",
                "op": "READ_SECTION",
                "outcome": "SUCCESS",
                "result": None,
                "error": None
            }],
            "artifacts": [],
            "root_hash": result["root_hash"],
            "parent_receipt_hash": None,
            "receipt_hash": None,
            "attestation": None,
            "receipt_index": 0
        }

        receipt_bytes = _canonical_json_bytes(receipt)
        receipt_hash = _sha256(receipt_bytes.decode('utf-8'))
        receipt["receipt_hash"] = receipt_hash

        receipt_path = receipts_dir / f"{run_id}_receipt_001.json"
        receipt_bytes = _canonical_json_bytes(receipt)
        receipt_path.write_bytes(receipt_bytes)

        return bundle_dir, receipts_dir, result

    finally:
        bundle_builder.close()


def create_compression_claim(bundle_id, run_id, artifact_count, uncompressed_tokens, compressed_tokens, total_bytes):
    """Create a compression claim for testing."""
    components = [
        {"name": "vector_db_only", "included": True},
        {"name": "symbol_lang", "included": True},
        {"name": "f3", "included": False},
        {"name": "cas", "included": True}
    ]

    claim = {
        "claim_version": "1.0.0",
        "run_id": run_id,
        "bundle_id": bundle_id,
        "components": components,
        "reported_metrics": {
            "compression_ratio": compressed_tokens / uncompressed_tokens if uncompressed_tokens > 0 else 0.0,
            "uncompressed_tokens": uncompressed_tokens,
            "compressed_tokens": compressed_tokens,
            "artifact_count": artifact_count,
            "total_bytes": total_bytes,
            "vector_db_tokens": uncompressed_tokens,
            "symbol_lang_tokens": compressed_tokens,
            "cas_tokens": total_bytes // 4 + artifact_count * 10
        },
        "notes": "Test claim"
    }

    # Compute claim_hash
    claim_copy = dict(claim)
    claim_hash = _sha256(_canonical_json_bytes(claim_copy).decode('utf-8'))
    claim["claim_hash"] = claim_hash

    return claim


def test_estimate_tokens():
    """Test token estimation function."""
    # Short text
    assert _estimate_tokens("Hello") == round(5 / 4)

    # Empty text
    assert _estimate_tokens("") == 0

    # Longer text
    text = "Hello world, this is a test."
    expected = round(len(text.encode('utf-8')) / 4)
    assert _estimate_tokens(text) == expected


def test_compression_verify_passes_on_matching_claim(indexed_repo):
    """Test compression verification passes on matching claim."""
    import tempfile
    import hashlib

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        run_id = "test_run_verify_pass"
        job_id = "test_job_verify_pass"

        # Create minimal bundle manually
        bundle_dir = tmp_path / "test_bundle"
        receipts_dir = tmp_path / "test_receipts"
        bundle_dir.mkdir(parents=True)
        receipts_dir.mkdir(parents=True)

        artifacts_dir = bundle_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create a simple artifact
        artifact_content = "Test content for compression.\n"
        artifact_hash = hashlib.sha256(artifact_content.encode('utf-8')).hexdigest()
        artifact_path = artifacts_dir / "test_artifact.txt"
        artifact_path.write_text(artifact_content)

        # Create bundle manifest
        bundle_json_input = {
            "bundle_version": "5.0.0",
            "bundle_id": "",
            "run_id": run_id,
            "job_id": job_id,
            "message_id": "test_msg",
            "plan_hash": "test_plan",
            "steps": [],
            "inputs": {"symbols": [], "files": [], "slices": []},
            "artifacts": [{
                "artifact_id": "test_artifact",
                "kind": "SECTION_SLICE",
                "ref": "test_ref",
                "slice": None,
                "path": "artifacts/test_artifact.txt",
                "sha256": artifact_hash,
                "bytes": len(artifact_content.encode('utf-8'))
            }],
            "hashes": {"root_hash": artifact_hash},
            "provenance": {}
        }
        bundle_id_input = hashlib.sha256(
            json.dumps(bundle_json_input, sort_keys=True, separators=(",", ":")).encode('utf-8')
        ).hexdigest()

        bundle_manifest = bundle_json_input
        bundle_manifest["bundle_id"] = bundle_id_input

        bundle_json_path = bundle_dir / "bundle.json"
        bundle_json_path.write_text(json.dumps(bundle_manifest, sort_keys=True, separators=(",", ":")) + "\n")

        # Create a simple receipt
        receipt = {
            "receipt_version": "1.0.0",
            "run_id": run_id,
            "job_id": job_id,
            "bundle_id": bundle_id_input,
            "plan_hash": "test_plan",
            "executor_version": "1.0.0",
            "outcome": "SUCCESS",
            "error": None,
            "steps": [],
            "artifacts": [{
                "artifact_id": "test_artifact",
                "sha256": artifact_hash,
                "bytes": len(artifact_content.encode('utf-8'))
            }],
            "root_hash": artifact_hash,
            "parent_receipt_hash": None,
            "receipt_hash": None,
            "attestation": None,
            "receipt_index": 0
        }

        receipt_bytes = _canonical_json_bytes(receipt)
        receipt_hash = hashlib.sha256(receipt_bytes.decode('utf-8')).hexdigest()
        receipt["receipt_hash"] = receipt_hash

        receipt_path = receipts_dir / f"{run_id}_receipt_001.json"
        receipt_path.write_bytes(_canonical_json_bytes(receipt))

        # Compute metrics
        artifact_count = 1
        uncompressed_tokens = _estimate_tokens(artifact_content)
        compressed_tokens = _estimate_tokens("test_ref")  # Symbol reference
        total_bytes = len(artifact_content.encode('utf-8'))  # 34 bytes

        claim = create_compression_claim(
            bundle_id_input,
            run_id,
            artifact_count,
            uncompressed_tokens,
            compressed_tokens,
            total_bytes
        )

        claim_path = tmp_path / "claim.json"
        claim_bytes = _canonical_json_bytes(claim)
        claim_path.write_bytes(claim_bytes)

        # Verify claim
        result = validate_compression_claim(
            bundle_path=str(bundle_dir),
            receipts_dir=str(receipts_dir),
            trust_policy_path=None,
            claim_json_path=str(claim_path),
            strict_trust=False,
            strict_identity=False,
            require_attestation=False
        )

        assert result["ok"] is True, f"Expected ok=True, got {result}"
        assert len(result.get("errors", [])) == 0, f"Expected no errors, got {result['errors']}"
        assert result["computed"] is not None, "Expected computed metrics"


def test_compression_verify_fails_on_metric_mismatch(indexed_repo):
    """Test compression verification fails on metric mismatch."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        run_id = "test_run_verify_fail"
        job_id = "test_job_verify_fail"
        create_completed_job(indexed_repo, run_id, "req_verify_fail", "test intent")

        bundle_dir, receipts_dir, bundle_result = create_bundle_and_receipts(
            indexed_repo, tmp_path, run_id, job_id
        )

        artifact_count = bundle_result["artifact_count"]
        uncompressed_tokens = 100
        compressed_tokens = 50

        claim = create_compression_claim(
            bundle_result["bundle_id"],
            run_id,
            artifact_count,
            uncompressed_tokens,
            compressed_tokens,
            1000
        )

        # Flip compressed_tokens to create mismatch
        claim["reported_metrics"]["compressed_tokens"] = 999
        claim_copy = dict(claim)
        claim_hash = _sha256(_canonical_json_bytes(claim_copy).decode('utf-8'))
        claim["claim_hash"] = claim_hash

        claim_path = tmp_path / "claim.json"
        claim_bytes = _canonical_json_bytes(claim)
        claim_path.write_bytes(claim_bytes)

        # Verify claim
        result = validate_compression_claim(
            bundle_path=str(bundle_dir),
            receipts_dir=str(receipts_dir),
            trust_policy_path=None,
            claim_json_path=str(claim_path),
            strict_trust=False,
            strict_identity=False,
            require_attestation=False
        )

        assert result["ok"] is False, f"Expected ok=False, got {result}"
        errors = result.get("errors", [])
        assert len(errors) > 0, f"Expected errors, got {errors}"

        # Check for metric mismatch error
        metric_errors = [e for e in errors if e.get("code") == "METRIC_MISMATCH"]
        assert len(metric_errors) > 0, f"Expected METRIC_MISMATCH error, got {[e['code'] for e in errors]}"


def test_compression_verify_fails_if_not_strictly_verified(indexed_repo):
    """Test compression verification fails if receipts missing."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        run_id = "test_run_verify_missing"
        job_id = "test_job_verify_missing"
        create_completed_job(indexed_repo, run_id, "req_verify_missing", "test intent")

        bundle_dir, receipts_dir, bundle_result = create_bundle_and_receipts(
            indexed_repo, tmp_path, run_id, job_id
        )

        # Delete receipts to simulate missing
        shutil.rmtree(receipts_dir)

        claim = create_compression_claim(
            bundle_result["bundle_id"],
            run_id,
            bundle_result["artifact_count"],
            100,
            50,
            1000
        )

        claim_path = tmp_path / "claim.json"
        claim_bytes = _canonical_json_bytes(claim)
        claim_path.write_bytes(claim_bytes)

        # Verify claim
        result = validate_compression_claim(
            bundle_path=str(bundle_dir),
            receipts_dir=str(receipts_dir),
            trust_policy_path=None,
            claim_json_path=str(claim_path),
            strict_trust=False,
            strict_identity=False,
            require_attestation=False
        )

        assert result["ok"] is False, f"Expected ok=False, got {result}"
        errors = result.get("errors", [])
        assert len(errors) > 0, f"Expected errors, got {errors}"

        # Check for missing receipts error
        missing_errors = [e for e in errors if "RECEIPT" in e.get("code", "")]
        assert len(missing_errors) > 0, f"Expected RECEIPT error, got {[e['code'] for e in errors]}"


def test_compression_outputs_deterministic(indexed_repo):
    """Test compression verification outputs deterministic JSON."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        run_id = "test_run_deterministic"
        job_id = "test_job_deterministic"
        create_completed_job(indexed_repo, run_id, "req_deterministic", "test intent")

        bundle_dir, receipts_dir, bundle_result = create_bundle_and_receipts(
            indexed_repo, tmp_path, run_id, job_id
        )

        claim = create_compression_claim(
            bundle_result["bundle_id"],
            run_id,
            bundle_result["artifact_count"],
            100,
            50,
            1000
        )

        claim_path = tmp_path / "claim.json"
        claim_bytes = _canonical_json_bytes(claim)
        claim_path.write_bytes(claim_bytes)

        # Run verify twice
        result1 = validate_compression_claim(
            bundle_path=str(bundle_dir),
            receipts_dir=str(receipts_dir),
            trust_policy_path=None,
            claim_json_path=str(claim_path),
            strict_trust=False,
            strict_identity=False,
            require_attestation=False
        )

        result2 = validate_compression_claim(
            bundle_path=str(bundle_dir),
            receipts_dir=str(receipts_dir),
            trust_policy_path=None,
            claim_json_path=str(claim_path),
            strict_trust=False,
            strict_identity=False,
            require_attestation=False
        )

        # Serialize both results to JSON
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)

        assert json1 == json2, "Results should be identical (deterministic)"
