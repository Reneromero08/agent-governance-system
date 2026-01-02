#!/usr/bin/env python3
"""
Receipt Chain Ordering Tests (Phase 6.10)
"""

import json
import hashlib
import tempfile
from pathlib import Path
import pytest


def create_minimal_receipt(receipt_index, run_id="test_run"):
    """Create a minimal receipt for testing."""
    return {
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


def receipt_canonical_bytes(receipt):
    """Convert dict to canonical JSON bytes."""
    json_str = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return (json_str + "\n").encode('utf-8')


def compute_receipt_hash(receipt):
    """Compute receipt hash."""
    receipt_copy = dict(receipt)

    if "receipt_hash" in receipt_copy:
        del receipt_copy["receipt_hash"]

    canonical_bytes = receipt_canonical_bytes(receipt_copy)
    return hashlib.sha256(canonical_bytes).hexdigest()


def write_receipt(out_path: Path, receipt: dict) -> None:
    """Write receipt to file."""
    receipt_bytes = receipt_canonical_bytes(receipt)
    out_path.write_bytes(receipt_bytes)


def test_receipt_chain_sorted_explicitly():
    """Receipt chain sorted explicitly produces deterministic order regardless of FS order."""
    from catalytic_chat.receipt import find_receipt_chain, verify_receipt_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        receipt1 = create_minimal_receipt(0)
        receipt2 = create_minimal_receipt(1)
        receipt3 = create_minimal_receipt(2)

        receipt1_hash = compute_receipt_hash(receipt1)
        receipt1["receipt_hash"] = receipt1_hash

        receipt2["parent_receipt_hash"] = receipt1_hash
        receipt2_hash = compute_receipt_hash(receipt2)
        receipt2["receipt_hash"] = receipt2_hash

        receipt3["parent_receipt_hash"] = receipt2_hash
        receipt3_hash = compute_receipt_hash(receipt3)
        receipt3["receipt_hash"] = receipt3_hash

        write_receipt(tmpdir / "test_run_003.json", receipt3)
        write_receipt(tmpdir / "test_run_001.json", receipt1)
        write_receipt(tmpdir / "test_run_002.json", receipt2)

        receipts = find_receipt_chain(tmpdir, "test_run")

        assert len(receipts) == 3, f"Expected 3 receipts, got {len(receipts)}"
        assert receipts[0]["receipt_hash"] == receipt1_hash
        assert receipts[1]["receipt_hash"] == receipt2_hash
        assert receipts[2]["receipt_hash"] == receipt3_hash

        merkle_root = verify_receipt_chain(receipts, verify_attestation=False)
        assert merkle_root is not None
        assert len(merkle_root) == 64


def test_receipt_chain_fails_on_duplicate_receipt_index():
    """Receipt chain verification fails when two receipts have same receipt_index."""
    from catalytic_chat.receipt import find_receipt_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        receipt1 = create_minimal_receipt(0)
        receipt2 = create_minimal_receipt(1)
        receipt3 = create_minimal_receipt(1)

        receipt1_hash = compute_receipt_hash(receipt1)
        receipt1["receipt_hash"] = receipt1_hash

        receipt2["parent_receipt_hash"] = receipt1_hash
        receipt2_hash = compute_receipt_hash(receipt2)
        receipt2["receipt_hash"] = receipt2_hash

        receipt3["parent_receipt_hash"] = receipt2_hash
        receipt3_hash = compute_receipt_hash(receipt3)
        receipt3["receipt_hash"] = receipt3_hash

        write_receipt(tmpdir / "test_run_001.json", receipt1)
        write_receipt(tmpdir / "test_run_002.json", receipt2)
        write_receipt(tmpdir / "test_run_003.json", receipt3)

        with pytest.raises(ValueError, match="Duplicate receipt_index"):
            find_receipt_chain(tmpdir, "test_run")


def test_receipt_chain_fails_on_mixed_receipt_index():
    """Receipt chain fails when some receipts have receipt_index and others don't."""
    from catalytic_chat.receipt import find_receipt_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        receipt1 = create_minimal_receipt(0)
        receipt2 = create_minimal_receipt(None)
        receipt3 = create_minimal_receipt(2)

        receipt1_hash = compute_receipt_hash(receipt1)
        receipt1["receipt_hash"] = receipt1_hash

        receipt2["parent_receipt_hash"] = receipt1_hash
        receipt2_hash = compute_receipt_hash(receipt2)
        receipt2["receipt_hash"] = receipt2_hash

        receipt3["parent_receipt_hash"] = receipt2_hash
        receipt3_hash = compute_receipt_hash(receipt3)
        receipt3["receipt_hash"] = receipt3_hash

        write_receipt(tmpdir / "test_run_001.json", receipt1)
        write_receipt(tmpdir / "test_run_002.json", receipt2)
        write_receipt(tmpdir / "test_run_003.json", receipt3)

        with pytest.raises(ValueError, match="All receipts must have receipt_index set or all must be null"):
            find_receipt_chain(tmpdir, "test_run")


def test_merkle_root_independent_of_fs_order():
    """Merkle root computation is independent of filesystem creation order."""
    from catalytic_chat.receipt import find_receipt_chain, verify_receipt_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dir_a = tmpdir / "dir_a"
        dir_b = tmpdir / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        receipt1 = create_minimal_receipt(0)
        receipt2 = create_minimal_receipt(1)
        receipt3 = create_minimal_receipt(2)

        receipt1_hash = compute_receipt_hash(receipt1)
        receipt1["receipt_hash"] = receipt1_hash

        receipt2["parent_receipt_hash"] = receipt1_hash
        receipt2_hash = compute_receipt_hash(receipt2)
        receipt2["receipt_hash"] = receipt2_hash

        receipt3["parent_receipt_hash"] = receipt2_hash
        receipt3_hash = compute_receipt_hash(receipt3)
        receipt3["receipt_hash"] = receipt3_hash

        write_receipt(dir_a / "test_run_001.json", receipt1)
        write_receipt(dir_a / "test_run_002.json", receipt2)
        write_receipt(dir_a / "test_run_003.json", receipt3)

        write_receipt(dir_b / "test_run_003.json", receipt3)
        write_receipt(dir_b / "test_run_001.json", receipt1)
        write_receipt(dir_b / "test_run_002.json", receipt2)

        receipts_a = find_receipt_chain(dir_a, "test_run")
        receipts_b = find_receipt_chain(dir_b, "test_run")

        merkle_root_a = verify_receipt_chain(receipts_a, verify_attestation=False)
        merkle_root_b = verify_receipt_chain(receipts_b, verify_attestation=False)

        assert merkle_root_a == merkle_root_b
        assert len(merkle_root_a) == 64


def test_verify_receipt_chain_strictly_monotonic():
    """verify_receipt_chain enforces strictly increasing receipt_index sequence."""
    from catalytic_chat.receipt import verify_receipt_chain

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        receipt1 = create_minimal_receipt(0)
        receipt2 = create_minimal_receipt(1)
        receipt3 = create_minimal_receipt(1)

        receipt1_hash = compute_receipt_hash(receipt1)
        receipt1["receipt_hash"] = receipt1_hash

        receipt2["parent_receipt_hash"] = receipt1_hash
        receipt2_hash = compute_receipt_hash(receipt2)
        receipt2["receipt_hash"] = receipt2_hash

        receipt3["parent_receipt_hash"] = receipt2_hash
        receipt3_hash = compute_receipt_hash(receipt3)
        receipt3["receipt_hash"] = receipt3_hash

        receipts = [receipt1, receipt2, receipt3]

        with pytest.raises(ValueError, match="strictly increasing"):
            verify_receipt_chain(receipts, verify_attestation=False)
