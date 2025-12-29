"""
Test restoration proof validator.

Run:
    pytest CAPABILITY/TESTBENCH/test_restore_proof.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add CATALYTIC-DPT to path for imports
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_path / "CATALYTIC-DPT"))

from CAPABILITY.PRIMITIVES.restore_proof import RestorationProofValidator


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def proof_schema_path(repo_root):
    return repo_root / "LAW" / "SCHEMAS" / "proof.schema.json"


@pytest.fixture
def validator(proof_schema_path):
    return RestorationProofValidator(proof_schema_path)


def test_identical_artifacts_verified_true(validator):
    """Test that identical pre/post states produce verified=true."""
    pre_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
            "file2.txt": "b" * 64,
        }
    }
    post_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
            "file2.txt": "b" * 64,
        }
    }

    proof = validator.generate_proof(
        run_id="test_run_001",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    assert proof["restoration_result"]["verified"] is True
    assert proof["restoration_result"]["condition"] == "RESTORED_IDENTICAL"
    assert "mismatches" not in proof["restoration_result"]
    assert proof["proof_version"] == "1.0.0"
    assert proof["run_id"] == "test_run_001"
    assert proof["catalytic_domains"] == ["CAPABILITY/PRIMITIVES/_scratch"]
    assert "proof_hash" in proof


def test_hash_mismatch_verified_false(validator):
    """Test that hash mismatch produces verified=false with correct condition."""
    pre_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
        }
    }
    post_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "b" * 64,
        }
    }

    proof = validator.generate_proof(
        run_id="test_run_002",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    assert proof["restoration_result"]["verified"] is False
    assert proof["restoration_result"]["condition"] == "RESTORATION_FAILED_HASH_MISMATCH"
    assert len(proof["restoration_result"]["mismatches"]) == 1
    mismatch = proof["restoration_result"]["mismatches"][0]
    assert mismatch["path"] == "file1.txt"
    assert mismatch["type"] == "hash_mismatch"
    assert mismatch["expected_hash"] == "a" * 64
    assert mismatch["actual_hash"] == "b" * 64


def test_missing_files_verified_false(validator):
    """Test that missing files produce verified=false."""
    pre_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
            "file2.txt": "b" * 64,
        }
    }
    post_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
        }
    }

    proof = validator.generate_proof(
        run_id="test_run_003",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    assert proof["restoration_result"]["verified"] is False
    assert proof["restoration_result"]["condition"] == "RESTORATION_FAILED_MISSING_FILES"
    assert len(proof["restoration_result"]["mismatches"]) == 1
    mismatch = proof["restoration_result"]["mismatches"][0]
    assert mismatch["path"] == "file2.txt"
    assert mismatch["type"] == "missing"
    assert mismatch["expected_hash"] == "b" * 64


def test_extra_files_verified_false(validator):
    """Test that extra files produce verified=false."""
    pre_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
        }
    }
    post_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
            "file2.txt": "b" * 64,
        }
    }

    proof = validator.generate_proof(
        run_id="test_run_004",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    assert proof["restoration_result"]["verified"] is False
    assert proof["restoration_result"]["condition"] == "RESTORATION_FAILED_EXTRA_FILES"
    assert len(proof["restoration_result"]["mismatches"]) == 1
    mismatch = proof["restoration_result"]["mismatches"][0]
    assert mismatch["path"] == "file2.txt"
    assert mismatch["type"] == "extra"
    assert mismatch["actual_hash"] == "b" * 64


def test_deterministic_output(validator):
    """Test that repeated runs produce identical output."""
    pre_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
            "file2.txt": "b" * 64,
        }
    }
    post_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
            "file2.txt": "b" * 64,
        }
    }

    proof1 = validator.generate_proof(
        run_id="test_run_005",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    proof2 = validator.generate_proof(
        run_id="test_run_005",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    # Compare canonical JSON
    canonical1 = json.dumps(proof1, sort_keys=True, separators=(",", ":"))
    canonical2 = json.dumps(proof2, sort_keys=True, separators=(",", ":"))
    assert canonical1 == canonical2


def test_proof_validates_against_schema(validator, proof_schema_path):
    """Test that generated proof validates against proof.schema.json."""
    from jsonschema import Draft7Validator

    pre_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
        }
    }
    post_state = {
        "CAPABILITY/PRIMITIVES/_scratch": {
            "file1.txt": "a" * 64,
        }
    }

    proof = validator.generate_proof(
        run_id="test_run_006",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    # Load schema and validate
    schema = json.loads(Path(proof_schema_path).read_text(encoding="utf-8"))
    v = Draft7Validator(schema)
    errors = list(v.iter_errors(proof))
    assert not errors, f"Proof validation failed: {errors[0].message if errors else ''}"


def test_empty_domains_verified_true(validator):
    """Test that empty pre/post states produce verified=true."""
    pre_state = {"CAPABILITY/PRIMITIVES/_scratch": {}}
    post_state = {"CAPABILITY/PRIMITIVES/_scratch": {}}

    proof = validator.generate_proof(
        run_id="test_run_007",
        catalytic_domains=["CAPABILITY/PRIMITIVES/_scratch"],
        pre_state=pre_state,
        post_state=post_state,
        timestamp="2025-12-25T00:00:00Z",
    )

    assert proof["restoration_result"]["verified"] is True
    assert proof["restoration_result"]["condition"] == "RESTORED_IDENTICAL"
