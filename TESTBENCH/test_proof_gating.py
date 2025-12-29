"""
Test proof-gated acceptance.

Validates that run acceptance is STRICTLY proof-driven:
- Valid PROOF.json with verified=true → run accepted
- Invalid PROOF.json with verified=false → run rejected

Run:
    pytest CATALYTIC-DPT/TESTBENCH/test_proof_gating.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add CATALYTIC-DPT and TOOLS to path
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_path / "CATALYTIC-DPT"))
sys.path.insert(0, str(repo_root_path / "TOOLS"))

from PRIMITIVES.restore_proof import RestorationProofValidator
from catalytic_validator import CatalyticLedgerValidator


def create_minimal_ledger(ledger_dir: Path, proof: dict) -> None:
    """Create minimal valid ledger with given PROOF.json (Phase 0.2 canonical set)."""
    # Canonical artifacts
    jobspec = {
        "job_id": "test_run_001",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test proof-gated acceptance",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": ["CATALYTIC-DPT/_scratch"],
        "determinism": "deterministic",
    }
    (ledger_dir / "JOBSPEC.json").write_text(json.dumps(jobspec, indent=2))

    status = {"status": "succeeded", "restoration_verified": True, "exit_code": 0, "validation_passed": True}
    (ledger_dir / "STATUS.json").write_text(json.dumps(status, indent=2))

    (ledger_dir / "INPUT_HASHES.json").write_text(json.dumps({}, indent=2))
    (ledger_dir / "OUTPUT_HASHES.json").write_text(json.dumps({}, indent=2))
    (ledger_dir / "DOMAIN_ROOTS.json").write_text(json.dumps({}, indent=2))
    (ledger_dir / "LEDGER.jsonl").write_text("{}\n")

    validator_id = {"validator_semver": "0.1.0", "validator_build_id": "test"}
    (ledger_dir / "VALIDATOR_ID.json").write_text(json.dumps(validator_id, indent=2))

    (ledger_dir / "PROOF.json").write_text(json.dumps(proof, indent=2))

    # Legacy artifacts (backwards compatibility)
    run_info = {
        "run_id": "test_run_001",
        "timestamp": "2025-12-25T00:00:00Z",
        "intent": "Test proof-gated acceptance",
        "catalytic_domains": ["CATALYTIC-DPT/_scratch"],
        "exit_code": 0,
    }
    (ledger_dir / "RUN_INFO.json").write_text(json.dumps(run_info, indent=2))

    pre_manifest = {"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}}
    (ledger_dir / "PRE_MANIFEST.json").write_text(json.dumps(pre_manifest, indent=2))

    post_manifest = {"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}}
    (ledger_dir / "POST_MANIFEST.json").write_text(json.dumps(post_manifest, indent=2))

    restore_diff = {"CATALYTIC-DPT/_scratch": {"added": {}, "removed": {}, "changed": {}}}
    (ledger_dir / "RESTORE_DIFF.json").write_text(json.dumps(restore_diff, indent=2))

    outputs = []
    (ledger_dir / "OUTPUTS.json").write_text(json.dumps(outputs, indent=2))


@pytest.fixture
def proof_schema_path():
    return Path(__file__).resolve().parents[2] / "CATALYTIC-DPT" / "SCHEMAS" / "proof.schema.json"


@pytest.fixture
def proof_validator(proof_schema_path):
    return RestorationProofValidator(proof_schema_path)


def test_valid_proof_accepted(proof_validator):
    """Test that valid PROOF.json with verified=true leads to acceptance."""
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Generate valid proof (identical pre/post)
        proof = proof_validator.generate_proof(
            run_id="test_run_001",
            catalytic_domains=["CATALYTIC-DPT/_scratch"],
            pre_state={"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}},
            post_state={"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}},
            timestamp="2025-12-25T00:00:00Z",
        )

        assert proof["restoration_result"]["verified"] is True
        assert proof["restoration_result"]["condition"] == "RESTORED_IDENTICAL"

        create_minimal_ledger(ledger_dir, proof)

        # Validate ledger
        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is True
        assert report["valid"] is True
        assert report["errors"] == []


def test_invalid_proof_rejected(proof_validator):
    """Test that invalid PROOF.json with verified=false leads to rejection."""
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Generate invalid proof (hash mismatch)
        proof = proof_validator.generate_proof(
            run_id="test_run_002",
            catalytic_domains=["CATALYTIC-DPT/_scratch"],
            pre_state={"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}},
            post_state={"CATALYTIC-DPT/_scratch": {"file.txt": "b" * 64}},
            timestamp="2025-12-25T00:00:00Z",
        )

        assert proof["restoration_result"]["verified"] is False
        assert proof["restoration_result"]["condition"] == "RESTORATION_FAILED_HASH_MISMATCH"

        create_minimal_ledger(ledger_dir, proof)

        # Validate ledger
        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert report["valid"] is False
        assert len(report["errors"]) > 0
        assert any("PROOF-GATED REJECTION" in err for err in report["errors"])


def test_missing_proof_rejected():
    """Test that missing PROOF.json leads to rejection."""
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Create all canonical files EXCEPT PROOF.json
        jobspec = {
            "job_id": "test_run_003",
            "phase": 0,
            "task_type": "primitive_implementation",
            "intent": "Test missing proof",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": ["CATALYTIC-DPT/_scratch"],
            "determinism": "deterministic",
        }
        (ledger_dir / "JOBSPEC.json").write_text(json.dumps(jobspec, indent=2))

        status = {"status": "succeeded", "restoration_verified": True, "exit_code": 0, "validation_passed": True}
        (ledger_dir / "STATUS.json").write_text(json.dumps(status, indent=2))

        (ledger_dir / "INPUT_HASHES.json").write_text(json.dumps({}, indent=2))
        (ledger_dir / "OUTPUT_HASHES.json").write_text(json.dumps({}, indent=2))
        (ledger_dir / "DOMAIN_ROOTS.json").write_text(json.dumps({}, indent=2))
        (ledger_dir / "LEDGER.jsonl").write_text("{}\n")

        validator_id = {"validator_semver": "0.1.0", "validator_build_id": "test"}
        (ledger_dir / "VALIDATOR_ID.json").write_text(json.dumps(validator_id, indent=2))

        # Legacy files
        run_info = {
            "run_id": "test_run_003",
            "timestamp": "2025-12-25T00:00:00Z",
            "intent": "Test missing proof",
            "catalytic_domains": ["CATALYTIC-DPT/_scratch"],
            "exit_code": 0,
        }
        (ledger_dir / "RUN_INFO.json").write_text(json.dumps(run_info, indent=2))

        pre_manifest = {"CATALYTIC-DPT/_scratch": {}}
        (ledger_dir / "PRE_MANIFEST.json").write_text(json.dumps(pre_manifest, indent=2))

        post_manifest = {"CATALYTIC-DPT/_scratch": {}}
        (ledger_dir / "POST_MANIFEST.json").write_text(json.dumps(post_manifest, indent=2))

        restore_diff = {"CATALYTIC-DPT/_scratch": {"added": {}, "removed": {}, "changed": {}}}
        (ledger_dir / "RESTORE_DIFF.json").write_text(json.dumps(restore_diff, indent=2))

        outputs = []
        (ledger_dir / "OUTPUTS.json").write_text(json.dumps(outputs, indent=2))

        # NOTE: PROOF.json is intentionally missing

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert report["valid"] is False
        # Check for missing file error
        assert any("PROOF.json" in err for err in report["errors"])


def test_proof_is_single_source_of_truth(proof_validator):
    """
    Test that PROOF.json is the ONLY source of truth.
    Even if RESTORE_DIFF shows clean, if PROOF.json says verified=false, reject.
    """
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Generate invalid proof
        proof = proof_validator.generate_proof(
            run_id="test_run_004",
            catalytic_domains=["CATALYTIC-DPT/_scratch"],
            pre_state={"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}},
            post_state={"CATALYTIC-DPT/_scratch": {"file.txt": "b" * 64}},  # Mismatch
            timestamp="2025-12-25T00:00:00Z",
        )

        # Create canonical artifacts
        jobspec = {
            "job_id": "test_run_004",
            "phase": 0,
            "task_type": "primitive_implementation",
            "intent": "Test proof as single source of truth",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": ["CATALYTIC-DPT/_scratch"],
            "determinism": "deterministic",
        }
        (ledger_dir / "JOBSPEC.json").write_text(json.dumps(jobspec, indent=2))

        status = {"status": "succeeded", "restoration_verified": True, "exit_code": 0, "validation_passed": True}
        (ledger_dir / "STATUS.json").write_text(json.dumps(status, indent=2))

        (ledger_dir / "INPUT_HASHES.json").write_text(json.dumps({}, indent=2))
        (ledger_dir / "OUTPUT_HASHES.json").write_text(json.dumps({}, indent=2))
        (ledger_dir / "DOMAIN_ROOTS.json").write_text(json.dumps({}, indent=2))
        (ledger_dir / "LEDGER.jsonl").write_text("{}\n")

        validator_id = {"validator_semver": "0.1.0", "validator_build_id": "test"}
        (ledger_dir / "VALIDATOR_ID.json").write_text(json.dumps(validator_id, indent=2))

        # But PROOF.json says verified=false
        (ledger_dir / "PROOF.json").write_text(json.dumps(proof, indent=2))

        # Legacy artifacts - RESTORE_DIFF claims clean (contradicts proof)
        run_info = {
            "run_id": "test_run_004",
            "timestamp": "2025-12-25T00:00:00Z",
            "intent": "Test proof as single source of truth",
            "catalytic_domains": ["CATALYTIC-DPT/_scratch"],
            "exit_code": 0,
        }
        (ledger_dir / "RUN_INFO.json").write_text(json.dumps(run_info, indent=2))

        pre_manifest = {"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}}
        (ledger_dir / "PRE_MANIFEST.json").write_text(json.dumps(pre_manifest, indent=2))

        post_manifest = {"CATALYTIC-DPT/_scratch": {"file.txt": "a" * 64}}
        (ledger_dir / "POST_MANIFEST.json").write_text(json.dumps(post_manifest, indent=2))

        # RESTORE_DIFF claims clean
        restore_diff = {"CATALYTIC-DPT/_scratch": {"added": {}, "removed": {}, "changed": {}}}
        (ledger_dir / "RESTORE_DIFF.json").write_text(json.dumps(restore_diff, indent=2))

        outputs = []
        (ledger_dir / "OUTPUTS.json").write_text(json.dumps(outputs, indent=2))

        # Validate ledger - should REJECT based on proof alone
        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert report["valid"] is False
        assert any("PROOF-GATED REJECTION" in err for err in report["errors"])
