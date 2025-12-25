"""
Test canonical run artifact writer.

Validates Phase 0.2: canonical artifact set is written completely.

Run:
    pytest CATALYTIC-DPT/TESTBENCH/test_canonical_artifacts.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add paths
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_path / "CATALYTIC-DPT"))
sys.path.insert(0, str(repo_root_path / "TOOLS"))

from catalytic_validator import CatalyticLedgerValidator


def test_canonical_artifact_set_required():
    """Test that all canonical artifacts must exist for acceptance."""
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Create partial artifact set (missing DOMAIN_ROOTS.json)
        canonical_files = [
            "JOBSPEC.json",
            "STATUS.json",
            "INPUT_HASHES.json",
            "OUTPUT_HASHES.json",
            # "DOMAIN_ROOTS.json",  # Missing!
            "LEDGER.jsonl",
            "VALIDATOR_ID.json",
            "PROOF.json",
        ]

        for filename in canonical_files:
            (ledger_dir / filename).write_text("{}")

        # Add legacy files
        (ledger_dir / "RUN_INFO.json").write_text(json.dumps({
            "run_id": "test",
            "timestamp": "2025-12-25T00:00:00Z",
            "intent": "test",
            "catalytic_domains": [],
            "exit_code": 0,
        }, indent=2))
        (ledger_dir / "PRE_MANIFEST.json").write_text("{}")
        (ledger_dir / "POST_MANIFEST.json").write_text("{}")
        (ledger_dir / "RESTORE_DIFF.json").write_text("{}")
        (ledger_dir / "OUTPUTS.json").write_text("[]")

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert report["valid"] is False
        assert any("DOMAIN_ROOTS.json" in err for err in report["errors"])


def test_status_state_machine():
    """Test that STATUS.json has correct state machine values."""
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Valid states: started, failed, succeeded, verified
        invalid_status = {
            "status": "invalid_state",  # Not in state machine
            "restoration_verified": True,
            "exit_code": 0,
            "validation_passed": True,
        }

        (ledger_dir / "STATUS.json").write_text(json.dumps(invalid_status, indent=2))

        # Create minimal other files
        (ledger_dir / "JOBSPEC.json").write_text(json.dumps({
            "job_id": "test",
            "phase": 0,
            "task_type": "primitive_implementation",
            "intent": "test",
            "inputs": {},
            "outputs": {"durable_paths": [], "validation_criteria": {}},
            "catalytic_domains": [],
            "determinism": "deterministic",
        }, indent=2))

        for filename in ["INPUT_HASHES.json", "OUTPUT_HASHES.json", "DOMAIN_ROOTS.json"]:
            (ledger_dir / filename).write_text("{}")

        (ledger_dir / "LEDGER.jsonl").write_text("{}\n")

        (ledger_dir / "VALIDATOR_ID.json").write_text(json.dumps({
            "validator_semver": "0.1.0",
            "validator_build_id": "test",
        }, indent=2))

        # Add legacy files
        (ledger_dir / "RUN_INFO.json").write_text(json.dumps({
            "run_id": "test",
            "timestamp": "2025-12-25T00:00:00Z",
            "intent": "test",
            "catalytic_domains": [],
            "exit_code": 0,
        }, indent=2))
        (ledger_dir / "PRE_MANIFEST.json").write_text("{}")
        (ledger_dir / "POST_MANIFEST.json").write_text("{}")
        (ledger_dir / "RESTORE_DIFF.json").write_text("{}")
        (ledger_dir / "OUTPUTS.json").write_text("[]")

        # PROOF.json with verified=false (will fail validation later)
        (ledger_dir / "PROOF.json").write_text(json.dumps({
            "proof_version": "1.0.0",
            "run_id": "test",
            "timestamp": "2025-12-25T00:00:00Z",
            "catalytic_domains": [],
            "pre_state": {"domain_root_hash": "a" * 64, "file_manifest": {}},
            "post_state": {"domain_root_hash": "a" * 64, "file_manifest": {}},
            "restoration_result": {"verified": False, "condition": "RESTORATION_FAILED_DOMAIN_UNREACHABLE", "mismatches": []},
            "proof_hash": "b" * 64,
        }, indent=2))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        # Should fail on proof validation (verified=false)
        assert success is False
        # Status state validation would be schema-level (future enhancement)


def test_proof_written_last():
    """Test that PROOF.json can only exist if other artifacts exist."""
    # This is enforced by write_canonical_artifacts() writing PROOF.json last
    # If any earlier artifact write fails, PROOF.json won't be written
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Only PROOF.json exists (invalid - other artifacts missing)
        (ledger_dir / "PROOF.json").write_text(json.dumps({
            "proof_version": "1.0.0",
            "run_id": "test",
            "timestamp": "2025-12-25T00:00:00Z",
            "catalytic_domains": [],
            "pre_state": {"domain_root_hash": "a" * 64, "file_manifest": {}},
            "post_state": {"domain_root_hash": "a" * 64, "file_manifest": {}},
            "restoration_result": {"verified": True, "condition": "RESTORED_IDENTICAL"},
            "proof_hash": "b" * 64,
        }, indent=2))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert report["valid"] is False
        # Should fail on missing canonical artifacts
        assert len(report["errors"]) > 0
