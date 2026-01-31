#!/usr/bin/env python3
"""
Tests for CatalyticLedgerValidator (CMP-01 compliance validation).

Covers:
- Valid run ledger acceptance
- Missing PROOF.json rejection
- PROOF.json with verified=false rejection
- Schema validation for PROOF.json
- Forbidden artifact detection (logs/, tmp/, transcript.json)
- DOMAIN_ROOTS mismatch detection
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest
import sys

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.TOOLS.catalytic.catalytic_validator import CatalyticLedgerValidator


def create_minimal_valid_ledger(ledger_dir: Path) -> None:
    """Create a minimal valid run ledger for testing."""
    ledger_dir.mkdir(parents=True, exist_ok=True)

    # Canonical files
    (ledger_dir / "JOBSPEC.json").write_text(json.dumps({
        "job_id": "test-job",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test run",
        "inputs": {},
        "outputs": {"durable_paths": [], "validation_criteria": {}},
        "catalytic_domains": ["CAPABILITY/PRIMITIVES/_scratch"],
        "determinism": "deterministic",
    }))

    (ledger_dir / "STATUS.json").write_text(json.dumps({
        "status": "succeeded",
        "restoration_verified": True,
        "exit_code": 0,
        "validation_passed": True,
    }))

    (ledger_dir / "INPUT_HASHES.json").write_text(json.dumps({}))
    (ledger_dir / "OUTPUT_HASHES.json").write_text(json.dumps({}))
    (ledger_dir / "DOMAIN_ROOTS.json").write_text(json.dumps({}))
    (ledger_dir / "LEDGER.jsonl").write_text("")

    (ledger_dir / "VALIDATOR_ID.json").write_text(json.dumps({
        "validator_semver": "0.1.0",
        "validator_build_id": "test",
        "python_version": "3.11.0",
    }))

    # Valid PROOF.json
    (ledger_dir / "PROOF.json").write_text(json.dumps({
        "proof_version": "1.0.0",
        "run_id": "test-run",
        "timestamp": "2025-01-01T00:00:00Z",
        "catalytic_domains": ["CAPABILITY/PRIMITIVES/_scratch"],
        "pre_state": {
            "domain_root_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "file_manifest": {},
        },
        "post_state": {
            "domain_root_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "file_manifest": {},
        },
        "restoration_result": {
            "verified": True,
            "condition": "RESTORED_IDENTICAL",
        },
        "proof_hash": "a" * 64,
    }))

    # Legacy files
    (ledger_dir / "RUN_INFO.json").write_text(json.dumps({
        "run_id": "test-run",
        "timestamp": "2025-01-01T00:00:00Z",
        "intent": "Test run",
        "catalytic_domains": ["CAPABILITY/PRIMITIVES/_scratch"],
        "exit_code": 0,
        "restoration_verified": True,
    }))

    (ledger_dir / "PRE_MANIFEST.json").write_text(json.dumps({}))
    (ledger_dir / "POST_MANIFEST.json").write_text(json.dumps({}))
    (ledger_dir / "RESTORE_DIFF.json").write_text(json.dumps({}))
    (ledger_dir / "OUTPUTS.json").write_text(json.dumps([]))


class TestCatalyticLedgerValidator:
    """Test suite for CatalyticLedgerValidator."""

    def test_validate_valid_run(self, tmp_path: Path) -> None:
        """Complete valid run ledger should pass validation."""
        ledger_dir = tmp_path / "run-valid"
        create_minimal_valid_ledger(ledger_dir)

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is True, f"Expected valid, got errors: {report['errors']}"
        assert report["valid"] is True

    def test_validate_missing_proof(self, tmp_path: Path) -> None:
        """Missing PROOF.json should fail validation."""
        ledger_dir = tmp_path / "run-no-proof"
        create_minimal_valid_ledger(ledger_dir)
        (ledger_dir / "PROOF.json").unlink()

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("PROOF.json" in e for e in report["errors"])

    def test_validate_proof_verified_false(self, tmp_path: Path) -> None:
        """PROOF.json with verified=false should fail validation."""
        ledger_dir = tmp_path / "run-unverified"
        create_minimal_valid_ledger(ledger_dir)

        # Update PROOF.json to have verified=false
        proof = json.loads((ledger_dir / "PROOF.json").read_text())
        proof["restoration_result"]["verified"] = False
        proof["restoration_result"]["condition"] = "RESTORATION_FAILED_HASH_MISMATCH"
        proof["restoration_result"]["mismatches"] = [
            {"path": "test.txt", "type": "hash_mismatch", "expected_hash": "a" * 64, "actual_hash": "b" * 64}
        ]
        (ledger_dir / "PROOF.json").write_text(json.dumps(proof))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("PROOF-GATED REJECTION" in e for e in report["errors"])

    def test_validate_schema_violation_missing_field(self, tmp_path: Path) -> None:
        """PROOF.json missing required field should fail schema validation."""
        ledger_dir = tmp_path / "run-bad-schema"
        create_minimal_valid_ledger(ledger_dir)

        # Remove required field from PROOF.json
        proof = json.loads((ledger_dir / "PROOF.json").read_text())
        del proof["proof_version"]
        (ledger_dir / "PROOF.json").write_text(json.dumps(proof))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("schema violation" in e.lower() for e in report["errors"])

    def test_validate_forbidden_artifacts_logs(self, tmp_path: Path) -> None:
        """logs/ directory in ledger should fail validation."""
        ledger_dir = tmp_path / "run-with-logs"
        create_minimal_valid_ledger(ledger_dir)

        # Add forbidden logs/ directory
        logs_dir = ledger_dir / "logs"
        logs_dir.mkdir()
        (logs_dir / "debug.log").write_text("some log content")

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("Forbidden artifact present: logs/" in e for e in report["errors"])

    def test_validate_forbidden_artifacts_tmp(self, tmp_path: Path) -> None:
        """tmp/ directory in ledger should fail validation."""
        ledger_dir = tmp_path / "run-with-tmp"
        create_minimal_valid_ledger(ledger_dir)

        # Add forbidden tmp/ directory
        tmp_dir = ledger_dir / "tmp"
        tmp_dir.mkdir()
        (tmp_dir / "scratch.txt").write_text("temp content")

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("Forbidden artifact present: tmp/" in e for e in report["errors"])

    def test_validate_forbidden_artifacts_transcript(self, tmp_path: Path) -> None:
        """transcript.json in ledger should fail validation."""
        ledger_dir = tmp_path / "run-with-transcript"
        create_minimal_valid_ledger(ledger_dir)

        # Add forbidden transcript.json
        (ledger_dir / "transcript.json").write_text(json.dumps({"messages": []}))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("Forbidden artifact present: transcript.json" in e for e in report["errors"])

    def test_validate_domain_roots_mismatch(self, tmp_path: Path) -> None:
        """DOMAIN_ROOTS not matching POST_MANIFEST should fail validation."""
        ledger_dir = tmp_path / "run-roots-mismatch"
        create_minimal_valid_ledger(ledger_dir)

        # Set up non-empty POST_MANIFEST
        post_manifest = {
            "CAPABILITY/PRIMITIVES/_scratch": {
                "test.txt": "a" * 64,
            }
        }
        (ledger_dir / "POST_MANIFEST.json").write_text(json.dumps(post_manifest))

        # Set DOMAIN_ROOTS to wrong value
        domain_roots = {
            "CAPABILITY/PRIMITIVES/_scratch": "b" * 64,  # Intentionally wrong
        }
        (ledger_dir / "DOMAIN_ROOTS.json").write_text(json.dumps(domain_roots))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("DOMAIN_ROOTS mismatch" in e for e in report["errors"])

    def test_validate_missing_run_info_field(self, tmp_path: Path) -> None:
        """RUN_INFO.json missing required field should fail validation."""
        ledger_dir = tmp_path / "run-bad-runinfo"
        create_minimal_valid_ledger(ledger_dir)

        # Remove required field from RUN_INFO.json
        run_info = json.loads((ledger_dir / "RUN_INFO.json").read_text())
        del run_info["intent"]
        (ledger_dir / "RUN_INFO.json").write_text(json.dumps(run_info))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("RUN_INFO missing field: intent" in e for e in report["errors"])


class TestValidatorEdgeCases:
    """Edge cases and error handling tests."""

    def test_validate_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON in artifacts should fail gracefully."""
        ledger_dir = tmp_path / "run-bad-json"
        create_minimal_valid_ledger(ledger_dir)

        # Write invalid JSON
        (ledger_dir / "RUN_INFO.json").write_text("not valid json {{{")

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("Invalid JSON" in e for e in report["errors"])

    def test_validate_empty_ledger_dir(self, tmp_path: Path) -> None:
        """Empty ledger directory should fail with missing files."""
        ledger_dir = tmp_path / "run-empty"
        ledger_dir.mkdir()

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("Missing required file" in e for e in report["errors"])

    def test_validate_nonexistent_dir(self, tmp_path: Path) -> None:
        """Nonexistent ledger directory should fail."""
        ledger_dir = tmp_path / "run-does-not-exist"

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
