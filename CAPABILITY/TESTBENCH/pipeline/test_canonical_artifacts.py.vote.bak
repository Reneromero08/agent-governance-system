from pathlib import Path
import sys
import json
from tempfile import TemporaryDirectory
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adjust path to find TOOLS
TOOLS_PATH = REPO_ROOT / "CAPABILITY" / "TOOLS"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

from CAPABILITY.TOOLS.catalytic.catalytic_validator import CatalyticLedgerValidator

def test_canonical_artifact_set_required():
    """Test that all canonical artifacts must exist for acceptance."""
    with TemporaryDirectory() as tmpdir:
        ledger_dir = Path(tmpdir) / "test_run"
        ledger_dir.mkdir()

        # Create ALL canonical artifacts first
        canonical_files = [
            "JOBSPEC.json",
            "STATUS.json",
            "INPUT_HASHES.json",
            "OUTPUT_HASHES.json",
            "DOMAIN_ROOTS.json",
            "LEDGER.jsonl",
            "VALIDATOR_ID.json",
            "PROOF.json",
        ]

        for filename in canonical_files:
            (ledger_dir / filename).write_text("{}")

        # Now remove one and verify failure
        (ledger_dir / "DOMAIN_ROOTS.json").unlink()

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert any("DOMAIN_ROOTS.json" in str(err) for err in report["errors"])

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

        # PROOF.json with verified=false (will fail validation later)
        (ledger_dir / "PROOF.json").write_text(json.dumps({
            "proof_version": "1.0.0",
            "run_id": "test",
            "timestamp": "2025-12-25T00:00:00Z",
            "catalytic_domains": [],
            "pre_state": {"domain_root_hash": "a" * 64, "file_manifest": {}},
            "post_state": {"domain_root_hash": "a" * 64, "file_manifest": {}},
            "restoration_result": {"verified": True, "condition": "RESTORATION_FAILED", "mismatches": []},
            "proof_hash": "b" * 64,
        }, indent=2))

        validator = CatalyticLedgerValidator(ledger_dir)
        success, report = validator.validate()

        assert success is False
        assert len(report["errors"]) > 0