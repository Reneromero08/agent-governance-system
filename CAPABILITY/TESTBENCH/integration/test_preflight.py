import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY/TOOLS"))

from CAPABILITY.PRIMITIVES.preflight import PreflightValidator


@pytest.fixture
def jobspec_schema_path():
    return Path(__file__).resolve().parents[3] / "LAW" / "SCHEMAS" / "jobspec.schema.json"


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def preflight_validator(jobspec_schema_path):
    return PreflightValidator(jobspec_schema_path)


def test_valid_jobspec_passes(preflight_validator, project_root):
    """Test that valid JobSpec passes preflight validation."""
    jobspec = {
        "job_id": "test-run-001",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test valid preflight",
        "inputs": {},
        "outputs": {
            "durable_paths": ["LAW/CONTRACTS/_runs/test-run-001/output.json"],
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is True
    assert errors == []


def test_invalid_jobspec_schema_fails(preflight_validator, project_root):
    """Test that invalid JobSpec fails preflight validation."""
    jobspec = {
        "job_id": "test-run-002",
        "phase": 0,
        # Missing required field: task_type
        "intent": "Test invalid schema",
        "inputs": {},
        "outputs": {
            "durable_paths": [],
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "JOBSPEC_SCHEMA_INVALID" for err in errors)


def test_path_traversal_fails(preflight_validator, project_root):
    """Test that path traversal in catalytic_domains fails preflight."""
    jobspec = {
        "job_id": "test-run-003",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test path traversal",
        "inputs": {},
        "outputs": {
            "durable_paths": [],
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/../CANON"],  # Path traversal!
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "PATH_TRAVERSAL" for err in errors)


def test_absolute_path_fails(preflight_validator, project_root):
    """Test that absolute paths fail preflight validation."""
    import os

    if os.name == 'nt':
        absolute_path = "C:/tmp/output.json"
    else:
        absolute_path = "/tmp/output.json"

    jobspec = {
        "job_id": "test-run-004",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test absolute path",
        "inputs": {},
        "outputs": {
            "durable_paths": [absolute_path],  # Absolute path!
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "PATH_ABSOLUTE" for err in errors)


def test_path_not_under_allowed_root_fails(preflight_validator, project_root):
    """Test that paths outside allowed roots fail preflight."""
    jobspec = {
        "job_id": "test-run-005",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test path not under allowed root",
        "inputs": {},
        "outputs": {
            "durable_paths": ["AGENTS.md"],  # Not under allowed roots!
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "PATH_NOT_ALLOWED" for err in errors)


def test_forbidden_path_fails(preflight_validator, project_root):
    """Test that forbidden paths fail preflight validation."""
    jobspec = {
        "job_id": "test-run-006",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test forbidden path",
        "inputs": {},
        "outputs": {
            "durable_paths": [],
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CANON"],  # Forbidden path!
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "PATH_FORBIDDEN" for err in errors)


def test_input_output_overlap_fails(preflight_validator, project_root):
    """Test that overlapping catalytic_domains and outputs fail preflight."""
    jobspec = {
        "job_id": "test-run-007",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test input/output overlap",
        "inputs": {},
        "outputs": {
            "durable_paths": ["LAW/CONTRACTS/_runs/_tmp/scratch"],  # Overlaps with catalytic_domains!
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "PATH_OVERLAP" for err in errors)


def test_multiple_errors_reported(preflight_validator, project_root):
    """Test that multiple validation errors are all reported."""
    jobspec = {
        "job_id": "test-run-008",
        "phase": 0,
        # Missing task_type
        "intent": "Test multiple errors",
        "inputs": {},
        "outputs": {
            "durable_paths": ["/tmp/output.json"],  # Absolute path
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/../CANON"],  # Path traversal
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) >= 2  # At least schema error and path error


def test_empty_catalytic_domains_valid(preflight_validator, project_root):
    """Test that empty catalytic_domains is valid (no scratch space needed)."""
    jobspec = {
        "job_id": "test-run-009",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test empty catalytic domains",
        "inputs": {},
        "outputs": {
            "durable_paths": ["LAW/CONTRACTS/_runs/test-run-009/output.json"],
            "validation_criteria": {},
        },
        "catalytic_domains": [],  # Empty is valid
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is True
    assert errors == []


def test_empty_outputs_valid(preflight_validator, project_root):
    """Test that empty outputs.durable_paths is valid."""
    jobspec = {
        "job_id": "test-run-010",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test empty outputs",
        "inputs": {},
        "outputs": {
            "durable_paths": [],  # Empty is valid
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is True
    assert errors == []


def test_bucket_violation_path_outside_buckets_fails(preflight_validator, project_root):
    """Test that path outside all buckets fails (X3 violation)."""
    jobspec = {
        "job_id": "test-run-011",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test bucket violation",
        "inputs": {},
        "outputs": {
            "durable_paths": ["SKILLS/output.json"],  # Old path not in 6-bucket structure
            "validation_criteria": {},
        },
        "catalytic_domains": [],
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is False
    assert len(errors) > 0
    assert any(err["code"] == "BUCKET_VIOLATION" for err in errors)


def test_path_in_valid_bucket_passes(preflight_validator, project_root):
    """Test that paths in valid buckets pass (X3 compliance)."""
    jobspec = {
        "job_id": "test-run-012",
        "phase": 0,
        "task_type": "primitive_implementation",
        "intent": "Test valid bucket",
        "inputs": {},
        "outputs": {
            "durable_paths": ["CAPABILITY/PRIMITIVES/_scratch/test.json"],  # In CAPABILITY bucket and allowed root
            "validation_criteria": {},
        },
        "catalytic_domains": ["LAW/CONTRACTS/_runs/_tmp/scratch"],  # In LAW bucket and allowed root
        "determinism": "deterministic",
    }

    valid, errors = preflight_validator.validate(jobspec, project_root)

    assert valid is True
    assert errors == []


def test_all_buckets_are_valid(preflight_validator, project_root):
    """Test that all 6 buckets are recognized as valid (X3)."""
    # Map bucket to an allowed path within that bucket
    bucket_paths = {
        "LAW": "LAW/CONTRACTS/_runs/test/output.json",
        "CAPABILITY": "CAPABILITY/PRIMITIVES/_scratch/test/output.json",
        "NAVIGATION": "NAVIGATION/CORTEX/_generated/test/output.json",
        "MEMORY": "MEMORY/LLM_PACKER/_packs/test/output.json",
        "THOUGHT": "THOUGHT/LAB/test/output.json",  # Not in allowed roots, but valid bucket
        "INBOX": "INBOX/reports/test/output.json",  # Not in allowed roots, but valid bucket
    }

    for bucket, path in bucket_paths.items():
        jobspec = {
            "job_id": f"test-run-bucket-{bucket.lower()}",
            "phase": 0,
            "task_type": "primitive_implementation",
            "intent": f"Test {bucket} bucket",
            "inputs": {},
            "outputs": {
                "durable_paths": [path],
                "validation_criteria": {},
            },
            "catalytic_domains": [],
            "determinism": "deterministic",
        }

        valid, errors = preflight_validator.validate(jobspec, project_root)

        # Buckets THOUGHT and INBOX may fail due to ALLOWED_ROOTS, but bucket enforcement should pass
        # Check that there's no BUCKET_VIOLATION error
        has_bucket_violation = any(err["code"] == "BUCKET_VIOLATION" for err in errors)
        assert not has_bucket_violation, f"Bucket {bucket} should pass bucket enforcement: {errors}"
