#!/usr/bin/env python3
"""
Pipeline Toolkit Tests

Tests for the unified pipeline-toolkit skill that consolidates:
- pipeline-dag-scheduler (schedule operation)
- pipeline-dag-receipts (receipts operation)
- pipeline-dag-restore (restore operation)

Run: python -m pytest CAPABILITY/TESTBENCH/skills/test_pipeline_toolkit.py -v
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOLKIT_DIR = REPO_ROOT / "CAPABILITY" / "SKILLS" / "pipeline" / "pipeline-toolkit"
FIXTURES_DIR = TOOLKIT_DIR / "fixtures"
RUN_SCRIPT = TOOLKIT_DIR / "run.py"

# Durable output directory within allowed roots (LAW/CONTRACTS/_runs)
# Note: _tmp is a tmp domain, so we use _test_pipeline_toolkit for durable writes
TEST_OUTPUT_BASE = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_test_pipeline_toolkit"


def _run_toolkit(
    input_data: Dict[str, Any],
    input_dir: Path,
    output_file: Path,
) -> tuple[int, Dict[str, Any] | None, str]:
    """Run the pipeline-toolkit with given input data.

    Args:
        input_data: The input payload for the toolkit
        input_dir: Directory to store the input file (can be anywhere)
        output_file: Path to write output (must be within durable roots)

    Returns:
        Tuple of (exit_code, output_data or None if failed to parse, stderr)
    """
    input_file = input_dir / "input.json"

    input_file.write_text(json.dumps(input_data, indent=2), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), str(input_file), str(output_file)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    output_data = None
    if output_file.exists():
        try:
            output_data = json.loads(output_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    return result.returncode, output_data, result.stderr


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory for test input artifacts."""
    work_dir = tmp_path / "pipeline_toolkit_test"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


@pytest.fixture
def durable_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory within durable roots and clean up after."""
    test_id = uuid.uuid4().hex[:8]
    output_dir = TEST_OUTPUT_BASE / f"test_{test_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir
    # Cleanup after test
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def basic_fixture() -> Dict[str, Any]:
    """Load the basic fixture from the toolkit fixtures directory."""
    fixture_path = FIXTURES_DIR / "basic" / "input.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


class TestScheduleOperation:
    """Tests for the 'schedule' operation."""

    def test_schedule_can_be_invoked(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that schedule operation can be invoked successfully."""
        input_data = {
            "operation": "schedule",
            "dag_spec_path": "test-dag.json",
            "runs_root": "CONTRACTS/_runs",
        }
        output_file = durable_output_dir / "schedule_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        assert output is not None, "Expected output JSON file to be created"
        assert "ok" in output, "Output should contain 'ok' field"
        assert output.get("details", {}).get("operation") == "schedule"

    def test_schedule_with_basic_fixture(
        self, tmp_work_dir: Path, durable_output_dir: Path, basic_fixture: Dict[str, Any]
    ) -> None:
        """Test schedule operation using the basic fixture."""
        # Basic fixture already has operation: schedule
        assert basic_fixture.get("operation") == "schedule"
        output_file = durable_output_dir / "basic_fixture_output.json"

        exit_code, output, stderr = _run_toolkit(basic_fixture, tmp_work_dir, output_file)

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        assert output is not None
        assert output.get("details", {}).get("operation") == "schedule"


class TestReceiptsOperation:
    """Tests for the 'receipts' operation."""

    def test_receipts_can_be_invoked(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that receipts operation can be invoked successfully."""
        input_data = {
            "operation": "receipts",
            "dag_spec_path": "test-dag.json",
            "runs_root": "CONTRACTS/_runs",
        }
        output_file = durable_output_dir / "receipts_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        assert output is not None, "Expected output JSON file to be created"
        assert "ok" in output, "Output should contain 'ok' field"
        assert output.get("details", {}).get("operation") == "receipts"


class TestRestoreOperation:
    """Tests for the 'restore' operation."""

    def test_restore_can_be_invoked(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that restore operation can be invoked successfully."""
        input_data = {
            "operation": "restore",
            "dag_spec_path": "test-dag.json",
            "runs_root": "CONTRACTS/_runs",
        }
        output_file = durable_output_dir / "restore_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        assert output is not None, "Expected output JSON file to be created"
        assert "ok" in output, "Output should contain 'ok' field"
        assert output.get("details", {}).get("operation") == "restore"


class TestErrorHandling:
    """Tests for error handling in pipeline-toolkit."""

    def test_invalid_operation_raises_error(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that an invalid operation name returns exit code 1."""
        input_data = {
            "operation": "invalid_operation",
            "dag_spec_path": "test-dag.json",
        }
        output_file = durable_output_dir / "invalid_op_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 1, f"Expected exit code 1 for invalid operation, got {exit_code}"

    def test_missing_operation_field_raises_error(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that missing 'operation' field returns exit code 1."""
        input_data = {
            "dag_spec_path": "test-dag.json",
            "runs_root": "CONTRACTS/_runs",
        }
        output_file = durable_output_dir / "missing_op_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 1, f"Expected exit code 1 for missing operation, got {exit_code}"

    def test_empty_input_raises_error(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that empty input JSON returns exit code 1."""
        input_data = {}
        output_file = durable_output_dir / "empty_input_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 1, f"Expected exit code 1 for empty input, got {exit_code}"


class TestOutputFormat:
    """Tests for output format consistency."""

    def test_all_operations_return_consistent_format(
        self, tmp_work_dir: Path, durable_output_dir: Path
    ) -> None:
        """Test that all operations return output with consistent structure."""
        operations = ["schedule", "receipts", "restore"]

        for operation in operations:
            input_data = {
                "operation": operation,
                "dag_spec_path": "test-dag.json",
            }
            output_file = durable_output_dir / f"{operation}_format_output.json"

            exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

            assert exit_code == 0, f"Operation '{operation}' failed with exit code {exit_code}. stderr: {stderr}"
            assert output is not None, f"Operation '{operation}' produced no output"

            # Verify required fields
            assert "ok" in output, f"Operation '{operation}' missing 'ok' field"
            assert "code" in output, f"Operation '{operation}' missing 'code' field"
            assert "details" in output, f"Operation '{operation}' missing 'details' field"

            # Verify details contains operation
            assert output["details"].get("operation") == operation

    def test_output_includes_dag_spec_path(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that output details include the dag_spec_path from input."""
        input_data = {
            "operation": "schedule",
            "dag_spec_path": "my/custom/dag.json",
        }
        output_file = durable_output_dir / "dag_spec_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        assert output is not None
        assert output["details"].get("dag_spec_path") == "my/custom/dag.json"

    def test_output_includes_runs_root(self, tmp_work_dir: Path, durable_output_dir: Path) -> None:
        """Test that output details include the runs_root from input."""
        input_data = {
            "operation": "schedule",
            "dag_spec_path": "test-dag.json",
            "runs_root": "custom/_runs",
        }
        output_file = durable_output_dir / "runs_root_output.json"

        exit_code, output, stderr = _run_toolkit(input_data, tmp_work_dir, output_file)

        assert exit_code == 0, f"Expected exit code 0, got {exit_code}. stderr: {stderr}"
        assert output is not None
        assert output["details"].get("runs_root") == "custom/_runs"


class TestFixtures:
    """Tests that verify fixtures are usable."""

    def test_basic_fixture_exists(self) -> None:
        """Test that the basic fixture exists."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        assert fixture_path.exists(), f"Basic fixture not found at {fixture_path}"

    def test_basic_fixture_is_valid_json(self) -> None:
        """Test that the basic fixture is valid JSON."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        content = fixture_path.read_text(encoding="utf-8")

        # Should not raise
        data = json.loads(content)

        assert isinstance(data, dict), "Fixture should be a JSON object"

    def test_basic_fixture_has_required_fields(self, basic_fixture: Dict[str, Any]) -> None:
        """Test that the basic fixture has required fields."""
        assert "operation" in basic_fixture, "Fixture missing 'operation' field"
        assert basic_fixture["operation"] in ["schedule", "receipts", "restore"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
