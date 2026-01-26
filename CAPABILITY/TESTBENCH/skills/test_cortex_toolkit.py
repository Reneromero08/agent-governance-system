#!/usr/bin/env python3
"""
Tests for cortex-toolkit skill.

Tests the unified CORTEX toolkit operations:
- verify_cas: Check CAS directory integrity
- summarize: Generate deterministic section summaries
- smoke_test: Run LLM Packer smoke tests

Note: build and verify_system1 were removed - cassette network now
handles semantic search via NAVIGATION/CORTEX/cassettes/
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# Project root setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Skill paths
SKILL_DIR = REPO_ROOT / "CAPABILITY" / "SKILLS" / "cortex-toolkit"
FIXTURES_DIR = SKILL_DIR / "fixtures"
RUN_SCRIPT = SKILL_DIR / "run.py"

# Import the skill module
spec = __import__("importlib.util").util.spec_from_file_location("cortex_toolkit_run", RUN_SCRIPT)
cortex_toolkit = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(cortex_toolkit)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test runs."""
    run_dir = tmp_path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def mock_writer():
    """Create a mock GuardedWriter for testing."""
    mock_gw = mock.MagicMock()
    mock_gw.mkdir_durable = mock.MagicMock()
    mock_gw.write_durable = mock.MagicMock()
    return mock_gw


@pytest.fixture
def basic_fixture() -> Dict[str, Any]:
    """Load the basic fixture from fixtures/basic/input.json."""
    fixture_path = FIXTURES_DIR / "basic" / "input.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    return {"operation": "verify_system1"}


@pytest.fixture
def temp_cas_dir(tmp_path: Path) -> Path:
    """Create a temporary CAS directory with valid blobs."""
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir(parents=True, exist_ok=True)

    # Create a valid CAS blob
    content = b"test content for CAS"
    hash_val = hashlib.sha256(content).hexdigest()
    prefix_dir = cas_dir / hash_val[:2]
    prefix_dir.mkdir(parents=True, exist_ok=True)
    blob_path = prefix_dir / hash_val
    blob_path.write_bytes(content)

    return cas_dir


# ============================================================================
# Test: Operation Dispatch
# ============================================================================

class TestOperationDispatch:
    """Test that each operation can be invoked through the dispatcher."""

    def test_operations_registry_contains_all_operations(self):
        """Verify all expected operations are in the registry."""
        # Note: build and verify_system1 removed - cassette network handles semantic search
        expected_ops = ["verify_cas", "summarize", "smoke_test"]
        for op in expected_ops:
            assert op in cortex_toolkit.OPERATIONS, f"Operation '{op}' not in OPERATIONS registry"

    def test_operations_are_callable(self):
        """Verify all operations are callable functions."""
        for op_name, op_func in cortex_toolkit.OPERATIONS.items():
            assert callable(op_func), f"Operation '{op_name}' is not callable"


# ============================================================================
# Test: Build Operation (REMOVED - cassette network handles semantic search)
# ============================================================================

@pytest.mark.skip(reason="build operation removed - cassette network handles semantic search")
class TestBuildOperation:
    """Test the build operation (REMOVED)."""

    def test_build_can_be_invoked(self, temp_run_dir: Path, mock_writer):
        """Test that build operation can be invoked."""
        payload = {"operation": "build", "expected_paths": [], "timeout_sec": 5}
        output_path = temp_run_dir / "output.json"

        # The build may fail (no cortex.build.py), but it should be invokable
        result = cortex_toolkit.op_build(payload, output_path, mock_writer)

        # Verify write_durable was called with output
        assert mock_writer.write_durable.called
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        # Output should have expected structure
        assert "ok" in output_data
        assert "errors" in output_data

    def test_build_with_missing_script_reports_error(self, temp_run_dir: Path, mock_writer):
        """Test that build reports error when script is missing."""
        payload = {
            "operation": "build",
            "build_script": "nonexistent/script.py",
            "expected_paths": []
        }
        output_path = temp_run_dir / "output.json"

        result = cortex_toolkit.op_build(payload, output_path, mock_writer)

        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert output_data["ok"] is False
        assert any("not_found" in err for err in output_data["errors"])


# ============================================================================
# Test: Verify CAS Operation
# ============================================================================

class TestVerifyCasOperation:
    """Test the verify_cas operation."""

    def test_verify_cas_can_be_invoked(self, temp_cas_dir: Path, temp_run_dir: Path, mock_writer):
        """Test that verify_cas operation can be invoked."""
        payload = {"operation": "verify_cas", "cas_root": str(temp_cas_dir)}
        output_path = temp_run_dir / "output.json"

        result = cortex_toolkit.op_verify_cas(payload, output_path, mock_writer)

        assert result == 0
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert output_data["status"] == "success"
        assert output_data["total_blobs"] == 1
        assert output_data["corrupt_blobs"] == []

    def test_verify_cas_missing_root_returns_error(self, temp_run_dir: Path, mock_writer):
        """Test that verify_cas returns error for missing cas_root."""
        payload = {"operation": "verify_cas"}  # Missing cas_root
        output_path = temp_run_dir / "output.json"

        result = cortex_toolkit.op_verify_cas(payload, output_path, mock_writer)

        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert output_data["status"] == "failure"
        assert "MISSING_CAS_ROOT" in output_data["error"]

    def test_verify_cas_detects_corrupt_blob(self, temp_cas_dir: Path, temp_run_dir: Path, mock_writer):
        """Test that verify_cas detects corrupted blobs."""
        # Create a corrupt blob (filename doesn't match content hash)
        corrupt_dir = temp_cas_dir / "ab"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        corrupt_path = corrupt_dir / ("ab" + "cd" * 31)  # 64 char filename
        corrupt_path.write_bytes(b"wrong content")

        payload = {"operation": "verify_cas", "cas_root": str(temp_cas_dir)}
        output_path = temp_run_dir / "output.json"

        result = cortex_toolkit.op_verify_cas(payload, output_path, mock_writer)

        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert output_data["status"] == "failure"
        assert len(output_data["corrupt_blobs"]) > 0


# ============================================================================
# Test: Verify System1 Operation (REMOVED - cassette network handles semantic search)
# ============================================================================

@pytest.mark.skip(reason="verify_system1 operation removed - cassette network handles semantic search")
class TestVerifySystem1Operation:
    """Test the verify_system1 operation (REMOVED).

    Tests verify the operation is registered and callable.
    The actual cortex verification test runs serially to avoid DB locking.
    """

    def test_verify_system1_in_operations_registry(self):
        """Test that verify_system1 is registered as a valid operation."""
        assert "verify_system1" in cortex_toolkit.OPERATIONS
        assert callable(cortex_toolkit.OPERATIONS["verify_system1"])

    def test_verify_system1_handles_missing_db(self, temp_run_dir: Path, mock_writer, monkeypatch):
        """Test that verify_system1 handles missing DB gracefully."""
        # Point to a non-existent DB to avoid race conditions with CI DB rebuild
        fake_db = temp_run_dir / "nonexistent.db"
        monkeypatch.setattr(cortex_toolkit, "DB_PATH", fake_db)

        payload = {"operation": "verify_system1"}
        output_path = temp_run_dir / "output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = cortex_toolkit.op_verify_system1(payload, output_path, mock_writer)

        # Should return non-zero for missing DB
        assert result == 1
        assert mock_writer.write_durable.called
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])
        assert output_data["success"] is False
        assert "does not exist" in output_data["description"]


# ============================================================================
# Test: Summarize Operation
# ============================================================================

class TestSummarizeOperation:
    """Test the summarize operation."""

    def test_summarize_can_be_invoked(self, temp_run_dir: Path, mock_writer):
        """Test that summarize operation can be invoked."""
        payload = {
            "operation": "summarize",
            "record": {
                "section_id": "test::section",
                "heading": "Test Section",
                "start_line": 1,
                "end_line": 10,
                "hash": "abc123def456"
            },
            "slice_text": "# Test Section\n\nThis is a test section with some content."
        }
        output_path = temp_run_dir / "output.json"

        result = cortex_toolkit.op_summarize(payload, output_path, mock_writer)

        assert result == 0
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert "safe_filename" in output_data
        assert "summary_md" in output_data
        assert "summary_sha256" in output_data

    def test_summarize_produces_deterministic_output(self, temp_run_dir: Path, mock_writer):
        """Test that summarize produces deterministic output."""
        payload = {
            "operation": "summarize",
            "record": {"section_id": "test::id", "start_line": 1, "end_line": 5},
            "slice_text": "Deterministic content"
        }
        output_path1 = temp_run_dir / "output1.json"
        output_path2 = temp_run_dir / "output2.json"

        cortex_toolkit.op_summarize(payload, output_path1, mock_writer)
        first_call = json.loads(mock_writer.write_durable.call_args[0][1])

        cortex_toolkit.op_summarize(payload, output_path2, mock_writer)
        second_call = json.loads(mock_writer.write_durable.call_args[0][1])

        assert first_call["summary_sha256"] == second_call["summary_sha256"]
        assert first_call["safe_filename"] == second_call["safe_filename"]

    def test_summarize_with_empty_record(self, temp_run_dir: Path, mock_writer):
        """Test that summarize handles empty record gracefully."""
        payload = {"operation": "summarize", "record": {}, "slice_text": ""}
        output_path = temp_run_dir / "output.json"

        result = cortex_toolkit.op_summarize(payload, output_path, mock_writer)

        assert result == 0
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert "safe_filename" in output_data
        assert "summary_sha256" in output_data


# ============================================================================
# Test: Smoke Test Operation
# ============================================================================

class TestSmokeTestOperation:
    """Test the smoke_test operation."""

    def test_smoke_test_validates_out_dir_constraint(self, temp_run_dir: Path, mock_writer):
        """Test that smoke_test enforces out_dir under _packs."""
        payload = {
            "operation": "smoke_test",
            "out_dir": "/tmp/invalid/path",  # Not under _packs
            "scope": "ags"
        }
        output_path = temp_run_dir / "output.json"

        with pytest.raises(ValueError, match="_packs"):
            cortex_toolkit.op_smoke_test(payload, output_path, mock_writer)

    def test_smoke_test_validates_runner_writes_constraint(self, mock_writer):
        """Test that smoke_test enforces output under _runs."""
        payload = {
            "operation": "smoke_test",
            "out_dir": "MEMORY/LLM_PACKER/_packs/_system/test",
            "scope": "ags"
        }
        # Output path not under _runs
        output_path = Path("/tmp/invalid/output.json")

        with pytest.raises(ValueError, match="_runs"):
            cortex_toolkit.op_smoke_test(payload, output_path, mock_writer)


# ============================================================================
# Test: Invalid Operation
# ============================================================================

class TestInvalidOperation:
    """Test error handling for invalid operations."""

    def test_invalid_operation_raises_error(self, tmp_path: Path):
        """Test that invalid operation name raises error."""
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"

        payload = {"operation": "invalid_operation_name"}
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        # Mock ensure_canon_compat to return True
        with mock.patch.object(cortex_toolkit, "ensure_canon_compat", return_value=True):
            with mock.patch.object(cortex_toolkit, "get_writer") as mock_get_writer:
                mock_get_writer.return_value = mock.MagicMock()
                result = cortex_toolkit.main(input_path, output_path)

        assert result == 1  # Should fail with exit code 1

    def test_unknown_operation_not_in_registry(self):
        """Test that unknown operations are not in the registry."""
        unknown_ops = ["unknown", "foo", "bar", "invalid"]
        for op in unknown_ops:
            assert op not in cortex_toolkit.OPERATIONS


# ============================================================================
# Test: Missing Operation Field
# ============================================================================

class TestMissingOperationField:
    """Test error handling when operation field is missing."""

    def test_missing_operation_field_raises_error(self, tmp_path: Path):
        """Test that missing operation field raises error."""
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"

        payload = {"some_field": "value"}  # No 'operation' field
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        with mock.patch.object(cortex_toolkit, "ensure_canon_compat", return_value=True):
            with mock.patch.object(cortex_toolkit, "get_writer") as mock_get_writer:
                mock_get_writer.return_value = mock.MagicMock()
                result = cortex_toolkit.main(input_path, output_path)

        assert result == 1  # Should fail with exit code 1

    def test_empty_operation_field_raises_error(self, tmp_path: Path):
        """Test that empty operation field raises error."""
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"

        payload = {"operation": ""}  # Empty operation
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        with mock.patch.object(cortex_toolkit, "ensure_canon_compat", return_value=True):
            with mock.patch.object(cortex_toolkit, "get_writer") as mock_get_writer:
                mock_get_writer.return_value = mock.MagicMock()
                result = cortex_toolkit.main(input_path, output_path)

        assert result == 1  # Should fail with exit code 1

    def test_null_operation_field_raises_error(self, tmp_path: Path):
        """Test that null operation field raises error."""
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"

        payload = {"operation": None}  # Null operation
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        with mock.patch.object(cortex_toolkit, "ensure_canon_compat", return_value=True):
            with mock.patch.object(cortex_toolkit, "get_writer") as mock_get_writer:
                mock_get_writer.return_value = mock.MagicMock()
                result = cortex_toolkit.main(input_path, output_path)

        assert result == 1  # Should fail with exit code 1


# ============================================================================
# Test: Fixture Loading
# ============================================================================

class TestFixtureLoading:
    """Test that fixtures can be loaded correctly."""

    def test_basic_fixture_exists(self):
        """Test that the basic fixture file exists."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        assert fixture_path.exists(), f"Basic fixture not found at {fixture_path}"

    def test_basic_fixture_is_valid_json(self):
        """Test that the basic fixture is valid JSON."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        if fixture_path.exists():
            content = fixture_path.read_text(encoding="utf-8")
            data = json.loads(content)
            assert isinstance(data, dict)

    def test_basic_fixture_has_operation(self):
        """Test that the basic fixture has an operation field."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        if fixture_path.exists():
            data = json.loads(fixture_path.read_text(encoding="utf-8"))
            assert "operation" in data
            assert data["operation"] in cortex_toolkit.OPERATIONS


# ============================================================================
# Test: Input/Output Handling
# ============================================================================

class TestInputOutputHandling:
    """Test input parsing and output writing."""

    def test_invalid_json_input_returns_error(self, tmp_path: Path):
        """Test that invalid JSON input returns error."""
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"

        input_path.write_text("not valid json {{{", encoding="utf-8")

        with mock.patch.object(cortex_toolkit, "ensure_canon_compat", return_value=True):
            result = cortex_toolkit.main(input_path, output_path)

        assert result == 1

    def test_nonexistent_input_file_returns_error(self, tmp_path: Path):
        """Test that nonexistent input file returns error."""
        input_path = tmp_path / "nonexistent.json"
        output_path = tmp_path / "output.json"

        with mock.patch.object(cortex_toolkit, "ensure_canon_compat", return_value=True):
            result = cortex_toolkit.main(input_path, output_path)

        assert result == 1


# ============================================================================
# Test: SHA256 File Hashing
# ============================================================================

class TestSha256FileHashing:
    """Test the internal SHA256 file hashing function."""

    def test_sha256_file_computes_correct_hash(self, tmp_path: Path):
        """Test that _sha256_file computes correct hash."""
        test_file = tmp_path / "test.txt"
        content = b"Hello, World!"
        test_file.write_bytes(content)

        result = cortex_toolkit._sha256_file(test_file)
        expected = hashlib.sha256(content).hexdigest()

        assert result == expected

    def test_sha256_file_empty_file(self, tmp_path: Path):
        """Test SHA256 of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        result = cortex_toolkit._sha256_file(test_file)
        expected = hashlib.sha256(b"").hexdigest()

        assert result == expected


# ============================================================================
# Test: Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Test helper functions in the module."""

    def test_resolve_out_dir_absolute(self):
        """Test _resolve_out_dir with absolute path."""
        if sys.platform == "win32":
            path = "C:\\some\\absolute\\path"
        else:
            path = "/some/absolute/path"

        result = cortex_toolkit._resolve_out_dir(path)
        assert result.is_absolute()

    def test_resolve_out_dir_relative(self):
        """Test _resolve_out_dir with relative path."""
        path = "relative/path"
        result = cortex_toolkit._resolve_out_dir(path)
        assert result.is_absolute()
        assert str(cortex_toolkit.PROJECT_ROOT) in str(result)
