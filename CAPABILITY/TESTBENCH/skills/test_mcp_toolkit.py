#!/usr/bin/env python3
"""
Tests for mcp-toolkit skill.

Tests that prove:
1. Each supported operation can be invoked
2. Invalid operations raise errors
3. Missing operation field raises errors
4. Fixtures are correctly loaded and used

Run: python -m pytest CAPABILITY/TESTBENCH/skills/test_mcp_toolkit.py -v
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Skill paths
MCP_TOOLKIT_DIR = REPO_ROOT / "CAPABILITY" / "SKILLS" / "mcp-toolkit"
FIXTURES_DIR = MCP_TOOLKIT_DIR / "fixtures"


@pytest.fixture
def temp_output_path(tmp_path: Path) -> Path:
    """Create a temporary output path for test results."""
    return tmp_path / "output.json"


@pytest.fixture
def mock_writer():
    """Create a mock GuardedWriter for testing."""
    writer = MagicMock()
    writer.mkdir_durable = MagicMock()
    writer.write_durable = MagicMock()
    return writer


@pytest.fixture
def basic_input_fixture() -> Dict[str, Any]:
    """Load the basic input fixture."""
    fixture_path = FIXTURES_DIR / "basic" / "input.json"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON from file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class TestMcpToolkitOperations:
    """Test each supported operation can be invoked."""

    def test_build_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test build operation can be invoked."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {"operation": "build", "task": {"id": "test-build-123"}}
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        _write_json(input_path, input_data)

        result = mcp_toolkit.op_build(input_data, output_path, mock_writer)

        assert result == 0
        mock_writer.write_durable.assert_called_once()
        # Verify output was written
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert written_data["status"] == "success"
        assert written_data["task_id"] == "test-build-123"

    def test_validate_access_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test validate_access operation can be invoked."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {
            "operation": "validate_access",
            "agent_action": "querying database for symbols",
            "agent_code_snippet": "",
            "files_accessed": [],
            "databases_queried": []
        }
        output_path = tmp_path / "output.json"

        result = mcp_toolkit.op_validate_access(input_data, output_path, mock_writer)

        assert result == 0
        mock_writer.write_durable.assert_called_once()
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert "validation_passed" in written_data
        assert "token_waste_detected" in written_data

    def test_verify_extension_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test verify_extension operation can be invoked."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {
            "operation": "verify_extension",
            "client": "vscode",
            "entrypoint_substring": "LAW/CONTRACTS/ags_mcp_entrypoint.py",
            "args": ["--test"]
        }
        output_path = tmp_path / "output.json"

        # Mock _get_cortex_query to avoid needing actual cortex
        with patch.object(mcp_toolkit, "_get_cortex_query", return_value=(None, "mocked")):
            result = mcp_toolkit.op_verify_extension(input_data, output_path, mock_writer)

        # Should return 1 when cortex unavailable, but operation was invoked
        assert result == 1
        mock_writer.write_durable.assert_called_once()
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert "ok" in written_data
        assert "instructions" in written_data

    def test_message_board_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test message_board operation can be invoked."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {"operation": "message_board"}
        output_path = tmp_path / "output.json"

        result = mcp_toolkit.op_message_board(input_data, output_path, mock_writer)

        assert result == 0
        mock_writer.write_durable.assert_called_once()
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert written_data["ok"] is False
        assert written_data["code"] == "NOT_IMPLEMENTED"

    def test_precommit_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test precommit operation can be invoked (dry_run mode)."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {
            "operation": "precommit",
            "dry_run": True,
            "require_running": False,
            "require_autostart": False
        }
        output_path = tmp_path / "output.json"

        result = mcp_toolkit.op_precommit(input_data, output_path, mock_writer)

        assert result == 0
        mock_writer.write_durable.assert_called_once()
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert written_data["ok"] is True
        assert "checks" in written_data
        # All checks should be skipped in dry_run
        for check in written_data["checks"].values():
            assert check.get("skipped") is True

    def test_smoke_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test smoke operation can be invoked."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {
            "operation": "smoke",
            "entrypoint_substring": "LAW/CONTRACTS/ags_mcp_entrypoint.py",
            "args": ["--test"],
            "bridge_smoke": {"enabled": False}
        }
        output_path = tmp_path / "output.json"

        # Mock _get_cortex_query to avoid needing actual cortex
        with patch.object(mcp_toolkit, "_get_cortex_query", return_value=(None, "mocked")):
            result = mcp_toolkit.op_smoke(input_data, output_path, mock_writer)

        # Should return 1 when cortex unavailable, but operation was invoked
        assert result == 1
        mock_writer.write_durable.assert_called_once()
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert "ok" in written_data
        assert "bridge_smoke" in written_data

    def test_adapt_operation_invokable(self, tmp_path: Path, mock_writer: MagicMock):
        """Test adapt operation can be invoked."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {"operation": "adapt", "task": {"id": "adapt-task-456"}}
        output_path = tmp_path / "output.json"

        result = mcp_toolkit.op_adapt(input_data, output_path, mock_writer)

        assert result == 0
        mock_writer.write_durable.assert_called_once()
        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert written_data["status"] == "success"
        assert written_data["task_id"] == "adapt-task-456"


class TestMcpToolkitErrors:
    """Test error handling for invalid inputs."""

    def test_invalid_operation_raises_error(self, tmp_path: Path, capsys):
        """Test that invalid operation returns error code and prints message."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {"operation": "invalid_operation"}
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        _write_json(input_path, input_data)

        result = mcp_toolkit.main(input_path, output_path)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown operation 'invalid_operation'" in captured.out
        assert "Valid:" in captured.out

    def test_missing_operation_field_raises_error(self, tmp_path: Path, capsys):
        """Test that missing operation field returns error code."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {"task": {"id": "test"}}  # No operation field
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        _write_json(input_path, input_data)

        result = mcp_toolkit.main(input_path, output_path)

        assert result == 1
        captured = capsys.readouterr()
        assert "'operation' field is required" in captured.out

    def test_empty_operation_field_raises_error(self, tmp_path: Path, capsys):
        """Test that empty operation field returns error code."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {"operation": ""}  # Empty operation
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        _write_json(input_path, input_data)

        result = mcp_toolkit.main(input_path, output_path)

        assert result == 1
        captured = capsys.readouterr()
        assert "'operation' field is required" in captured.out

    def test_invalid_json_input_raises_error(self, tmp_path: Path, capsys):
        """Test that invalid JSON input returns error code."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        input_path.write_text("{ invalid json }", encoding="utf-8")

        result = mcp_toolkit.main(input_path, output_path)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error reading input JSON" in captured.out

    def test_nonexistent_input_file_raises_error(self, tmp_path: Path, capsys):
        """Test that nonexistent input file returns error code."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_path = tmp_path / "nonexistent.json"
        output_path = tmp_path / "output.json"

        result = mcp_toolkit.main(input_path, output_path)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error reading input JSON" in captured.out


class TestMcpToolkitFixtures:
    """Test that fixtures from the skill are correctly used."""

    def test_basic_fixture_exists(self):
        """Test that basic input fixture exists."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        assert fixture_path.exists(), f"Basic fixture not found at {fixture_path}"

    def test_basic_fixture_is_valid_json(self):
        """Test that basic fixture is valid JSON."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        if not fixture_path.exists():
            pytest.skip(f"Fixture not found: {fixture_path}")

        data = _load_json(fixture_path)
        assert isinstance(data, dict)
        assert "operation" in data

    def test_basic_fixture_operation(self, tmp_path: Path, mock_writer: MagicMock, basic_input_fixture: Dict[str, Any]):
        """Test that basic fixture can be executed."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        output_path = tmp_path / "output.json"
        operation = basic_input_fixture.get("operation")

        assert operation in mcp_toolkit.OPERATIONS, f"Fixture operation '{operation}' not in OPERATIONS"

        # Execute the operation from fixture
        op_func = mcp_toolkit.OPERATIONS[operation]
        result = op_func(basic_input_fixture, output_path, mock_writer)

        # Should complete without exception
        assert result in (0, 1)  # Either success or expected failure
        mock_writer.write_durable.assert_called_once()


class TestMcpToolkitValidateAccess:
    """Test validate_access operation in detail."""

    def test_detects_database_query_pattern(self, tmp_path: Path, mock_writer: MagicMock):
        """Test that database query patterns are detected."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {
            "operation": "validate_access",
            "agent_action": "SELECT * FROM symbols WHERE name LIKE '%test%'",
            "agent_code_snippet": "",
            "files_accessed": [],
            "databases_queried": []
        }
        output_path = tmp_path / "output.json"

        mcp_toolkit.op_validate_access(input_data, output_path, mock_writer)

        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert written_data["recommended_mcp_tool"] == "cortex_query"

    def test_detects_file_read_pattern(self, tmp_path: Path, mock_writer: MagicMock):
        """Test that file read patterns are detected."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        input_data = {
            "operation": "validate_access",
            "agent_action": "read_file LAW/CANON/CONTRACT.md",
            "agent_code_snippet": "",
            "files_accessed": [],
            "databases_queried": []
        }
        output_path = tmp_path / "output.json"

        mcp_toolkit.op_validate_access(input_data, output_path, mock_writer)

        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert written_data["recommended_mcp_tool"] == "canon_read"

    def test_calculates_token_waste_metrics(self, tmp_path: Path, mock_writer: MagicMock):
        """Test that token waste metrics are calculated."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        # Large code snippet should trigger token waste detection
        large_code = "x = 1\n" * 100
        input_data = {
            "operation": "validate_access",
            "agent_action": "query database",
            "agent_code_snippet": large_code,
            "files_accessed": [],
            "databases_queried": []
        }
        output_path = tmp_path / "output.json"

        mcp_toolkit.op_validate_access(input_data, output_path, mock_writer)

        call_args = mock_writer.write_durable.call_args
        written_data = json.loads(call_args[0][1])
        assert "token_waste_metrics" in written_data
        metrics = written_data["token_waste_metrics"]
        assert "code_tokens" in metrics
        assert metrics["code_tokens"] > 0


class TestMcpToolkitOperationsMapping:
    """Test that OPERATIONS mapping is correct."""

    def test_all_operations_in_mapping(self):
        """Test that all documented operations are in OPERATIONS mapping."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        expected_operations = [
            "build",
            "validate_access",
            "verify_extension",
            "message_board",
            "precommit",
            "smoke",
            "adapt"
        ]

        for op in expected_operations:
            assert op in mcp_toolkit.OPERATIONS, f"Operation '{op}' not in OPERATIONS mapping"

    def test_operations_are_callable(self):
        """Test that all operations in mapping are callable."""
        import importlib
        mcp_toolkit = importlib.import_module("CAPABILITY.SKILLS.mcp-toolkit.run")

        for op_name, op_func in mcp_toolkit.OPERATIONS.items():
            assert callable(op_func), f"Operation '{op_name}' is not callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
