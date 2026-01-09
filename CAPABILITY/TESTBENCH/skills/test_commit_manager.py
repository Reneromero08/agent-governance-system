#!/usr/bin/env python3
"""
Tests for commit-manager skill.

Tests the unified commit operations skill that consolidates:
- commit-queue: Manage commit queue (enqueue/list/process)
- commit-summary-log: Generate commit summaries or message templates
- artifact-escape-hatch: Emergency artifact recovery (escape hatch check)

Supported operations: queue, summarize, recover

Run: python -m pytest CAPABILITY/TESTBENCH/skills/test_commit_manager.py -v
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
sys.path.insert(0, str(REPO_ROOT))

SKILL_DIR = REPO_ROOT / "CAPABILITY" / "SKILLS" / "commit-manager"
FIXTURES_DIR = SKILL_DIR / "fixtures"

# Use durable paths inside the project for input/output
# Note: _tmp is a catalytic domain, not durable - use a test-specific subdirectory under _runs
TEST_RUNS_DIR = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_testbench_commit_manager"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def skill_input_output() -> Generator[tuple[Path, Path], None, None]:
    """Create temp input and output file paths within durable roots."""
    test_id = str(uuid.uuid4())[:8]
    test_dir = TEST_RUNS_DIR / test_id
    test_dir.mkdir(parents=True, exist_ok=True)

    input_path = test_dir / "input.json"
    output_path = test_dir / "output.json"

    yield input_path, output_path

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def ensure_test_runs_dir_exists() -> Generator[None, None, None]:
    """Ensure the test directory exists for tests."""
    TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_runs_dir_after_session() -> Generator[None, None, None]:
    """Clean up the test directory after all tests complete."""
    yield
    # Clean up after all tests
    shutil.rmtree(TEST_RUNS_DIR, ignore_errors=True)
    # Also clean up any commit queues created by tests
    commit_queue_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "commit_queue"
    if commit_queue_dir.exists():
        for f in commit_queue_dir.glob("test-*.jsonl"):
            f.unlink(missing_ok=True)


def run_skill(input_data: Dict[str, Any], input_path: Path, output_path: Path) -> tuple[int, Dict[str, Any], str]:
    """Run the commit-manager skill with the given input.

    Returns:
        tuple of (returncode, output_data dict, stderr string)
    """
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SKILL_DIR / "run.py"), str(input_path), str(output_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text(encoding="utf-8"))

    return result.returncode, output_data, result.stderr


def load_fixture(fixture_name: str) -> Dict[str, Any]:
    """Load a fixture input.json from the fixtures directory."""
    fixture_path = FIXTURES_DIR / fixture_name / "input.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


# ============================================================================
# Test: Each operation can be invoked
# ============================================================================


class TestOperationInvocation:
    """Test that each operation (queue, summarize, recover) can be invoked."""

    def test_queue_operation_can_be_invoked(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that 'queue' operation can be invoked with list action."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "queue",
            "action": "list",
            "queue_id": "test-queue",
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        assert returncode == 0, f"Expected returncode 0, got {returncode}. stderr: {stderr}"
        assert output_data.get("ok") is True
        assert output_data.get("action") == "list"
        assert "entries" in output_data

    def test_queue_enqueue_can_be_invoked(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that 'queue' operation with enqueue action can be invoked."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "queue",
            "action": "enqueue",
            "queue_id": "test-queue-enqueue",
            "entry": {
                "message": "test: add new feature",
                "files": ["path/to/file.py"],
                "author": "test-author",
                "created_at": "2025-01-07T12:00:00Z",
            },
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        assert returncode == 0, f"Expected returncode 0, got {returncode}. stderr: {stderr}"
        assert output_data.get("ok") is True
        assert output_data.get("action") == "enqueue"
        assert "entry_id" in output_data

    def test_summarize_operation_can_be_invoked(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that 'summarize' operation can be invoked with template action."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "summarize",
            "action": "template",
            "type": "feat",
            "scope": "core",
            "subject": "add new feature",
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        assert returncode == 0, f"Expected returncode 0, got {returncode}. stderr: {stderr}"
        assert output_data.get("ok") is True
        assert "message" in output_data
        assert output_data["message"] == "feat(core): add new feature"

    def test_recover_operation_can_be_invoked(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that 'recover' operation can be invoked."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "recover",
            "check_type": "artifact-escape-hatch",
            "description": "Test escape hatch check",
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        # Note: returncode may be 0 or 1 depending on whether escaped artifacts are found
        assert "escaped_artifacts" in output_data, f"Missing escaped_artifacts. stderr: {stderr}"
        assert "escape_check_passed" in output_data

    def test_recover_with_basic_fixture(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test 'recover' operation using the basic fixture."""
        input_path, output_path = skill_input_output

        fixture_data = load_fixture("basic")
        assert fixture_data["operation"] == "recover"

        returncode, output_data, stderr = run_skill(fixture_data, input_path, output_path)

        assert "escaped_artifacts" in output_data, f"Missing escaped_artifacts. stderr: {stderr}"
        assert "escape_check_passed" in output_data


# ============================================================================
# Test: Invalid operation raises error
# ============================================================================


class TestInvalidOperation:
    """Test that invalid operation values raise errors."""

    def test_unknown_operation_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that an unknown operation value causes an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "invalid_operation",
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for invalid operation"

    def test_empty_operation_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that an empty operation value causes an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "",
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for empty operation"

    def test_null_operation_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that a null operation value causes an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": None,
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for null operation"

    def test_numeric_operation_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that a numeric operation value causes an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": 123,
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for numeric operation"


# ============================================================================
# Test: Missing operation field raises error
# ============================================================================


class TestMissingOperationField:
    """Test that missing operation field raises errors."""

    def test_missing_operation_field_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that omitting the operation field causes an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "some_other_field": "value",
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for missing operation"

    def test_empty_input_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that an empty input object causes an error."""
        input_path, output_path = skill_input_output

        input_data: Dict[str, Any] = {}

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for empty input"


# ============================================================================
# Test: Operation-specific validation
# ============================================================================


@pytest.mark.xdist_group("serial_commit_manager")
class TestQueueOperationValidation:
    """Test validation specific to the queue operation."""

    def test_queue_invalid_action_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that an invalid queue action returns an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "queue",
            "action": "invalid_action",
            "queue_id": "test-queue",
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        assert returncode == 0, f"Expected returncode 0. stderr: {stderr}"  # Still runs, but returns error in output
        assert output_data.get("ok") is False
        assert output_data.get("error") == "ACTION_INVALID"

    def test_queue_enqueue_missing_entry_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that enqueue without entry returns an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "queue",
            "action": "enqueue",
            "queue_id": "test-queue",
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        assert returncode == 0, f"Expected returncode 0. stderr: {stderr}"
        assert output_data.get("ok") is False
        assert output_data.get("error") == "ENTRY_REQUIRED"

    def test_queue_enqueue_missing_message_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that enqueue without message returns an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "queue",
            "action": "enqueue",
            "queue_id": "test-queue",
            "entry": {
                "files": ["path/to/file.py"],
                "created_at": "2025-01-07T12:00:00Z",
            },
        }

        returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

        assert returncode == 0, f"Expected returncode 0. stderr: {stderr}"
        assert output_data.get("ok") is False
        assert output_data.get("error") == "MESSAGE_REQUIRED"

    def test_queue_invalid_queue_id_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that an invalid queue_id (with special chars) returns an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "queue",
            "action": "list",
            "queue_id": "../invalid/path",
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for invalid queue_id"


@pytest.mark.xdist_group("serial_commit_manager")
class TestSummarizeOperationValidation:
    """Test validation specific to the summarize operation."""

    def test_summarize_template_missing_type_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that template action without type returns an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "summarize",
            "action": "template",
            "scope": "core",
            "subject": "add new feature",
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for missing type"

    def test_summarize_template_invalid_type_raises_error(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that template action with invalid type returns an error."""
        input_path, output_path = skill_input_output

        input_data = {
            "operation": "summarize",
            "action": "template",
            "type": "invalid_type",
            "scope": "core",
            "subject": "add new feature",
        }

        returncode, _, _ = run_skill(input_data, input_path, output_path)

        assert returncode != 0, "Expected non-zero returncode for invalid type"

    def test_summarize_all_valid_types(self, skill_input_output: tuple[Path, Path]) -> None:
        """Test that all valid commit types work."""
        input_path, output_path = skill_input_output

        valid_types = ["feat", "fix", "docs", "chore", "refactor", "test"]

        for commit_type in valid_types:
            input_data = {
                "operation": "summarize",
                "action": "template",
                "type": commit_type,
                "scope": "core",
                "subject": "test subject",
            }

            returncode, output_data, stderr = run_skill(input_data, input_path, output_path)

            assert returncode == 0, f"Expected returncode 0 for type '{commit_type}', got {returncode}. stderr: {stderr}"
            assert output_data.get("ok") is True, f"Expected ok=True for type '{commit_type}'"
            assert output_data.get("message").startswith(f"{commit_type}(core):")


# ============================================================================
# Test: Integration with fixtures
# ============================================================================


class TestFixtures:
    """Test using fixtures from the skill's fixtures directory."""

    def test_basic_fixture_exists(self) -> None:
        """Test that the basic fixture exists and is valid JSON."""
        fixture_path = FIXTURES_DIR / "basic" / "input.json"
        assert fixture_path.exists(), f"Basic fixture not found at {fixture_path}"

        content = fixture_path.read_text(encoding="utf-8")
        data = json.loads(content)

        assert "operation" in data, "Fixture must have 'operation' field"

    def test_basic_fixture_has_valid_operation(self) -> None:
        """Test that the basic fixture has a valid operation."""
        fixture_data = load_fixture("basic")

        valid_operations = {"queue", "summarize", "recover"}
        assert fixture_data["operation"] in valid_operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
