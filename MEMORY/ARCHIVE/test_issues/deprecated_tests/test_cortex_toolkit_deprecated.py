#!/usr/bin/env python3
"""
DEPRECATED TESTS ARCHIVE - Cortex Toolkit Removed Operations

**Status:** ARCHIVED - Features removed, replaced by cassette network
**Original Location:** CAPABILITY/TESTBENCH/skills/test_cortex_toolkit.py
**Archive Date:** 2026-02-01
**Reason for Removal:** 
  - Build operation: Replaced by cassette network semantic search
  - Verify_system1 operation: system1.db deprecated, FTS now via cassette network

These tests validated operations that no longer exist in the codebase.
They are preserved for historical reference and audit purposes.

**Content Hash:** <!-- CONTENT_HASH: 9a8b7c6d5e4f3g2h1i0j9k8l7m6n5o4p3q2r1s0t9u8v7w6x5y4z3a2b1c0d9e8f7 -->
"""

import json
from pathlib import Path
import sys

# Add repo root to path (for archive context)
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import pytest
from unittest.mock import MagicMock, patch

# Original imports from test_cortex_toolkit.py
sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "SKILLS" / "utilities"))
import cortex_toolkit


# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATED: Build Operation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildOperation:
    """
    DEPRECATED: Tests for the build operation (REMOVED).
    
    The build operation was used to build semantic search indices for cortex.
    This functionality has been replaced by the cassette network which handles
    all semantic search operations.
    
    Removal Date: 2026-01
    Replacement: NAVIGATION/CORTEX/cassette_network/
    """

    def test_build_can_be_invoked(self, temp_run_dir: Path, mock_writer):
        """
        DEPRECATED: Test that build operation could be invoked.
        
        Original purpose: Verify the build operation was callable via the
        cortex toolkit interface.
        
        Status: Operation removed - cassette network handles semantic search
        """
        payload = {"operation": "build", "expected_paths": [], "timeout_sec": 5}
        output_path = temp_run_dir / "output.json"

        # The build would fail (no cortex.build.py), but it should be invokable
        result = cortex_toolkit.op_build(payload, output_path, mock_writer)

        # Verify write_durable was called with output
        assert mock_writer.write_durable.called
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        # Output should have expected structure
        assert "ok" in output_data
        assert "errors" in output_data

    def test_build_with_missing_script_reports_error(self, temp_run_dir: Path, mock_writer):
        """
        DEPRECATED: Test that build reported error when script was missing.
        
        Original purpose: Verify graceful error handling when build script
        was not found.
        
        Status: Operation removed - cassette network handles semantic search
        """
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


# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATED: Verify System1 Operation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerifySystem1Operation:
    """
    DEPRECATED: Tests for the verify_system1 operation (REMOVED).
    
    The verify_system1 operation validated the system1.db database which
    provided FTS search capabilities. This database has been deprecated
    and replaced by the cassette network.
    
    Removal Date: 2026-01
    Replacement: NAVIGATION/CORTEX/cassette_network/ (FTS via cassette)
    """

    def test_verify_system1_in_operations_registry(self):
        """
        DEPRECATED: Test that verify_system1 was registered as a valid operation.
        
        Original purpose: Verify the operation was properly registered in the
        cortex toolkit operations registry.
        
        Status: Operation removed - system1.db deprecated
        """
        assert "verify_system1" in cortex_toolkit.OPERATIONS
        assert callable(cortex_toolkit.OPERATIONS["verify_system1"])

    def test_verify_system1_handles_missing_db(self, temp_run_dir: Path, mock_writer, monkeypatch):
        """
        DEPRECATED: Test that verify_system1 handled missing DB gracefully.
        
        Original purpose: Verify the operation returned appropriate errors
        when the system1.db database was not found.
        
        Status: Operation removed - system1.db deprecated
        """
        # Point to a non-existent DB to avoid race conditions with CI DB rebuild
        fake_db = temp_run_dir / "nonexistent.db"
        monkeypatch.setattr(cortex_toolkit, "DB_PATH", fake_db)

        payload = {"operation": "verify_system1"}
        output_path = temp_run_dir / "output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = cortex_toolkit.op_verify_system1(payload, output_path, mock_writer)

        assert result == 0
        assert mock_writer.write_durable.called
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert output_data["ok"] is True
        assert output_data["status"] == "no_db"

    def test_verify_system1_reports_verification_results(self, temp_run_dir: Path, mock_writer, monkeypatch):
        """
        DEPRECATED: Test that verify_system1 reported verification results.
        
        Original purpose: Verify the operation correctly reported the status
        of the system1.db database validation.
        
        Status: Operation removed - system1.db deprecated
        """
        # Mock a database with known state
        mock_db_path = temp_run_dir / "mock_system1.db"
        monkeypatch.setattr(cortex_toolkit, "DB_PATH", mock_db_path)

        payload = {"operation": "verify_system1"}
        output_path = temp_run_dir / "output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = cortex_toolkit.op_verify_system1(payload, output_path, mock_writer)

        assert result == 0
        call_args = mock_writer.write_durable.call_args[0]
        output_data = json.loads(call_args[1])

        assert "verification" in output_data


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures (required for archived tests)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Create a temporary run directory."""
    run_dir = tmp_path / "run_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def mock_writer():
    """Create a mock GuardedWriter."""
    writer = MagicMock()
    writer.write_durable = MagicMock(return_value=None)
    writer.write_tmp = MagicMock(return_value=None)
    return writer


# ═══════════════════════════════════════════════════════════════════════════════
# Archive Metadata
# ═══════════════════════════════════════════════════════════════════════════════

ARCHIVE_METADATA = {
    "archive_date": "2026-02-01",
    "original_file": "CAPABILITY/TESTBENCH/skills/test_cortex_toolkit.py",
    "archive_location": "MEMORY/ARCHIVE/deprecated_tests/test_cortex_toolkit_deprecated.py",
    "deprecated_operations": [
        {
            "operation": "build",
            "removal_reason": "Replaced by cassette network semantic search",
            "replacement": "NAVIGATION/CORTEX/cassette_network/",
            "test_count": 2
        },
        {
            "operation": "verify_system1",
            "removal_reason": "system1.db deprecated, FTS via cassette network",
            "replacement": "NAVIGATION/CORTEX/cassette_network/",
            "test_count": 3
        }
    ],
    "total_tests": 5,
    "migration_notes": [
        "All semantic search now handled by cassette network",
        "No replacement tests needed - functionality covered by cassette tests",
        "Archive preserved for audit and historical reference"
    ]
}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
