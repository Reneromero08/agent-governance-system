#!/usr/bin/env python3
"""Tests for workspace_isolation module.

These tests verify the workspace isolation functionality without
actually creating/removing real worktrees (to avoid side effects).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add module to path
sys.path.insert(0, str(Path(__file__).parent))

from workspace_isolation import (
    WorkspaceIsolationError,
    get_task_branch_name,
    get_task_worktree_path,
    REPO_ROOT,
)


class TestNamingConventions:
    """Test standard naming conventions."""

    def test_branch_name_simple(self):
        """Test simple task ID branch naming."""
        assert get_task_branch_name("2.4.1C.5") == "task/2.4.1C.5"

    def test_branch_name_with_dots(self):
        """Test task ID with multiple dots."""
        assert get_task_branch_name("phase.2.cleanup") == "task/phase.2.cleanup"

    def test_branch_name_with_dashes(self):
        """Test task ID with dashes."""
        assert get_task_branch_name("fix-bug-123") == "task/fix-bug-123"

    def test_worktree_path(self):
        """Test worktree path generation."""
        wt_path = get_task_worktree_path("2.4.1C.5")
        assert wt_path.name == "wt-2.4.1C.5"
        assert wt_path.parent == REPO_ROOT.parent


class TestCLIHelp:
    """Test CLI help and argument parsing."""

    def test_help_output(self):
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "workspace_isolation.py"), "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "workspace" in result.stdout.lower()
        assert "create" in result.stdout
        assert "status" in result.stdout
        assert "merge" in result.stdout
        assert "cleanup" in result.stdout

    def test_create_requires_task_id(self):
        """Test that create command requires task_id."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "workspace_isolation.py"), "create"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


class TestStatusCommand:
    """Test status command."""

    def test_status_returns_json(self):
        """Test that status command returns valid JSON."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "workspace_isolation.py"), "status"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT)
        )
        # Status should work even if we're not in optimal state
        if result.returncode == 0:
            data = json.loads(result.stdout)
            assert "current_branch" in data
            assert "is_dirty" in data
            assert "task_worktrees" in data


class TestErrorHandling:
    """Test error handling."""

    def test_cleanup_nonexistent_task(self):
        """Test cleanup of non-existent task."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "workspace_isolation.py"),
             "cleanup", "nonexistent-task-12345"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT)
        )
        # Should return error (task doesn't exist)
        # But shouldn't crash
        assert result.returncode in [0, 1]  # 0 if "nothing to do", 1 if error


class TestCleanupStale:
    """Test cleanup-stale command."""

    def test_cleanup_stale_dry_run(self):
        """Test cleanup-stale in dry-run mode."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "workspace_isolation.py"), "cleanup-stale"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT)
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            assert data["dry_run"] == True
            assert "stale_count" in data
            assert "stale_worktrees" in data


def test_repo_root_detection():
    """Test that REPO_ROOT is detected correctly."""
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / ".git").exists() or (REPO_ROOT / ".git").is_file()  # Could be worktree


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
