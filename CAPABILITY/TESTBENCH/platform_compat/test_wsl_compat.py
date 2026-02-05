#!/usr/bin/env python3
"""
Tests for WSL cross-platform compatibility module.

These tests verify platform detection, path conversion, and
temp directory selection work correctly across environments.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CAPABILITY.PRIMITIVES.wsl_compat import (
    is_wsl,
    get_platform_type,
    get_python_executable,
    windows_to_wsl_path,
    wsl_to_windows_path,
    normalize_path_for_platform,
    get_temp_directory,
    get_repo_root,
    ensure_temp_directory,
)


class TestIsWsl:
    """Tests for WSL detection."""

    def test_detects_wsl_via_env_var(self):
        """WSL_DISTRO_NAME environment variable indicates WSL."""
        with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}):
            assert is_wsl() is True

    def test_detects_non_wsl_without_env_var(self):
        """Without WSL_DISTRO_NAME and without /proc/version, not WSL."""
        env_without_wsl = {k: v for k, v in os.environ.items() if k != "WSL_DISTRO_NAME"}
        with patch.dict(os.environ, env_without_wsl, clear=True):
            with patch("builtins.open", side_effect=OSError("File not found")):
                assert is_wsl() is False

    def test_detects_wsl_via_proc_version(self):
        """/proc/version containing 'microsoft' indicates WSL."""
        env_without_wsl = {k: v for k, v in os.environ.items() if k != "WSL_DISTRO_NAME"}
        with patch.dict(os.environ, env_without_wsl, clear=True):
            mock_version = "Linux version 5.4.0-microsoft-standard-WSL2"
            with patch("builtins.open", mock_open(read_data=mock_version)):
                assert is_wsl() is True


class TestGetPlatformType:
    """Tests for platform type detection."""

    def test_darwin_detection(self):
        """macOS returns 'darwin'."""
        with patch.object(sys, "platform", "darwin"):
            assert get_platform_type() == "darwin"

    def test_windows_detection(self):
        """Windows returns 'windows'."""
        with patch.object(sys, "platform", "win32"):
            assert get_platform_type() == "windows"

    def test_linux_detection(self):
        """Linux without WSL returns 'linux'."""
        env_without_wsl = {k: v for k, v in os.environ.items() if k != "WSL_DISTRO_NAME"}
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, env_without_wsl, clear=True):
                with patch("builtins.open", side_effect=OSError("File not found")):
                    assert get_platform_type() == "linux"

    def test_wsl_detection(self):
        """Linux with WSL indicators returns 'wsl'."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}):
                assert get_platform_type() == "wsl"


class TestPathConversion:
    """Tests for Windows/WSL path conversion."""

    def test_windows_to_wsl_c_drive(self):
        """C: drive converts to /mnt/c/."""
        result = windows_to_wsl_path("C:\\Users\\test\\file.txt")
        assert result == "/mnt/c/Users/test/file.txt"

    def test_windows_to_wsl_d_drive(self):
        """D: drive converts to /mnt/d/."""
        result = windows_to_wsl_path("D:\\CCC 2.0\\AI\\repo")
        assert result == "/mnt/d/CCC 2.0/AI/repo"

    def test_windows_to_wsl_lowercase_drive(self):
        """Lowercase drive letter is normalized."""
        result = windows_to_wsl_path("c:\\Users\\test")
        assert result == "/mnt/c/Users/test"

    def test_windows_to_wsl_forward_slashes(self):
        """Forward slashes in Windows path are handled."""
        result = windows_to_wsl_path("C:/Users/test/file.txt")
        assert result == "/mnt/c/Users/test/file.txt"

    def test_windows_to_wsl_unc_rejected(self):
        """UNC paths raise ValueError."""
        with pytest.raises(ValueError, match="UNC paths not supported"):
            windows_to_wsl_path("\\\\server\\share\\file.txt")

    def test_windows_to_wsl_unix_passthrough(self):
        """Unix paths pass through unchanged."""
        result = windows_to_wsl_path("/home/user/file.txt")
        assert result == "/home/user/file.txt"

    def test_wsl_to_windows_c_drive(self):
        """WSL /mnt/c/ converts to C:\\."""
        result = wsl_to_windows_path("/mnt/c/Users/test/file.txt")
        assert result == "C:\\Users\\test\\file.txt"

    def test_wsl_to_windows_d_drive(self):
        """WSL /mnt/d/ converts to D:\\."""
        result = wsl_to_windows_path("/mnt/d/CCC 2.0/AI/repo")
        assert result == "D:\\CCC 2.0\\AI\\repo"

    def test_wsl_to_windows_non_mnt_passthrough(self):
        """Non-/mnt/ paths pass through unchanged."""
        result = wsl_to_windows_path("/home/user/file.txt")
        assert result == "/home/user/file.txt"


class TestPathConversionRoundtrip:
    """Tests for path conversion roundtrip integrity."""

    @pytest.mark.parametrize("windows_path", [
        "C:\\Users\\test\\file.txt",
        "D:\\CCC 2.0\\AI\\agent-governance-system\\file.py",
        "E:\\Projects\\data.json",
    ])
    def test_roundtrip_windows_to_wsl_to_windows(self, windows_path):
        """Windows -> WSL -> Windows roundtrip preserves path."""
        wsl_path = windows_to_wsl_path(windows_path)
        result = wsl_to_windows_path(wsl_path)
        assert result == windows_path

    @pytest.mark.parametrize("wsl_path", [
        "/mnt/c/Users/test/file.txt",
        "/mnt/d/CCC 2.0/AI/repo/file.py",
    ])
    def test_roundtrip_wsl_to_windows_to_wsl(self, wsl_path):
        """WSL -> Windows -> WSL roundtrip preserves path."""
        windows_path = wsl_to_windows_path(wsl_path)
        result = windows_to_wsl_path(windows_path)
        assert result == wsl_path


class TestNormalizePathForPlatform:
    """Tests for platform-aware path normalization."""

    def test_wsl_normalizes_windows_path(self):
        """In WSL, Windows paths are converted to WSL format."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="wsl"):
            result = normalize_path_for_platform("C:\\Users\\test\\file.txt")
            assert result == "/mnt/c/Users/test/file.txt"

    def test_windows_normalizes_wsl_path(self):
        """In Windows, WSL paths are converted to Windows format."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="windows"):
            result = normalize_path_for_platform("/mnt/c/Users/test/file.txt")
            assert result == "C:\\Users\\test\\file.txt"

    def test_linux_keeps_unix_path(self):
        """In Linux, Unix paths are unchanged."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="linux"):
            result = normalize_path_for_platform("/home/user/file.txt")
            assert result == "/home/user/file.txt"


class TestGetTempDirectory:
    """Tests for temporary directory selection."""

    def test_wsl_uses_tmp_subdir(self):
        """WSL uses /tmp/pytest_tmp."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="wsl"):
            result = get_temp_directory()
            assert result == Path("/tmp/pytest_tmp")

    def test_linux_uses_tmp_subdir(self):
        """Linux uses /tmp/pytest_tmp."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="linux"):
            result = get_temp_directory()
            assert result == Path("/tmp/pytest_tmp")

    def test_darwin_uses_tmp_subdir(self):
        """macOS uses /tmp/pytest_tmp."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="darwin"):
            result = get_temp_directory()
            assert result == Path("/tmp/pytest_tmp")

    def test_windows_uses_repo_relative(self):
        """Windows uses repo-relative path based on module location."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="windows"):
            result = get_temp_directory()
            # Should end with the expected subdirectory structure
            assert result.parts[-4:] == ("LAW", "CONTRACTS", "_runs", "pytest_tmp")


class TestGetPythonExecutable:
    """Tests for Python executable resolution."""

    def test_wsl_prefers_linux_python(self):
        """In WSL, prefer Linux Python path."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="wsl"):
            with patch("os.path.isfile", return_value=True):
                result = get_python_executable()
                assert result in ["/usr/bin/python3", "/usr/bin/python"]

    def test_windows_uses_sys_executable(self):
        """In Windows, use sys.executable."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="windows"):
            result = get_python_executable()
            assert result == sys.executable

    def test_linux_uses_sys_executable(self):
        """In Linux, use sys.executable."""
        with patch("CAPABILITY.PRIMITIVES.wsl_compat.get_platform_type", return_value="linux"):
            result = get_python_executable()
            assert result == sys.executable


class TestGetRepoRoot:
    """Tests for repository root resolution."""

    def test_repo_root_is_valid_directory(self):
        """get_repo_root returns a valid directory."""
        result = get_repo_root()
        assert result.is_dir()

    def test_repo_root_contains_law_directory(self):
        """Repo root should contain LAW directory."""
        result = get_repo_root()
        assert (result / "LAW").is_dir()


class TestEnsureTempDirectory:
    """Tests for temp directory creation."""

    def test_ensure_returns_path(self):
        """ensure_temp_directory returns a Path."""
        result = ensure_temp_directory()
        assert isinstance(result, Path)

    def test_ensure_creates_directory(self):
        """ensure_temp_directory creates the directory if needed."""
        result = ensure_temp_directory()
        assert result.exists()
