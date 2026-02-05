#!/usr/bin/env python3
"""
WSL Cross-Platform Compatibility Module

Provides platform detection and path normalization for proof generation
across Windows, WSL, Linux, and macOS environments.

This module enables proof runners to work correctly regardless of
whether they're executed in WSL or native Windows.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Literal

PlatformType = Literal["wsl", "windows", "linux", "darwin"]


def is_wsl() -> bool:
    """
    Detect if running inside Windows Subsystem for Linux.

    Detection methods:
    1. Check WSL_DISTRO_NAME environment variable (most reliable)
    2. Check /proc/version for Microsoft/WSL strings

    Returns:
        True if running in WSL, False otherwise.
    """
    # Method 1: Environment variable (most reliable)
    if os.environ.get("WSL_DISTRO_NAME"):
        return True

    # Method 2: Check /proc/version
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            version_info = f.read().lower()
            if "microsoft" in version_info or "wsl" in version_info:
                return True
    except (OSError, IOError):
        pass

    return False


def get_platform_type() -> PlatformType:
    """
    Get the current platform type.

    Returns:
        One of: 'wsl', 'windows', 'linux', 'darwin'
    """
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform == "win32":
        return "windows"
    elif is_wsl():
        return "wsl"
    else:
        return "linux"


def get_python_executable() -> str:
    """
    Get the platform-appropriate Python executable path.

    For WSL, this returns the Linux Python path to avoid
    invoking Windows Python from within WSL.

    Returns:
        Path to Python executable suitable for current platform.
    """
    platform = get_platform_type()

    if platform == "wsl":
        # In WSL, prefer the Linux Python
        for python_path in ["/usr/bin/python3", "/usr/bin/python"]:
            if os.path.isfile(python_path):
                return python_path
        # Fallback to sys.executable if Linux Python not found
        return sys.executable
    else:
        # For all other platforms, use the current interpreter
        return sys.executable


def windows_to_wsl_path(path: str) -> str:
    """
    Convert a Windows path to WSL path format.

    Examples:
        C:\\Users\\name\\file.txt -> /mnt/c/Users/name/file.txt
        D:\\CCC 2.0\\AI\\repo -> /mnt/d/CCC 2.0/AI/repo

    Args:
        path: Windows-style path (e.g., C:\\Users\\...)

    Returns:
        WSL-style path (e.g., /mnt/c/Users/...)
    """
    # Handle UNC paths (not supported in WSL mount conversion)
    if path.startswith("\\\\"):
        raise ValueError(f"UNC paths not supported: {path}")

    # Normalize path separators
    path = path.replace("\\", "/")

    # Match drive letter pattern
    match = re.match(r"^([A-Za-z]):/(.*)$", path)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2)
        return f"/mnt/{drive}/{rest}"

    # Already a Unix path or relative path
    return path


def wsl_to_windows_path(path: str) -> str:
    """
    Convert a WSL path to Windows path format.

    Examples:
        /mnt/c/Users/name/file.txt -> C:\\Users\\name\\file.txt
        /mnt/d/CCC 2.0/AI/repo -> D:\\CCC 2.0\\AI\\repo

    Args:
        path: WSL-style path (e.g., /mnt/c/Users/...)

    Returns:
        Windows-style path (e.g., C:\\Users\\...)
    """
    match = re.match(r"^/mnt/([a-z])/(.*)$", path)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"

    # Not a /mnt/ path, return as-is
    return path


def normalize_path_for_platform(path: str) -> str:
    """
    Normalize a path for the current platform.

    If running in WSL with a Windows path, converts to WSL format.
    If running in Windows with a WSL path, converts to Windows format.

    Args:
        path: Path in any format

    Returns:
        Path normalized for current platform
    """
    platform = get_platform_type()

    if platform == "wsl":
        # Check if it's a Windows path
        if re.match(r"^[A-Za-z]:[/\\]", path):
            return windows_to_wsl_path(path)
    elif platform == "windows":
        # Check if it's a WSL path
        if path.startswith("/mnt/"):
            return wsl_to_windows_path(path)

    return path


def _get_repo_root_internal() -> Path:
    """
    Internal helper to get repo root from file location.

    Returns:
        Path to repository root
    """
    # Navigate from this file to repo root
    # wsl_compat.py -> PRIMITIVES -> CAPABILITY -> repo_root
    this_file = Path(__file__).resolve()
    return this_file.parents[2]


def get_temp_directory() -> Path:
    """
    Get the platform-appropriate temporary directory for proof runs.

    WSL: Uses /tmp/pytest_tmp (Linux standard with subdirectory)
    Windows: Uses repo-relative tmp to avoid long path issues
    Linux/macOS: Uses /tmp/pytest_tmp

    Returns:
        Path to temporary directory
        Caller should ensure this exists by calling mkdir(parents=True, exist_ok=True)
    """
    platform = get_platform_type()

    if platform == "wsl" or platform == "linux" or platform == "darwin":
        return Path("/tmp") / "pytest_tmp"
    else:
        # Windows: use repo-relative path to avoid issues
        # Caller should ensure this exists
        return _get_repo_root_internal() / "LAW" / "CONTRACTS" / "_runs" / "pytest_tmp"


def get_repo_root() -> Path:
    """
    Get the repository root, handling WSL path conversion if needed.

    Returns:
        Path to repository root
    """
    # Navigate from this file to repo root
    # wsl_compat.py -> PRIMITIVES -> CAPABILITY -> repo_root
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2]

    return repo_root


def ensure_temp_directory() -> Path:
    """
    Ensure the temporary directory exists and return its path.

    Returns:
        Path to existing temporary directory
    """
    tmp_dir = get_temp_directory()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir
