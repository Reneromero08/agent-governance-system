"""
Test to ensure foundational directories exist and cannot be accidentally deleted.

This test acts as a governance guardrail that fails if critical directories
are missing, preventing agents from "helpfully" deleting these important paths.
"""
import os
from pathlib import Path


def test_foundational_directories_exist():
    """Test that foundational directories exist at repo root."""
    # Define the required foundational directories
    required_dirs = [
        "CAPABILITY/CAS",
        "CAPABILITY/ARTIFACTS"
    ]
    
    # Get the repository root directory (assumes running from repo root)
    repo_root = Path().resolve()
    
    # Check each required directory
    for dir_path in required_dirs:
        full_path = repo_root / dir_path
        
        # Verify the directory exists
        assert full_path.exists(), f"Required directory missing: {dir_path}"
        assert full_path.is_dir(), f"Path exists but is not a directory: {dir_path}"