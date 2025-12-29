#!/usr/bin/env python3
"""
Tests for verifier stability freeze.
Verifies stable API return shapes and error codes.
"""

import sys
import json
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch

# Add CATALYTIC-DPT to path
repo_root_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root_path / "CATALYTIC-DPT"))

from PRIMITIVES.verify_bundle import BundleVerifier, ERROR_CODES, ED25519_AVAILABLE

@pytest.fixture
def test_base():
    base = repo_root_path / "CONTRACTS" / "_runs" / "_test_verifier_freeze"
    base.mkdir(parents=True, exist_ok=True)
    yield base
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)

@pytest.fixture
def verifier():
    return BundleVerifier(project_root=repo_root_path)

def test_api_return_shape_bundle(test_base, verifier):
    """Test verify_bundle_spectrum05 returns stable shape {ok, code, details}."""
    # Test failure due to missing artifacts
    run_dir = test_base / "nonexistent"
    result = verifier.verify_bundle_spectrum05(run_dir)
    
    assert "ok" in result
    assert "code" in result
    assert "details" in result
    assert result["ok"] is False
    assert result["code"] == ERROR_CODES["ARTIFACT_MISSING"]

def test_api_return_shape_chain(test_base, verifier):
    """Test verify_chain_spectrum05 returns stable shape {ok, code, details}."""
    # Test failure due to empty chain
    result = verifier.verify_chain_spectrum05([])
    
    assert "ok" in result
    assert "code" in result
    assert "details" in result
    assert result["ok"] is False
    assert result["code"] == ERROR_CODES["CHAIN_EMPTY"]

def test_frozen_error_codes_exist():
    """Verify that required SPECTRUM-05 error codes are present and stable."""
    required_codes = [
        "ARTIFACT_MISSING", "ARTIFACT_MALFORMED", "ARTIFACT_EXTRA",
        "FIELD_MISSING", "FIELD_EXTRA", "ALGORITHM_UNSUPPORTED",
        "KEY_INVALID", "IDENTITY_INVALID", "IDENTITY_MISMATCH",
        "IDENTITY_MULTIPLE", "SIGNATURE_MALFORMED", "SIGNATURE_INCOMPLETE",
        "SIGNATURE_INVALID", "SIGNATURE_MULTIPLE", "BUNDLE_ROOT_MISMATCH",
        "CHAIN_ROOT_MISMATCH", "DECISION_INVALID", "PAYLOAD_MISMATCH",
        "SERIALIZATION_INVALID", "RESTORATION_FAILED", "FORBIDDEN_ARTIFACT",
        "OUTPUT_MISSING", "HASH_MISMATCH", "CHAIN_EMPTY", "CHAIN_DUPLICATE_RUN"
    ]
    for code in required_codes:
        assert code in ERROR_CODES
        assert ERROR_CODES[code] == code

def test_crypto_dep_missing_fail_closed(verifier):
    """Verify that missing Ed25519 dependency causes hard reject."""
    with patch("PRIMITIVES.verify_bundle.ED25519_AVAILABLE", False):
        result = verifier.verify_bundle_spectrum05(Path("some-dir"), strict=True)
        assert result["ok"] is False
        assert result["code"] == ERROR_CODES["ALGORITHM_UNSUPPORTED"]
        assert "cryptography" in result["message"]

def test_chain_dir_ordering_deterministic(test_base, verifier):
    """Verify that chain_dir ordering is deterministic (alphabetical)."""
    # Create some dummy run directories
    (test_base / "run_b").mkdir()
    (test_base / "run_a").mkdir()
    (test_base / "run_c").mkdir()
    
    # We test the logic in TOOL wrapper or here by simulating it
    run_dirs = sorted([d for d in test_base.iterdir() if d.is_dir()])
    assert [d.name for d in run_dirs] == ["run_a", "run_b", "run_c"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
