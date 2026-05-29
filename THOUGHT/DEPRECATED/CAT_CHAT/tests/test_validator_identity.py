#!/usr/bin/env python3
"""
Tests for validator_identity module (Phase 6.7)
"""

import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from catalytic_chat.validator_identity import get_build_id, get_validator_identity, parse_build_id, ValidatorIdentityError


def test_parse_build_id_valid_git():
    """Test parsing valid git build IDs."""
    result = parse_build_id("git:abc1234")
    assert result["type"] == "git"
    assert result["value"] == "abc1234"


def test_parse_build_id_valid_git_full():
    """Test parsing valid git build ID with full SHA."""
    result = parse_build_id("git:a1b2c3d4e5f67890abcdef1234567890abcdef12")
    assert result["type"] == "git"
    assert result["value"] == "a1b2c3d4e5f67890abcdef1234567890abcdef12"


def test_parse_build_id_valid_file():
    """Test parsing valid file build IDs."""
    result = parse_build_id("file:a1b2c3d4e5f67890abcdef1234567890")
    assert result["type"] == "file"
    assert result["value"] == "a1b2c3d4e5f67890abcdef1234567890"


def test_parse_build_id_invalid_no_colon():
    """Test parsing fails without colon."""
    with pytest.raises(ValidatorIdentityError, match="invalid build_id format"):
        parse_build_id("invalid_no_colon")


def test_parse_build_id_invalid_git_too_short():
    """Test parsing fails for git with too short SHA."""
    with pytest.raises(ValidatorIdentityError, match="git build_id must be at least 7 chars"):
        parse_build_id("git:abc123")


def test_parse_build_id_invalid_file_too_short():
    """Test parsing fails for file with too short hash."""
    with pytest.raises(ValidatorIdentityError, match="file build_id must be at least 16 chars"):
        parse_build_id("file:a1b2c3d4e5f6789")


def test_parse_build_id_invalid_unknown_type():
    """Test parsing fails for unknown type."""
    with pytest.raises(ValidatorIdentityError, match="unknown build_id type"):
        parse_build_id("unknown:value")


def test_parse_build_id_not_string():
    """Test parsing fails for non-string."""
    with pytest.raises(ValidatorIdentityError, match="build_id must be a string"):
        parse_build_id(12345)


def test_get_validator_identity_valid_key():
    """Test getting validator identity with valid key."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]

    identity = get_validator_identity(key_bytes)

    assert "validator_id" in identity
    assert "scheme" in identity
    assert identity["scheme"] == "ed25519"
    assert "public_key" in identity
    assert "build_id" in identity

    assert len(identity["public_key"]) == 64
    assert identity["validator_id"] == identity["build_id"]


def test_get_validator_identity_custom_validator_id():
    """Test getting validator identity with custom validator_id."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        pytest.skip("PyNaCl not installed")

    signing_key = SigningKey.generate()
    key_bytes = signing_key.encode()[:32]
    custom_id = "my_custom_validator"

    identity = get_validator_identity(key_bytes, validator_id=custom_id)

    assert identity["validator_id"] == custom_id
    assert identity["build_id"] != custom_id


def test_get_validator_identity_invalid_key_not_bytes():
    """Test getting validator identity fails for non-bytes key."""
    with pytest.raises(ValidatorIdentityError, match="signing_key must be bytes"):
        get_validator_identity("not_bytes")


def test_get_validator_identity_invalid_key_too_short():
    """Test getting validator identity fails for too short key."""
    with pytest.raises(ValidatorIdentityError, match="signing_key must be 32 or 64 bytes"):
        get_validator_identity(b"short")


def test_build_id_deterministic():
    """Test A: get_build_id returns identical output on multiple calls."""
    build_id_1 = get_build_id()
    build_id_2 = get_build_id()

    assert build_id_1 == build_id_2


def test_build_id_deterministic_with_repo_root():
    """Test B: get_build_id is deterministic with explicit repo_root."""
    repo_root = Path(__file__).parent.parent.parent.parent

    build_id_1 = get_build_id(repo_root)
    build_id_2 = get_build_id(repo_root)

    assert build_id_1 == build_id_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
