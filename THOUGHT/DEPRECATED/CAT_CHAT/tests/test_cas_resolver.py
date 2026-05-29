"""
Tests for CAS Resolver (Phase E.1)

Verifies:
- Exact hash lookup
- Hash validation
- Hash verification (NO TRUST BYPASS)
- Content mismatch handling
"""

import pytest
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
# Add catalytic_chat to path
test_dir = Path(__file__).parent
cat_chat_dir = test_dir.parent
sys.path.insert(0, str(cat_chat_dir))

from catalytic_chat.cas_resolver import CASResolver, CASResult, CASResolverError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = Mock()
    return executor


@pytest.fixture
def cas_resolver(tmp_path, mock_tool_executor):
    """Create a CASResolver with mocked MCP."""
    resolver = CASResolver(
        repo_root=tmp_path,
        tool_executor=mock_tool_executor
    )
    return resolver


def compute_hash(content: str) -> str:
    """Helper to compute SHA-256 hash."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# =============================================================================
# E.1.1 Hash Validation Tests
# =============================================================================

class TestHashValidation:
    """Test hash format validation."""

    def test_valid_hex_hash(self, cas_resolver):
        """Valid 64-char hex hash should be accepted."""
        valid_hash = "a" * 64
        assert cas_resolver.is_valid_hash(valid_hash)

    def test_valid_prefixed_hash(self, cas_resolver):
        """sha256: prefixed hash should be accepted."""
        valid_hash = "sha256:" + "b" * 64
        assert cas_resolver.is_valid_hash(valid_hash)

    def test_invalid_short_hash(self, cas_resolver):
        """Hash shorter than 64 chars should be rejected."""
        assert not cas_resolver.is_valid_hash("abc123")

    def test_invalid_long_hash(self, cas_resolver):
        """Hash longer than 64 chars should be rejected."""
        assert not cas_resolver.is_valid_hash("a" * 65)

    def test_invalid_non_hex_hash(self, cas_resolver):
        """Non-hex characters should be rejected."""
        assert not cas_resolver.is_valid_hash("g" * 64)

    def test_empty_hash(self, cas_resolver):
        """Empty string should be rejected."""
        assert not cas_resolver.is_valid_hash("")

    def test_none_hash(self, cas_resolver):
        """None should be rejected (via normalize_hash)."""
        assert cas_resolver.normalize_hash(None) is None


class TestHashNormalization:
    """Test hash normalization."""

    def test_normalize_uppercase(self, cas_resolver):
        """Uppercase hash should be normalized to lowercase."""
        upper = "A" * 64
        normalized = cas_resolver.normalize_hash(upper)
        assert normalized == "a" * 64

    def test_normalize_with_prefix(self, cas_resolver):
        """sha256: prefix should be stripped."""
        prefixed = "sha256:" + "c" * 64
        normalized = cas_resolver.normalize_hash(prefixed)
        assert normalized == "c" * 64

    def test_normalize_mixed_case(self, cas_resolver):
        """Mixed case should be lowercased."""
        mixed = "AbCdEf" + "0" * 58
        normalized = cas_resolver.normalize_hash(mixed)
        assert normalized == "abcdef" + "0" * 58


# =============================================================================
# E.1.2 CAS Lookup Tests
# =============================================================================

class TestCASLookup:
    """Test CAS lookup functionality."""

    def test_lookup_returns_content_when_found(self, cas_resolver, mock_tool_executor):
        """Lookup should return content when hash matches."""
        content = "Hello, World!"
        content_hash = compute_hash(content)

        # Mock MCP response
        mock_tool_executor.execute_tool.return_value = {
            "content": [{"type": "text", "text": content}]
        }

        result = cas_resolver.lookup(content_hash)

        assert result is not None
        assert result.content == content
        assert result.content_hash == content_hash
        assert result.verified is True

    def test_lookup_returns_none_for_invalid_hash(self, cas_resolver):
        """Lookup should return None for invalid hash format."""
        result = cas_resolver.lookup("not-a-valid-hash")
        assert result is None

    def test_lookup_returns_none_when_not_found(self, cas_resolver, mock_tool_executor):
        """Lookup should return None when content not found."""
        mock_tool_executor.execute_tool.return_value = {"content": []}

        result = cas_resolver.lookup("a" * 64)
        assert result is None


# =============================================================================
# E.1.3 Hash Verification (NO TRUST BYPASS)
# =============================================================================

class TestHashVerification:
    """Test that hash verification cannot be bypassed."""

    def test_verification_fails_on_hash_mismatch(self, cas_resolver, mock_tool_executor):
        """Lookup should fail when content hash doesn't match requested hash."""
        actual_content = "Actual content"
        claimed_hash = compute_hash("Different content")

        # Mock returns content that doesn't match the requested hash
        mock_tool_executor.execute_tool.return_value = {
            "content": [{"type": "text", "text": actual_content}]
        }

        result = cas_resolver.lookup(claimed_hash)

        # Should return None because hash verification fails
        assert result is None

    def test_verify_content_correct_hash(self, cas_resolver):
        """verify_content should return True for matching hash."""
        content = "Test content"
        correct_hash = compute_hash(content)

        assert cas_resolver.verify_content(content, correct_hash)

    def test_verify_content_wrong_hash(self, cas_resolver):
        """verify_content should return False for mismatched hash."""
        content = "Test content"
        wrong_hash = compute_hash("Different content")

        assert not cas_resolver.verify_content(content, wrong_hash)

    def test_no_trust_bypass_even_with_prefixed_hash(self, cas_resolver, mock_tool_executor):
        """Hash verification must happen even with sha256: prefix."""
        actual_content = "Actual"
        wrong_hash = "sha256:" + compute_hash("Wrong")

        mock_tool_executor.execute_tool.return_value = {
            "content": [{"type": "text", "text": actual_content}]
        }

        result = cas_resolver.lookup(wrong_hash)
        assert result is None


# =============================================================================
# E.1.4 Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in CAS resolver."""

    def test_mcp_error_returns_none(self, cas_resolver, mock_tool_executor):
        """MCP access error should return None, not raise."""
        from catalytic_chat.mcp_integration import McpAccessError

        mock_tool_executor.execute_tool.side_effect = McpAccessError("Test error")

        result = cas_resolver.lookup("a" * 64)
        assert result is None

    def test_generic_exception_returns_none(self, cas_resolver, mock_tool_executor):
        """Generic exception should return None (fail closed)."""
        mock_tool_executor.execute_tool.side_effect = Exception("Unknown error")

        result = cas_resolver.lookup("a" * 64)
        assert result is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
