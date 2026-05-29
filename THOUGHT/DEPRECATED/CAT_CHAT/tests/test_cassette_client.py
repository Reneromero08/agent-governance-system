"""
Tests for CassetteClient (Phase B.1)

Verifies:
- Read-only interface (no write methods)
- Query functionality
- Symbol resolution via cassettes
- FTS query normalization
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import sys
# Add catalytic_chat to path
test_dir = Path(__file__).parent
cat_chat_dir = test_dir.parent
sys.path.insert(0, str(cat_chat_dir))

from catalytic_chat.cassette_client import (
    CassetteClient,
    CassetteResult,
    CassetteClientError
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor that returns canned responses."""
    executor = Mock()
    return executor


@pytest.fixture
def cassette_client(tmp_path, mock_tool_executor):
    """Create a CassetteClient with mocked MCP."""
    client = CassetteClient(
        repo_root=tmp_path,
        tool_executor=mock_tool_executor
    )
    return client


# =============================================================================
# B.1 Core Tests: Read-Only Interface
# =============================================================================

class TestReadOnlyInterface:
    """Verify CassetteClient is read-only by design."""

    def test_client_has_no_write_methods(self):
        """CassetteClient must have no write methods."""
        write_keywords = ['write', 'save', 'create', 'insert', 'update', 'delete']
        public_methods = [m for m in dir(CassetteClient) if not m.startswith('_')]

        write_methods = []
        for method in public_methods:
            for keyword in write_keywords:
                if keyword in method.lower():
                    write_methods.append(method)

        assert write_methods == [], f"Found write-like methods: {write_methods}"

    def test_client_has_expected_read_methods(self):
        """CassetteClient should have expected read methods."""
        expected_methods = ['query', 'resolve_symbol', 'get_network_status', 'normalize_fts_query']
        public_methods = [m for m in dir(CassetteClient) if not m.startswith('_')]

        for method in expected_methods:
            assert method in public_methods, f"Missing expected method: {method}"


# =============================================================================
# B.1 Query Tests
# =============================================================================

class TestCassetteQuery:
    """Test cassette network querying."""

    def test_query_returns_results(self, cassette_client, mock_tool_executor):
        """Query should return CassetteResult objects."""
        # Mock MCP response
        mock_tool_executor.execute_tool.return_value = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "query": "test query",
                    "results": [
                        {
                            "content": "Test content",
                            "score": 0.95,
                            "cassette_id": "canon",
                            "path": "LAW/CANON/test.md",
                            "hash": "abc123"
                        }
                    ]
                })
            }]
        }

        results = cassette_client.query("test query", limit=5)

        assert len(results) == 1
        assert isinstance(results[0], CassetteResult)
        assert results[0].content == "Test content"
        assert results[0].score == 0.95
        assert results[0].cassette_id == "canon"
        assert results[0].path == "LAW/CANON/test.md"

    def test_query_with_cassette_filter(self, cassette_client, mock_tool_executor):
        """Query should pass cassette filter to MCP."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{"type": "text", "text": json.dumps({"results": []})}]
        }

        cassette_client.query("test", cassettes=["canon", "governance"])

        call_args = mock_tool_executor.execute_tool.call_args
        assert call_args[0][0] == "cassette_network_query"
        assert call_args[0][1]["cassettes"] == ["canon", "governance"]

    def test_query_empty_returns_empty_list(self, cassette_client):
        """Empty query should return empty list."""
        results = cassette_client.query("")
        assert results == []

        results = cassette_client.query("   ")
        assert results == []

    def test_query_handles_mcp_error(self, cassette_client, mock_tool_executor):
        """Query should return empty list on MCP error."""
        from catalytic_chat.mcp_integration import McpAccessError
        mock_tool_executor.execute_tool.side_effect = McpAccessError("Test error")

        results = cassette_client.query("test")
        assert results == []

    def test_query_handles_invalid_json(self, cassette_client, mock_tool_executor):
        """Query should return empty list on invalid JSON response."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{"type": "text", "text": "not valid json"}]
        }

        results = cassette_client.query("test")
        assert results == []


# =============================================================================
# B.1 FTS Normalization Tests
# =============================================================================

class TestFTSNormalization:
    """Test FTS5 query normalization."""

    def test_normalize_strips_at_prefix(self):
        """Should strip @ prefix from symbols."""
        result = CassetteClient.normalize_fts_query("@SYMBOL")
        assert result == "SYMBOL"

    def test_normalize_extracts_path_component(self):
        """Should extract last path component for path-like queries."""
        result = CassetteClient.normalize_fts_query("CANON/INVARIANTS")
        assert result == "INVARIANTS"

        result = CassetteClient.normalize_fts_query("@THOUGHT/LAB/NOTES")
        assert result == "NOTES"

    def test_normalize_escapes_special_chars(self):
        """Should escape FTS5 special characters."""
        # Asterisk, quote, caret should be replaced with space
        result = CassetteClient.normalize_fts_query('test*query"here^now')
        assert '*' not in result
        assert '"' not in result
        assert '^' not in result

    def test_normalize_handles_operators(self):
        """Should convert AND/OR/NOT to lowercase."""
        result = CassetteClient.normalize_fts_query("test AND query OR NOT this")
        assert "AND" not in result
        assert "OR" not in result
        assert "NOT" not in result
        assert "and" in result
        assert "or" in result
        assert "not" in result

    def test_normalize_collapses_spaces(self):
        """Should collapse multiple spaces."""
        result = CassetteClient.normalize_fts_query("test   multiple    spaces")
        assert "  " not in result

    def test_normalize_empty_string(self):
        """Should handle empty string."""
        result = CassetteClient.normalize_fts_query("")
        assert result == ""


# =============================================================================
# B.1 Symbol Resolution Tests
# =============================================================================

class TestSymbolResolution:
    """Test symbol resolution via cassettes."""

    def test_resolve_symbol_basic(self, cassette_client, mock_tool_executor):
        """Should resolve basic symbol."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "results": [{"content": "Resolved content", "score": 0.9, "cassette_id": "canon", "path": "test.md"}]
                })
            }]
        }

        content = cassette_client.resolve_symbol("INVARIANTS")
        assert content == "Resolved content"

    def test_resolve_symbol_with_path(self, cassette_client, mock_tool_executor):
        """Should resolve path-like symbol and target correct cassette."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "results": [{"content": "Canon content", "score": 0.9, "cassette_id": "canon", "path": "test.md"}]
                })
            }]
        }

        content = cassette_client.resolve_symbol("@CANON/INVARIANTS")

        # Should have passed cassette filter
        call_args = mock_tool_executor.execute_tool.call_args
        assert call_args[0][1].get("cassettes") == ["canon"]

    def test_resolve_symbol_not_found(self, cassette_client, mock_tool_executor):
        """Should return None when symbol not found."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{"type": "text", "text": json.dumps({"results": []})}]
        }

        content = cassette_client.resolve_symbol("@NONEXISTENT")
        assert content is None

    def test_resolve_symbol_concatenates_results(self, cassette_client, mock_tool_executor):
        """Should concatenate multiple results."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "results": [
                        {"content": "First part", "score": 0.9, "cassette_id": "canon", "path": "a.md"},
                        {"content": "Second part", "score": 0.8, "cassette_id": "canon", "path": "b.md"}
                    ]
                })
            }]
        }

        content = cassette_client.resolve_symbol("SYMBOL")
        assert "First part" in content
        assert "Second part" in content
        assert "---" in content  # Separator


# =============================================================================
# B.1 Path Mapping Tests
# =============================================================================

class TestPathMapping:
    """Test path prefix to cassette ID mapping."""

    def test_path_to_cassette_mapping(self):
        """Verify path prefix mappings."""
        expected = {
            "LAW": "canon",
            "CANON": "canon",
            "CONTEXT": "governance",
            "CAPABILITY": "capability",
            "NAVIGATION": "navigation",
            "DIRECTION": "direction",
            "THOUGHT": "thought",
            "MEMORY": "memory",
            "INBOX": "inbox",
        }

        for path_prefix, cassette_id in expected.items():
            assert CassetteClient.PATH_TO_CASSETTE.get(path_prefix) == cassette_id

    def test_parse_symbol_extracts_cassette(self):
        """_parse_symbol should extract cassette from path."""
        client = CassetteClient()

        query, cassette = client._parse_symbol("@CANON/INVARIANTS")
        assert query == "INVARIANTS"
        assert cassette == "canon"

        query, cassette = client._parse_symbol("@THOUGHT/LAB/NOTES")
        assert query == "NOTES"
        assert cassette == "thought"

    def test_parse_symbol_simple(self):
        """_parse_symbol should handle simple symbols."""
        client = CassetteClient()

        query, cassette = client._parse_symbol("INVARIANTS")
        assert query == "INVARIANTS"
        assert cassette is None

        query, cassette = client._parse_symbol("@SIMPLE")
        assert query == "SIMPLE"
        assert cassette is None


# =============================================================================
# B.1 Network Status Tests
# =============================================================================

class TestNetworkStatus:
    """Test network status retrieval."""

    def test_get_network_status_success(self, cassette_client, mock_tool_executor):
        """Should return network status on success."""
        mock_tool_executor.execute_tool.return_value = {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "total_cassettes": 9,
                    "total_chunks": 1000
                })
            }]
        }

        status = cassette_client.get_network_status()
        assert status["total_cassettes"] == 9

    def test_get_network_status_error(self, cassette_client, mock_tool_executor):
        """Should return error info on failure."""
        from catalytic_chat.mcp_integration import McpAccessError
        mock_tool_executor.execute_tool.side_effect = McpAccessError("Network error")

        status = cassette_client.get_network_status()
        assert status["available"] is False
        assert "error" in status
