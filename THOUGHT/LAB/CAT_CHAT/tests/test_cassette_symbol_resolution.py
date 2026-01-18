"""
Tests for Cassette Symbol Resolution (Phase B.2)

Verifies:
- Symbol resolution order (cassette_network_symbol -> cassette_network -> symbol_registry)
- Targeted cassette search for @SYMBOL refs
- Fallback to local symbol registry
- Integration with CortexExpansionResolver
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

from catalytic_chat.cortex_expansion_resolver import (
    CortexExpansionResolver,
    CortexRetrievalError,
    RetrievalResult
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = Mock()
    # Default: return empty results
    executor.execute_tool.return_value = {
        "content": [{"type": "text", "text": json.dumps({"results": []})}]
    }
    return executor


@pytest.fixture
def resolver_with_mock(tmp_path, mock_tool_executor):
    """Create a CortexExpansionResolver with mocked MCP."""
    resolver = CortexExpansionResolver(
        repo_root=tmp_path,
        tool_executor=mock_tool_executor,
        fail_on_unresolved=False  # Don't raise on failure for testing
    )
    return resolver


# =============================================================================
# B.2 Resolution Order Tests
# =============================================================================

class TestResolutionOrder:
    """Test the symbol resolution order."""

    def test_cassette_network_symbol_is_in_path(self, resolver_with_mock, mock_tool_executor):
        """Cassette network symbol search should be in retrieval path for @SYMBOL."""
        # All searches return empty
        result = resolver_with_mock.resolve_expansion("@CANON/INVARIANTS")

        # Should have tried cassette_network_symbol
        assert "cassette_network_symbol" in result.retrieval_path

    def test_cassette_network_symbol_tried_before_general(self, resolver_with_mock, mock_tool_executor):
        """Cassette network symbol should be tried before general cassette network."""
        result = resolver_with_mock.resolve_expansion("@CANON/INVARIANTS")

        # Get indices
        path = result.retrieval_path
        if "cassette_network_symbol" in path and "cassette_network" in path:
            symbol_idx = path.index("cassette_network_symbol")
            general_idx = path.index("cassette_network")
            assert symbol_idx < general_idx, "cassette_network_symbol should come before cassette_network"

    def test_symbol_registry_is_last_resort(self, resolver_with_mock, mock_tool_executor):
        """Symbol registry should be last resort for @SYMBOL refs."""
        result = resolver_with_mock.resolve_expansion("@SYMBOL")

        # Should have tried symbol_registry last (for @SYMBOL refs)
        assert result.retrieval_path[-1] == "symbol_registry" or "symbol_registry" in result.retrieval_path

    def test_resolution_order_for_path_symbol(self, resolver_with_mock, mock_tool_executor):
        """Path-like symbols (with /) should try cassette_network_symbol."""
        result = resolver_with_mock.resolve_expansion("THOUGHT/LAB/NOTES")

        # Should have tried cassette_network_symbol since it has /
        assert "cassette_network_symbol" in result.retrieval_path


# =============================================================================
# B.2 Targeted Cassette Search Tests
# =============================================================================

class TestTargetedCassetteSearch:
    """Test that symbol resolution targets correct cassettes."""

    def test_canon_symbol_queries_canon_cassette(self, resolver_with_mock, mock_tool_executor):
        """@CANON/INVARIANTS should search 'INVARIANTS' in canon cassette."""
        # Make cassette_network_symbol return content
        def execute_side_effect(tool_name, args):
            if tool_name == "cassette_network_query":
                query = args.get("query", "")
                cassettes = args.get("cassettes", [])
                if "INVARIANTS" in query and cassettes == ["canon"]:
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{
                                    "content": "Canon invariants content",
                                    "score": 0.95,
                                    "cassette_id": "canon",
                                    "path": "LAW/CANON/INVARIANTS.md"
                                }]
                            })
                        }]
                    }
            return {"content": [{"type": "text", "text": json.dumps({"results": []})}]}

        mock_tool_executor.execute_tool.side_effect = execute_side_effect

        result = resolver_with_mock.resolve_expansion("@CANON/INVARIANTS")

        # Should have found content from canon
        assert result.content != ""
        assert result.source == "cassette"
        assert "cassette_network_symbol" in result.retrieval_path

    def test_thought_symbol_queries_thought_cassette(self, resolver_with_mock, mock_tool_executor):
        """@THOUGHT/LAB/NOTES should search in thought cassette."""
        def execute_side_effect(tool_name, args):
            if tool_name == "cassette_network_query":
                cassettes = args.get("cassettes", [])
                if cassettes == ["thought"]:
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{
                                    "content": "Thought content",
                                    "score": 0.9,
                                    "cassette_id": "thought",
                                    "path": "THOUGHT/LAB/NOTES.md"
                                }]
                            })
                        }]
                    }
            return {"content": [{"type": "text", "text": json.dumps({"results": []})}]}

        mock_tool_executor.execute_tool.side_effect = execute_side_effect

        result = resolver_with_mock.resolve_expansion("@THOUGHT/LAB/NOTES")

        assert result.content != ""
        assert result.source == "cassette"


# =============================================================================
# B.2 Fallback Tests
# =============================================================================

class TestFallbackBehavior:
    """Test fallback to local symbol registry."""

    def test_fallback_to_local_when_cassettes_empty(self, resolver_with_mock, mock_tool_executor):
        """Should fall back to local symbol registry when cassettes return nothing."""
        # Mock local symbol resolver
        with patch.object(resolver_with_mock, '_try_symbol_resolver') as mock_local:
            mock_local.return_value = "Local symbol content"

            result = resolver_with_mock.resolve_expansion("@LOCAL_SYMBOL")

            # Should have fallen back to symbol_registry
            assert "symbol_registry" in result.retrieval_path
            assert result.content == "Local symbol content"
            assert result.source == "symbol_registry"

    def test_unresolved_returns_empty_when_not_failing(self, resolver_with_mock, mock_tool_executor):
        """Should return empty result when nothing found and fail_on_unresolved=False."""
        result = resolver_with_mock.resolve_expansion("@NONEXISTENT_SYMBOL")

        assert result.source == "unresolved"
        assert result.content == ""

    def test_unresolved_raises_when_failing(self, tmp_path, mock_tool_executor):
        """Should raise CortexRetrievalError when fail_on_unresolved=True."""
        resolver = CortexExpansionResolver(
            repo_root=tmp_path,
            tool_executor=mock_tool_executor,
            fail_on_unresolved=True
        )

        with pytest.raises(CortexRetrievalError) as exc_info:
            resolver.resolve_expansion("@NONEXISTENT_SYMBOL")

        assert "@NONEXISTENT_SYMBOL" in str(exc_info.value)


# =============================================================================
# B.2 CassetteClient Property Tests
# =============================================================================

class TestCassetteClientProperty:
    """Test the cassette_client lazy property."""

    def test_cassette_client_is_lazy_loaded(self, tmp_path, mock_tool_executor):
        """cassette_client should be lazy loaded."""
        resolver = CortexExpansionResolver(
            repo_root=tmp_path,
            tool_executor=mock_tool_executor,
            fail_on_unresolved=False
        )

        # Should not have _cassette_client yet
        assert not hasattr(resolver, '_cassette_client') or resolver._cassette_client is None

        # Access property
        client = resolver.cassette_client

        # Now should have it
        assert client is not None
        assert resolver._cassette_client is not None

    def test_cassette_client_reuses_tool_executor(self, tmp_path, mock_tool_executor):
        """CassetteClient should use the same tool_executor."""
        resolver = CortexExpansionResolver(
            repo_root=tmp_path,
            tool_executor=mock_tool_executor,
            fail_on_unresolved=False
        )

        client = resolver.cassette_client

        # Should share the tool executor
        assert client.tool_executor is mock_tool_executor


# =============================================================================
# B.2 Stats Tracking Tests
# =============================================================================

class TestStatsTracking:
    """Test that stats are properly tracked."""

    def test_cassette_hit_increments_on_symbol_resolution(self, resolver_with_mock, mock_tool_executor):
        """cassette_hits should increment when cassette_network_symbol succeeds."""
        def execute_side_effect(tool_name, args):
            if tool_name == "cassette_network_query":
                cassettes = args.get("cassettes", [])
                if cassettes:  # Targeted search
                    return {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "results": [{
                                    "content": "Found content",
                                    "score": 0.9,
                                    "cassette_id": "canon",
                                    "path": "test.md"
                                }]
                            })
                        }]
                    }
            return {"content": [{"type": "text", "text": json.dumps({"results": []})}]}

        mock_tool_executor.execute_tool.side_effect = execute_side_effect

        initial_hits = resolver_with_mock._stats["cassette_hits"]
        resolver_with_mock.resolve_expansion("@CANON/TEST")
        assert resolver_with_mock._stats["cassette_hits"] == initial_hits + 1

    def test_total_queries_increments(self, resolver_with_mock, mock_tool_executor):
        """total_queries should increment on each resolve call."""
        initial = resolver_with_mock._stats["total_queries"]
        resolver_with_mock.resolve_expansion("query1")
        resolver_with_mock.resolve_expansion("query2")
        assert resolver_with_mock._stats["total_queries"] == initial + 2


# =============================================================================
# B.2 Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full resolution flow."""

    def test_resolve_to_expansion_returns_context_expansion(self, resolver_with_mock, mock_tool_executor):
        """resolve_to_expansion should return ContextExpansion."""
        def execute_side_effect(tool_name, args):
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [{
                            "content": "Test content",
                            "score": 0.9,
                            "cassette_id": "canon",
                            "path": "test.md"
                        }]
                    })
                }]
            }

        mock_tool_executor.execute_tool.side_effect = execute_side_effect

        from catalytic_chat.context_assembler import ContextExpansion
        expansion = resolver_with_mock.resolve_to_expansion("@TEST", priority=5)

        assert isinstance(expansion, ContextExpansion)
        assert expansion.symbol_id == "@TEST"
        assert expansion.priority == 5

    def test_resolve_batch_processes_multiple_symbols(self, resolver_with_mock, mock_tool_executor):
        """resolve_batch should process multiple symbols."""
        call_count = [0]

        def execute_side_effect(tool_name, args):
            call_count[0] += 1
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "results": [{
                            "content": f"Content {call_count[0]}",
                            "score": 0.9,
                            "cassette_id": "test",
                            "path": "test.md"
                        }]
                    })
                }]
            }

        mock_tool_executor.execute_tool.side_effect = execute_side_effect

        expansions = resolver_with_mock.resolve_batch(["@SYM1", "@SYM2", "@SYM3"])

        assert len(expansions) == 3
