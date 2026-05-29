"""
Tests for Retrieval Order (Phase E.1)

Verifies:
- Strict retrieval order enforcement
- SPC returns immediately (no vector)
- Cassette FTS before vector
- CAS before vector
- Vector only after all else fails
- ELO metadata tracking without ranking influence
"""

import pytest
import hashlib
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
    RetrievalResult,
    CortexRetrievalError
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = Mock()
    # Default to returning empty results
    executor.execute_tool.return_value = {"content": []}
    return executor


@pytest.fixture
def resolver(tmp_path, mock_tool_executor):
    """Create a CortexExpansionResolver with mocked components."""
    resolver = CortexExpansionResolver(
        repo_root=tmp_path,
        tool_executor=mock_tool_executor,
        fail_on_unresolved=False,  # Don't raise, just return empty
        enable_spc=True,
        enable_vector_fallback=True,
        enable_elo_observer=True,
        session_id="test-session"
    )
    return resolver


def compute_hash(content: str) -> str:
    """Helper to compute SHA-256 hash."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# =============================================================================
# E.1.1 Retrieval Order Enforcement
# =============================================================================

class TestRetrievalOrder:
    """Test strict retrieval order."""

    def test_spc_returns_immediately(self, resolver, mock_tool_executor):
        """SPC hit should return immediately without trying later steps."""
        # Mock SPC to return content
        mock_spc = Mock()
        mock_spc.is_spc_pointer.return_value = True
        mock_spc.resolve_pointer.return_value = {"content": "SPC content"}
        mock_spc.get_expansion_text.return_value = "SPC content"
        resolver._spc_bridge = mock_spc

        result = resolver.resolve_expansion("C", remaining_budget=5000)

        assert result.source == "spc"
        assert result.content == "SPC content"
        # Cassette network should NOT have been called
        assert "cassette_network" not in result.retrieval_path

    def test_cassette_fts_before_vector(self, resolver, mock_tool_executor):
        """Cassette FTS should be tried before vector fallback."""
        # Disable SPC
        resolver._enable_spc = False

        # Mock cassette network to return content
        def execute_tool_side_effect(tool_name, args):
            if tool_name == "cassette_network_query":
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"results": [{"content": "Cassette content"}]})
                    }]
                }
            return {"content": []}

        mock_tool_executor.execute_tool.side_effect = execute_tool_side_effect

        result = resolver.resolve_expansion("test query", remaining_budget=5000)

        assert result.source == "cassette"
        # Vector fallback should NOT be in path
        assert "vector_fallback" not in result.retrieval_path

    def test_local_index_before_cas(self, resolver, mock_tool_executor):
        """Local index should be tried before CAS."""
        # Disable SPC
        resolver._enable_spc = False

        # Mock local symbol resolver to return content
        mock_symbol_resolver = Mock()
        mock_symbol_resolver.resolve.return_value = ("Local content", False)
        resolver._symbol_resolver = mock_symbol_resolver

        result = resolver.resolve_expansion("@TestSymbol", remaining_budget=5000)

        assert result.source == "symbol_registry"
        # CAS and Vector should NOT be in path after successful local resolution
        assert "cas_lookup" not in result.retrieval_path
        assert "vector_fallback" not in result.retrieval_path

    def test_cas_before_vector(self, resolver, mock_tool_executor):
        """CAS should be tried before vector fallback."""
        # Disable SPC
        resolver._enable_spc = False

        # Use a hash-like symbol
        content = "CAS content"
        content_hash = compute_hash(content)

        # Mock CAS to return content
        def execute_tool_side_effect(tool_name, args):
            if tool_name == "cassette_network_query" and "hash:" in args.get("query", ""):
                return {
                    "content": [{"type": "text", "text": content}]
                }
            return {"content": []}

        mock_tool_executor.execute_tool.side_effect = execute_tool_side_effect

        result = resolver.resolve_expansion(content_hash, remaining_budget=5000)

        assert result.source == "cas"
        # Vector should NOT be in path
        assert "vector_fallback" not in result.retrieval_path

    def test_vector_only_after_all_fail(self, resolver, mock_tool_executor):
        """Vector fallback should only be used after all else fails."""
        # Disable SPC
        resolver._enable_spc = False

        # Mock all early paths to fail, vector to succeed
        vector_content = "Vector content"
        vector_hash = compute_hash(vector_content)

        def execute_tool_side_effect(tool_name, args):
            if tool_name == "semantic_search":
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps([{
                            "content": vector_content,
                            "similarity": 0.9,
                            "hash": vector_hash
                        }])
                    }]
                }
            return {"content": []}

        mock_tool_executor.execute_tool.side_effect = execute_tool_side_effect

        result = resolver.resolve_expansion("obscure query", remaining_budget=5000)

        assert result.source == "vector_fallback"
        # Should have tried all earlier paths first
        assert "cassette_network" in result.retrieval_path


# =============================================================================
# E.1.2 Vector Fallback Budget
# =============================================================================

class TestVectorBudget:
    """Test vector fallback budget handling."""

    def test_no_vector_without_budget(self, resolver, mock_tool_executor):
        """Vector fallback should not be tried if no budget."""
        # Disable SPC and all early paths
        resolver._enable_spc = False

        # Don't provide remaining_budget
        result = resolver.resolve_expansion("test query")

        # Vector fallback should not be in path (no budget provided)
        assert "vector_fallback" not in result.retrieval_path

    def test_vector_with_budget(self, resolver, mock_tool_executor):
        """Vector fallback should be tried if budget provided."""
        # Disable SPC
        resolver._enable_spc = False

        vector_content = "Vector content"
        vector_hash = compute_hash(vector_content)

        def execute_tool_side_effect(tool_name, args):
            if tool_name == "semantic_search":
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps([{
                            "content": vector_content,
                            "similarity": 0.9,
                            "hash": vector_hash
                        }])
                    }]
                }
            return {"content": []}

        mock_tool_executor.execute_tool.side_effect = execute_tool_side_effect

        result = resolver.resolve_expansion("test query", remaining_budget=5000)

        # Vector fallback should be in path
        assert "vector_fallback" in result.retrieval_path


# =============================================================================
# E.3 ELO Metadata Tracking
# =============================================================================

class TestEloMetadataTracking:
    """Test ELO metadata tracking without ranking influence."""

    def test_elo_observer_called_on_success(self, resolver, mock_tool_executor):
        """ELO observer should be called after successful retrieval."""
        # Disable SPC
        resolver._enable_spc = False

        # Mock cassette to return content
        def execute_tool_side_effect(tool_name, args):
            if tool_name == "cassette_network_query":
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"results": [{"content": "Test content"}]})
                    }]
                }
            return {"content": []}

        mock_tool_executor.execute_tool.side_effect = execute_tool_side_effect

        # Mock ELO observer
        mock_observer = Mock()
        resolver._elo_observer = mock_observer

        result = resolver.resolve_expansion("test query", remaining_budget=5000)

        # ELO observer should have been called
        mock_observer.on_retrieval_complete.assert_called()

    def test_elo_does_not_affect_result_order(self, resolver, mock_tool_executor):
        """ELO tracking should not change result content or order."""
        # Disable SPC
        resolver._enable_spc = False

        expected_content = "Expected content"

        def execute_tool_side_effect(tool_name, args):
            if tool_name == "cassette_network_query":
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"results": [{"content": expected_content}]})
                    }]
                }
            return {"content": []}

        mock_tool_executor.execute_tool.side_effect = execute_tool_side_effect

        # Run twice and verify same result
        result1 = resolver.resolve_expansion("test query", remaining_budget=5000)
        result2 = resolver.resolve_expansion("test query", remaining_budget=5000)

        # Content should be identical (ELO didn't change it)
        assert result1.content == result2.content
        assert expected_content in result1.content


# =============================================================================
# Stats Tracking
# =============================================================================

class TestStatsTracking:
    """Test retrieval stats tracking."""

    def test_stats_track_new_hit_types(self, resolver, mock_tool_executor):
        """Stats should include cas_hits and vector_fallback_hits."""
        stats = resolver.get_stats()

        assert "cas_hits" in stats
        assert "vector_fallback_hits" in stats
        assert stats["cas_hits"] == 0
        assert stats["vector_fallback_hits"] == 0

    def test_hit_rate_includes_all_sources(self, resolver, mock_tool_executor):
        """Hit rate should include all hit sources."""
        # Simulate some hits
        resolver._stats["spc_hits"] = 10
        resolver._stats["cassette_hits"] = 5
        resolver._stats["cas_hits"] = 3
        resolver._stats["vector_fallback_hits"] = 2
        resolver._stats["total_queries"] = 25

        stats = resolver.get_stats()

        # Hit rate should be (10+5+3+2)/25 = 0.8
        expected_hits = 10 + 5 + 3 + 2
        expected_rate = expected_hits / 25
        assert stats["hit_rate"] == expected_rate


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
