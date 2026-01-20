"""
Tests for Vector Fallback Resolver (Phase E.2)

Verifies:
- Agent is FREE to search until satisfied
- Budget is SAFETY BOUNDARY, not fill target
- Similarity threshold filtering
- Hash verification (NO TRUST BYPASS)
- Search logging for analysis
"""

import pytest
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import sys
# Add catalytic_chat to path
test_dir = Path(__file__).parent
cat_chat_dir = test_dir.parent
sys.path.insert(0, str(cat_chat_dir))

from catalytic_chat.vector_fallback import (
    VectorFallbackResolver,
    VectorBudgetConfig,
    VectorResult,
    VectorSearchLog,
    VectorSearchLogger,
    DEFAULT_MIN_SIMILARITY
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_tool_executor():
    """Create a mock tool executor."""
    executor = Mock()
    return executor


@pytest.fixture
def vector_resolver(tmp_path, mock_tool_executor):
    """Create a VectorFallbackResolver with mocked MCP."""
    resolver = VectorFallbackResolver(
        repo_root=tmp_path,
        tool_executor=mock_tool_executor,
        log_path=tmp_path / "test_search.jsonl"
    )
    return resolver


def compute_hash(content: str) -> str:
    """Helper to compute SHA-256 hash."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# =============================================================================
# Config Tests
# =============================================================================

class TestVectorBudgetConfig:
    """Test budget configuration."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = VectorBudgetConfig()
        assert config.min_similarity == DEFAULT_MIN_SIMILARITY

    def test_custom_similarity(self):
        """Config should accept custom similarity threshold."""
        config = VectorBudgetConfig(min_similarity=0.7)
        assert config.min_similarity == 0.7

    def test_invalid_min_similarity_negative(self):
        """Negative min_similarity should raise."""
        with pytest.raises(ValueError):
            VectorBudgetConfig(min_similarity=-0.1)

    def test_invalid_min_similarity_over_one(self):
        """min_similarity over 1.0 should raise."""
        with pytest.raises(ValueError):
            VectorBudgetConfig(min_similarity=1.5)

    def test_config_to_dict(self):
        """Config should serialize to dict."""
        config = VectorBudgetConfig(min_similarity=0.6)
        d = config.to_dict()
        assert d["min_similarity"] == 0.6

    def test_config_from_dict(self):
        """Config should deserialize from dict."""
        d = {"min_similarity": 0.7}
        config = VectorBudgetConfig.from_dict(d)
        assert config.min_similarity == 0.7

    def test_config_save_and_load(self, tmp_path):
        """Config should save to and load from JSON file."""
        config_path = tmp_path / "test_config.json"

        # Save
        config = VectorBudgetConfig(min_similarity=0.55)
        config.save(config_path)

        # Load
        loaded = VectorBudgetConfig.load(config_path)
        assert loaded.min_similarity == 0.55

    def test_config_load_missing_file_returns_defaults(self, tmp_path):
        """Loading from missing file should return defaults."""
        config = VectorBudgetConfig.load(tmp_path / "nonexistent.json")
        assert config.min_similarity == DEFAULT_MIN_SIMILARITY


# =============================================================================
# Similarity Threshold Tests
# =============================================================================

class TestSimilarityThreshold:
    """Test similarity threshold filtering."""

    def test_results_below_threshold_rejected(self, vector_resolver, mock_tool_executor):
        """Results below min_similarity should be filtered out."""
        high_sim_content = "High similarity content"
        low_sim_content = "Low similarity content"

        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([
                    {
                        "content": high_sim_content,
                        "similarity": 0.8,
                        "hash": compute_hash(high_sim_content)
                    },
                    {
                        "content": low_sim_content,
                        "similarity": 0.3,  # Below default 0.5 threshold
                        "hash": compute_hash(low_sim_content)
                    }
                ])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        results = vector_resolver.search("test query", remaining_budget=5000)

        # Only high similarity result should be included
        assert len(results) == 1
        assert results[0].content == high_sim_content

    def test_results_at_threshold_included(self, vector_resolver, mock_tool_executor):
        """Results exactly at min_similarity should be included."""
        content = "Threshold content"
        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([{
                    "content": content,
                    "similarity": 0.5,  # Exactly at threshold
                    "hash": compute_hash(content)
                }])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        results = vector_resolver.search("test query", remaining_budget=5000)

        assert len(results) == 1

    def test_custom_similarity_threshold(self, tmp_path, mock_tool_executor):
        """Custom similarity threshold should be respected."""
        config = VectorBudgetConfig(min_similarity=0.8)
        resolver = VectorFallbackResolver(
            repo_root=tmp_path,
            tool_executor=mock_tool_executor,
            config=config
        )

        content = "Medium similarity"
        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([{
                    "content": content,
                    "similarity": 0.7,  # Above 0.5 but below 0.8
                    "hash": compute_hash(content)
                }])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        results = resolver.search("test query", remaining_budget=5000)

        # Should be filtered because 0.7 < 0.8
        assert len(results) == 0


# =============================================================================
# Hash Verification (NO TRUST BYPASS)
# =============================================================================

class TestVectorHashVerification:
    """Test that hash verification cannot be bypassed."""

    def test_mismatched_hash_filtered(self, vector_resolver, mock_tool_executor):
        """Results with wrong hash should be filtered."""
        content = "Actual content"
        wrong_hash = compute_hash("Different content")

        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([{
                    "content": content,
                    "similarity": 0.9,
                    "hash": wrong_hash  # Wrong hash!
                }])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        results = vector_resolver.search("test query", remaining_budget=5000)

        # Should be empty because hash verification fails
        assert len(results) == 0

    def test_correct_hash_included(self, vector_resolver, mock_tool_executor):
        """Results with correct hash should be included."""
        content = "Verified content"
        correct_hash = compute_hash(content)

        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([{
                    "content": content,
                    "similarity": 0.9,
                    "hash": correct_hash
                }])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        results = vector_resolver.search("test query", remaining_budget=5000)

        assert len(results) == 1
        assert results[0].verified is True


# =============================================================================
# Budget as Safety Boundary (NOT fill target)
# =============================================================================

class TestBudgetAsSafetyBoundary:
    """Test that budget is safety boundary, not fill target."""

    def test_returns_all_valid_results_within_budget(self, vector_resolver, mock_tool_executor):
        """Should return all valid results if within budget."""
        # Create 3 small results that easily fit in budget
        results_data = []
        for i in range(3):
            content = f"Result {i}"
            results_data.append({
                "content": content,
                "similarity": 0.9,
                "hash": compute_hash(content)
            })

        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps(results_data)
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        # Large budget - all results should be returned
        results = vector_resolver.search("test query", remaining_budget=10000)

        # All 3 should be included - we're not trying to "fill" anything
        assert len(results) == 3

    def test_stops_at_safety_boundary(self, vector_resolver, mock_tool_executor):
        """Should stop when hitting safety boundary."""
        # Create results that would exceed a small budget
        results_data = []
        for i in range(10):
            content = f"Content {i}: " + "x" * 400  # ~100 tokens each
            results_data.append({
                "content": content,
                "similarity": 0.9,
                "hash": compute_hash(content)
            })

        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps(results_data)
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        # Small budget - should hit safety boundary
        results = vector_resolver.search("test query", remaining_budget=200)

        # Should have stopped at boundary, not returned all 10
        assert len(results) < 10
        assert len(results) >= 1

    def test_no_results_when_zero_budget(self, vector_resolver, mock_tool_executor):
        """Should return empty when no budget available."""
        results = vector_resolver.search("test query", remaining_budget=0)
        assert len(results) == 0

        # Verify MCP was not called
        mock_tool_executor.execute_tool.assert_not_called()

    def test_agent_finds_what_it_needs_early(self, vector_resolver, mock_tool_executor):
        """Agent typically finds what it needs on first result."""
        # Single high-quality result
        content = "Exactly what the agent needs"
        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([{
                    "content": content,
                    "similarity": 0.95,
                    "hash": compute_hash(content)
                }])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        results = vector_resolver.search("test query", remaining_budget=10000)

        # Got what we needed - just 1 result, nowhere near budget
        assert len(results) == 1
        assert results[0].content == content


# =============================================================================
# Error Handling
# =============================================================================

class TestVectorErrorHandling:
    """Test error handling in vector fallback."""

    def test_mcp_error_returns_empty(self, vector_resolver, mock_tool_executor):
        """MCP access error should return empty list."""
        from catalytic_chat.mcp_integration import McpAccessError

        mock_tool_executor.execute_tool.side_effect = McpAccessError("Test error")

        results = vector_resolver.search("test query", remaining_budget=5000)
        assert results == []

    def test_generic_exception_returns_empty(self, vector_resolver, mock_tool_executor):
        """Generic exception should return empty list (fail closed)."""
        mock_tool_executor.execute_tool.side_effect = Exception("Unknown error")

        results = vector_resolver.search("test query", remaining_budget=5000)
        assert results == []


# =============================================================================
# Search Logging
# =============================================================================

class TestSearchLogging:
    """Test search logging for analysis."""

    def test_search_logs_recorded(self, vector_resolver, mock_tool_executor):
        """Search should record logs."""
        content = "Test content"
        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([{
                    "content": content,
                    "similarity": 0.9,
                    "hash": compute_hash(content)
                }])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        vector_resolver.search("test query", remaining_budget=5000)

        logs = vector_resolver.get_search_logs()
        assert len(logs) == 1
        assert logs[0].query == "test query"
        assert logs[0].remaining_budget == 5000

    def test_log_contains_config(self, vector_resolver, mock_tool_executor):
        """Logs should contain config for analysis."""
        mock_tool_executor.execute_tool.return_value = {"content": []}

        vector_resolver.search("test query", remaining_budget=5000)

        logs = vector_resolver.get_search_logs()
        assert "min_similarity" in logs[0].config

    def test_log_tracks_filtering(self, vector_resolver, mock_tool_executor):
        """Logs should track filtering reasons."""
        high_sim = {"content": "High", "similarity": 0.9, "hash": compute_hash("High")}
        low_sim = {"content": "Low", "similarity": 0.3, "hash": compute_hash("Low")}

        mock_response = {
            "content": [{
                "type": "text",
                "text": json.dumps([high_sim, low_sim])
            }]
        }
        mock_tool_executor.execute_tool.return_value = mock_response

        vector_resolver.search("test query", remaining_budget=5000)

        logs = vector_resolver.get_search_logs()
        assert logs[0].raw_results_count == 2
        assert logs[0].filtered_by_similarity == 1

    def test_clear_logs(self, vector_resolver, mock_tool_executor):
        """Should be able to clear logs."""
        mock_tool_executor.execute_tool.return_value = {"content": []}

        vector_resolver.search("test query", remaining_budget=5000)
        assert len(vector_resolver.get_search_logs()) == 1

        vector_resolver.clear_search_logs()
        assert len(vector_resolver.get_search_logs()) == 0

    def test_logger_writes_to_file(self, tmp_path):
        """Logger should write to JSONL file."""
        log_path = tmp_path / "test_log.jsonl"
        logger = VectorSearchLogger(log_path)

        entry = VectorSearchLog(
            timestamp="2024-01-01T00:00:00Z",
            query="test",
            remaining_budget=5000,
            allocation=5000,
            min_similarity=0.5,
            raw_results_count=10,
            filtered_by_similarity=3,
            filtered_by_verification=1,
            final_results_count=6,
            tokens_used=800,
            budget_utilization_pct=16.0,  # 800/5000 = 16%
            config={"min_similarity": 0.5}
        )
        logger.log(entry)

        # Verify file was written
        assert log_path.exists()
        with open(log_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            assert data["query"] == "test"
            assert data["remaining_budget"] == 5000


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
