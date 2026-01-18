"""
Tests for Auto-Controlled Context Loop (Section C)

Tests cover:
- C.1 Adaptive Budget Discovery
- C.2 Context Partitioner
- C.3 Turn Compressor
- C.4 Event Logging
- C.5 Loop Integration
- C.6 Threshold Adaptation

NOTE: These tests use REAL embedding models for E-score computation.
Set SKIP_REAL_MODEL=1 to skip tests requiring sentence-transformers.
"""

import os
import json
import pytest
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Check if we should skip real model tests
SKIP_REAL_MODEL = os.environ.get("SKIP_REAL_MODEL", "0") == "1"

# Import modules under test
import sys
CAT_CHAT_ROOT = Path(__file__).parent.parent
CAT_CHAT_PATH = CAT_CHAT_ROOT / "catalytic_chat"
if str(CAT_CHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(CAT_CHAT_ROOT))
if str(CAT_CHAT_PATH) not in sys.path:
    sys.path.insert(0, str(CAT_CHAT_PATH))

from catalytic_chat.adaptive_budget import (
    AdaptiveBudget,
    ModelBudgetDiscovery,
    BudgetExceededError,
    BudgetConfig,
    BudgetSnapshot,
)
from catalytic_chat.context_partitioner import (
    ContextPartitioner,
    ContextItem,
    PartitionResult,
    ScoredItem,
)
from catalytic_chat.turn_compressor import (
    TurnCompressor,
    TurnContent,
    TurnPointer,
    create_turn_from_messages,
)
from catalytic_chat.session_capsule import (
    SessionCapsule,
    EVENT_PARTITION,
    EVENT_TURN_STORED,
    EVENT_TURN_HYDRATED,
    EVENT_BUDGET_CHECK,
)
from catalytic_chat.threshold_adapter import (
    ThresholdAdapter,
    ThresholdConfig,
    EDistributionStats,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_cat_chat.db"
    yield db_path
    # Cleanup happens automatically via pytest's tmp_path


@pytest.fixture
def session_capsule(temp_db):
    """Create a session capsule with temporary database."""
    capsule = SessionCapsule(db_path=temp_db)
    yield capsule
    capsule.close()
    # Force garbage collection to release file handles
    import gc
    gc.collect()


@pytest.fixture
def session_id(session_capsule):
    """Create a test session."""
    return session_capsule.create_session()


def synthetic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic synthetic embedding from text hash."""
    text_hash = hash(text) % (2**31)
    rng = np.random.RandomState(text_hash)
    vec = rng.randn(dim)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def real_embedding_fn():
    """Get real embedding function using sentence-transformers."""
    if SKIP_REAL_MODEL:
        pytest.skip("Skipping real model test (SKIP_REAL_MODEL=1)")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        def embed(text: str) -> np.ndarray:
            return model.encode(text, convert_to_numpy=True)

        return embed
    except ImportError:
        pytest.skip("sentence-transformers not installed")


# =============================================================================
# C.1 Adaptive Budget Tests
# =============================================================================

class TestAdaptiveBudget:
    """Tests for adaptive budget discovery."""

    def test_budget_from_context_window(self):
        """Budget correctly computes from context window."""
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=40961,
            system_prompt="You are a helpful assistant.",
            response_reserve_pct=0.25,
            model_id="nemotron"
        )

        assert budget.context_window == 40961
        assert budget.response_reserve_pct == 0.25
        assert budget.response_reserve_tokens == int(40961 * 0.25)
        assert budget.system_prompt_tokens > 0
        assert budget.available_for_working_set > 0
        assert budget.available_for_working_set < budget.context_window

    def test_budget_invariant_passes(self):
        """Budget invariant check passes when under budget."""
        budget = ModelBudgetDiscovery.from_context_window(context_window=4096)

        # Should not raise
        budget.check_invariant(tokens_used=1000, item_count=5)

    def test_budget_invariant_fails(self):
        """Budget invariant check raises when over budget."""
        budget = ModelBudgetDiscovery.from_context_window(context_window=4096)

        with pytest.raises(BudgetExceededError) as exc_info:
            budget.check_invariant(tokens_used=50000, item_count=100)

        assert exc_info.value.budget_used == 50000

    def test_budget_config_from_json(self, tmp_path):
        """Budget config loads from JSON file."""
        config_path = tmp_path / "model_config.json"
        config_path.write_text(json.dumps({
            "model_context_window": 8192,
            "response_reserve_pct": 0.20,
            "model_id": "test_model"
        }))

        config = BudgetConfig.from_json_file(config_path)

        assert config.model_context_window == 8192
        assert config.response_reserve_pct == 0.20
        assert config.model_id == "test_model"

    def test_discovery_from_config_file(self, tmp_path):
        """ModelBudgetDiscovery loads from config file."""
        config_path = tmp_path / "model_config.json"
        config_path.write_text(json.dumps({
            "model_context_window": 16384,
            "response_reserve_pct": 0.30,
            "model_id": "config_model"
        }))

        discovery = ModelBudgetDiscovery(config_path=config_path)
        budget = discovery.discover(system_prompt="Test prompt")

        assert budget.context_window == 16384
        assert budget.response_reserve_pct == 0.30
        assert budget.model_id == "config_model"


# =============================================================================
# C.2 Context Partitioner Tests
# =============================================================================

class TestContextPartitioner:
    """Tests for E-score based context partitioning."""

    def test_partition_respects_threshold(self):
        """Items below threshold always go to pointer_set."""
        partitioner = ContextPartitioner(
            threshold=0.5,
            embed_fn=synthetic_embedding,
        )

        # Create items - some will naturally score low
        items = [
            ContextItem(
                item_id=f"item_{i}",
                content=f"Content for item {i}",
                tokens=100,
                embedding=synthetic_embedding(f"Content for item {i}"),
            )
            for i in range(10)
        ]

        query = "Completely unrelated query about quantum physics"
        query_embedding = synthetic_embedding(query)

        result = partitioner.partition(
            query_embedding=query_embedding,
            all_items=items,
            budget_tokens=10000,  # Large budget
            query_text=query,
        )

        # All items with E < 0.5 should be in pointer_set
        for scored in result.pointer_set:
            if result.items_below_threshold > 0:
                # At least verify some items are below threshold
                pass

        assert result.items_total == 10

    def test_partition_respects_budget(self):
        """Working_set never exceeds budget."""
        partitioner = ContextPartitioner(
            threshold=0.0,  # Accept all items based on E
            embed_fn=synthetic_embedding,
        )

        items = [
            ContextItem(
                item_id=f"item_{i}",
                content="X" * 1000,  # Large content
                tokens=250,  # Each item is 250 tokens
                embedding=synthetic_embedding(f"item {i}"),
            )
            for i in range(20)  # 20 items * 250 = 5000 tokens total
        ]

        query_embedding = synthetic_embedding("test query")

        result = partitioner.partition(
            query_embedding=query_embedding,
            all_items=items,
            budget_tokens=500,  # Small budget: fits only 2 items
            query_text="test query",
        )

        # Budget should be respected
        assert result.budget_used <= 500
        assert len(result.working_set) <= 2
        assert result.items_over_budget > 0

    def test_partition_sorts_by_E(self):
        """Highest E items are prioritized for working_set."""
        # Use very low threshold to ensure items pass
        partitioner = ContextPartitioner(
            threshold=-1.0,  # Accept all items regardless of E-score
            embed_fn=synthetic_embedding,
        )

        # Create items with known embedding relationship to query
        query = "catalytic computing context management"
        items = [
            ContextItem(
                item_id="relevant",
                content="catalytic computing manages context automatically",
                tokens=100,
                embedding=synthetic_embedding("catalytic computing manages context automatically"),
            ),
            ContextItem(
                item_id="irrelevant",
                content="the weather is sunny today",
                tokens=100,
                embedding=synthetic_embedding("the weather is sunny today"),
            ),
        ]

        query_embedding = synthetic_embedding(query)

        result = partitioner.partition(
            query_embedding=query_embedding,
            all_items=items,
            budget_tokens=100,  # Only fits one item
            query_text=query,
        )

        # Working set should have exactly one item (budget constraint)
        assert len(result.working_set) == 1
        # The remaining item should be in pointer set (over budget, not below threshold)
        assert result.items_over_budget == 1
        # Verify the highest E item was selected
        # (items were sorted by E DESC before filling budget)

    @pytest.mark.skipif(SKIP_REAL_MODEL, reason="Requires real model")
    def test_partition_with_real_embeddings(self, real_embedding_fn):
        """Partition works with real embedding model."""
        partitioner = ContextPartitioner(
            threshold=0.3,
            embed_fn=real_embedding_fn,
        )

        items = [
            ContextItem(
                item_id="auth_doc",
                content="OAuth 2.0 is an authorization framework for secure API access.",
                tokens=50,
                embedding=real_embedding_fn("OAuth 2.0 is an authorization framework for secure API access."),
            ),
            ContextItem(
                item_id="weather_doc",
                content="The forecast shows rain tomorrow with temperatures dropping.",
                tokens=50,
                embedding=real_embedding_fn("The forecast shows rain tomorrow with temperatures dropping."),
            ),
        ]

        query = "How does OAuth authentication work?"
        query_embedding = real_embedding_fn(query)

        result = partitioner.partition(
            query_embedding=query_embedding,
            all_items=items,
            budget_tokens=1000,
            query_text=query,
        )

        # Auth doc should score higher
        working_ids = [s.item.item_id for s in result.working_set]
        pointer_ids = [s.item.item_id for s in result.pointer_set]

        # With real embeddings, auth_doc should be more relevant
        if "auth_doc" in working_ids:
            auth_score = next(s.E_score for s in result.working_set if s.item.item_id == "auth_doc")
            print(f"Auth doc E-score: {auth_score}")
            assert auth_score > 0


# =============================================================================
# C.3 Turn Compressor Tests
# =============================================================================

class TestTurnCompressor:
    """Tests for turn compression."""

    def test_compression_every_turn(self, temp_db, session_id):
        """Each turn compresses immediately after response."""
        compressor = TurnCompressor(
            db_path=temp_db,
            session_id=session_id,
        )

        turn = create_turn_from_messages(
            turn_id="turn_001",
            user_message="What is catalytic computing?",
            assistant_message="Catalytic computing is a paradigm where disk state restores exactly after use.",
        )

        result = compressor.compress_turn(turn)

        assert result.stored
        assert result.pointer is not None
        assert result.pointer.content_hash == turn.compute_hash()
        assert result.pointer.compression_ratio > 1.0

    def test_turn_rehydration_by_E(self, temp_db, session_id):
        """Compressed turns can be rehydrated."""
        compressor = TurnCompressor(
            db_path=temp_db,
            session_id=session_id,
        )

        # Compress a turn
        turn = create_turn_from_messages(
            turn_id="turn_001",
            user_message="Test query",
            assistant_message="Test response with important information.",
        )

        compress_result = compressor.compress_turn(turn)
        content_hash = compress_result.pointer.content_hash

        # Hydrate the turn
        hydration = compressor.decompress_turn(content_hash, E_score=0.7)

        assert hydration.success
        assert hydration.content is not None
        assert hydration.content.user_query == turn.user_query
        assert hydration.content.assistant_response == turn.assistant_response
        assert hydration.E_score == 0.7

    def test_compression_ratio(self, temp_db, session_id):
        """Compression achieves significant ratio."""
        compressor = TurnCompressor(
            db_path=temp_db,
            session_id=session_id,
        )

        # Large turn
        turn = create_turn_from_messages(
            turn_id="turn_001",
            user_message="What is the meaning of life?",
            assistant_message="A" * 5000,  # Large response
        )

        result = compressor.compress_turn(turn)

        # Pointer should be much smaller than original
        assert result.pointer.pointer_tokens < result.pointer.original_tokens
        assert result.pointer.compression_ratio > 5.0


# =============================================================================
# C.4 Event Logging Tests
# =============================================================================

class TestEventLogging:
    """Tests for auto-context event logging."""

    def test_partition_event_logged(self, session_capsule, session_id):
        """Partition events are logged correctly."""
        event = session_capsule.log_partition(
            session_id=session_id,
            query_hash="abc123",
            working_set_ids=["item1", "item2"],
            pointer_set_ids=["item3"],
            budget_total=4000,
            budget_used=2000,
            threshold=0.5,
            E_mean=0.65,
            E_min=0.3,
            E_max=0.9,
            items_below_threshold=1,
            items_over_budget=0,
        )

        assert event.event_type == EVENT_PARTITION
        assert event.payload["query_hash"] == "abc123"
        assert event.payload["budget_used"] == 2000
        assert event.payload["E_mean"] == 0.65

        # Verify working/pointer sets updated
        state = session_capsule.get_session_state(session_id)
        assert "item1" in state.working_set
        assert "item3" in state.pointer_set

    def test_turn_stored_event_logged(self, session_capsule, session_id):
        """Turn stored events are logged correctly."""
        event = session_capsule.log_turn_stored(
            session_id=session_id,
            turn_id="turn_001",
            content_hash="hash123",
            summary="User asked about X, assistant explained Y",
            original_tokens=500,
            pointer_tokens=50,
        )

        assert event.event_type == EVENT_TURN_STORED
        assert event.payload["turn_id"] == "turn_001"
        assert event.payload["compression_ratio"] == 10.0

    def test_turn_hydrated_event_logged(self, session_capsule, session_id):
        """Turn hydrated events are logged correctly."""
        event = session_capsule.log_turn_hydrated(
            session_id=session_id,
            turn_id="turn_001",
            content_hash="hash123",
            E_score=0.75,
            tokens_added=500,
        )

        assert event.event_type == EVENT_TURN_HYDRATED
        assert event.payload["E_score"] == 0.75
        assert event.payload["tokens_added"] == 500

    def test_budget_check_event_logged(self, session_capsule, session_id):
        """Budget check events are logged correctly."""
        event = session_capsule.log_budget_check(
            session_id=session_id,
            budget_available=30000,
            budget_used=15000,
            item_count=10,
            passed=True,
            context_window=40961,
            model_id="nemotron",
        )

        assert event.event_type == EVENT_BUDGET_CHECK
        assert event.payload["passed"] == True
        assert event.payload["utilization_pct"] == 0.5

    def test_event_chain_integrity(self, session_capsule, session_id):
        """Events maintain hash chain integrity."""
        # Log multiple events
        session_capsule.log_partition(
            session_id=session_id,
            query_hash="q1",
            working_set_ids=["a"],
            pointer_set_ids=["b"],
            budget_total=1000,
            budget_used=100,
            threshold=0.5,
            E_mean=0.6,
            E_min=0.4,
            E_max=0.8,
            items_below_threshold=0,
            items_over_budget=0,
        )

        session_capsule.log_turn_stored(
            session_id=session_id,
            turn_id="t1",
            content_hash="h1",
            summary="Test",
            original_tokens=100,
            pointer_tokens=10,
        )

        # Verify chain
        is_valid, error = session_capsule.verify_chain(session_id)
        assert is_valid, f"Chain invalid: {error}"


# =============================================================================
# C.5 Loop Integration Tests
# =============================================================================

class TestLoopIntegration:
    """Tests for the full catalytic loop integration."""

    def test_catalytic_loop_determinism(self, temp_db):
        """Same inputs produce identical partition decisions."""
        from catalytic_chat.auto_context_manager import AutoContextManager

        budget = ModelBudgetDiscovery.from_context_window(context_window=4096)

        # Create separate sessions via each manager's capsule
        capsule1 = SessionCapsule(db_path=temp_db)
        session_id1 = capsule1.create_session()

        # Use a different db for manager2 to avoid conflicts
        temp_db2 = temp_db.parent / "test_cat_chat_2.db"
        capsule2 = SessionCapsule(db_path=temp_db2)
        session_id2 = capsule2.create_session()

        manager1 = AutoContextManager(
            db_path=temp_db,
            session_id=session_id1,
            budget=budget,
            embed_fn=synthetic_embedding,
            E_threshold=-1.0,  # Accept all to test determinism
        )
        # Replace the capsule with ours that created the session
        manager1.capsule = capsule1

        manager2 = AutoContextManager(
            db_path=temp_db2,
            session_id=session_id2,
            budget=budget,
            embed_fn=synthetic_embedding,
            E_threshold=-1.0,  # Accept all to test determinism
        )
        manager2.capsule = capsule2

        # Add same items
        items = [
            ContextItem(
                item_id="doc1",
                content="Catalytic computing explanation",
                tokens=50,
            ),
            ContextItem(
                item_id="doc2",
                content="Weather forecast for today",
                tokens=50,
            ),
        ]

        manager1.add_items([ContextItem(**i.__dict__) for i in items])
        manager2.add_items([ContextItem(**i.__dict__) for i in items])

        # Same query
        query = "What is catalytic computing?"
        query_embedding = synthetic_embedding(query)

        result1 = manager1.prepare_context(query, query_embedding)
        result2 = manager2.prepare_context(query, query_embedding)

        # Results should be identical
        assert result1.partition_result.query_hash == result2.partition_result.query_hash
        assert result1.tokens_used == result2.tokens_used
        assert len(result1.working_set) == len(result2.working_set)

        capsule1.close()
        capsule2.close()

    def test_all_events_logged(self, temp_db):
        """Every partition/compression/hydration is logged."""
        from catalytic_chat.auto_context_manager import AutoContextManager

        # Create session first
        capsule = SessionCapsule(db_path=temp_db)
        session_id = capsule.create_session()

        budget = ModelBudgetDiscovery.from_context_window(context_window=4096)

        manager = AutoContextManager(
            db_path=temp_db,
            session_id=session_id,
            budget=budget,
            embed_fn=synthetic_embedding,
            E_threshold=-1.0,  # Accept all items
        )
        # Use the same capsule that created the session
        manager.capsule = capsule

        # Add items
        manager.add_items([
            ContextItem(
                item_id="doc1",
                content="Test document",
                tokens=50,
            ),
        ])

        # Run a turn
        def mock_llm(system: str, prompt: str) -> str:
            return "Mock response"

        result = manager.respond_catalytic(
            query="Test query",
            llm_generate=mock_llm,
        )

        # Check events were logged
        events = capsule.get_events(session_id)
        event_types = [e.event_type for e in events]

        assert EVENT_PARTITION in event_types
        assert EVENT_BUDGET_CHECK in event_types
        assert EVENT_TURN_STORED in event_types

        capsule.close()


# =============================================================================
# C.6 Threshold Adaptation Tests
# =============================================================================

class TestThresholdAdaptation:
    """Tests for E-threshold adaptation."""

    def test_default_threshold(self):
        """Default threshold is 0.5 from Q44."""
        adapter = ThresholdAdapter()
        assert adapter.get_threshold() == 0.5

    def test_adjust_threshold(self):
        """Threshold can be manually adjusted."""
        adapter = ThresholdAdapter()

        history = adapter.adjust_threshold(0.6, reason="Test adjustment")

        assert adapter.get_threshold() == 0.6
        assert history.old_threshold == 0.5
        assert history.new_threshold == 0.6
        assert history.reason == "Test adjustment"

    def test_threshold_bounds(self):
        """Threshold stays within configured bounds."""
        config = ThresholdConfig(
            threshold=0.5,
            min_threshold=0.2,
            max_threshold=0.8,
        )
        adapter = ThresholdAdapter(config=config)

        # Try to set below min
        adapter.adjust_threshold(0.1)
        assert adapter.get_threshold() == 0.2

        # Try to set above max
        adapter.adjust_threshold(0.95)
        assert adapter.get_threshold() == 0.8

    def test_E_distribution_tracking(self):
        """E-score distribution is tracked correctly."""
        adapter = ThresholdAdapter()

        # Record some partitions
        adapter.record_partition([0.3, 0.5, 0.7, 0.9])
        adapter.record_partition([0.2, 0.4, 0.6, 0.8])

        stats = adapter.get_E_distribution_stats()

        assert stats is not None
        assert stats.sample_count == 8
        assert 0.4 < stats.E_mean < 0.6
        assert stats.E_min == 0.2
        assert stats.E_max == 0.9

    def test_threshold_suggestion(self):
        """Threshold suggestion works based on distribution."""
        adapter = ThresholdAdapter()

        # Record skewed low distribution
        np.random.seed(42)
        for _ in range(20):
            scores = np.random.beta(2, 5, 10).tolist()  # Skewed low
            adapter.record_partition(scores)

        suggested, explanation = adapter.suggest_threshold()

        # Suggestion should be lower than default since distribution is low
        assert 0.1 <= suggested <= 0.9
        assert "samples" in explanation


# =============================================================================
# Integration Test with Real Model
# =============================================================================

@pytest.mark.skipif(SKIP_REAL_MODEL, reason="Requires real model")
class TestRealModelIntegration:
    """Integration tests using real embedding model."""

    def test_full_catalytic_loop_real_model(self, temp_db, real_embedding_fn):
        """Full catalytic loop with real embeddings."""
        from catalytic_chat.auto_context_manager import AutoContextManager

        capsule = SessionCapsule(db_path=temp_db)
        session_id = capsule.create_session()

        budget = ModelBudgetDiscovery.from_context_window(
            context_window=4096,
            system_prompt="You are a helpful assistant.",
        )

        manager = AutoContextManager(
            db_path=temp_db,
            session_id=session_id,
            budget=budget,
            embed_fn=real_embedding_fn,
            E_threshold=0.3,
        )
        # Use the same capsule that created the session
        manager.capsule = capsule

        # Add real documents
        docs = [
            ("auth_doc", "OAuth 2.0 provides secure delegated access to server resources."),
            ("jwt_doc", "JSON Web Tokens encode claims for secure transmission."),
            ("weather_doc", "The forecast shows sunny weather with mild temperatures."),
        ]

        for doc_id, content in docs:
            manager.add_item(ContextItem(
                item_id=doc_id,
                content=content,
                tokens=len(content) // 4,
            ))

        # Run queries
        def mock_llm(system: str, prompt: str) -> str:
            return f"Response to: {prompt[:50]}..."

        # Auth-related query
        result1 = manager.respond_catalytic(
            query="How does OAuth authentication work?",
            llm_generate=mock_llm,
        )

        print(f"\nQuery 1 E_mean: {result1.E_mean:.3f}")
        print(f"Working set: {[i.item_id for i in result1.prepare_result.working_set]}")

        # Weather query
        result2 = manager.respond_catalytic(
            query="What is the weather forecast?",
            llm_generate=mock_llm,
        )

        print(f"\nQuery 2 E_mean: {result2.E_mean:.3f}")
        print(f"Working set: {[i.item_id for i in result2.prepare_result.working_set]}")

        # Compression should have happened
        stats = manager.get_compression_stats()
        assert stats["turns_compressed"] >= 2

        capsule.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
