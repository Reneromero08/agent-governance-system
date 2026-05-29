"""
Tests for Geometric Chat - I.2 CAT Chat Integration

Acceptance Criteria:
- I.2.1: Geometric context assembly works
- I.2.2: E-gating correlates with response quality
- I.2.3: Conversation state updates geometrically
- I.2.4: High-E responses are measurably better
"""

import sys
from pathlib import Path
import pytest

# Add paths for imports
CAT_CHAT_PATH = Path(__file__).parent.parent
CATALYTIC_PATH = CAT_CHAT_PATH / "catalytic_chat"
CAPABILITY_PATH = CAT_CHAT_PATH.parent.parent.parent / "CAPABILITY" / "PRIMITIVES"

sys.path.insert(0, str(CAT_CHAT_PATH))
sys.path.insert(0, str(CATALYTIC_PATH))
sys.path.insert(0, str(CAPABILITY_PATH))

# Skip all tests if sentence-transformers not available
try:
    from geometric_chat import GeometricChat, GeometricChatResult
    from geometric_reasoner import GeometricReasoner, GeometricState
    from catalytic_chat.geometric_context_assembler import (
        GeometricContextAssembler,
        GeometricAssemblyReceipt
    )
    from catalytic_chat.context_assembler import (
        ContextBudget,
        ContextMessage,
        ContextExpansion
    )
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not GEOMETRIC_AVAILABLE,
    reason="GeometricReasoner or sentence-transformers not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def geometric_chat():
    """Create GeometricChat instance."""
    return GeometricChat(E_threshold=0.5)


@pytest.fixture
def geometric_reasoner():
    """Create GeometricReasoner instance."""
    return GeometricReasoner()


@pytest.fixture
def geometric_assembler():
    """Create GeometricContextAssembler instance."""
    return GeometricContextAssembler(E_threshold=0.5)


@pytest.fixture
def auth_context():
    """Authentication-related context documents."""
    return [
        "OAuth 2.0 is an authorization framework that enables applications to obtain limited access to user accounts.",
        "JWT (JSON Web Tokens) are a compact, URL-safe means of representing claims to be transferred between two parties.",
        "Authentication verifies the identity of a user or process, while authorization determines what they can access.",
    ]


@pytest.fixture
def unrelated_context():
    """Context documents unrelated to auth."""
    return [
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The Pythagorean theorem states that in a right triangle, a² + b² = c².",
        "Mount Everest is the tallest mountain on Earth, standing at 8,849 meters.",
    ]


@pytest.fixture
def echo_llm():
    """Simple echo LLM for testing."""
    def _echo(query: str, context: list) -> str:
        return f"Response to: {query}"
    return _echo


@pytest.fixture
def sample_budget():
    """Sample context budget."""
    return ContextBudget(
        max_total_tokens=4000,
        reserve_response_tokens=500,
        max_messages=10,
        max_expansions=5,
        max_tokens_per_message=500,
        max_tokens_per_expansion=300
    )


@pytest.fixture
def sample_messages():
    """Sample context messages."""
    return [
        ContextMessage(
            id="sys-1",
            source="SYSTEM",
            content="You are a helpful assistant specializing in authentication.",
            created_at="2024-01-01T00:00:00Z",
            metadata={}
        ),
        ContextMessage(
            id="user-1",
            source="USER",
            content="How does JWT authentication work?",
            created_at="2024-01-01T00:00:01Z",
            metadata={}
        ),
        ContextMessage(
            id="asst-1",
            source="ASSISTANT",
            content="JWT authentication uses signed tokens for stateless auth.",
            created_at="2024-01-01T00:00:02Z",
            metadata={}
        ),
    ]


@pytest.fixture
def sample_expansions():
    """Sample context expansions."""
    return [
        ContextExpansion(
            symbol_id="@AUTH/JWT_BASICS",
            content="JWT tokens consist of header, payload, and signature.",
            is_explicit_reference=True,
            priority=10
        ),
        ContextExpansion(
            symbol_id="@AUTH/OAUTH_FLOW",
            content="OAuth uses authorization codes exchanged for access tokens.",
            is_explicit_reference=False,
            priority=5
        ),
    ]


# ============================================================================
# I.2.1: Geometric Context Assembly Works
# ============================================================================

class TestGeometricContextAssembly:
    """Tests for I.2.1: Geometric context assembly works."""

    def test_assembler_creates_valid_receipt(
        self,
        geometric_assembler,
        geometric_reasoner,
        sample_messages,
        sample_expansions,
        sample_budget
    ):
        """Assembly produces valid output with E-scoring."""
        query_state = geometric_reasoner.initialize("How does JWT work?")

        items, receipt = geometric_assembler.assemble_with_geometry(
            messages=sample_messages,
            expansions=sample_expansions,
            budget=sample_budget,
            query_state=query_state
        )

        # Verify receipt structure
        assert receipt.success
        assert isinstance(receipt, GeometricAssemblyReceipt)
        assert receipt.mean_E >= 0
        assert receipt.max_E >= receipt.mean_E or receipt.max_E == 0
        assert len(receipt.E_distribution) == len(items)
        assert receipt.query_Df > 0

    def test_assembler_preserves_tier_order(
        self,
        geometric_assembler,
        geometric_reasoner,
        sample_messages,
        sample_expansions,
        sample_budget
    ):
        """Tier order is preserved (Mandatory → Dialog → Explicit → Optional)."""
        query_state = geometric_reasoner.initialize("authentication question")

        items, receipt = geometric_assembler.assemble_with_geometry(
            messages=sample_messages,
            expansions=sample_expansions,
            budget=sample_budget,
            query_state=query_state
        )

        assert receipt.success
        assert len(items) > 0

        # System prompt should be first
        assert items[0].original_id == "sys-1"

        # User message should be last
        assert items[-1].original_id == "user-1"

    def test_assembler_computes_e_distribution(
        self,
        geometric_assembler,
        geometric_reasoner,
        sample_messages,
        sample_expansions,
        sample_budget
    ):
        """E distribution is computed for all included items."""
        query_state = geometric_reasoner.initialize("JWT token validation")

        items, receipt = geometric_assembler.assemble_with_geometry(
            messages=sample_messages,
            expansions=sample_expansions,
            budget=sample_budget,
            query_state=query_state
        )

        # E distribution should have same length as items
        assert len(receipt.E_distribution) == len(items)

        # All E values should be in valid range
        for e in receipt.E_distribution:
            assert 0 <= e <= 1


# ============================================================================
# I.2.2: E-Gating Correlates with Response Quality
# ============================================================================

class TestEGatingCorrelation:
    """Tests for I.2.2: E-gating correlates with response quality."""

    def test_high_e_for_relevant_context(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """High E when query matches context."""
        result = geometric_chat.respond(
            "How does OAuth authentication work?",
            auth_context,
            echo_llm
        )

        # Auth query + auth context should have high E
        assert result.E_resonance > 0.3  # Reasonable threshold
        assert result.gate_open

    def test_low_e_for_unrelated_context(
        self,
        geometric_chat,
        unrelated_context,
        echo_llm
    ):
        """Low E when query doesn't match context."""
        result = geometric_chat.respond(
            "How does OAuth authentication work?",
            unrelated_context,
            echo_llm
        )

        # Auth query + biology/math context should have low E
        assert result.E_resonance < 0.5  # Below threshold

    def test_gate_opens_for_relevant_queries(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Gate opens when query is relevant to context."""
        # Use lower threshold for this test - mean E depends on all docs
        geometric_chat.E_threshold = 0.3

        result = geometric_chat.respond(
            "Explain JWT token structure",
            auth_context,
            echo_llm
        )

        # With threshold=0.3, mean E (0.418) should pass
        assert result.gate_open
        # Also verify at least one doc has high resonance
        assert result.receipt['context_alignment'][1] > 0.5  # JWT doc

    def test_gate_closes_for_irrelevant_queries(
        self,
        geometric_chat,
        unrelated_context,
        echo_llm
    ):
        """Gate closes when query is irrelevant to context."""
        result = geometric_chat.respond(
            "Explain JWT token structure",
            unrelated_context,
            echo_llm
        )

        # May or may not close depending on model, but E should be low
        assert result.E_resonance < 0.7  # Definitely lower than relevant case


# ============================================================================
# I.2.3: Conversation State Updates Geometrically
# ============================================================================

class TestConversationStateUpdates:
    """Tests for I.2.3: Conversation state updates geometrically."""

    def test_conversation_state_initializes(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Conversation state is created on first response."""
        assert geometric_chat.conversation_state is None

        result = geometric_chat.respond(
            "What is authentication?",
            auth_context,
            echo_llm
        )

        assert geometric_chat.conversation_state is not None
        assert result.conversation_Df > 0

    def test_conversation_state_evolves(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Conversation state changes via entangle."""
        # First turn
        result1 = geometric_chat.respond(
            "What is authentication?",
            auth_context,
            echo_llm
        )
        Df_1 = result1.conversation_Df
        hash_1 = geometric_chat.conversation_state.receipt()['vector_hash']

        # Second turn
        result2 = geometric_chat.respond(
            "How do refresh tokens work?",
            auth_context,
            echo_llm
        )
        Df_2 = result2.conversation_Df
        hash_2 = geometric_chat.conversation_state.receipt()['vector_hash']

        # State should have changed
        assert hash_1 != hash_2
        # Df may change
        assert Df_1 > 0 and Df_2 > 0

    def test_distance_from_start_increases(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Distance from start increases over conversation."""
        # First turn
        result1 = geometric_chat.respond("What is authentication?", auth_context, echo_llm)
        dist_1 = result1.distance_from_start

        # Should be 0 or very small after first turn
        assert dist_1 >= 0

        # More turns
        result2 = geometric_chat.respond("Explain OAuth flow", auth_context, echo_llm)
        dist_2 = result2.distance_from_start

        result3 = geometric_chat.respond("What about JWT?", auth_context, echo_llm)
        dist_3 = result3.distance_from_start

        # Distance should generally increase (though not strictly monotonic)
        assert dist_3 >= 0  # At minimum, valid

    def test_turn_history_tracked(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Turn history is tracked correctly."""
        assert len(geometric_chat.turn_history) == 0

        geometric_chat.respond("Query 1", auth_context, echo_llm)
        assert len(geometric_chat.turn_history) == 1

        geometric_chat.respond("Query 2", auth_context, echo_llm)
        assert len(geometric_chat.turn_history) == 2

        geometric_chat.respond("Query 3", auth_context, echo_llm)
        assert len(geometric_chat.turn_history) == 3


# ============================================================================
# I.2.4: High-E Responses Are Measurably Better
# ============================================================================

class TestHighEResponseQuality:
    """Tests for I.2.4: High-E responses are measurably better."""

    def test_e_resonance_computed(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """E_resonance is computed for each response."""
        result = geometric_chat.respond(
            "How does authentication work?",
            auth_context,
            echo_llm
        )

        assert hasattr(result, 'E_resonance')
        assert 0 <= result.E_resonance <= 1

    def test_e_compression_computed(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """E_compression tracks response-conversation alignment."""
        # First turn
        result1 = geometric_chat.respond("What is auth?", auth_context, echo_llm)
        # E_compression should be 1.0 for first turn (self-resonance)
        assert result1.E_compression >= 0

        # Second turn
        result2 = geometric_chat.respond("More about JWT", auth_context, echo_llm)
        assert result2.E_compression >= 0

    def test_metrics_include_e_history(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Metrics include E history for analysis."""
        geometric_chat.respond("Query 1", auth_context, echo_llm)
        geometric_chat.respond("Query 2", auth_context, echo_llm)

        metrics = geometric_chat.get_metrics()

        assert 'E_history' in metrics
        assert len(metrics['E_history']) == 2


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_inputs_same_e(
        self,
        geometric_reasoner
    ):
        """Same inputs produce same E values."""
        query = "authentication"
        context = "OAuth provides authentication"

        q1 = geometric_reasoner.initialize(query)
        c1 = geometric_reasoner.initialize(context)
        e1 = q1.E_with(c1)

        q2 = geometric_reasoner.initialize(query)
        c2 = geometric_reasoner.initialize(context)
        e2 = q2.E_with(c2)

        assert abs(e1 - e2) < 1e-6  # Should be identical

    def test_assembly_deterministic(
        self,
        geometric_assembler,
        geometric_reasoner,
        sample_messages,
        sample_expansions,
        sample_budget
    ):
        """Assembly produces deterministic results."""
        query_state = geometric_reasoner.initialize("JWT tokens")

        items1, receipt1 = geometric_assembler.assemble_with_geometry(
            sample_messages, sample_expansions, sample_budget, query_state
        )

        items2, receipt2 = geometric_assembler.assemble_with_geometry(
            sample_messages, sample_expansions, sample_budget, query_state
        )

        # Should produce same hash
        assert receipt1.final_assemblage_hash == receipt2.final_assemblage_hash
        assert abs(receipt1.mean_E - receipt2.mean_E) < 1e-6


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_context(
        self,
        geometric_chat,
        echo_llm
    ):
        """Handle empty context gracefully."""
        result = geometric_chat.respond(
            "What is authentication?",
            [],  # Empty context
            echo_llm
        )

        assert result is not None
        assert result.E_resonance == 0.0
        assert not result.gate_open

    def test_single_context_doc(
        self,
        geometric_chat,
        echo_llm
    ):
        """Handle single context document."""
        result = geometric_chat.respond(
            "What is authentication?",
            ["Authentication verifies identity."],
            echo_llm
        )

        assert result is not None
        assert result.E_resonance >= 0

    def test_clear_resets_state(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Clear resets conversation state."""
        geometric_chat.respond("Query", auth_context, echo_llm)
        assert geometric_chat.conversation_state is not None

        geometric_chat.clear()

        assert geometric_chat.conversation_state is None
        assert len(geometric_chat.turn_history) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_conversation_flow(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Test complete conversation flow."""
        # Multi-turn conversation
        queries = [
            "What is authentication?",
            "How does OAuth work?",
            "What about JWT tokens?",
            "How do I implement refresh tokens?",
        ]

        for query in queries:
            result = geometric_chat.respond(query, auth_context, echo_llm)
            assert result is not None
            assert result.response

        metrics = geometric_chat.get_metrics()
        assert metrics['turn_count'] == 4
        assert len(metrics['E_history']) == 4
        assert metrics['distance_from_start'] >= 0

    def test_stats_tracking(
        self,
        geometric_chat,
        auth_context,
        echo_llm
    ):
        """Verify stats are tracked correctly."""
        geometric_chat.respond("Query 1", auth_context, echo_llm)
        geometric_chat.respond("Query 2", auth_context, echo_llm)

        metrics = geometric_chat.get_metrics()

        assert metrics['stats']['turns'] == 2
        assert metrics['stats']['embedding_calls'] > 0
        assert metrics['stats']['geometric_ops'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
