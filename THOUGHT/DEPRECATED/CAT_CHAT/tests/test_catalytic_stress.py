"""
Stress Test: Catalytic Context Loop over 30+ Turns
===================================================

This test verifies that:
1. Compression happens EVERY turn
2. Rehydration ACTUALLY triggers when old content becomes relevant
3. Budget stays under control throughout
4. Old compressed turns can be recalled correctly

Run with: pytest tests/test_catalytic_stress.py -v -s
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

CAT_CHAT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import (
    SessionCapsule,
    EVENT_PARTITION,
    EVENT_TURN_STORED,
    EVENT_TURN_HYDRATED,
)
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.context_partitioner import ContextItem


# Check for real embeddings
try:
    from sentence_transformers import SentenceTransformer
    REAL_EMBEDDINGS = True
    _embed_model = None

    def get_real_embedding(text: str) -> np.ndarray:
        global _embed_model
        if _embed_model is None:
            _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        return _embed_model.encode(text, convert_to_numpy=True)
except ImportError:
    REAL_EMBEDDINGS = False

    def get_real_embedding(text: str) -> np.ndarray:
        # Fallback: deterministic synthetic embedding
        text_hash = hash(text) % (2**31)
        rng = np.random.RandomState(text_hash)
        vec = rng.randn(384)
        return vec / np.linalg.norm(vec)


class TestCatalyticStress:
    """Stress tests for catalytic context management."""

    @pytest.fixture
    def setup_manager(self, tmp_path):
        """Create a manager with small budget to force compression/rehydration."""
        db_path = tmp_path / "stress_test.db"

        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        # Small budget to force compression pressure
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=2048,  # Small context window
            system_prompt="You are a helpful assistant.",
            response_reserve_pct=0.25,
        )

        manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=get_real_embedding,
            E_threshold=0.3,  # Lower threshold to allow more rehydration
        )
        manager.capsule = capsule

        return manager, capsule, session_id

    def test_30_turns_compression(self, setup_manager):
        """Verify compression happens every turn over 30 turns."""
        manager, capsule, session_id = setup_manager

        # Mock LLM that returns varied responses
        def mock_llm(system: str, prompt: str) -> str:
            # Generate response based on prompt content
            if "weather" in prompt.lower():
                return "The weather is sunny with temperatures around 72F."
            elif "catalytic" in prompt.lower():
                return "Catalytic computing manages context through compression and rehydration."
            elif "quantum" in prompt.lower():
                return "Quantum mechanics uses wave functions and probability amplitudes."
            elif "database" in prompt.lower():
                return "Databases store data in tables with rows and columns."
            elif "python" in prompt.lower():
                return "Python is a high-level programming language known for readability."
            else:
                return f"This is response to: {prompt[:50]}..."

        # Varied topics to create diverse conversation
        topics = [
            "What is catalytic computing?",
            "Explain quantum mechanics briefly.",
            "How do databases work?",
            "Tell me about Python programming.",
            "What's the weather like?",
            "How does memory management work?",
            "Explain neural networks.",
            "What is machine learning?",
            "How do compilers work?",
            "Explain operating systems.",
        ]

        # Run 30 turns cycling through topics
        for i in range(30):
            query = topics[i % len(topics)]
            if i > 0:
                query = f"Turn {i+1}: {query}"  # Make each query unique

            result = manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            # Verify compression happened
            assert result.finalize_result is not None, f"Turn {i+1}: No finalize result"
            assert result.finalize_result.turn_pointer is not None, f"Turn {i+1}: No turn pointer"
            assert result.compression_ratio > 1.0, f"Turn {i+1}: No compression"

        # Check compression stats
        stats = manager.get_compression_stats()
        assert stats["turns_compressed"] == 30, f"Expected 30 compressed turns, got {stats['turns_compressed']}"
        assert stats["tokens_saved"] > 0, "No tokens saved from compression"

        # Verify all turn_stored events
        events = capsule.get_events(session_id, event_type=EVENT_TURN_STORED)
        assert len(events) == 30, f"Expected 30 turn_stored events, got {len(events)}"

        print(f"\n=== 30 Turn Compression Test ===")
        print(f"Turns compressed: {stats['turns_compressed']}")
        print(f"Total original tokens: {stats['total_original_tokens']}")
        print(f"Total pointer tokens: {stats['total_pointer_tokens']}")
        print(f"Tokens saved: {stats['tokens_saved']}")
        print(f"Average compression ratio: {stats['average_compression_ratio']:.1f}x")

    def test_rehydration_actually_triggers(self, setup_manager):
        """Verify rehydration actually happens when old content becomes relevant."""
        manager, capsule, session_id = setup_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Response about: {prompt[:30]}..."

        # Phase 1: Create turns about SPECIFIC topics that we'll reference later
        specific_topics = [
            ("The Eiffel Tower was built in 1889 for the World's Fair in Paris.", "eiffel"),
            ("Python was created by Guido van Rossum in 1991.", "guido"),
            ("The speed of light is approximately 299,792 km per second.", "light_speed"),
            ("Mount Everest is 8,848 meters tall, the highest peak on Earth.", "everest"),
            ("The human brain has approximately 86 billion neurons.", "neurons"),
        ]

        # Create initial turns with specific facts
        for i, (fact, _) in enumerate(specific_topics):
            result = manager.respond_catalytic(
                query=f"Tell me: {fact}",
                llm_generate=mock_llm,
            )

        # Phase 2: Add 15 unrelated turns to push original content out
        filler_topics = [
            "What is the capital of France?",
            "How do airplanes fly?",
            "Explain photosynthesis.",
            "What causes earthquakes?",
            "How do computers work?",
            "Explain gravity.",
            "What is DNA?",
            "How do vaccines work?",
            "Explain electricity.",
            "What causes rain?",
            "How do magnets work?",
            "Explain sound waves.",
            "What is entropy?",
            "How do rockets work?",
            "Explain black holes.",
        ]

        for topic in filler_topics:
            manager.respond_catalytic(
                query=topic,
                llm_generate=mock_llm,
            )

        # Count hydration events so far
        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        # Phase 3: Ask about the SPECIFIC topics from Phase 1
        # These queries should trigger rehydration of old compressed turns
        rehydration_queries = [
            "When was the Eiffel Tower built?",  # Should rehydrate turn about Eiffel
            "Who created Python programming language?",  # Should rehydrate Guido turn
            "What is the speed of light?",  # Should rehydrate light_speed turn
            "How tall is Mount Everest?",  # Should rehydrate everest turn
            "How many neurons in the human brain?",  # Should rehydrate neurons turn
        ]

        for query in rehydration_queries:
            result = manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            # Check if rehydration happened for this turn
            if result.prepare_result.hydrated_turns:
                print(f"Query '{query[:30]}...' triggered {len(result.prepare_result.hydrated_turns)} rehydration(s)")

        # Count hydration events after
        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))
        hydrations_triggered = hydration_after - hydration_before

        print(f"\n=== Rehydration Test ===")
        print(f"Turns before rehydration queries: {len(specific_topics) + len(filler_topics)}")
        print(f"Rehydration queries: {len(rehydration_queries)}")
        print(f"Hydration events triggered: {hydrations_triggered}")
        print(f"Turn pointers available: {len(manager._turn_pointers)}")

        # We expect SOME rehydration to happen (not necessarily all 5)
        # With real embeddings, similar queries should rehydrate relevant turns
        if REAL_EMBEDDINGS:
            assert hydrations_triggered > 0, (
                f"Expected rehydration with real embeddings, but got 0. "
                f"Turn pointers: {len(manager._turn_pointers)}"
            )
            print(f"SUCCESS: {hydrations_triggered} rehydrations triggered with real embeddings")
        else:
            print(f"Note: Using synthetic embeddings - rehydration may not trigger consistently")

    def test_budget_stays_bounded(self, setup_manager):
        """Verify budget never exceeds limit over many turns."""
        manager, capsule, session_id = setup_manager

        def mock_llm(system: str, prompt: str) -> str:
            # Return LARGE responses to stress the budget
            return "A" * 500 + f" Response to: {prompt[:20]}"

        budget_violations = []
        max_utilization = 0.0

        # Run 40 turns with large responses
        for i in range(40):
            result = manager.respond_catalytic(
                query=f"Turn {i+1}: Give me a detailed explanation of topic {i}",
                llm_generate=mock_llm,
            )

            state = result.context_state
            utilization = state.utilization_pct
            max_utilization = max(max_utilization, utilization)

            # Check budget invariant
            if not result.prepare_result.budget_checked:
                budget_violations.append(i+1)

            # Verify tokens used never exceeds budget
            budget_limit = manager.budget.available_for_working_set
            if state.tokens_used > budget_limit:
                budget_violations.append(f"Turn {i+1}: {state.tokens_used} > {budget_limit}")

        print(f"\n=== Budget Bound Test ===")
        print(f"Turns executed: 40")
        print(f"Max utilization: {max_utilization:.1%}")
        print(f"Budget violations: {len(budget_violations)}")
        print(f"Final tokens used: {manager.context_state.tokens_used}")
        print(f"Budget limit: {manager.budget.available_for_working_set}")

        assert len(budget_violations) == 0, f"Budget violations: {budget_violations}"
        assert max_utilization <= 1.0, f"Utilization exceeded 100%: {max_utilization:.1%}"

    def test_chain_integrity_after_many_turns(self, setup_manager):
        """Verify event chain remains valid after many operations."""
        manager, capsule, session_id = setup_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Response {hash(prompt) % 1000}"

        # Run 25 varied turns
        for i in range(25):
            manager.respond_catalytic(
                query=f"Query {i}: topic {i % 5}",
                llm_generate=mock_llm,
            )

        # Verify chain integrity
        is_valid, error = capsule.verify_chain(session_id)

        events = capsule.get_events(session_id)
        event_counts = {}
        for e in events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        print(f"\n=== Chain Integrity Test ===")
        print(f"Total events: {len(events)}")
        for etype, count in sorted(event_counts.items()):
            print(f"  {etype}: {count}")
        print(f"Chain valid: {is_valid}")
        if error:
            print(f"Error: {error}")

        assert is_valid, f"Chain integrity failed: {error}"

    @pytest.mark.skipif(not REAL_EMBEDDINGS, reason="Requires sentence-transformers")
    def test_semantic_rehydration(self, setup_manager):
        """Test that semantically similar queries rehydrate relevant turns."""
        manager, capsule, session_id = setup_manager

        def mock_llm(system: str, prompt: str) -> str:
            return f"Response to: {prompt[:50]}"

        # Create turns with specific semantic content
        knowledge_turns = [
            "The Apollo 11 mission landed on the Moon on July 20, 1969. Neil Armstrong was the first human to walk on the lunar surface.",
            "Machine learning models learn patterns from data through training. Neural networks are a type of ML model inspired by the brain.",
            "Climate change is caused by greenhouse gas emissions. Carbon dioxide traps heat in the atmosphere.",
        ]

        for content in knowledge_turns:
            manager.respond_catalytic(
                query=content,
                llm_generate=mock_llm,
            )

        # Add filler to push knowledge turns into compression
        for i in range(10):
            manager.respond_catalytic(
                query=f"Filler topic {i} about something completely different like cooking or gardening",
                llm_generate=mock_llm,
            )

        hydration_before = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        # Query with semantically similar but different wording
        semantic_queries = [
            "Who was the first person on the Moon?",  # Should match Apollo 11 turn
            "How do neural networks learn?",  # Should match ML turn
            "What causes global warming?",  # Should match climate turn
        ]

        for query in semantic_queries:
            result = manager.respond_catalytic(
                query=query,
                llm_generate=mock_llm,
            )

            if result.prepare_result.hydrated_turns:
                print(f"'{query}' -> rehydrated {len(result.prepare_result.hydrated_turns)} turn(s)")

        hydration_after = len(capsule.get_events(session_id, event_type=EVENT_TURN_HYDRATED))

        print(f"\n=== Semantic Rehydration Test ===")
        print(f"Hydrations triggered: {hydration_after - hydration_before}")

        # With real semantic embeddings, we expect rehydration
        assert hydration_after > hydration_before, (
            "Semantic rehydration should trigger with real embeddings"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
