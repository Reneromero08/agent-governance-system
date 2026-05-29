#!/usr/bin/env python3
"""
Verify Message-Level Catalytic Behavior
========================================

This test verifies that the fix for message-level catalytic storage is working:
1. Individual messages (user AND assistant) are stored with embeddings
2. E-scores are computed on individual messages, not just turn pointers
3. Relevant messages are retrieved based on semantic similarity

Run: python verify_message_level_catalytic.py
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Add parent to path for imports
CAT_CHAT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CAT_CHAT_ROOT))

from catalytic_chat.session_capsule import SessionCapsule
from catalytic_chat.auto_context_manager import AutoContextManager
from catalytic_chat.adaptive_budget import ModelBudgetDiscovery
from catalytic_chat.debug import CatChatDebugger


def mock_embedding(text: str) -> np.ndarray:
    """Simple mock embedding based on word presence for testing."""
    # Key concepts we want to test recall for
    concepts = {
        "catalytic": np.array([1.0, 0.0, 0.0, 0.0]),
        "weather": np.array([0.0, 1.0, 0.0, 0.0]),
        "pizza": np.array([0.0, 0.0, 1.0, 0.0]),
        "quantum": np.array([0.0, 0.0, 0.0, 1.0]),
    }

    text_lower = text.lower()
    embedding = np.zeros(4)

    for word, vec in concepts.items():
        if word in text_lower:
            embedding += vec

    # Add small random component if no match
    if np.linalg.norm(embedding) == 0:
        embedding = np.random.randn(4) * 0.1

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def mock_llm(system: str, prompt: str) -> str:
    """Mock LLM that echoes context keywords."""
    return f"I see you're asking about: {prompt[:100]}..."


def main():
    print("=" * 60)
    print("MESSAGE-LEVEL CATALYTIC VERIFICATION TEST")
    print("=" * 60)

    # Create temp database
    tmpdir = Path(tempfile.gettempdir()) / "catalytic_verify"
    tmpdir.mkdir(exist_ok=True)
    db_path = tmpdir / "verify_test.db"

    # Clean old test db
    if db_path.exists():
        db_path.unlink()

    # Initialize
    capsule = SessionCapsule(db_path=db_path)
    session_id = capsule.create_session()

    budget = ModelBudgetDiscovery.from_context_window(
        context_window=4096,
        system_prompt="Test system",
        response_reserve_pct=0.25,
        model_id="test"
    )

    manager = AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=mock_embedding,
        E_threshold=0.3,
    )
    manager.capsule = capsule

    # Turn 1: Talk about catalytic computing
    print("\n[Turn 1] User: What is catalytic computing?")
    result1 = manager.respond_catalytic(
        query="What is catalytic computing?",
        llm_generate=mock_llm,
        system_prompt="Test"
    )
    print(f"Response: {result1.response[:80]}...")

    # Turn 2: Talk about weather (unrelated)
    print("\n[Turn 2] User: What's the weather like?")
    result2 = manager.respond_catalytic(
        query="What's the weather like?",
        llm_generate=mock_llm,
        system_prompt="Test"
    )
    print(f"Response: {result2.response[:80]}...")

    # Turn 3: Talk about pizza (unrelated)
    print("\n[Turn 3] User: Tell me about pizza toppings.")
    result3 = manager.respond_catalytic(
        query="Tell me about pizza toppings.",
        llm_generate=mock_llm,
        system_prompt="Test"
    )
    print(f"Response: {result3.response[:80]}...")

    # Turn 4: Ask about catalytic again (should recall Turn 1 messages)
    print("\n[Turn 4] User: Explain catalytic more.")
    result4 = manager.respond_catalytic(
        query="Explain catalytic more.",
        llm_generate=mock_llm,
        system_prompt="Test"
    )
    print(f"Response: {result4.response[:80]}...")

    # VERIFICATION: Check if message-level items are in pointer set
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check pointer set contains individual messages
    pointer_set = manager._pointer_set
    user_messages = [p for p in pointer_set if p.item_type == "user_message"]
    asst_messages = [p for p in pointer_set if p.item_type == "assistant_message"]

    print(f"\nPointer set contents:")
    print(f"  User messages: {len(user_messages)}")
    print(f"  Assistant messages: {len(asst_messages)}")
    print(f"  Total items: {len(pointer_set)}")

    # Check each message has embedding
    messages_with_embeddings = sum(1 for p in pointer_set if p.embedding is not None)
    print(f"  Items with embeddings: {messages_with_embeddings}")

    # Verify E-score based recall worked for Turn 4
    # The prepare_context should have pulled catalytic-related messages into working set
    print(f"\nTurn 4 E_mean: {result4.E_mean:.4f}")
    print(f"Turn 4 working set size: {len(result4.prepare_result.working_set)} items")

    # Show what was in the working set
    working_set_items = result4.prepare_result.working_set
    print("\nWorking set (high-E items surfaced):")
    for item in working_set_items:
        print(f"  - [{item.item_type}] {item.content[:60]}...")

    # Check if catalytic message was surfaced
    catalytic_in_working = any(
        "catalytic" in item.content.lower()
        for item in working_set_items
    )

    print("\n" + "-" * 60)
    if catalytic_in_working:
        print("PASS: Catalytic-related message was surfaced in working set")
        print("      Message-level E-score retrieval is working!")
    else:
        print("FAIL: Catalytic message not found in working set")
        print("      Check pointer_set embeddings and E-score computation")
    print("-" * 60)

    # Also verify via debug tools
    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION (via debug tools)")
    print("=" * 60)

    debugger = CatChatDebugger(db_path)
    debugger.show_event_summary(session_id)
    debugger.show_messages(session_id, limit=10)

    capsule.close()

    print("\n" + "=" * 60)
    print("Test complete. Check output above for PASS/FAIL status.")
    print("=" * 60)


if __name__ == "__main__":
    main()
