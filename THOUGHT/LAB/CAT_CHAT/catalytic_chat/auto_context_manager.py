"""
Auto Context Manager
====================

Orchestrates the full auto-controlled context loop for catalytic chat.

This is the main integration point that ties together:
- Adaptive budget discovery
- E-score based context partitioning
- Continuous turn compression
- Session event logging

Key Design Principle: Context is always catalytic. Every turn:
1. Re-score ALL items against current query
2. Partition into working_set (high-E, fits budget) and pointer_set
3. After response, compress the turn immediately
4. Log all decisions for deterministic replay

Phase C.5 of Auto-Controlled Context Loop implementation.
"""

import hashlib
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

from .adaptive_budget import (
    AdaptiveBudget,
    ModelBudgetDiscovery,
    BudgetExceededError,
    BudgetSnapshot,
)
from .context_partitioner import (
    ContextPartitioner,
    ContextItem,
    PartitionResult,
    ScoredItem,
)
from .turn_compressor import (
    TurnCompressor,
    TurnContent,
    TurnPointer,
    CompressionResult,
    HydrationResult,
    create_turn_from_messages,
)
from .session_capsule import (
    SessionCapsule,
    SessionEvent,
    EVENT_PARTITION,
    EVENT_TURN_STORED,
    EVENT_TURN_HYDRATED,
    EVENT_BUDGET_CHECK,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ContextState:
    """
    Current state of auto-managed context.

    Tracks what's in working_set vs pointer_set and budget usage.
    """
    working_set: List[ContextItem]
    pointer_set: List[ContextItem]
    turn_pointers: List[TurnPointer]  # Compressed turns available for hydration
    budget: AdaptiveBudget
    tokens_used: int
    turn_index: int

    @property
    def tokens_remaining(self) -> int:
        """Tokens still available in working set."""
        return max(0, self.budget.available_for_working_set - self.tokens_used)

    @property
    def utilization_pct(self) -> float:
        """Budget utilization as percentage."""
        available = self.budget.available_for_working_set
        if available == 0:
            return 1.0 if self.tokens_used > 0 else 0.0
        return self.tokens_used / available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "working_set_count": len(self.working_set),
            "pointer_set_count": len(self.pointer_set),
            "turn_pointers_count": len(self.turn_pointers),
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "utilization_pct": self.utilization_pct,
            "turn_index": self.turn_index,
            "budget": self.budget.to_dict(),
        }


@dataclass
class PrepareContextResult:
    """Result of preparing context for a query."""
    working_set: List[ContextItem]
    partition_result: PartitionResult
    hydrated_turns: List[HydrationResult]
    tokens_used: int
    budget_checked: bool

    def get_context_text(self, separator: str = "\n\n") -> str:
        """Get concatenated context text for LLM."""
        return separator.join(item.content for item in self.working_set)


@dataclass
class FinalizeResult:
    """Result of finalizing a turn."""
    compression_result: CompressionResult
    turn_pointer: TurnPointer
    events_logged: List[str]  # Event IDs


@dataclass
class CatalyticChatResult:
    """Result of a full catalytic chat turn."""
    response: str
    prepare_result: PrepareContextResult
    finalize_result: Optional[FinalizeResult]
    turn_index: int
    context_state: ContextState

    # Metrics
    E_mean: float
    tokens_in_context: int
    tokens_compressed: int
    compression_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "turn_index": self.turn_index,
            "E_mean": self.E_mean,
            "tokens_in_context": self.tokens_in_context,
            "tokens_compressed": self.tokens_compressed,
            "compression_ratio": self.compression_ratio,
            "working_set_count": len(self.prepare_result.working_set),
            "pointer_set_count": len(self.prepare_result.partition_result.pointer_set),
            "hydrated_turns": len(self.prepare_result.hydrated_turns),
            "context_state": self.context_state.to_dict(),
        }


# =============================================================================
# Auto Context Manager
# =============================================================================

class AutoContextManager:
    """
    Orchestrates the full auto-controlled context loop.

    Usage:
        manager = AutoContextManager(
            db_path=Path("_generated/cat_chat.db"),
            session_id="session_123",
            budget=budget,
            embed_fn=my_embedding_function
        )

        # Prepare context for a query
        result = manager.prepare_context(query, query_embedding)

        # Get context for LLM
        context_text = result.get_context_text()

        # After LLM response, finalize the turn
        manager.finalize_turn(query, response)
    """

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        budget: AdaptiveBudget,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        summarize_fn: Optional[Callable[[str], str]] = None,
        token_estimator: Optional[Callable[[str], int]] = None,
        E_threshold: float = 0.5
    ):
        """
        Initialize auto context manager.

        Args:
            db_path: Path to SQLite database
            session_id: Current session ID
            budget: Adaptive budget configuration
            embed_fn: Function to compute embeddings (required for E-scoring)
            summarize_fn: Function to summarize turns (optional)
            token_estimator: Function to estimate tokens (default: len//4)
            E_threshold: E-score threshold for partitioning (default: 0.5)
        """
        self.db_path = Path(db_path)
        self.session_id = session_id
        self.budget = budget
        self.embed_fn = embed_fn
        self.token_estimator = token_estimator or (lambda s: len(s) // 4)
        self.E_threshold = E_threshold

        # Initialize components
        self.partitioner = ContextPartitioner(
            threshold=E_threshold,
            embed_fn=embed_fn,
            token_estimator=self.token_estimator,
        )

        self.compressor = TurnCompressor(
            db_path=db_path,
            session_id=session_id,
            summarize_fn=summarize_fn,
            token_estimator=self.token_estimator,
        )

        self.capsule = SessionCapsule(db_path=db_path)

        # State tracking
        self._working_set: List[ContextItem] = []
        self._pointer_set: List[ContextItem] = []
        self._turn_pointers: List[TurnPointer] = []
        self._turn_index: int = 0

    @property
    def context_state(self) -> ContextState:
        """Get current context state."""
        tokens_used = sum(item.tokens for item in self._working_set)
        return ContextState(
            working_set=list(self._working_set),
            pointer_set=list(self._pointer_set),
            turn_pointers=list(self._turn_pointers),
            budget=self.budget,
            tokens_used=tokens_used,
            turn_index=self._turn_index,
        )

    def add_item(self, item: ContextItem) -> None:
        """
        Add an item to the context pool (starts in pointer_set).

        Items are partitioned into working_set on the next query.
        """
        self._pointer_set.append(item)

    def add_items(self, items: List[ContextItem]) -> None:
        """Add multiple items to the context pool."""
        self._pointer_set.extend(items)

    def prepare_context(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> PrepareContextResult:
        """
        Prepare context for a query.

        1. Compute query embedding if not provided
        2. Score ALL items (working + pointer + turn pointers)
        3. Partition based on E-scores and budget
        4. Hydrate high-E compressed turns
        5. Return assembled working set

        Args:
            query: User query text
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            PrepareContextResult with assembled context
        """
        # Compute embedding if needed
        if query_embedding is None and self.embed_fn:
            query_embedding = self.embed_fn(query)
        elif query_embedding is None:
            # No embedding function - use random for testing
            query_embedding = np.random.randn(384)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Collect all items for partitioning
        all_items = list(self._working_set) + list(self._pointer_set)

        # Track which turns are already hydrated in working set
        hydrated_turn_ids = {
            item.metadata.get("turn_id")
            for item in self._working_set
            if item.item_type == "hydrated_turn"
        }

        # Add turn pointers as potential items (skip already-hydrated turns)
        for tp in self._turn_pointers:
            # Skip if this turn is already hydrated in working set
            if tp.turn_id in hydrated_turn_ids:
                continue

            pointer_content = tp.to_pointer_content()
            embedding = self.embed_fn(pointer_content) if self.embed_fn else None

            # Use ORIGINAL tokens for budget accounting, not pointer tokens
            # This ensures hydration doesn't exceed budget
            all_items.append(ContextItem(
                item_id=f"turn_ptr_{tp.turn_id}",
                content=pointer_content,
                tokens=tp.original_tokens,  # Full size for budget accounting
                embedding=embedding,
                item_type="turn_pointer",
                metadata={
                    "content_hash": tp.content_hash,
                    "original_tokens": tp.original_tokens,
                    "pointer_tokens": tp.pointer_tokens,
                },
            ))

        # Partition all items
        budget_available = self.budget.available_for_working_set
        partition_result = self.partitioner.partition(
            query_embedding=query_embedding,
            all_items=all_items,
            budget_tokens=budget_available,
            query_text=query,
        )

        # Extract working and pointer sets
        new_working_set = [s.item for s in partition_result.working_set]
        new_pointer_set = [s.item for s in partition_result.pointer_set]

        # Check for turn pointers that scored high enough to hydrate
        hydrated_turns: List[HydrationResult] = []
        final_working_set: List[ContextItem] = []

        for item in new_working_set:
            if item.item_type == "turn_pointer":
                # This is a compressed turn that needs hydration
                content_hash = item.metadata.get("content_hash")
                if content_hash:
                    # Find the E-score for this item
                    E_score = 0.0
                    for s in partition_result.working_set:
                        if s.item.item_id == item.item_id:
                            E_score = s.E_score
                            break

                    # Hydrate the turn
                    hydration = self.compressor.decompress_turn(content_hash, E_score)
                    hydrated_turns.append(hydration)

                    if hydration.success and hydration.content:
                        # Replace pointer with full content
                        final_working_set.append(ContextItem(
                            item_id=f"turn_{hydration.turn_id}",
                            content=hydration.content.full_content,
                            tokens=hydration.tokens_added,
                            embedding=self.embed_fn(hydration.content.full_content) if self.embed_fn else None,
                            item_type="hydrated_turn",
                            metadata={
                                "turn_id": hydration.turn_id,
                                "E_score": E_score,
                            },
                        ))

                        # Log hydration event
                        self.capsule.log_turn_hydrated(
                            session_id=self.session_id,
                            turn_id=hydration.turn_id,
                            content_hash=content_hash,
                            E_score=E_score,
                            tokens_added=hydration.tokens_added,
                        )
                    else:
                        # Hydration failed - keep pointer
                        final_working_set.append(item)
            else:
                final_working_set.append(item)

        # Update internal state
        self._working_set = final_working_set
        self._pointer_set = [
            item for item in new_pointer_set
            if item.item_type != "turn_pointer"
        ]

        # Log partition event
        tokens_used = sum(item.tokens for item in final_working_set)
        self.capsule.log_partition(
            session_id=self.session_id,
            query_hash=partition_result.query_hash,
            working_set_ids=[item.item_id for item in final_working_set],
            pointer_set_ids=[item.item_id for item in self._pointer_set],
            budget_total=budget_available,
            budget_used=tokens_used,
            threshold=self.E_threshold,
            E_mean=partition_result.E_mean,
            E_min=partition_result.E_min,
            E_max=partition_result.E_max,
            items_below_threshold=partition_result.items_below_threshold,
            items_over_budget=partition_result.items_over_budget,
        )

        # Check budget invariant
        try:
            self.budget.check_invariant(tokens_used, len(final_working_set))
            budget_passed = True
        except BudgetExceededError:
            budget_passed = False

        self.capsule.log_budget_check(
            session_id=self.session_id,
            budget_available=budget_available,
            budget_used=tokens_used,
            item_count=len(final_working_set),
            passed=budget_passed,
            context_window=self.budget.context_window,
            model_id=self.budget.model_id,
        )

        return PrepareContextResult(
            working_set=final_working_set,
            partition_result=partition_result,
            hydrated_turns=hydrated_turns,
            tokens_used=tokens_used,
            budget_checked=budget_passed,
        )

    def finalize_turn(
        self,
        user_query: str,
        assistant_response: str,
        turn_id: Optional[str] = None
    ) -> FinalizeResult:
        """
        Finalize a turn by storing messages and compressing to catalytic space.

        Called after LLM response. Stores EACH message individually with
        embedding for E-score based recall, PLUS compresses the turn.

        Args:
            user_query: The user's query
            assistant_response: The assistant's response
            turn_id: Optional turn ID (auto-generated if not provided)

        Returns:
            FinalizeResult with compression details
        """
        self._turn_index += 1

        if turn_id is None:
            turn_id = f"turn_{self._turn_index:04d}"

        # CATALYTIC: Store individual messages with embeddings
        user_msg_id = f"msg_user_{turn_id}"
        asst_msg_id = f"msg_asst_{turn_id}"

        # Compute embeddings
        user_embedding = self.embed_fn(user_query) if self.embed_fn else None
        asst_embedding = self.embed_fn(assistant_response) if self.embed_fn else None

        # Add messages to pointer set for E-score based retrieval
        # This is the KEY catalytic behavior - every message becomes
        # a candidate for future E-score based recall
        self._pointer_set.append(ContextItem(
            item_id=user_msg_id,
            content=f"[User] {user_query}",
            tokens=self.token_estimator(user_query),
            embedding=user_embedding,
            item_type="user_message",
            metadata={"turn_id": turn_id, "role": "user"},
        ))

        self._pointer_set.append(ContextItem(
            item_id=asst_msg_id,
            content=f"[Assistant] {assistant_response}",
            tokens=self.token_estimator(assistant_response),
            embedding=asst_embedding,
            item_type="assistant_message",
            metadata={"turn_id": turn_id, "role": "assistant"},
        ))

        # Create turn content for compression
        turn = create_turn_from_messages(
            turn_id=turn_id,
            user_message=user_query,
            assistant_message=assistant_response,
        )

        # Compress turn and store full content for hydration
        # Use skip_storage=True to avoid sequence number race with capsule
        compression_result = self.compressor.compress_turn(turn, skip_storage=True)
        pointer = compression_result.pointer

        # Add pointer to pool for future hydration
        self._turn_pointers.append(pointer)

        # Cache content in compressor for hydration (no DB write)
        self.compressor._content_cache[pointer.content_hash] = turn

        # Log all events through capsule (single source of truth)
        # This ensures correct sequence numbering
        self.capsule.log_user_message(self.session_id, user_query)
        self.capsule.log_assistant_response(self.session_id, assistant_response)

        # Log turn stored with full content for hydration
        event = self.capsule.append_event(self.session_id, EVENT_TURN_STORED, {
            "turn_id": turn_id,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "timestamp": turn.timestamp,
            "content_hash": pointer.content_hash,
            "summary": pointer.summary,
            "original_tokens": pointer.original_tokens,
            "pointer_tokens": pointer.pointer_tokens,
            "compression_ratio": pointer.compression_ratio,
        })

        return FinalizeResult(
            compression_result=compression_result,
            turn_pointer=pointer,
            events_logged=[event.event_id],
        )

    def respond_catalytic(
        self,
        query: str,
        llm_generate: Callable[[str, str], str],
        query_embedding: Optional[np.ndarray] = None,
        system_prompt: str = ""
    ) -> CatalyticChatResult:
        """
        Full catalytic chat turn: prepare -> generate -> finalize.

        This is the main entry point for auto-managed context chat.

        TRULY CATALYTIC: Every message (user AND assistant) is stored
        individually with its embedding for future E-score based recall.

        Args:
            query: User query
            llm_generate: Function(system_prompt, user_context) -> response
            query_embedding: Pre-computed query embedding (optional)
            system_prompt: System prompt for LLM

        Returns:
            CatalyticChatResult with full turn details
        """
        # Prepare context (uses E-scores against all stored messages)
        prepare_result = self.prepare_context(query, query_embedding)

        # Assemble context for LLM
        context_text = prepare_result.get_context_text()

        # Generate response
        full_prompt = f"{context_text}\n\nUser: {query}" if context_text else query
        response = llm_generate(system_prompt, full_prompt)

        # CATALYTIC: finalize_turn stores BOTH messages with embeddings
        finalize_result = self.finalize_turn(query, response)

        # Compute metrics
        E_mean = prepare_result.partition_result.E_mean
        tokens_in_context = prepare_result.tokens_used
        tokens_compressed = finalize_result.turn_pointer.original_tokens
        pointer_tokens = finalize_result.turn_pointer.pointer_tokens
        compression_ratio = tokens_compressed / max(pointer_tokens, 1)

        return CatalyticChatResult(
            response=response,
            prepare_result=prepare_result,
            finalize_result=finalize_result,
            turn_index=self._turn_index,
            context_state=self.context_state,
            E_mean=E_mean,
            tokens_in_context=tokens_in_context,
            tokens_compressed=tokens_compressed,
            compression_ratio=compression_ratio,
        )

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics for all turns."""
        return self.compressor.get_compression_stats()

    def adjust_threshold(self, new_threshold: float) -> None:
        """Adjust E-score threshold."""
        self.E_threshold = new_threshold
        self.partitioner.adjust_threshold(new_threshold)

    # =========================================================================
    # Debug Methods - Easy inspection during development
    # =========================================================================

    def debug_show_state(self) -> None:
        """Print current context state for debugging."""
        state = self.context_state
        print(f"\n--- Context State (Turn {state.turn_index}) ---")
        print(f"Working set: {len(state.working_set)} items, {state.tokens_used} tokens")
        print(f"Pointer set: {len(state.pointer_set)} items")
        print(f"Turn pointers: {len(state.turn_pointers)} compressed turns")
        print(f"Budget: {state.utilization_pct:.1%} used ({state.tokens_used}/{state.budget.available_for_working_set})")

    def debug_show_turns(self) -> None:
        """Print all compressed turn pointers."""
        print(f"\n--- Compressed Turns ({len(self._turn_pointers)}) ---")
        for tp in self._turn_pointers:
            print(f"  {tp.turn_id}: {tp.summary[:50]}... ({tp.original_tokens} tok, {tp.compression_ratio:.1f}x)")

    def debug_get_all_messages(self) -> List[Dict[str, Any]]:
        """
        Get all stored messages in simple format for inspection.

        Returns list of {role: 'user'/'assistant', content: str, turn_id: str}
        """
        from .debug import CatChatDebugger
        debugger = CatChatDebugger(self.db_path)
        messages = debugger.get_all_messages(self.session_id)
        return [
            {"role": m.role, "content": m.content, "turn_id": m.turn_id}
            for m in messages
        ]

    def debug_report(self) -> None:
        """Print a full diagnostic report."""
        from .debug import CatChatDebugger
        debugger = CatChatDebugger(self.db_path)
        debugger.report(self.session_id)


# =============================================================================
# Factory Functions
# =============================================================================

def create_auto_context_manager(
    db_path: Path,
    session_id: str,
    model_context_window: int,
    system_prompt: str = "",
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    E_threshold: float = 0.5
) -> AutoContextManager:
    """
    Factory function to create an AutoContextManager.

    Args:
        db_path: Path to SQLite database
        session_id: Session ID
        model_context_window: Model's context window size
        system_prompt: System prompt text
        embed_fn: Embedding function (optional)
        E_threshold: E-score threshold

    Returns:
        Configured AutoContextManager
    """
    budget = ModelBudgetDiscovery.from_context_window(
        context_window=model_context_window,
        system_prompt=system_prompt,
    )

    return AutoContextManager(
        db_path=db_path,
        session_id=session_id,
        budget=budget,
        embed_fn=embed_fn,
        E_threshold=E_threshold,
    )


if __name__ == "__main__":
    # Quick sanity test
    print("Auto Context Manager - Sanity Test")
    print("=" * 50)

    import tempfile

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Create session
        capsule = SessionCapsule(db_path=db_path)
        session_id = capsule.create_session()

        # Create manager
        budget = ModelBudgetDiscovery.from_context_window(
            context_window=4096,
            system_prompt="You are a helpful assistant.",
        )

        def mock_embed(text: str) -> np.ndarray:
            """Mock embedding function."""
            text_hash = hash(text) % (2**31)
            rng = np.random.RandomState(text_hash)
            vec = rng.randn(384)
            return vec / np.linalg.norm(vec)

        def mock_llm(system: str, prompt: str) -> str:
            """Mock LLM function."""
            return f"Mock response to: {prompt[:50]}..."

        manager = AutoContextManager(
            db_path=db_path,
            session_id=session_id,
            budget=budget,
            embed_fn=mock_embed,
            E_threshold=0.5,
        )

        # Add some context items
        manager.add_items([
            ContextItem(
                item_id="doc1",
                content="Catalytic computing is about clean and catalytic space.",
                tokens=20,
                item_type="document",
            ),
            ContextItem(
                item_id="doc2",
                content="The weather is sunny today.",
                tokens=10,
                item_type="document",
            ),
        ])

        print(f"\nInitial state: {manager.context_state.to_dict()}")

        # Run a catalytic chat turn
        result = manager.respond_catalytic(
            query="What is catalytic computing?",
            llm_generate=mock_llm,
            system_prompt="You are a helpful assistant.",
        )

        print(f"\nTurn 1 result:")
        print(f"  Response: {result.response}")
        print(f"  E_mean: {result.E_mean:.4f}")
        print(f"  Tokens in context: {result.tokens_in_context}")
        print(f"  Compression ratio: {result.compression_ratio:.1f}x")
        print(f"  Working set: {len(result.prepare_result.working_set)} items")

        # Run another turn
        result2 = manager.respond_catalytic(
            query="Tell me about the weather.",
            llm_generate=mock_llm,
        )

        print(f"\nTurn 2 result:")
        print(f"  Response: {result2.response}")
        print(f"  E_mean: {result2.E_mean:.4f}")
        print(f"  Turn pointers available: {len(manager._turn_pointers)}")

        # Show compression stats
        stats = manager.get_compression_stats()
        print(f"\nCompression stats: {stats}")

        capsule.close()
