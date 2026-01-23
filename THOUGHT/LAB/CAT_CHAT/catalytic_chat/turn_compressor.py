"""
Turn Compressor
===============

Continuous turn compression for the auto-controlled context loop.

Key Design Principle: Compression happens EVERY turn, not based on a window.
After each response, the current turn is immediately compressed to a hash pointer.
Rehydration happens via E-score on the next query if relevant.

Storage:
- Full turn content stored in session_events (catalytic space)
- Hash pointer (~50 tokens) remains for potential rehydration

Phase C.3 of Auto-Controlled Context Loop implementation.
Phase J.3 Integration: Automatic hierarchy maintenance via HierarchyBuilder.
"""

import hashlib
import json
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple, TYPE_CHECKING
import numpy as np

from .vector_persistence import VectorPersistence

if TYPE_CHECKING:
    from .hierarchy_builder import HierarchyBuilder


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TurnContent:
    """
    Full turn content before compression.

    A turn consists of a user query and assistant response.
    """
    turn_id: str
    user_query: str
    assistant_response: str
    timestamp: str  # ISO format
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_content(self) -> str:
        """Combined content of the turn."""
        return f"User: {self.user_query}\n\nAssistant: {self.assistant_response}"

    def compute_hash(self) -> str:
        """Compute deterministic hash of turn content."""
        canonical = json.dumps({
            "turn_id": self.turn_id,
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "turn_id": self.turn_id,
            "user_query": self.user_query,
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "content_hash": self.compute_hash(),
        }


@dataclass
class TurnPointer:
    """
    Compressed pointer to a stored turn.

    This is the small (~50 token) representation that stays in the potential
    working set while the full content is in catalytic space.
    """
    turn_id: str
    content_hash: str  # SHA-256 hash of full content
    summary: str  # Brief summary (~1 sentence)
    original_tokens: int  # Token count of original content
    pointer_tokens: int  # Token count of this pointer
    timestamp: str

    @property
    def compression_ratio(self) -> float:
        """Ratio of original to pointer size."""
        if self.pointer_tokens == 0:
            return float("inf")
        return self.original_tokens / self.pointer_tokens

    def to_pointer_content(self) -> str:
        """Generate pointer text for context inclusion."""
        return f"[Turn {self.turn_id}] {self.summary} (hash:{self.content_hash[:8]})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "turn_id": self.turn_id,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "original_tokens": self.original_tokens,
            "pointer_tokens": self.pointer_tokens,
            "compression_ratio": self.compression_ratio,
            "timestamp": self.timestamp,
        }


@dataclass
class CompressionResult:
    """Result of compressing a turn."""
    pointer: TurnPointer
    stored: bool  # Whether content was stored successfully
    storage_event_id: Optional[str] = None  # Event ID in session_events

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "pointer": self.pointer.to_dict(),
            "stored": self.stored,
            "storage_event_id": self.storage_event_id,
        }


@dataclass
class HydrationResult:
    """Result of hydrating (decompressing) a turn."""
    turn_id: str
    content: Optional[TurnContent]
    success: bool
    E_score: float = 0.0  # E-score that triggered hydration
    tokens_added: int = 0  # Tokens added to working set
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "turn_id": self.turn_id,
            "success": self.success,
            "E_score": self.E_score,
            "tokens_added": self.tokens_added,
            "error": self.error,
            "content_hash": self.content.compute_hash() if self.content else None,
        }


# =============================================================================
# Turn Compressor
# =============================================================================

class TurnCompressor:
    """
    Compresses turns to hash pointers and stores full content in catalytic space.

    Usage:
        compressor = TurnCompressor(
            db_path=Path("_generated/cat_chat.db"),
            session_id="session_123",
            summarize_fn=my_summarizer  # Optional
        )

        # Compress a turn
        result = compressor.compress_turn(turn_content)

        # Later: hydrate if E-score is high
        hydrated = compressor.decompress_turn(pointer.content_hash)
    """

    # Event types for session_events table
    EVENT_TURN_STORED = "turn_stored"
    EVENT_TURN_HYDRATED = "turn_hydrated"

    def __init__(
        self,
        db_path: Path,
        session_id: str,
        summarize_fn: Optional[Callable[[str], str]] = None,
        token_estimator: Optional[Callable[[str], int]] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        vector_persistence: Optional[VectorPersistence] = None,
        hierarchy_builder: Optional["HierarchyBuilder"] = None,
    ):
        """
        Initialize turn compressor.

        Args:
            db_path: Path to SQLite database (cat_chat.db)
            session_id: Current session ID for event logging
            summarize_fn: Function to generate turn summary (default: first 50 chars)
            token_estimator: Function to estimate tokens (default: len//4)
            embed_fn: Optional function to compute embeddings (J.0.2)
            vector_persistence: Optional VectorPersistence instance for embedding storage (J.0.2)
            hierarchy_builder: Optional HierarchyBuilder for automatic tree maintenance (J.3)
        """
        self.db_path = Path(db_path)
        self.session_id = session_id
        self.summarize_fn = summarize_fn or self._default_summarize
        self.token_estimator = token_estimator or (lambda s: len(s) // 4)

        # J.0.2: Optional embedding support
        self._embed_fn = embed_fn
        self._vector_persistence = vector_persistence

        # J.3: Optional hierarchy builder for automatic tree maintenance
        self._hierarchy_builder = hierarchy_builder

        # Cache of compressed turns for fast lookup
        self._pointer_cache: Dict[str, TurnPointer] = {}
        self._content_cache: Dict[str, TurnContent] = {}

    def _default_summarize(self, content: str, max_length: int = 100) -> str:
        """
        Default summarization: extract key phrase from content.

        A real implementation would use an LLM or extractive summarization.
        """
        # Simple extraction: first sentence or truncated content
        lines = content.strip().split("\n")
        first_line = lines[0] if lines else content[:max_length]

        # Clean up
        summary = first_line.strip()
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."

        return summary

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def compress_turn(self, turn: TurnContent, skip_storage: bool = False) -> CompressionResult:
        """
        Compress a turn to a hash pointer.

        1. Compute content hash
        2. Generate summary
        3. Optionally store full content in session_events
        4. Return pointer for working set

        Args:
            turn: Full turn content to compress
            skip_storage: If True, skip database storage (caller will handle logging)

        Returns:
            CompressionResult with pointer and storage status
        """
        content_hash = turn.compute_hash()

        # Generate summary
        summary = self.summarize_fn(turn.full_content)

        # Estimate tokens
        original_tokens = self.token_estimator(turn.full_content)
        pointer_content = f"[Turn {turn.turn_id}] {summary} (hash:{content_hash[:8]})"
        pointer_tokens = self.token_estimator(pointer_content)

        # Create pointer
        pointer = TurnPointer(
            turn_id=turn.turn_id,
            content_hash=content_hash,
            summary=summary,
            original_tokens=original_tokens,
            pointer_tokens=pointer_tokens,
            timestamp=turn.timestamp,
        )

        # Store full content in database (if not skipping)
        stored = False
        storage_event_id = None

        if not skip_storage:
            try:
                storage_event_id = self._store_turn_content(turn, content_hash)
                stored = True
            except Exception as e:
                # Log error but don't fail - pointer is still valid
                print(f"Warning: Failed to store turn content: {e}")
        else:
            # When skipping storage, mark as stored since caller handles it
            stored = True

        # Cache the pointer and content
        self._pointer_cache[content_hash] = pointer
        self._content_cache[content_hash] = turn

        return CompressionResult(
            pointer=pointer,
            stored=stored,
            storage_event_id=storage_event_id,
        )

    def _store_turn_content(self, turn: TurnContent, content_hash: str) -> str:
        """Store turn content in session_events table."""
        conn = self._get_connection()
        try:
            # Generate event ID
            event_id = f"turn_{turn.turn_id}_{content_hash[:8]}"

            # Get next sequence number
            cursor = conn.execute(
                "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM session_events WHERE session_id = ?",
                (self.session_id,)
            )
            sequence_num = cursor.fetchone()[0]

            # Get chain head for hash linking
            cursor = conn.execute(
                "SELECT chain_hash FROM session_events WHERE session_id = ? ORDER BY sequence_num DESC LIMIT 1",
                (self.session_id,)
            )
            row = cursor.fetchone()
            prev_hash = row["chain_hash"] if row else "0" * 64

            # Build payload
            payload = turn.to_dict()
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

            # Compute chain hash
            chain_hash = hashlib.sha256(
                (content_hash + prev_hash).encode()
            ).hexdigest()

            # Insert event
            conn.execute("""
                INSERT INTO session_events
                (event_id, session_id, event_type, sequence_num, timestamp,
                 payload_json, content_hash, prev_hash, chain_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                self.session_id,
                self.EVENT_TURN_STORED,
                sequence_num,
                turn.timestamp,
                payload_json,
                content_hash,
                prev_hash,
                chain_hash,
            ))

            conn.commit()

            # J.0.2: Store embedding at compression time
            embedding = None
            if self._embed_fn is not None and self._vector_persistence is not None:
                try:
                    # Compute embedding (one API call)
                    embedding = self._embed_fn(turn.full_content)

                    # Store to persistence layer
                    self._vector_persistence.store_embedding(
                        event_id=event_id,
                        session_id=self.session_id,
                        content_hash=content_hash,
                        embedding=embedding
                    )
                except Exception as e:
                    # Log warning but don't fail turn storage
                    # Embedding can be backfilled later via J.0.4 migration
                    logging.warning(
                        "Failed to store embedding for turn %s: %s", event_id, e
                    )

            # J.3: Update hierarchy if builder is configured
            if self._hierarchy_builder is not None and embedding is not None:
                try:
                    token_count = self.token_estimator(turn.full_content)
                    self._hierarchy_builder.on_turn_compressed(
                        event_id=event_id,
                        turn_vec=embedding,
                        content_hash=content_hash,
                        sequence_num=sequence_num,
                        token_count=token_count,
                    )
                except Exception as e:
                    # Log warning but don't fail turn storage
                    # Hierarchy can be rebuilt later via build_initial_hierarchy
                    logging.warning(
                        "Failed to update hierarchy for turn %s: %s", event_id, e
                    )

            return event_id

        finally:
            conn.close()

    def decompress_turn(
        self,
        content_hash: str,
        E_score: float = 0.0
    ) -> HydrationResult:
        """
        Decompress (hydrate) a turn from its hash pointer.

        Args:
            content_hash: SHA-256 hash of the turn content
            E_score: E-score that triggered this hydration

        Returns:
            HydrationResult with full content or error
        """
        # Check cache first
        if content_hash in self._content_cache:
            content = self._content_cache[content_hash]
            return HydrationResult(
                turn_id=content.turn_id,
                content=content,
                success=True,
                E_score=E_score,
                tokens_added=self.token_estimator(content.full_content),
            )

        # Load from database
        try:
            content = self._load_turn_content(content_hash)
            if content is None:
                return HydrationResult(
                    turn_id="unknown",
                    content=None,
                    success=False,
                    E_score=E_score,
                    error=f"Turn not found: {content_hash[:16]}",
                )

            # Cache for future access
            self._content_cache[content_hash] = content

            # Log hydration event
            self._log_hydration_event(content, E_score)

            return HydrationResult(
                turn_id=content.turn_id,
                content=content,
                success=True,
                E_score=E_score,
                tokens_added=self.token_estimator(content.full_content),
            )

        except Exception as e:
            return HydrationResult(
                turn_id="unknown",
                content=None,
                success=False,
                E_score=E_score,
                error=str(e),
            )

    def _load_turn_content(self, content_hash: str) -> Optional[TurnContent]:
        """Load turn content from session_events table."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                SELECT payload_json FROM session_events
                WHERE event_type = ? AND content_hash = ?
                LIMIT 1
            """, (self.EVENT_TURN_STORED, content_hash))

            row = cursor.fetchone()
            if not row:
                return None

            payload = json.loads(row["payload_json"])
            return TurnContent(
                turn_id=payload["turn_id"],
                user_query=payload["user_query"],
                assistant_response=payload["assistant_response"],
                timestamp=payload["timestamp"],
                metadata=payload.get("metadata", {}),
            )

        finally:
            conn.close()

    def _log_hydration_event(self, content: TurnContent, E_score: float) -> None:
        """Log hydration event to session_events."""
        conn = self._get_connection()
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            event_id = f"hydrate_{content.turn_id}_{timestamp[:19].replace(':', '')}"

            # Get next sequence number
            cursor = conn.execute(
                "SELECT COALESCE(MAX(sequence_num), -1) + 1 FROM session_events WHERE session_id = ?",
                (self.session_id,)
            )
            sequence_num = cursor.fetchone()[0]

            # Get chain head
            cursor = conn.execute(
                "SELECT chain_hash FROM session_events WHERE session_id = ? ORDER BY sequence_num DESC LIMIT 1",
                (self.session_id,)
            )
            row = cursor.fetchone()
            prev_hash = row["chain_hash"] if row else "0" * 64

            # Build payload
            payload = {
                "turn_id": content.turn_id,
                "content_hash": content.compute_hash(),
                "E_score": E_score,
                "tokens_added": self.token_estimator(content.full_content),
            }
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            content_hash = hashlib.sha256(payload_json.encode()).hexdigest()
            chain_hash = hashlib.sha256((content_hash + prev_hash).encode()).hexdigest()

            conn.execute("""
                INSERT INTO session_events
                (event_id, session_id, event_type, sequence_num, timestamp,
                 payload_json, content_hash, prev_hash, chain_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                self.session_id,
                self.EVENT_TURN_HYDRATED,
                sequence_num,
                timestamp,
                payload_json,
                content_hash,
                prev_hash,
                chain_hash,
            ))

            conn.commit()

        finally:
            conn.close()

    def get_all_pointers(self) -> List[TurnPointer]:
        """Get all cached turn pointers for this session."""
        return list(self._pointer_cache.values())

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        pointers = self.get_all_pointers()
        if not pointers:
            return {
                "turns_compressed": 0,
                "total_original_tokens": 0,
                "total_pointer_tokens": 0,
                "average_compression_ratio": 0.0,
            }

        total_original = sum(p.original_tokens for p in pointers)
        total_pointer = sum(p.pointer_tokens for p in pointers)
        avg_ratio = total_original / total_pointer if total_pointer > 0 else 0.0

        return {
            "turns_compressed": len(pointers),
            "total_original_tokens": total_original,
            "total_pointer_tokens": total_pointer,
            "average_compression_ratio": avg_ratio,
            "tokens_saved": total_original - total_pointer,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_turn_from_messages(
    turn_id: str,
    user_message: str,
    assistant_message: str,
    timestamp: Optional[str] = None
) -> TurnContent:
    """Create a TurnContent from user and assistant messages."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return TurnContent(
        turn_id=turn_id,
        user_query=user_message,
        assistant_response=assistant_message,
        timestamp=timestamp,
    )


def pointers_to_context_items(
    pointers: List[TurnPointer],
    embed_fn: Optional[Callable[[str], np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """
    Convert turn pointers to context items for partitioning.

    Returns list of dicts compatible with ContextItem creation.
    """
    from .context_partitioner import ContextItem

    items = []
    for p in pointers:
        content = p.to_pointer_content()
        embedding = embed_fn(content) if embed_fn else None

        items.append(ContextItem(
            item_id=f"turn_ptr_{p.turn_id}",
            content=content,
            tokens=p.pointer_tokens,
            embedding=embedding,
            item_type="turn_pointer",
            metadata={
                "content_hash": p.content_hash,
                "original_tokens": p.original_tokens,
                "summary": p.summary,
            }
        ))

    return items


if __name__ == "__main__":
    # Quick sanity test (without database)
    print("Turn Compressor - Sanity Test")
    print("=" * 50)

    # Create a test turn
    turn = TurnContent(
        turn_id="turn_001",
        user_query="What is catalytic computing and how does it manage context?",
        assistant_response=(
            "Catalytic computing is a paradigm where large disk state (catalytic space) "
            "must restore exactly after use. Context is managed through:\n"
            "1. Clean space: bounded token window\n"
            "2. Pointer compression: symbols instead of full content\n"
            "3. Hash verification: every expansion is verified"
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print(f"Original content ({len(turn.full_content)} chars):")
    print(f"  User: {turn.user_query[:50]}...")
    print(f"  Assistant: {turn.assistant_response[:50]}...")

    # Simulate compression (without DB)
    content_hash = turn.compute_hash()
    summary = turn.full_content[:100] + "..."
    original_tokens = len(turn.full_content) // 4
    pointer_content = f"[Turn {turn.turn_id}] {summary[:50]}... (hash:{content_hash[:8]})"
    pointer_tokens = len(pointer_content) // 4

    pointer = TurnPointer(
        turn_id=turn.turn_id,
        content_hash=content_hash,
        summary=summary[:50] + "...",
        original_tokens=original_tokens,
        pointer_tokens=pointer_tokens,
        timestamp=turn.timestamp,
    )

    print(f"\nCompressed to pointer ({pointer.pointer_tokens} tokens):")
    print(f"  {pointer.to_pointer_content()}")
    print(f"\nCompression ratio: {pointer.compression_ratio:.1f}x")
    print(f"Tokens saved: {pointer.original_tokens - pointer.pointer_tokens}")
