"""
ELO Observer (Phase E.3)

Tracks retrieval usage as METADATA ONLY - does NOT modify ranking.

Key Design Principles:
1. OBSERVATION ONLY: Called AFTER retrieval completes
2. NO RANKING INFLUENCE: ELO scores are never used to reorder results
3. METADATA TRACKING: Logs usage patterns for analytics/governance
4. APPEND-ONLY: All updates logged to elo_updates.jsonl

This module implements E.3 from CAT_CHAT_ROADMAP_2.0.md:
- Track usage patterns
- Do NOT modify ranking based on ELO

CRITICAL: The observer is called AFTER results are returned.
It NEVER participates in ranking decisions.
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Dict

# Try to import from CAPABILITY, but allow standalone use
try:
    from CAPABILITY.PRIMITIVES.elo_db import EloDatabase
    from CAPABILITY.PRIMITIVES.elo_engine import EloEngine
except ImportError:
    # For standalone testing or when CAPABILITY not in path
    EloDatabase = None
    EloEngine = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RetrievalEvent:
    """
    Record of a retrieval event for ELO observation.

    Attributes:
        timestamp: ISO8601 timestamp
        session_id: Session that triggered the retrieval
        source: Retrieval source (spc, cassette_fts, local_index, cas, vector_fallback)
        entity_id: ID of retrieved entity (hash, path, symbol)
        rank: Position in results (1 = first)
        was_used: Whether the result was actually included in context
        E_score: Optional E-score if available
    """
    timestamp: str
    session_id: str
    source: str
    entity_id: str
    rank: int
    was_used: bool
    E_score: Optional[float] = None


# =============================================================================
# ELO Observer
# =============================================================================

class EloObserver:
    """
    Observes retrieval events and updates ELO metadata.

    CRITICAL: This class is called AFTER retrieval completes.
    It NEVER influences ranking decisions.

    Usage:
        observer = EloObserver(elo_db_path, updates_log_path)
        # ... retrieval happens ...
        observer.on_retrieval_complete(
            session_id="session-123",
            source="vector_fallback",
            entity_id="abc123...",
            rank=1,
            was_used=True
        )
    """

    # Outcome scores based on rank (from VECTOR_ELO_SPEC.md)
    RANK_OUTCOMES = {
        1: 1.0,      # Rank 1 = full boost
        2: 0.75,     # Ranks 2-5 = partial boost
        3: 0.75,
        4: 0.75,
        5: 0.75,
    }
    DEFAULT_OUTCOME = 0.625  # Ranks 6+ = smaller boost

    # Map retrieval sources to entity types
    SOURCE_TO_ENTITY_TYPE = {
        "spc": "symbol",
        "cassette_fts": "file",
        "local_index": "symbol",
        "cas": "vector",
        "vector_fallback": "vector",
    }

    def __init__(
        self,
        elo_db_path: Optional[Path] = None,
        updates_log_path: Optional[Path] = None,
        enable_elo_updates: bool = True
    ):
        """
        Initialize ELO observer.

        Args:
            elo_db_path: Path to ELO database (optional, updates disabled if None)
            updates_log_path: Path to updates log (optional)
            enable_elo_updates: If False, only log events, don't update ELO
        """
        self._events: List[RetrievalEvent] = []
        self._enable_elo_updates = enable_elo_updates and EloDatabase is not None

        # Initialize ELO engine if available
        self._engine: Optional[Any] = None
        if self._enable_elo_updates and elo_db_path and updates_log_path:
            try:
                db = EloDatabase(str(elo_db_path))
                self._engine = EloEngine(db, str(updates_log_path))
            except Exception:
                # ELO system not available, continue without it
                self._enable_elo_updates = False

        # Event log path (separate from ELO updates)
        self._event_log_path = updates_log_path

    def on_retrieval_complete(
        self,
        session_id: str,
        source: str,
        entity_id: str,
        rank: int,
        was_used: bool,
        E_score: Optional[float] = None
    ) -> None:
        """
        Record retrieval event and update ELO metadata.

        Called AFTER retrieval to track usage patterns.
        NEVER influences ranking.

        Args:
            session_id: Session that triggered the retrieval
            source: Retrieval source (spc, cassette_fts, local_index, cas, vector_fallback)
            entity_id: ID of retrieved entity
            rank: Position in results (1 = first)
            was_used: Whether the result was included in context
            E_score: Optional E-score if available
        """
        # Create event record
        event = RetrievalEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            source=source,
            entity_id=entity_id,
            rank=rank,
            was_used=was_used,
            E_score=E_score
        )
        self._events.append(event)

        # Update ELO only if result was actually used
        if was_used and self._enable_elo_updates and self._engine:
            self._update_elo(event)

    def _update_elo(self, event: RetrievalEvent) -> None:
        """
        Update ELO score for retrieval event.

        Args:
            event: The retrieval event
        """
        # Determine outcome based on rank
        outcome = self.RANK_OUTCOMES.get(event.rank, self.DEFAULT_OUTCOME)

        # Determine entity type from source
        entity_type = self.SOURCE_TO_ENTITY_TYPE.get(event.source, "vector")

        # Build reason string
        reason = f"{event.source}_rank_{event.rank}"

        try:
            self._engine.update_elo(
                entity_type=entity_type,
                entity_id=event.entity_id,
                outcome=outcome,
                reason=reason
            )
        except Exception:
            # ELO update failed, continue without it
            pass

    def get_events(self) -> List[RetrievalEvent]:
        """
        Get recorded events.

        For testing/debugging only.

        Returns:
            Copy of recorded events
        """
        return list(self._events)

    def get_events_for_session(self, session_id: str) -> List[RetrievalEvent]:
        """
        Get events for a specific session.

        Args:
            session_id: Session ID to filter by

        Returns:
            Events for the specified session
        """
        return [e for e in self._events if e.session_id == session_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get observer statistics.

        Returns:
            Dict with event counts by source
        """
        stats: Dict[str, int] = {}
        for event in self._events:
            key = event.source
            stats[key] = stats.get(key, 0) + 1

        return {
            "total_events": len(self._events),
            "by_source": stats,
            "elo_updates_enabled": self._enable_elo_updates
        }

    def clear_events(self) -> None:
        """
        Clear recorded events.

        For testing only.
        """
        self._events.clear()
