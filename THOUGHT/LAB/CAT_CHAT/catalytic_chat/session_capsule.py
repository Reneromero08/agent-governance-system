"""
Session Capsule (Phase 3.4.1-3.4.2)

Hash-chained, append-only event log for CAT Chat session persistence.

Provides:
- Deterministic session state capture
- Hash-chained event integrity
- Save/resume capability
- Working set and pointer set tracking

Event Types:
- session_start: Session initialization
- user_message: User input
- assistant_response: Assistant output
- tool_call: MCP tool invocation
- tool_result: MCP tool result
- expansion: Symbol expansion
- assembly: Context assembly receipt
- session_end: Session termination
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from catalytic_chat.context_assembler import AssemblyReceipt


# Event type constants
EVENT_SESSION_START = "session_start"
EVENT_USER_MESSAGE = "user_message"
EVENT_ASSISTANT_RESPONSE = "assistant_response"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_EXPANSION = "expansion"
EVENT_ASSEMBLY = "assembly"
EVENT_SESSION_END = "session_end"

# Phase C.4: Auto-Controlled Context Loop event types
EVENT_PARTITION = "partition"  # Context re-partitioning based on E-scores
EVENT_TURN_STORED = "turn_stored"  # Turn compression to catalytic space
EVENT_TURN_HYDRATED = "turn_hydrated"  # Turn rehydration from catalytic space
EVENT_BUDGET_CHECK = "budget_check"  # Budget invariant verification

VALID_EVENT_TYPES = {
    EVENT_SESSION_START,
    EVENT_USER_MESSAGE,
    EVENT_ASSISTANT_RESPONSE,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    EVENT_EXPANSION,
    EVENT_ASSEMBLY,
    EVENT_SESSION_END,
    # Auto-Controlled Context Loop events
    EVENT_PARTITION,
    EVENT_TURN_STORED,
    EVENT_TURN_HYDRATED,
    EVENT_BUDGET_CHECK,
}


def _canonical_json(obj: Any) -> bytes:
    """Produce canonical JSON bytes for hashing."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash."""
    return hashlib.sha256(data).hexdigest()


def _now_iso() -> str:
    """Get ISO8601 timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class SessionEvent:
    """A single event in the session log."""
    event_id: str
    session_id: str
    event_type: str
    sequence_num: int
    timestamp: str
    payload: Dict[str, Any]
    content_hash: str
    prev_hash: str  # Hash of previous event (empty for first event)
    chain_hash: str  # Hash of (content_hash + prev_hash)


@dataclass
class SessionState:
    """Current state of a session."""
    session_id: str
    created_at: str
    last_event_at: str
    event_count: int
    chain_head: str  # Hash of latest event
    working_set: List[str]  # IDs in working set
    pointer_set: List[str]  # IDs in pointer set
    corpus_snapshot_id: Optional[str]
    is_active: bool


class SessionCapsuleError(Exception):
    """Session capsule operation error."""
    pass


class SessionCapsule:
    """
    Hash-chained, append-only session event log.

    Invariants:
    - Events are append-only (no updates, no deletes)
    - Each event chains to previous via hash
    - Chain integrity is verifiable
    - Session can be saved and resumed
    """

    # Genesis hash for first event in chain
    GENESIS_HASH = "0" * 64

    def __init__(
        self,
        db_path: Optional[Path] = None,
        repo_root: Optional[Path] = None
    ):
        """
        Initialize session capsule.

        Args:
            db_path: Path to SQLite database (default: cat_chat.db)
            repo_root: Repository root for default paths
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]
        self.repo_root = repo_root

        if db_path is None:
            from .paths import get_cat_chat_db
            db_path = get_cat_chat_db(repo_root)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()

        conn.executescript("""
            -- Sessions table
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_event_at TEXT NOT NULL,
                event_count INTEGER DEFAULT 0,
                chain_head TEXT NOT NULL,
                corpus_snapshot_id TEXT,
                is_active INTEGER DEFAULT 1
            );

            -- Session events (append-only)
            CREATE TABLE IF NOT EXISTS session_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                UNIQUE (session_id, sequence_num)
            );

            -- Working set tracking
            CREATE TABLE IF NOT EXISTS session_working_set (
                session_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (session_id, item_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Pointer set tracking
            CREATE TABLE IF NOT EXISTS session_pointer_set (
                session_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                PRIMARY KEY (session_id, item_id),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_events_session
                ON session_events(session_id, sequence_num);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON session_events(event_type);

            -- Append-only trigger (prevent updates/deletes)
            CREATE TRIGGER IF NOT EXISTS prevent_event_update
            BEFORE UPDATE ON session_events
            BEGIN
                SELECT RAISE(ABORT, 'Events are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS prevent_event_delete
            BEFORE DELETE ON session_events
            BEGIN
                SELECT RAISE(ABORT, 'Events are append-only');
            END;
        """)

        conn.commit()

    def create_session(
        self,
        session_id: Optional[str] = None,
        corpus_snapshot_id: Optional[str] = None
    ) -> str:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (generated if not provided)
            corpus_snapshot_id: Optional corpus snapshot hash

        Returns:
            Session ID
        """
        conn = self._get_conn()
        now = _now_iso()

        if session_id is None:
            session_id = f"session_{_compute_hash(now.encode())[:16]}"

        conn.execute("""
            INSERT INTO sessions (
                session_id, created_at, last_event_at,
                event_count, chain_head, corpus_snapshot_id, is_active
            ) VALUES (?, ?, ?, 0, ?, ?, 1)
        """, (session_id, now, now, self.GENESIS_HASH, corpus_snapshot_id))

        conn.commit()

        # Log session start event
        self.append_event(
            session_id,
            EVENT_SESSION_START,
            {"corpus_snapshot_id": corpus_snapshot_id}
        )

        return session_id

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: Dict[str, Any]
    ) -> SessionEvent:
        """
        Append an event to the session log.

        Args:
            session_id: Session ID
            event_type: Event type (must be in VALID_EVENT_TYPES)
            payload: Event payload

        Returns:
            The created SessionEvent

        Raises:
            SessionCapsuleError: On invalid input or chain integrity failure
        """
        if event_type not in VALID_EVENT_TYPES:
            raise SessionCapsuleError(f"Invalid event type: {event_type}")

        conn = self._get_conn()
        now = _now_iso()

        # Get current session state
        cursor = conn.execute("""
            SELECT event_count, chain_head, is_active
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()

        if row is None:
            raise SessionCapsuleError(f"Session not found: {session_id}")

        if not row["is_active"]:
            raise SessionCapsuleError(f"Session is not active: {session_id}")

        current_count = row["event_count"]
        prev_hash = row["chain_head"]
        sequence_num = current_count + 1

        # Compute hashes
        content_bytes = _canonical_json(payload)
        content_hash = _compute_hash(content_bytes)
        chain_hash = _compute_hash(f"{content_hash}{prev_hash}".encode())

        event_id = f"evt_{session_id}_{sequence_num}"

        # Insert event
        conn.execute("""
            INSERT INTO session_events (
                event_id, session_id, event_type, sequence_num,
                timestamp, payload_json, content_hash, prev_hash, chain_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, session_id, event_type, sequence_num,
            now, json.dumps(payload, sort_keys=True),
            content_hash, prev_hash, chain_hash
        ))

        # Update session
        conn.execute("""
            UPDATE sessions SET
                last_event_at = ?,
                event_count = ?,
                chain_head = ?
            WHERE session_id = ?
        """, (now, sequence_num, chain_hash, session_id))

        conn.commit()

        return SessionEvent(
            event_id=event_id,
            session_id=session_id,
            event_type=event_type,
            sequence_num=sequence_num,
            timestamp=now,
            payload=payload,
            content_hash=content_hash,
            prev_hash=prev_hash,
            chain_hash=chain_hash
        )

    def log_user_message(self, session_id: str, content: str) -> SessionEvent:
        """Log a user message."""
        return self.append_event(session_id, EVENT_USER_MESSAGE, {
            "content": content
        })

    def log_assistant_response(self, session_id: str, content: str) -> SessionEvent:
        """Log an assistant response."""
        return self.append_event(session_id, EVENT_ASSISTANT_RESPONSE, {
            "content": content
        })

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> SessionEvent:
        """Log a tool call."""
        return self.append_event(session_id, EVENT_TOOL_CALL, {
            "tool_name": tool_name,
            "arguments": arguments
        })

    def log_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any
    ) -> SessionEvent:
        """Log a tool result."""
        return self.append_event(session_id, EVENT_TOOL_RESULT, {
            "tool_name": tool_name,
            "result": result
        })

    def log_expansion(
        self,
        session_id: str,
        symbol_id: str,
        content_hash: str,
        source: str
    ) -> SessionEvent:
        """Log an expansion resolution."""
        return self.append_event(session_id, EVENT_EXPANSION, {
            "symbol_id": symbol_id,
            "content_hash": content_hash,
            "source": source
        })

    def log_assembly(
        self,
        session_id: str,
        receipt: AssemblyReceipt
    ) -> SessionEvent:
        """Log a context assembly."""
        # Update working set and pointer set
        conn = self._get_conn()
        now = _now_iso()

        for item_id in receipt.working_set:
            conn.execute("""
                INSERT OR IGNORE INTO session_working_set
                (session_id, item_id, added_at) VALUES (?, ?, ?)
            """, (session_id, item_id, now))

        for item_id in receipt.pointer_set:
            conn.execute("""
                INSERT OR IGNORE INTO session_pointer_set
                (session_id, item_id, added_at) VALUES (?, ?, ?)
            """, (session_id, item_id, now))

        conn.commit()

        return self.append_event(session_id, EVENT_ASSEMBLY, {
            "final_assemblage_hash": receipt.final_assemblage_hash,
            "token_usage_total": receipt.token_usage_total,
            "items_included": receipt.items_included,
            "working_set": receipt.working_set,
            "pointer_set": receipt.pointer_set,
            "corpus_snapshot_id": receipt.corpus_snapshot_id
        })

    # =========================================================================
    # Phase C.4: Auto-Controlled Context Loop Event Logging
    # =========================================================================

    def log_partition(
        self,
        session_id: str,
        query_hash: str,
        working_set_ids: List[str],
        pointer_set_ids: List[str],
        budget_total: int,
        budget_used: int,
        threshold: float,
        E_mean: float,
        E_min: float,
        E_max: float,
        items_below_threshold: int,
        items_over_budget: int
    ) -> SessionEvent:
        """
        Log a context partition event.

        Records the result of re-partitioning all items based on E-scores
        against the current query.
        """
        # Update working set and pointer set tables
        conn = self._get_conn()
        now = _now_iso()

        # Clear and repopulate (partition is a full replacement)
        conn.execute(
            "DELETE FROM session_working_set WHERE session_id = ?",
            (session_id,)
        )
        conn.execute(
            "DELETE FROM session_pointer_set WHERE session_id = ?",
            (session_id,)
        )

        # Deduplicate item IDs (safety net for any upstream bugs)
        unique_working = list(dict.fromkeys(working_set_ids))
        unique_pointer = list(dict.fromkeys(pointer_set_ids))

        for item_id in unique_working:
            conn.execute("""
                INSERT INTO session_working_set
                (session_id, item_id, added_at) VALUES (?, ?, ?)
            """, (session_id, item_id, now))

        for item_id in unique_pointer:
            conn.execute("""
                INSERT INTO session_pointer_set
                (session_id, item_id, added_at) VALUES (?, ?, ?)
            """, (session_id, item_id, now))

        conn.commit()

        return self.append_event(session_id, EVENT_PARTITION, {
            "query_hash": query_hash,
            "working_set": working_set_ids,
            "pointer_set": pointer_set_ids,
            "budget_total": budget_total,
            "budget_used": budget_used,
            "threshold": threshold,
            "E_mean": E_mean,
            "E_min": E_min,
            "E_max": E_max,
            "items_below_threshold": items_below_threshold,
            "items_over_budget": items_over_budget,
        })

    def log_turn_stored(
        self,
        session_id: str,
        turn_id: str,
        content_hash: str,
        summary: str,
        original_tokens: int,
        pointer_tokens: int
    ) -> SessionEvent:
        """
        Log turn compression to catalytic space.

        Records when a turn's full content is stored and replaced with a pointer.
        """
        return self.append_event(session_id, EVENT_TURN_STORED, {
            "turn_id": turn_id,
            "content_hash": content_hash,
            "summary": summary,
            "original_tokens": original_tokens,
            "pointer_tokens": pointer_tokens,
            "compression_ratio": original_tokens / max(pointer_tokens, 1),
        })

    def log_turn_hydrated(
        self,
        session_id: str,
        turn_id: str,
        content_hash: str,
        E_score: float,
        tokens_added: int
    ) -> SessionEvent:
        """
        Log turn rehydration from catalytic space.

        Records when a turn's full content is retrieved because it scored
        high enough on E-score relevance.
        """
        return self.append_event(session_id, EVENT_TURN_HYDRATED, {
            "turn_id": turn_id,
            "content_hash": content_hash,
            "E_score": E_score,
            "tokens_added": tokens_added,
        })

    def log_budget_check(
        self,
        session_id: str,
        budget_available: int,
        budget_used: int,
        item_count: int,
        passed: bool,
        context_window: int,
        model_id: str
    ) -> SessionEvent:
        """
        Log budget invariant check.

        Records verification of INV-CATALYTIC-04 (Clean Space Bound).
        """
        return self.append_event(session_id, EVENT_BUDGET_CHECK, {
            "budget_available": budget_available,
            "budget_used": budget_used,
            "item_count": item_count,
            "passed": passed,
            "utilization_pct": budget_used / max(budget_available, 1),
            "context_window": context_window,
            "model_id": model_id,
        })

    def get_working_set_tokens(
        self,
        session_id: str,
        token_estimator: Optional[callable] = None
    ) -> int:
        """
        Compute current working set token usage.

        Note: This requires the actual content to be stored somewhere.
        For now, we track the latest partition's budget_used.
        """
        events = self.get_events(session_id, event_type=EVENT_PARTITION, limit=1)
        if events:
            # Get from most recent partition event (they're ordered ASC)
            all_partitions = self.get_events(session_id, event_type=EVENT_PARTITION)
            if all_partitions:
                latest = all_partitions[-1]
                return latest.payload.get("budget_used", 0)
        return 0

    def end_session(self, session_id: str) -> SessionEvent:
        """End a session."""
        conn = self._get_conn()

        event = self.append_event(session_id, EVENT_SESSION_END, {})

        conn.execute("""
            UPDATE sessions SET is_active = 0 WHERE session_id = ?
        """, (session_id,))
        conn.commit()

        return event

    def get_session_state(self, session_id: str) -> SessionState:
        """Get current session state."""
        conn = self._get_conn()

        cursor = conn.execute("""
            SELECT * FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()

        if row is None:
            raise SessionCapsuleError(f"Session not found: {session_id}")

        # Get working set
        cursor = conn.execute("""
            SELECT item_id FROM session_working_set WHERE session_id = ?
        """, (session_id,))
        working_set = [r["item_id"] for r in cursor.fetchall()]

        # Get pointer set
        cursor = conn.execute("""
            SELECT item_id FROM session_pointer_set WHERE session_id = ?
        """, (session_id,))
        pointer_set = [r["item_id"] for r in cursor.fetchall()]

        return SessionState(
            session_id=row["session_id"],
            created_at=row["created_at"],
            last_event_at=row["last_event_at"],
            event_count=row["event_count"],
            chain_head=row["chain_head"],
            working_set=working_set,
            pointer_set=pointer_set,
            corpus_snapshot_id=row["corpus_snapshot_id"],
            is_active=bool(row["is_active"])
        )

    def get_events(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[SessionEvent]:
        """
        Get events from a session.

        Args:
            session_id: Session ID
            event_type: Optional filter by event type
            limit: Optional limit on results

        Returns:
            List of SessionEvent objects
        """
        conn = self._get_conn()

        query = "SELECT * FROM session_events WHERE session_id = ?"
        params: List[Any] = [session_id]

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY sequence_num ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)

        events = []
        for row in cursor.fetchall():
            events.append(SessionEvent(
                event_id=row["event_id"],
                session_id=row["session_id"],
                event_type=row["event_type"],
                sequence_num=row["sequence_num"],
                timestamp=row["timestamp"],
                payload=json.loads(row["payload_json"]),
                content_hash=row["content_hash"],
                prev_hash=row["prev_hash"],
                chain_hash=row["chain_hash"]
            ))

        return events

    def verify_chain(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify the hash chain integrity for a session.

        Args:
            session_id: Session ID to verify

        Returns:
            Tuple of (is_valid, error_message)
        """
        events = self.get_events(session_id)

        if not events:
            return True, None

        expected_prev = self.GENESIS_HASH

        for event in events:
            # Verify prev_hash chain
            if event.prev_hash != expected_prev:
                return False, (
                    f"Chain broken at event {event.sequence_num}: "
                    f"expected prev_hash {expected_prev[:16]}..., "
                    f"got {event.prev_hash[:16]}..."
                )

            # Verify content hash
            content_bytes = _canonical_json(event.payload)
            expected_content_hash = _compute_hash(content_bytes)
            if event.content_hash != expected_content_hash:
                return False, (
                    f"Content hash mismatch at event {event.sequence_num}"
                )

            # Verify chain hash
            expected_chain = _compute_hash(
                f"{event.content_hash}{event.prev_hash}".encode()
            )
            if event.chain_hash != expected_chain:
                return False, (
                    f"Chain hash mismatch at event {event.sequence_num}"
                )

            expected_prev = event.chain_hash

        return True, None

    def export_session(self, session_id: str) -> Dict[str, Any]:
        """
        Export a session as JSON-serializable dict for save/restore.

        Args:
            session_id: Session ID to export

        Returns:
            Dict containing full session state
        """
        state = self.get_session_state(session_id)
        events = self.get_events(session_id)

        return {
            "version": "1.0",
            "session_id": session_id,
            "state": asdict(state),
            "events": [asdict(e) for e in events],
            "exported_at": _now_iso()
        }

    def import_session(self, data: Dict[str, Any]) -> str:
        """
        Import a session from exported data.

        Args:
            data: Exported session data

        Returns:
            Session ID of imported session

        Raises:
            SessionCapsuleError: If import fails
        """
        if data.get("version") != "1.0":
            raise SessionCapsuleError(
                f"Unsupported session version: {data.get('version')}"
            )

        conn = self._get_conn()

        state = data["state"]
        events = data["events"]
        session_id = state["session_id"]

        # Check if session already exists
        cursor = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        )
        if cursor.fetchone():
            raise SessionCapsuleError(
                f"Session already exists: {session_id}"
            )

        # Create session record
        conn.execute("""
            INSERT INTO sessions (
                session_id, created_at, last_event_at,
                event_count, chain_head, corpus_snapshot_id, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            state["created_at"],
            state["last_event_at"],
            state["event_count"],
            state["chain_head"],
            state["corpus_snapshot_id"],
            state["is_active"]
        ))

        # Insert events
        for event in events:
            conn.execute("""
                INSERT INTO session_events (
                    event_id, session_id, event_type, sequence_num,
                    timestamp, payload_json, content_hash, prev_hash, chain_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event["event_id"],
                event["session_id"],
                event["event_type"],
                event["sequence_num"],
                event["timestamp"],
                json.dumps(event["payload"], sort_keys=True),
                event["content_hash"],
                event["prev_hash"],
                event["chain_hash"]
            ))

        # Insert working set
        for item_id in state["working_set"]:
            conn.execute("""
                INSERT INTO session_working_set
                (session_id, item_id, added_at) VALUES (?, ?, ?)
            """, (session_id, item_id, state["last_event_at"]))

        # Insert pointer set
        for item_id in state["pointer_set"]:
            conn.execute("""
                INSERT INTO session_pointer_set
                (session_id, item_id, added_at) VALUES (?, ?, ?)
            """, (session_id, item_id, state["last_event_at"]))

        conn.commit()

        # Verify imported chain
        is_valid, error = self.verify_chain(session_id)
        if not is_valid:
            raise SessionCapsuleError(f"Imported session has invalid chain: {error}")

        return session_id

    def list_sessions(self, active_only: bool = False) -> List[SessionState]:
        """List all sessions."""
        conn = self._get_conn()

        query = "SELECT session_id FROM sessions"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY created_at DESC"

        cursor = conn.execute(query)
        sessions = []
        for row in cursor.fetchall():
            sessions.append(self.get_session_state(row["session_id"]))

        return sessions

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
