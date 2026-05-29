"""
CAT_CHAT Debug Utilities
========================

Easy inspection of what's being logged during development.

Usage:
    from catalytic_chat.debug import CatChatDebugger

    debugger = CatChatDebugger(db_path)
    debugger.show_recent_turns(5)
    debugger.show_messages(session_id)
    debugger.show_event_summary()
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class StoredMessage:
    """A message extracted from storage."""
    turn_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    content_hash: str
    tokens: int


@dataclass
class TurnRecord:
    """A complete turn (user + assistant)."""
    turn_id: str
    user_query: str
    assistant_response: str
    timestamp: str
    content_hash: str
    original_tokens: int
    pointer_tokens: int
    compression_ratio: float


class CatChatDebugger:
    """
    Debug utility for inspecting CAT_CHAT database state.

    Makes it easy to see what's actually being stored.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # Message/Turn Inspection
    # =========================================================================

    def get_all_turns(self, session_id: Optional[str] = None) -> List[TurnRecord]:
        """
        Get all stored turns.

        Args:
            session_id: Optional filter by session

        Returns:
            List of TurnRecord with full content
        """
        conn = self._get_conn()
        try:
            query = """
                SELECT session_id, payload_json, timestamp
                FROM session_events
                WHERE event_type = 'turn_stored'
            """
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY sequence_num ASC"

            cursor = conn.execute(query, params)

            turns = []
            for row in cursor.fetchall():
                payload = json.loads(row["payload_json"])
                turns.append(TurnRecord(
                    turn_id=payload.get("turn_id", "unknown"),
                    user_query=payload.get("user_query", ""),
                    assistant_response=payload.get("assistant_response", ""),
                    timestamp=payload.get("timestamp", row["timestamp"]),
                    content_hash=payload.get("content_hash", ""),
                    original_tokens=payload.get("original_tokens", 0),
                    pointer_tokens=payload.get("pointer_tokens", 0),
                    compression_ratio=payload.get("compression_ratio", 0.0),
                ))

            return turns
        finally:
            conn.close()

    def get_all_messages(self, session_id: Optional[str] = None) -> List[StoredMessage]:
        """
        Get all messages (user and assistant) as flat list.

        Args:
            session_id: Optional filter by session

        Returns:
            List of StoredMessage in chronological order
        """
        turns = self.get_all_turns(session_id)
        messages = []

        for turn in turns:
            # User message
            messages.append(StoredMessage(
                turn_id=turn.turn_id,
                role="user",
                content=turn.user_query,
                timestamp=turn.timestamp,
                content_hash=turn.content_hash,
                tokens=len(turn.user_query) // 4,
            ))

            # Assistant message
            messages.append(StoredMessage(
                turn_id=turn.turn_id,
                role="assistant",
                content=turn.assistant_response,
                timestamp=turn.timestamp,
                content_hash=turn.content_hash,
                tokens=len(turn.assistant_response) // 4,
            ))

        return messages

    def show_recent_turns(self, n: int = 5, session_id: Optional[str] = None) -> None:
        """Print the most recent N turns."""
        turns = self.get_all_turns(session_id)[-n:]

        print(f"\n{'='*60}")
        print(f"RECENT TURNS ({len(turns)} shown)")
        print(f"{'='*60}")

        for turn in turns:
            print(f"\n--- {turn.turn_id} ({turn.timestamp[:19]}) ---")
            print(f"[USER] {turn.user_query[:200]}{'...' if len(turn.user_query) > 200 else ''}")
            print(f"[ASSISTANT] {turn.assistant_response[:300]}{'...' if len(turn.assistant_response) > 300 else ''}")
            print(f"[STATS] {turn.original_tokens} tokens, {turn.compression_ratio:.1f}x compression")

    def show_messages(self, session_id: Optional[str] = None, limit: int = 20) -> None:
        """Print messages in chat format."""
        messages = self.get_all_messages(session_id)[-limit:]

        print(f"\n{'='*60}")
        print(f"MESSAGES ({len(messages)} shown)")
        print(f"{'='*60}")

        for msg in messages:
            role_prefix = "USER" if msg.role == "user" else "ASST"
            print(f"\n[{role_prefix}] {msg.content[:400]}{'...' if len(msg.content) > 400 else ''}")

    # =========================================================================
    # Event Inspection
    # =========================================================================

    def get_event_summary(self, session_id: Optional[str] = None) -> Dict[str, int]:
        """Get count of each event type."""
        conn = self._get_conn()
        try:
            query = "SELECT event_type, COUNT(*) as cnt FROM session_events"
            params = []

            if session_id:
                query += " WHERE session_id = ?"
                params.append(session_id)

            query += " GROUP BY event_type ORDER BY cnt DESC"

            cursor = conn.execute(query, params)
            return {row["event_type"]: row["cnt"] for row in cursor.fetchall()}
        finally:
            conn.close()

    def show_event_summary(self, session_id: Optional[str] = None) -> None:
        """Print event type summary."""
        summary = self.get_event_summary(session_id)

        print(f"\n{'='*60}")
        print("EVENT SUMMARY")
        print(f"{'='*60}")

        total = sum(summary.values())
        for event_type, count in summary.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {event_type:25} {count:5} ({pct:5.1f}%)")
        print(f"  {'TOTAL':25} {total:5}")

    def get_partition_events(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get E-score partition events for analysis."""
        conn = self._get_conn()
        try:
            query = """
                SELECT payload_json, timestamp
                FROM session_events
                WHERE event_type = 'partition'
            """
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY sequence_num ASC"

            cursor = conn.execute(query, params)
            return [json.loads(row["payload_json"]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def show_e_score_history(self, session_id: Optional[str] = None) -> None:
        """Print E-score history across turns."""
        partitions = self.get_partition_events(session_id)

        print(f"\n{'='*60}")
        print("E-SCORE HISTORY")
        print(f"{'='*60}")
        print(f"{'Turn':>6} {'E_mean':>8} {'E_min':>8} {'E_max':>8} {'Working':>8} {'Pointer':>8}")
        print("-" * 60)

        for i, p in enumerate(partitions):
            print(f"{i+1:>6} {p.get('E_mean', 0):.4f}  {p.get('E_min', 0):.4f}  "
                  f"{p.get('E_max', 0):.4f}  {len(p.get('working_set', [])):>8} "
                  f"{len(p.get('pointer_set', [])):>8}")

    def get_hydration_events(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get turn hydration events."""
        conn = self._get_conn()
        try:
            query = """
                SELECT payload_json, timestamp
                FROM session_events
                WHERE event_type = 'turn_hydrated'
            """
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)

            query += " ORDER BY sequence_num ASC"

            cursor = conn.execute(query, params)
            return [json.loads(row["payload_json"]) for row in cursor.fetchall()]
        finally:
            conn.close()

    def show_hydrations(self, session_id: Optional[str] = None) -> None:
        """Print hydration events (when old turns were recalled)."""
        hydrations = self.get_hydration_events(session_id)

        print(f"\n{'='*60}")
        print(f"HYDRATION EVENTS ({len(hydrations)} total)")
        print(f"{'='*60}")

        if not hydrations:
            print("No turns have been hydrated yet.")
            return

        for h in hydrations:
            print(f"  Turn: {h.get('turn_id', 'unknown'):15} "
                  f"E={h.get('E_score', 0):.4f}  "
                  f"+{h.get('tokens_added', 0)} tokens")

    # =========================================================================
    # Session Inspection
    # =========================================================================

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT session_id, created_at, event_count, is_active
                FROM sessions
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def show_sessions(self) -> None:
        """Print all sessions."""
        sessions = self.list_sessions()

        print(f"\n{'='*60}")
        print(f"SESSIONS ({len(sessions)} total)")
        print(f"{'='*60}")

        for s in sessions:
            status = "ACTIVE" if s["is_active"] else "ended"
            print(f"  {s['session_id'][:30]:30} {s['event_count']:5} events  [{status}]")

    # =========================================================================
    # Schema Inspection
    # =========================================================================

    def show_schema(self) -> None:
        """Print database schema."""
        conn = self._get_conn()
        try:
            cursor = conn.execute("""
                SELECT name, sql FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)

            print(f"\n{'='*60}")
            print("DATABASE SCHEMA")
            print(f"{'='*60}")

            for row in cursor.fetchall():
                print(f"\n-- {row['name']} --")
                if row['sql']:
                    # Simplify the output
                    lines = row['sql'].split('\n')
                    for line in lines[:10]:  # First 10 lines
                        print(f"  {line.strip()}")
                    if len(lines) > 10:
                        print(f"  ... ({len(lines)-10} more lines)")
        finally:
            conn.close()

    # =========================================================================
    # Full Diagnostic Report
    # =========================================================================

    def report(self, session_id: Optional[str] = None) -> None:
        """Print a full diagnostic report."""
        print("\n" + "=" * 70)
        print("CAT_CHAT DIAGNOSTIC REPORT")
        print("=" * 70)
        print(f"Database: {self.db_path}")

        self.show_sessions()
        self.show_event_summary(session_id)
        self.show_recent_turns(3, session_id)
        self.show_e_score_history(session_id)
        self.show_hydrations(session_id)

        print("\n" + "=" * 70)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI for CAT_CHAT debugging."""
    import argparse

    parser = argparse.ArgumentParser(description="CAT_CHAT Debug Utility")
    parser.add_argument("db_path", help="Path to cat_chat.db")
    parser.add_argument("--session", "-s", help="Filter by session ID")
    parser.add_argument("--turns", "-t", type=int, default=5,
                        help="Number of recent turns to show")
    parser.add_argument("--messages", "-m", action="store_true",
                        help="Show messages in chat format")
    parser.add_argument("--events", "-e", action="store_true",
                        help="Show event summary only")
    parser.add_argument("--schema", action="store_true",
                        help="Show database schema")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Full diagnostic report")

    args = parser.parse_args()

    debugger = CatChatDebugger(Path(args.db_path))

    if args.report:
        debugger.report(args.session)
    elif args.schema:
        debugger.show_schema()
    elif args.events:
        debugger.show_event_summary(args.session)
    elif args.messages:
        debugger.show_messages(args.session)
    else:
        debugger.show_sessions()
        debugger.show_recent_turns(args.turns, args.session)
        debugger.show_e_score_history(args.session)


if __name__ == "__main__":
    main()