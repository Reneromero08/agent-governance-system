"""
Session Audit Logging for Agent Governance System.

Tracks all file/symbol access within a session and writes audit entries
to a JSONL file for governance and compliance purposes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class SessionAuditEntry:
    """Represents a completed session audit record."""

    session_id: str
    agent_id: str
    start_time: str
    end_time: str
    files_accessed: List[str]
    symbols_expanded: List[str]
    adrs_read: List[str]
    search_queries: int
    semantic_searches: int
    keyword_searches: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "files_accessed": self.files_accessed,
            "symbols_expanded": self.symbols_expanded,
            "adrs_read": self.adrs_read,
            "search_queries": self.search_queries,
            "semantic_searches": self.semantic_searches,
            "keyword_searches": self.keyword_searches,
        }


class SessionAuditor:
    """
    Session-level audit logger for tracking file and symbol access.

    Tracks:
    - Files accessed during the session
    - Symbols expanded (e.g., @Symbol references)
    - ADRs read
    - Search query counts (semantic vs keyword)

    Usage:
        auditor = SessionAuditor(log_dir="/path/to/logs", agent_id="antigravity")
        auditor.start_session()
        auditor.record_file_access("CANON/INVARIANTS.md")
        auditor.record_symbol_expansion("@CoreInvariants")
        auditor.record_adr_read("ADR-027")
        auditor.record_search(is_semantic=True)
        auditor.end_session()
    """

    def __init__(self, log_dir: str, agent_id: str):
        """
        Initialize the session auditor.

        Args:
            log_dir: Directory where session_audit.jsonl will be written.
            agent_id: Identifier for the agent (e.g., "antigravity", "user").
        """
        self.log_dir = Path(log_dir)
        self.agent_id = agent_id
        self.log_file = self.log_dir / "session_audit.jsonl"

        # Session state (initialized on start_session)
        self._session_id: Optional[str] = None
        self._start_time: Optional[str] = None
        self._files_accessed: List[str] = []
        self._symbols_expanded: List[str] = []
        self._adrs_read: List[str] = []
        self._semantic_searches: int = 0
        self._keyword_searches: int = 0
        self._session_active: bool = False

    def start_session(self) -> str:
        """
        Start a new audit session.

        Generates a unique session ID and records the start time.

        Returns:
            The generated session ID.

        Raises:
            RuntimeError: If a session is already active.
        """
        if self._session_active:
            raise RuntimeError("Session already active. Call end_session() first.")

        self._session_id = str(uuid.uuid4())
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._files_accessed = []
        self._symbols_expanded = []
        self._adrs_read = []
        self._semantic_searches = 0
        self._keyword_searches = 0
        self._session_active = True

        return self._session_id

    def record_file_access(self, file_path: str) -> None:
        """
        Record that a file was accessed during this session.

        Args:
            file_path: Path to the file that was accessed.

        Raises:
            RuntimeError: If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        # Normalize path and avoid duplicates
        normalized = str(file_path).replace("\\", "/")
        if normalized not in self._files_accessed:
            self._files_accessed.append(normalized)

    def record_symbol_expansion(self, symbol: str) -> None:
        """
        Record that a symbol was expanded during this session.

        Args:
            symbol: The symbol that was expanded (e.g., "@CoreInvariants").

        Raises:
            RuntimeError: If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        if symbol not in self._symbols_expanded:
            self._symbols_expanded.append(symbol)

    def record_adr_read(self, adr_id: str) -> None:
        """
        Record that an ADR was read during this session.

        Args:
            adr_id: The ADR identifier (e.g., "ADR-027").

        Raises:
            RuntimeError: If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        if adr_id not in self._adrs_read:
            self._adrs_read.append(adr_id)

    def record_search(self, is_semantic: bool) -> None:
        """
        Record a search query during this session.

        Args:
            is_semantic: True for semantic search, False for keyword search.

        Raises:
            RuntimeError: If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        if is_semantic:
            self._semantic_searches += 1
        else:
            self._keyword_searches += 1

    def end_session(self) -> SessionAuditEntry:
        """
        End the current session and write the audit entry to the log file.

        Returns:
            The completed SessionAuditEntry.

        Raises:
            RuntimeError: If no session is active.
        """
        if not self._session_active:
            raise RuntimeError("No active session. Call start_session() first.")

        end_time = datetime.now(timezone.utc).isoformat()

        entry = SessionAuditEntry(
            session_id=self._session_id,
            agent_id=self.agent_id,
            start_time=self._start_time,
            end_time=end_time,
            files_accessed=list(self._files_accessed),
            symbols_expanded=list(self._symbols_expanded),
            adrs_read=list(self._adrs_read),
            search_queries=self._semantic_searches + self._keyword_searches,
            semantic_searches=self._semantic_searches,
            keyword_searches=self._keyword_searches,
        )

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Append entry to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")

        # Reset session state
        self._session_id = None
        self._start_time = None
        self._files_accessed = []
        self._symbols_expanded = []
        self._adrs_read = []
        self._semantic_searches = 0
        self._keyword_searches = 0
        self._session_active = False

        return entry

    @property
    def is_active(self) -> bool:
        """Return True if a session is currently active."""
        return self._session_active

    @property
    def current_session_id(self) -> Optional[str]:
        """Return the current session ID, or None if no session is active."""
        return self._session_id if self._session_active else None


def _run_self_test() -> None:
    """
    Self-test function that validates the SessionAuditor functionality.

    Creates an auditor, starts a session, records sample activity,
    ends the session, and validates the resulting JSON entry.
    """
    import tempfile
    import os

    print("=== SessionAuditor Self-Test ===")
    print()

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        log_file = log_dir / "session_audit.jsonl"

        print(f"Test log directory: {log_dir}")
        print()

        # Create auditor
        auditor = SessionAuditor(log_dir=str(log_dir), agent_id="test-agent")
        print("[OK] Created SessionAuditor instance")

        # Start session
        session_id = auditor.start_session()
        print(f"[OK] Started session: {session_id}")

        # Record sample activity
        auditor.record_file_access("CANON/INVARIANTS.md")
        auditor.record_file_access("CAPABILITY/PRIMITIVES/session_auditor.py")
        auditor.record_file_access("CANON/INVARIANTS.md")  # Duplicate - should not be added
        print("[OK] Recorded file accesses (2 unique files)")

        auditor.record_symbol_expansion("@CoreInvariants")
        auditor.record_symbol_expansion("@SessionAuditor")
        auditor.record_symbol_expansion("@CoreInvariants")  # Duplicate
        print("[OK] Recorded symbol expansions (2 unique symbols)")

        auditor.record_adr_read("ADR-027")
        auditor.record_adr_read("ADR-001")
        print("[OK] Recorded ADR reads (2 ADRs)")

        # Record searches
        for _ in range(5):
            auditor.record_search(is_semantic=True)
        for _ in range(3):
            auditor.record_search(is_semantic=False)
        print("[OK] Recorded search queries (5 semantic, 3 keyword)")

        # End session
        entry = auditor.end_session()
        print("[OK] Ended session")
        print()

        # Validate the entry
        print("=== Validating Audit Entry ===")
        print()

        assert entry.session_id == session_id, "Session ID mismatch"
        print(f"  session_id: {entry.session_id}")

        assert entry.agent_id == "test-agent", "Agent ID mismatch"
        print(f"  agent_id: {entry.agent_id}")

        assert entry.start_time is not None, "Start time missing"
        print(f"  start_time: {entry.start_time}")

        assert entry.end_time is not None, "End time missing"
        print(f"  end_time: {entry.end_time}")

        assert len(entry.files_accessed) == 2, f"Expected 2 files, got {len(entry.files_accessed)}"
        print(f"  files_accessed: {entry.files_accessed}")

        assert len(entry.symbols_expanded) == 2, f"Expected 2 symbols, got {len(entry.symbols_expanded)}"
        print(f"  symbols_expanded: {entry.symbols_expanded}")

        assert len(entry.adrs_read) == 2, f"Expected 2 ADRs, got {len(entry.adrs_read)}"
        print(f"  adrs_read: {entry.adrs_read}")

        assert entry.search_queries == 8, f"Expected 8 total searches, got {entry.search_queries}"
        print(f"  search_queries: {entry.search_queries}")

        assert entry.semantic_searches == 5, f"Expected 5 semantic searches, got {entry.semantic_searches}"
        print(f"  semantic_searches: {entry.semantic_searches}")

        assert entry.keyword_searches == 3, f"Expected 3 keyword searches, got {entry.keyword_searches}"
        print(f"  keyword_searches: {entry.keyword_searches}")

        print()
        print("[OK] All entry fields validated")
        print()

        # Validate JSON file
        print("=== Validating JSONL File ===")
        print()

        assert log_file.exists(), "Log file was not created"
        print(f"[OK] Log file exists: {log_file}")

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        print(f"[OK] Log file contains 1 entry")

        # Parse and validate JSON
        parsed = json.loads(lines[0])
        print(f"[OK] Entry is valid JSON")

        # Verify all required fields present
        required_fields = [
            "session_id", "agent_id", "start_time", "end_time",
            "files_accessed", "symbols_expanded", "adrs_read",
            "search_queries", "semantic_searches", "keyword_searches"
        ]
        for field_name in required_fields:
            assert field_name in parsed, f"Missing field: {field_name}"
        print(f"[OK] All required fields present")

        print()
        print("=== Raw JSON Entry ===")
        print(json.dumps(parsed, indent=2))
        print()

        print("=== Self-Test PASSED ===")


if __name__ == "__main__":
    _run_self_test()
