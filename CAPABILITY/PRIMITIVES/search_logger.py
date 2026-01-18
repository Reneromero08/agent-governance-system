"""Search logging utility for MCP server search operations.

This module provides a SearchLogger class that captures all search operations
and writes them to a JSONL log file for analysis and auditing.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical_json_line(record: dict[str, Any]) -> bytes:
    """Convert a record to canonical JSON format with newline."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"


class SearchLogger:
    """
    Logger for MCP server search operations.

    Writes search logs in JSONL format, one JSON object per line.
    Each entry contains session_id, timestamp, tool, query, and results.
    """

    def __init__(self, log_dir: str | Path):
        """
        Initialize the SearchLogger.

        Args:
            log_dir: Directory where search_log.jsonl will be created.
                     The directory will be created if it does not exist.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "search_log.jsonl"

    def log_search(
        self,
        session_id: str,
        tool: str,
        query: str,
        results: list[dict[str, Any]]
    ) -> None:
        """
        Log a search operation.

        Args:
            session_id: UUID identifying the session
            tool: Search tool used (semantic_search, grep_search, cortex_query)
            query: The user's query text
            results: List of result objects with hash, file_path, rank, similarity
        """
        record = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "query": query,
            "results": results
        }

        line = _canonical_json_line(record)

        # Use low-level file operations for append-only writes
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        fd = os.open(str(self.log_path), flags, 0o644)
        try:
            written = 0
            while written < len(line):
                n = os.write(fd, line[written:])
                if n <= 0:
                    raise RuntimeError("failed to append search log entry")
                written += n
            os.fsync(fd)
        finally:
            os.close(fd)

    def read_all(self) -> list[dict[str, Any]]:
        """
        Read all log entries from the log file.

        Returns:
            List of log entry dictionaries
        """
        if not self.log_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(self.log_path, "rb") as f:
            for idx, raw_line in enumerate(f, start=1):
                if not raw_line.endswith(b"\n"):
                    raise ValueError(f"partial log line at {idx}")
                line = raw_line[:-1]
                if line == b"":
                    raise ValueError(f"empty log line at {idx}")
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception as e:
                    raise ValueError(f"invalid JSON at line {idx}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"log line {idx} is not an object")
                records.append(obj)
        return records


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # Self-test: Create logger, log 3 sample entries, validate
    import tempfile
    import shutil

    print("SearchLogger Self-Test")
    print("=" * 50)

    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp(prefix="search_logger_test_"))

    try:
        # Initialize logger
        logger = SearchLogger(test_dir)
        print(f"[OK] Created logger with log_dir: {test_dir}")

        # Generate a session ID
        session_id = str(uuid.uuid4())

        # Log entry 1: semantic_search
        logger.log_search(
            session_id=session_id,
            tool="semantic_search",
            query="How do I configure authentication?",
            results=[
                {
                    "hash": _compute_content_hash("auth config content"),
                    "file_path": "docs/auth/config.md",
                    "rank": 1,
                    "similarity": 0.92
                },
                {
                    "hash": _compute_content_hash("auth setup guide"),
                    "file_path": "docs/auth/setup.md",
                    "rank": 2,
                    "similarity": 0.85
                }
            ]
        )
        print("[OK] Logged semantic_search entry")

        # Log entry 2: grep_search
        logger.log_search(
            session_id=session_id,
            tool="grep_search",
            query="def authenticate",
            results=[
                {
                    "hash": _compute_content_hash("def authenticate(user):"),
                    "file_path": "src/auth/handler.py",
                    "rank": 1,
                    "similarity": 1.0
                }
            ]
        )
        print("[OK] Logged grep_search entry")

        # Log entry 3: cortex_query
        logger.log_search(
            session_id=session_id,
            tool="cortex_query",
            query="list all invariants",
            results=[
                {
                    "hash": _compute_content_hash("INV-001 content"),
                    "file_path": "CANON/INVARIANTS.md",
                    "rank": 1,
                    "similarity": 0.88
                },
                {
                    "hash": _compute_content_hash("INV-002 content"),
                    "file_path": "CANON/INVARIANTS.md",
                    "rank": 2,
                    "similarity": 0.82
                },
                {
                    "hash": _compute_content_hash("invariant tests"),
                    "file_path": "tests/test_invariants.py",
                    "rank": 3,
                    "similarity": 0.75
                }
            ]
        )
        print("[OK] Logged cortex_query entry")

        # Read back and validate
        entries = logger.read_all()
        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        print(f"[OK] Read back {len(entries)} entries")

        # Validate each entry has required fields
        required_fields = {"session_id", "timestamp", "tool", "query", "results"}
        for i, entry in enumerate(entries, start=1):
            missing = required_fields - set(entry.keys())
            assert not missing, f"Entry {i} missing fields: {missing}"

            # Validate results structure
            for r_idx, result in enumerate(entry["results"]):
                result_fields = {"hash", "file_path", "rank", "similarity"}
                result_missing = result_fields - set(result.keys())
                assert not result_missing, f"Entry {i} result {r_idx} missing: {result_missing}"

        print("[OK] All entries have valid structure")

        # Validate tools
        tools = [e["tool"] for e in entries]
        assert tools == ["semantic_search", "grep_search", "cortex_query"], f"Unexpected tools: {tools}"
        print("[OK] Tool types match expected values")

        # Validate JSON format (each line is valid JSON)
        with open(logger.log_path, "rb") as f:
            lines = f.readlines()
            assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
            for i, line in enumerate(lines, start=1):
                assert line.endswith(b"\n"), f"Line {i} does not end with newline"
                json.loads(line.decode("utf-8"))  # Will raise if invalid
        print("[OK] All lines are valid JSON with proper newlines")

        print("=" * 50)
        print("All tests PASSED!")
        print(f"Log file: {logger.log_path}")

        # Print sample content
        print("\nSample log content (first entry):")
        print(json.dumps(entries[0], indent=2))

    finally:
        # Clean up temporary directory
        shutil.rmtree(test_dir)
        print(f"\n[Cleanup] Removed test directory: {test_dir}")
