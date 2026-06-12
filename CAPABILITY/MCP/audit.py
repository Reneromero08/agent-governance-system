#!/usr/bin/env python3
"""
Session audit tracking for the AGS MCP server (E.1.2).

Extracted from server.py. Wraps CAPABILITY/PRIMITIVES/session_auditor.py and
maps MCP tool calls to ELO-relevant events (file accesses, ADR reads, symbol
expansions, search counts). Every failure degrades to a stderr warning;
auditing must never break a tool call.
"""

import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, Optional


class SessionAuditTracker:
    """Tracks file/symbol/search access for ELO scoring (E.1.2)."""

    def __init__(
        self,
        capability_root: Path,
        navigation_root: Path,
        project_root: Path,
        canon_resolver: Optional[Callable[[str], Optional[Path]]] = None,
    ):
        """Initialize the underlying SessionAuditor and start a session.

        The SessionAuditor tracks:
        - Files accessed via MCP tools (canon_read, context_search, etc.)
        - ADR reads
        - Symbol expansions (codebook_lookup)
        - Search query counts (semantic vs keyword)

        canon_resolver maps a bare canon name (e.g. CONTRACT) to its file
        path under LAW/CANON (bucket layout); used to audit canon_read.
        """
        self.session_auditor = None
        self.available = False
        self._project_root = project_root
        self._canon_resolver = canon_resolver
        try:
            # Add CAPABILITY to path for import
            sys.path.insert(0, str(capability_root))
            from PRIMITIVES.session_auditor import SessionAuditor

            # Initialize auditor with log dir and agent ID
            log_dir = navigation_root / "CORTEX" / "_generated"
            log_dir.mkdir(parents=True, exist_ok=True)

            self.session_auditor = SessionAuditor(
                log_dir=str(log_dir),
                agent_id="mcp-server"
            )
            # Start the session
            self.session_auditor.start_session()
            self.available = True
            print(f"[INFO] Session auditor initialized (session: {self.session_auditor.current_session_id})", file=sys.stderr)
        except Exception as e:
            self.session_auditor = None
            self.available = False
            print(f"[WARNING] Session auditor unavailable: {e}", file=sys.stderr)

    def end_session(self) -> None:
        """End the session audit and write to log file."""
        if self.session_auditor and self.session_auditor.is_active:
            try:
                entry = self.session_auditor.end_session()
                print(f"[INFO] Session audit complete: {entry.search_queries} searches, {len(entry.files_accessed)} files", file=sys.stderr)
            except Exception as e:
                print(f"[WARNING] Failed to end session audit: {e}", file=sys.stderr)

    def file_access(self, file_path: str) -> None:
        """Record a file access for ELO tracking."""
        if self.session_auditor and self.available:
            try:
                # Normalize to relative path from project root
                path = Path(file_path)
                if path.is_absolute():
                    try:
                        rel_path = path.relative_to(self._project_root)
                        file_path = str(rel_path)
                    except ValueError:
                        pass  # Keep absolute path if not under project root
                self.session_auditor.record_file_access(file_path)
            except Exception as e:
                print(f"[WARNING] Audit failure in file_access: {e}", file=sys.stderr)

    def adr_read(self, adr_id: str) -> None:
        """Record an ADR read for ELO tracking."""
        if self.session_auditor and self.available:
            try:
                self.session_auditor.record_adr_read(adr_id)
            except Exception as e:
                print(f"[WARNING] Audit failure in adr_read: {e}", file=sys.stderr)

    def symbol_expansion(self, symbol: str) -> None:
        """Record a symbol expansion for ELO tracking."""
        if self.session_auditor and self.available:
            try:
                self.session_auditor.record_symbol_expansion(symbol)
            except Exception as e:
                print(f"[WARNING] Audit failure in symbol_expansion: {e}", file=sys.stderr)

    def search(self, is_semantic: bool) -> None:
        """Record a search for ELO tracking."""
        if self.session_auditor and self.available:
            try:
                self.session_auditor.record_search(is_semantic=is_semantic)
            except Exception as e:
                print(f"[WARNING] Audit failure in search: {e}", file=sys.stderr)

    def track_tool_access(self, tool_name: str, arguments: Dict, result: Dict) -> None:
        """Track file/symbol/search access based on tool call (E.1.2).

        Maps tool calls to ELO-relevant events:
        - canon_read -> file access + potential ADR read
        - context_search/context_review -> file access (from results)
        - codebook_lookup -> symbol expansion
        - cassette_network_query/memory_query/skill_discovery -> semantic search
        - find_related -> semantic search
        """
        try:
            # Canon read - track file access
            if tool_name == "canon_read":
                file_name = arguments.get("file", "")
                if file_name and self._canon_resolver:
                    canon_path = self._canon_resolver(file_name.upper())
                    if canon_path:
                        self.file_access(
                            str(canon_path.relative_to(self._project_root).as_posix())
                        )

            # Context search/review - track file accesses from results
            elif tool_name in ("context_search", "context_review"):
                self.search(is_semantic=False)  # Keyword search
                # Extract file paths from result
                try:
                    content = result.get("content", [])
                    if content and content[0].get("type") == "text":
                        records = json.loads(content[0].get("text", "[]"))
                        for record in records:
                            if isinstance(record, dict) and "path" in record:
                                self.file_access(record["path"])
                                # Check if it's an ADR
                                path = record.get("path", "")
                                if "ADR-" in path:
                                    adr_match = re.search(r"ADR-(\d+)", path)
                                    if adr_match:
                                        self.adr_read(f"ADR-{adr_match.group(1)}")
                except Exception as e:
                    print(f"[WARNING] Audit failure in track_tool_access (ADR read): {e}", file=sys.stderr)

            # Codebook lookup - track symbol expansion
            elif tool_name == "codebook_lookup":
                symbol_id = arguments.get("id", "")
                if symbol_id:
                    self.symbol_expansion(symbol_id)
                # Also track as keyword search if expand=True
                if arguments.get("expand"):
                    self.search(is_semantic=False)

            # Semantic search tools
            elif tool_name in ("cassette_network_query", "memory_query", "skill_discovery",
                               "find_related", "semantic_neighbors"):
                self.search(is_semantic=True)
                # Extract file paths from cassette_network_query results
                if tool_name == "cassette_network_query":
                    try:
                        content = result.get("content", [])
                        if content and content[0].get("type") == "text":
                            data = json.loads(content[0].get("text", "{}"))
                            for res in data.get("results", []):
                                path = res.get("path", res.get("file_path", ""))
                                if path:
                                    self.file_access(path)
                    except Exception as e:
                        print(f"[WARNING] Audit failure in track_tool_access (cassette network query): {e}", file=sys.stderr)

            # ADR creation - track ADR read (reviewing existing ADRs)
            elif tool_name == "adr_create":
                # The ADR being created
                adr_id = arguments.get("id", "")
                if adr_id:
                    self.adr_read(adr_id)

        except Exception as e:
            print(f"[WARNING] Audit failure in track_tool_access (overall): {e}", file=sys.stderr)
