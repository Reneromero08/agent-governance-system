"""
Cassette Client (Phase B.1)

Thin read-only client for querying the main cassette network.
CAT_CHAT uses this to read content from NAVIGATION/CORTEX/cassettes/*.db
while keeping all CAT_CHAT state local in _generated/cat_chat.db.

Key Design Decisions:
1. Read-only: No write methods exist (enforces LAB isolation)
2. Uses MCP: Queries via ChatToolExecutor, not direct DB access
3. Lazy-loaded: Instantiated on first use in CortexExpansionResolver
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from catalytic_chat.mcp_integration import ChatToolExecutor, McpAccessError


@dataclass
class CassetteResult:
    """Result from cassette network query."""
    content: str
    score: float
    cassette_id: str
    path: str
    chunk_id: Optional[int] = None
    hash: Optional[str] = None


class CassetteClientError(Exception):
    """Raised when cassette client operations fail."""
    pass


class CassetteClient:
    """
    Thin read-only client for querying main cassettes.

    Uses MCP's cassette_network_query tool under the hood.
    Does NOT instantiate SemanticNetworkHub directly (security boundary).

    This class has NO write methods by design - CAT_CHAT only reads
    from main cassettes, all writes stay in _generated/cat_chat.db.
    """

    # Map path prefixes to cassette IDs
    PATH_TO_CASSETTE: Dict[str, str] = {
        "LAW": "canon",
        "CANON": "canon",
        "CONTEXT": "governance",
        "CAPABILITY": "capability",
        "NAVIGATION": "navigation",
        "DIRECTION": "direction",
        "THOUGHT": "thought",
        "MEMORY": "memory",
        "INBOX": "inbox",
    }

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        tool_executor: Optional[ChatToolExecutor] = None
    ):
        """
        Initialize cassette client.

        Args:
            repo_root: Repository root path
            tool_executor: ChatToolExecutor instance (created if None)
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]
        self.repo_root = repo_root

        self._tool_executor = tool_executor

    @property
    def tool_executor(self) -> ChatToolExecutor:
        """Lazy load tool executor."""
        if self._tool_executor is None:
            self._tool_executor = ChatToolExecutor(self.repo_root)
        return self._tool_executor

    def query(
        self,
        query: str,
        limit: int = 10,
        cassettes: Optional[List[str]] = None,
        capability: Optional[str] = None
    ) -> List[CassetteResult]:
        """
        Query main cassettes via MCP.

        Args:
            query: Search query (will be normalized for FTS5)
            limit: Max results
            cassettes: Filter to specific cassettes (e.g., ["canon", "governance"])
            capability: Filter by capability (e.g., "research", "ast")

        Returns:
            List of CassetteResult with content, score, cassette_id, path
        """
        if not query or not query.strip():
            return []

        # Normalize query for FTS5
        normalized_query = self.normalize_fts_query(query)

        # Build arguments
        args: Dict[str, Any] = {
            "query": normalized_query,
            "limit": limit
        }
        if cassettes:
            args["cassettes"] = cassettes
        if capability:
            args["capability"] = capability

        try:
            result = self.tool_executor.execute_tool(
                "cassette_network_query",
                args
            )

            if not result or "content" not in result:
                return []

            content = result["content"]
            if not isinstance(content, list) or len(content) == 0:
                return []

            text = content[0].get("text", "")
            if not text:
                return []

            # Parse JSON response
            data = json.loads(text)
            raw_results = data.get("results", [])

            # Convert to CassetteResult objects
            results = []
            for r in raw_results:
                results.append(CassetteResult(
                    content=r.get("content", r.get("text", r.get("snippet", ""))),
                    score=float(r.get("score", r.get("similarity", 0.0))),
                    cassette_id=r.get("cassette_id", ""),
                    path=r.get("path", r.get("file_path", "")),
                    chunk_id=r.get("chunk_id"),
                    hash=r.get("hash", r.get("content_hash"))
                ))

            return results

        except McpAccessError:
            return []
        except json.JSONDecodeError:
            return []

    def resolve_symbol(
        self,
        symbol: str,
        limit: int = 5
    ) -> Optional[str]:
        """
        Resolve a symbol by searching main cassettes.

        Extracts the meaningful part from symbol references like:
        - @CANON/INVARIANTS -> searches "INVARIANTS" in canon cassette
        - @THOUGHT/LAB/NOTES -> searches "NOTES" in thought cassette
        - INVARIANTS -> searches "INVARIANTS" in all cassettes

        Args:
            symbol: Symbol like "@CANON/INVARIANTS" or "INVARIANTS"

        Returns:
            Content string or None if not found
        """
        # Parse symbol to extract query and target cassette
        query, target_cassette = self._parse_symbol(symbol)

        if not query:
            return None

        # Build cassette filter
        cassettes = [target_cassette] if target_cassette else None

        # Query cassettes
        results = self.query(query, limit=limit, cassettes=cassettes)

        if not results:
            return None

        # Return concatenated content from top results
        contents = [r.content for r in results if r.content]
        if not contents:
            return None

        return "\n\n---\n\n".join(contents)

    def _parse_symbol(self, symbol: str) -> tuple[str, Optional[str]]:
        """
        Parse symbol to extract query and target cassette.

        Args:
            symbol: Symbol reference

        Returns:
            (query_string, cassette_id_or_none)
        """
        # Strip @ prefix if present
        if symbol.startswith("@"):
            symbol = symbol[1:]

        # Handle path-like symbols: CANON/INVARIANTS, THOUGHT/LAB/NOTES
        if "/" in symbol:
            parts = symbol.split("/")
            # First part might map to a cassette
            first_part = parts[0].upper()
            cassette_id = self.PATH_TO_CASSETTE.get(first_part)

            # Last part is the search query
            query = parts[-1]

            return query, cassette_id
        else:
            # Simple symbol - search all cassettes
            return symbol, None

    def get_network_status(self) -> Dict[str, Any]:
        """
        Get status of main cassette network.

        Returns:
            Dict with cassette counts, availability info
        """
        try:
            result = self.tool_executor.execute_tool(
                "semantic_stats",
                {}
            )

            if not result or "content" not in result:
                return {"available": False, "error": "No response"}

            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")
                if text:
                    return json.loads(text)

            return {"available": False, "error": "Invalid response format"}

        except McpAccessError as e:
            return {"available": False, "error": str(e)}
        except json.JSONDecodeError:
            return {"available": False, "error": "Invalid JSON response"}

    @staticmethod
    def normalize_fts_query(query: str) -> str:
        """
        Normalize query for FTS5 MATCH.

        FTS5 special characters that need escaping: * " - : ^ ( ) AND OR NOT

        Args:
            query: Raw query string

        Returns:
            FTS5-safe query string
        """
        if not query:
            return ""

        # Strip @ prefix (common in symbol references)
        if query.startswith("@"):
            query = query[1:]

        # Extract last path component for path-like queries
        if "/" in query and not " " in query:
            query = query.split("/")[-1]

        # Escape FTS5 special characters
        # We escape: " * ^ -
        # We convert AND OR NOT to lowercase (so they're not operators)
        result = query

        # Replace special characters with spaces (safer than escaping)
        result = re.sub(r'["\*\^]', ' ', result)

        # Handle leading minus (negation) - only escape if at word start
        result = re.sub(r'(^|\s)-', r'\1', result)

        # Convert FTS operators to lowercase
        result = re.sub(r'\bAND\b', 'and', result)
        result = re.sub(r'\bOR\b', 'or', result)
        result = re.sub(r'\bNOT\b', 'not', result)

        # Collapse multiple spaces
        result = re.sub(r'\s+', ' ', result).strip()

        return result


# Read-only verification - this module intentionally has no write methods
assert not any(m.startswith('write') or m.startswith('save') or m.startswith('create')
               for m in dir(CassetteClient) if not m.startswith('_')), \
    "CassetteClient must be read-only"
