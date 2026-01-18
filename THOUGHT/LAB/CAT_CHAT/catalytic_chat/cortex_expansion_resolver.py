"""
CORTEX Expansion Resolver (Phase 3.2.5)

Provides CORTEX-first retrieval for expansion resolution.

Retrieval Order:
1. CORTEX (semantic search / cassette network)
2. Symbol Registry (local symbol resolution)
3. Fail-closed if unresolvable

This module bridges the context assembly pipeline with the CORTEX infrastructure.
"""

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from catalytic_chat.context_assembler import ContextExpansion
from catalytic_chat.mcp_integration import ChatToolExecutor, McpAccessError


@dataclass
class RetrievalResult:
    """Result of a CORTEX retrieval operation."""
    symbol_id: str
    content: str
    source: str  # "cortex", "symbol_registry", "cache"
    retrieval_path: List[str]  # Path taken: ["cortex_query", "cassette_network"]
    content_hash: str
    retrieved_at: str


class CortexRetrievalError(Exception):
    """Raised when CORTEX retrieval fails and cannot recover."""
    pass


class CortexExpansionResolver:
    """
    CORTEX-first expansion resolver for context assembly.

    Implements 3.2.5 retrieval order:
    1. CORTEX semantic search
    2. Cassette network query
    3. Symbol registry fallback
    4. Fail-closed if unresolvable
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        tool_executor: Optional[ChatToolExecutor] = None,
        symbol_resolver: Optional[Any] = None,
        fail_on_unresolved: bool = True
    ):
        """
        Initialize CORTEX expansion resolver.

        Args:
            repo_root: Repository root path
            tool_executor: ChatToolExecutor instance (lazy loaded if None)
            symbol_resolver: SymbolResolver instance (lazy loaded if None)
            fail_on_unresolved: If True, raise error on unresolvable symbols
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]
        self.repo_root = repo_root

        self._tool_executor = tool_executor
        self._symbol_resolver = symbol_resolver
        self.fail_on_unresolved = fail_on_unresolved

        # Stats tracking
        self._stats = {
            "cortex_hits": 0,
            "cassette_hits": 0,
            "symbol_hits": 0,
            "failures": 0,
            "total_queries": 0
        }

    @property
    def tool_executor(self) -> ChatToolExecutor:
        """Lazy load tool executor."""
        if self._tool_executor is None:
            self._tool_executor = ChatToolExecutor(self.repo_root)
        return self._tool_executor

    @property
    def symbol_resolver(self):
        """Lazy load symbol resolver."""
        if self._symbol_resolver is None:
            from catalytic_chat.symbol_resolver import SymbolResolver
            from catalytic_chat.symbol_registry import SymbolRegistry

            registry = SymbolRegistry(self.repo_root, "sqlite")
            self._symbol_resolver = SymbolResolver(
                self.repo_root, "sqlite", registry
            )
        return self._symbol_resolver

    def _get_timestamp(self) -> str:
        """Get ISO8601 timestamp."""
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _try_cortex_query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Try CORTEX query for semantic search.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of results or empty list on failure
        """
        try:
            result = self.tool_executor.execute_tool(
                "cortex_query",
                {"query": query, "limit": limit}
            )
            if result and "content" in result:
                # Parse the content if it's a list of results
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    # MCP returns content as list of content blocks
                    text = content[0].get("text", "")
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return []
            return []
        except McpAccessError:
            return []

    def _try_cassette_network(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Try cassette network query.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of results or empty list on failure
        """
        try:
            result = self.tool_executor.execute_tool(
                "cassette_network_query",
                {"query": query, "limit": limit}
            )
            if result and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text", "")
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return []
            return []
        except McpAccessError:
            return []

    def _try_semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Try semantic search as final CORTEX fallback.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of results or empty list on failure
        """
        try:
            result = self.tool_executor.execute_tool(
                "semantic_search",
                {"query": query, "limit": limit}
            )
            if result and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text", "")
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return []
            return []
        except McpAccessError:
            return []

    def _try_symbol_resolver(self, symbol_id: str) -> Optional[str]:
        """
        Try symbol resolver for local resolution.

        Args:
            symbol_id: Symbol ID to resolve

        Returns:
            Content or None on failure
        """
        try:
            content, _cache_hit = self.symbol_resolver.resolve(symbol_id)
            return content
        except Exception:
            return None

    def resolve_expansion(
        self,
        symbol_id: str,
        query_hint: Optional[str] = None,
        is_explicit_reference: bool = True
    ) -> RetrievalResult:
        """
        Resolve a symbol to content using CORTEX-first retrieval.

        Retrieval order:
        1. CORTEX query (if query_hint provided or symbol looks like a query)
        2. Cassette network
        3. Semantic search
        4. Symbol registry
        5. Fail-closed

        Args:
            symbol_id: Symbol ID or query to resolve
            query_hint: Optional search query hint
            is_explicit_reference: Whether this is an explicit user reference

        Returns:
            RetrievalResult with content and metadata

        Raises:
            CortexRetrievalError: If fail_on_unresolved and cannot resolve
        """
        self._stats["total_queries"] += 1
        retrieval_path = []

        # Determine query string
        query = query_hint if query_hint else symbol_id

        # 1. Try CORTEX query
        results = self._try_cortex_query(query)
        retrieval_path.append("cortex_query")
        if results:
            self._stats["cortex_hits"] += 1
            content = self._format_results(results)
            return RetrievalResult(
                symbol_id=symbol_id,
                content=content,
                source="cortex",
                retrieval_path=retrieval_path,
                content_hash=self._compute_hash(content),
                retrieved_at=self._get_timestamp()
            )

        # 2. Try cassette network
        results = self._try_cassette_network(query)
        retrieval_path.append("cassette_network")
        if results:
            self._stats["cassette_hits"] += 1
            content = self._format_results(results)
            return RetrievalResult(
                symbol_id=symbol_id,
                content=content,
                source="cassette",
                retrieval_path=retrieval_path,
                content_hash=self._compute_hash(content),
                retrieved_at=self._get_timestamp()
            )

        # 3. Try semantic search
        results = self._try_semantic_search(query)
        retrieval_path.append("semantic_search")
        if results:
            self._stats["cortex_hits"] += 1
            content = self._format_results(results)
            return RetrievalResult(
                symbol_id=symbol_id,
                content=content,
                source="cortex",
                retrieval_path=retrieval_path,
                content_hash=self._compute_hash(content),
                retrieved_at=self._get_timestamp()
            )

        # 4. Try symbol resolver (for @SYMBOL references)
        if symbol_id.startswith("@"):
            retrieval_path.append("symbol_registry")
            content = self._try_symbol_resolver(symbol_id)
            if content:
                self._stats["symbol_hits"] += 1
                return RetrievalResult(
                    symbol_id=symbol_id,
                    content=content,
                    source="symbol_registry",
                    retrieval_path=retrieval_path,
                    content_hash=self._compute_hash(content),
                    retrieved_at=self._get_timestamp()
                )

        # 5. Fail-closed
        self._stats["failures"] += 1
        if self.fail_on_unresolved:
            raise CortexRetrievalError(
                f"Failed to resolve '{symbol_id}' via CORTEX or symbol registry. "
                f"Retrieval path: {' -> '.join(retrieval_path)}"
            )

        # Return empty result if not failing
        return RetrievalResult(
            symbol_id=symbol_id,
            content="",
            source="unresolved",
            retrieval_path=retrieval_path,
            content_hash=self._compute_hash(""),
            retrieved_at=self._get_timestamp()
        )

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into content string."""
        if not results:
            return ""

        formatted_parts = []
        for r in results[:5]:  # Limit to top 5
            # Handle different result formats
            if "content" in r:
                formatted_parts.append(r["content"])
            elif "text" in r:
                formatted_parts.append(r["text"])
            elif "snippet" in r:
                formatted_parts.append(r["snippet"])
            else:
                # Serialize the whole result
                formatted_parts.append(json.dumps(r, indent=2))

        return "\n\n---\n\n".join(formatted_parts)

    def resolve_to_expansion(
        self,
        symbol_id: str,
        query_hint: Optional[str] = None,
        is_explicit_reference: bool = True,
        priority: int = 0
    ) -> ContextExpansion:
        """
        Resolve symbol to a ContextExpansion for the assembler.

        Args:
            symbol_id: Symbol ID to resolve
            query_hint: Optional search hint
            is_explicit_reference: Whether explicitly referenced
            priority: Priority for optional extras

        Returns:
            ContextExpansion ready for context assembly
        """
        result = self.resolve_expansion(
            symbol_id, query_hint, is_explicit_reference
        )

        return ContextExpansion(
            symbol_id=symbol_id,
            content=result.content,
            is_explicit_reference=is_explicit_reference,
            priority=priority
        )

    def resolve_batch(
        self,
        symbols: List[str],
        is_explicit: bool = True
    ) -> List[ContextExpansion]:
        """
        Resolve multiple symbols to expansions.

        Args:
            symbols: List of symbol IDs to resolve
            is_explicit: Whether these are explicit references

        Returns:
            List of ContextExpansion objects
        """
        expansions = []
        for i, symbol in enumerate(symbols):
            try:
                exp = self.resolve_to_expansion(
                    symbol,
                    is_explicit_reference=is_explicit,
                    priority=len(symbols) - i  # Higher priority for earlier items
                )
                expansions.append(exp)
            except CortexRetrievalError:
                if self.fail_on_unresolved:
                    raise
                # Skip unresolved if not failing
                continue
        return expansions

    def compute_corpus_snapshot_id(self) -> str:
        """
        Compute a corpus snapshot ID for deterministic replay.

        Combines:
        - CORTEX index state (via stats query)
        - Symbol registry state

        Returns:
            SHA-256 hash representing current corpus state
        """
        parts = []

        # Get CORTEX stats if available
        try:
            result = self.tool_executor.execute_tool(
                "semantic_stats",
                {}
            )
            if result and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    parts.append(content[0].get("text", ""))
        except McpAccessError:
            parts.append("cortex_unavailable")

        # Get symbol registry hash
        try:
            # The symbol_resolver has access to the indexer which has index_hash
            index_hash = getattr(
                self.symbol_resolver.section_indexer,
                'index_hash',
                'unknown'
            )
            parts.append(f"index:{index_hash}")
        except Exception:
            parts.append("index:unknown")

        # Combine and hash
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]

    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        return {
            **self._stats,
            "hit_rate": (
                (self._stats["cortex_hits"] + self._stats["cassette_hits"] +
                 self._stats["symbol_hits"]) / max(1, self._stats["total_queries"])
            )
        }
