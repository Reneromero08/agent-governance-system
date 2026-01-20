"""
CORTEX Expansion Resolver (Phase 3.2.5 + Phase E)

Provides governed retrieval for expansion resolution.

Retrieval Order (Phase E - Vector Fallback Chain):
1. SPC resolution (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR) - Phase D
2. Main Cassette FTS (cassette network)
3. Local Index (symbol registry)
4. CAS (exact hash lookup) - Phase E.1
5. Vector Fallback (governed semantic search) - Phase E.2
6. Fail-closed if unresolvable

Phase E additions:
- E.1: Strict retrieval order enforcement
- E.2: Vector governance with hard token budgets
- E.3: ELO metadata tracking (no ranking influence)

This module bridges the context assembly pipeline with the CORTEX infrastructure.
"""

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from catalytic_chat.context_assembler import ContextExpansion
from catalytic_chat.mcp_integration import ChatToolExecutor, McpAccessError


# Forward references for type hints - actual imports are lazy
CassetteClient = None
CASResolver = None
VectorFallbackResolver = None
EloObserver = None


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
    Governed expansion resolver for context assembly.

    Implements retrieval order (Phase E - Vector Fallback Chain):
    1. SPC resolution (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR) - Phase D
    2. Main Cassette FTS (cassette network)
    3. Local Index (symbol registry for @SYMBOL refs)
    4. CAS (exact hash lookup) - Phase E.1
    5. Vector Fallback (governed semantic search) - Phase E.2
    6. Fail-closed if unresolvable

    Phase E governance:
    - Hard token budgets on vector retrieval (max 2000 tokens)
    - No trust bypass - all results hash-verified
    - ELO metadata tracking (does NOT affect ranking)
    """

    # SHA-256 hash pattern for CAS lookup
    HASH_PATTERN = re.compile(r'^(?:sha256:)?([0-9a-fA-F]{64})$')

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        tool_executor: Optional[ChatToolExecutor] = None,
        symbol_resolver: Optional[Any] = None,
        fail_on_unresolved: bool = True,
        enable_spc: bool = True,
        enable_vector_fallback: bool = True,
        enable_elo_observer: bool = True,
        session_id: Optional[str] = None
    ):
        """
        Initialize CORTEX expansion resolver.

        Args:
            repo_root: Repository root path
            tool_executor: ChatToolExecutor instance (lazy loaded if None)
            symbol_resolver: SymbolResolver instance (lazy loaded if None)
            fail_on_unresolved: If True, raise error on unresolvable symbols
            enable_spc: If True, enable SPC pointer resolution (Phase D)
            enable_vector_fallback: If True, enable vector fallback (Phase E)
            enable_elo_observer: If True, enable ELO metadata tracking (Phase E.3)
            session_id: Session ID for ELO tracking
        """
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[4]
        self.repo_root = repo_root

        self._tool_executor = tool_executor
        self._symbol_resolver = symbol_resolver
        self.fail_on_unresolved = fail_on_unresolved

        # Phase D: SPC integration
        self._enable_spc = enable_spc
        self._spc_bridge = None

        # Phase E: Vector Fallback Chain
        self._enable_vector_fallback = enable_vector_fallback
        self._enable_elo_observer = enable_elo_observer
        self._cas_resolver = None
        self._vector_fallback = None
        self._elo_observer = None
        self._session_id = session_id or "default-session"

        # Stats tracking (Phase E adds cas_hits, vector_fallback_hits)
        self._stats = {
            "spc_hits": 0,           # Phase D: SPC resolution hits
            "cortex_hits": 0,
            "cassette_hits": 0,
            "symbol_hits": 0,
            "cas_hits": 0,           # Phase E: CAS exact hash hits
            "vector_fallback_hits": 0,  # Phase E: Vector fallback hits
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

    @property
    def cassette_client(self):
        """Lazy load cassette client for main cassette network access."""
        if not hasattr(self, '_cassette_client') or self._cassette_client is None:
            from catalytic_chat.cassette_client import CassetteClient
            self._cassette_client = CassetteClient(
                repo_root=self.repo_root,
                tool_executor=self.tool_executor
            )
        return self._cassette_client

    @property
    def spc_bridge(self):
        """Lazy load SPC bridge for pointer resolution (Phase D)."""
        if self._spc_bridge is None and self._enable_spc:
            try:
                from catalytic_chat.spc_bridge import SPCBridge
                self._spc_bridge = SPCBridge(self.repo_root)
                # Auto-sync on first access
                self._spc_bridge.sync_handshake("resolver-session")
            except ImportError:
                # SPC not available, disable
                self._enable_spc = False
                return None
        return self._spc_bridge

    @property
    def cas_resolver(self):
        """Lazy load CAS resolver for exact hash lookup (Phase E.1)."""
        if self._cas_resolver is None:
            from catalytic_chat.cas_resolver import CASResolver
            self._cas_resolver = CASResolver(
                repo_root=self.repo_root,
                tool_executor=self.tool_executor
            )
        return self._cas_resolver

    @property
    def vector_fallback(self):
        """Lazy load vector fallback resolver (Phase E.2)."""
        if self._vector_fallback is None and self._enable_vector_fallback:
            try:
                from catalytic_chat.vector_fallback import VectorFallbackResolver
                self._vector_fallback = VectorFallbackResolver(
                    repo_root=self.repo_root,
                    tool_executor=self.tool_executor
                )
            except ImportError:
                self._enable_vector_fallback = False
                return None
        return self._vector_fallback

    @property
    def elo_observer(self):
        """Lazy load ELO observer for metadata tracking (Phase E.3)."""
        if self._elo_observer is None and self._enable_elo_observer:
            try:
                from catalytic_chat.elo_observer import EloObserver
                # ELO observer without DB connection for now (logs only)
                self._elo_observer = EloObserver(enable_elo_updates=False)
            except ImportError:
                self._enable_elo_observer = False
                return None
        return self._elo_observer

    def _looks_like_hash(self, symbol_id: str) -> bool:
        """Check if symbol_id looks like a SHA-256 hash."""
        return bool(self.HASH_PATTERN.match(symbol_id))

    def _notify_elo_observer(
        self,
        source: str,
        entity_id: str,
        rank: int = 1,
        was_used: bool = True
    ) -> None:
        """
        Notify ELO observer of retrieval event (Phase E.3).

        CRITICAL: This is called AFTER retrieval. It NEVER affects ranking.

        Args:
            source: Retrieval source (spc, cassette_fts, local_index, cas, vector_fallback)
            entity_id: ID of retrieved entity
            rank: Position in results (1 = first)
            was_used: Whether result was included in context
        """
        if not self._enable_elo_observer:
            return

        observer = self.elo_observer
        if observer is None:
            return

        observer.on_retrieval_complete(
            session_id=self._session_id,
            source=source,
            entity_id=entity_id,
            rank=rank,
            was_used=was_used
        )

    def _try_spc_resolution(self, symbol_id: str) -> Optional[str]:
        """
        Try SPC resolution for pointer-like symbols (Phase D.2).

        Supports:
        - SYMBOL_PTR: C, I, V, or CJK glyphs
        - HASH_PTR: sha256:abc123...
        - COMPOSITE_PTR: C3, C&I, L.C.3, C:build

        Args:
            symbol_id: Symbol or pointer to resolve

        Returns:
            Expanded content string, or None if not an SPC pointer
        """
        if not self._enable_spc:
            return None

        bridge = self.spc_bridge
        if bridge is None:
            return None

        if not bridge.is_spc_pointer(symbol_id):
            return None

        result = bridge.resolve_pointer(symbol_id)
        if result:
            return bridge.get_expansion_text(result)
        return None

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
                            data = json.loads(text)
                            # Extract results list from response dict
                            if isinstance(data, dict):
                                return data.get("results", [])
                            return data if isinstance(data, list) else []
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
                            data = json.loads(text)
                            # Extract results list from response dict
                            if isinstance(data, dict):
                                return data.get("results", [])
                            return data if isinstance(data, list) else []
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
                            data = json.loads(text)
                            # Extract results list from response dict
                            if isinstance(data, dict):
                                return data.get("results", [])
                            return data if isinstance(data, list) else []
                        except json.JSONDecodeError:
                            return []
            return []
        except McpAccessError:
            return []

    def _try_cassette_network_symbol(self, symbol_id: str, limit: int = 3) -> Optional[str]:
        """
        Try cassette network with symbol-aware query (Phase B.2).

        For @CANON/INVARIANTS:
        1. Extract base name: "INVARIANTS"
        2. Extract path hint: "CANON" -> cassette "canon"
        3. Query cassette network with targeted search

        This is more targeted than raw text search, improving resolution
        for well-formed symbol references.

        Args:
            symbol_id: Symbol ID like "@CANON/INVARIANTS"
            limit: Max results to consider

        Returns:
            Content string or None if not found
        """
        try:
            content = self.cassette_client.resolve_symbol(symbol_id, limit=limit)
            return content
        except Exception:
            return None

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
        is_explicit_reference: bool = True,
        remaining_budget: Optional[int] = None
    ) -> RetrievalResult:
        """
        Resolve a symbol to content using governed retrieval chain.

        Retrieval order (Phase E - Vector Fallback Chain):
        1. SPC resolution (SYMBOL_PTR, HASH_PTR, COMPOSITE_PTR) - Phase D
        2. Main Cassette FTS (cassette network)
        3. Local Index (symbol registry for @SYMBOL refs)
        4. CAS (exact hash lookup) - Phase E.1
        5. Vector Fallback (governed semantic search) - Phase E.2
        6. Fail-closed

        Args:
            symbol_id: Symbol ID or query to resolve
            query_hint: Optional search query hint
            is_explicit_reference: Whether this is an explicit user reference
            remaining_budget: Tokens available for vector fallback (Phase E)

        Returns:
            RetrievalResult with content and metadata

        Raises:
            CortexRetrievalError: If fail_on_unresolved and cannot resolve
        """
        self._stats["total_queries"] += 1
        retrieval_path = []

        # 1. Try SPC resolution FIRST (Phase D - highest priority)
        if self._enable_spc:
            retrieval_path.append("spc_resolve")
            content = self._try_spc_resolution(symbol_id)
            if content:
                self._stats["spc_hits"] += 1
                content_hash = self._compute_hash(content)
                self._notify_elo_observer("spc", symbol_id, rank=1)
                return RetrievalResult(
                    symbol_id=symbol_id,
                    content=content,
                    source="spc",
                    retrieval_path=retrieval_path,
                    content_hash=content_hash,
                    retrieved_at=self._get_timestamp()
                )

        # Determine query string
        query = query_hint if query_hint else symbol_id

        # 2. Try Main Cassette FTS (cassette network)
        # Try symbol-aware search first for @SYMBOL references
        if symbol_id.startswith("@") or "/" in symbol_id:
            retrieval_path.append("cassette_network_symbol")
            content = self._try_cassette_network_symbol(symbol_id)
            if content:
                self._stats["cassette_hits"] += 1
                content_hash = self._compute_hash(content)
                self._notify_elo_observer("cassette_fts", symbol_id, rank=1)
                return RetrievalResult(
                    symbol_id=symbol_id,
                    content=content,
                    source="cassette",
                    retrieval_path=retrieval_path,
                    content_hash=content_hash,
                    retrieved_at=self._get_timestamp()
                )

        # Try general FTS
        results = self._try_cassette_network(query)
        retrieval_path.append("cassette_network")
        if results:
            self._stats["cassette_hits"] += 1
            content = self._format_results(results)
            content_hash = self._compute_hash(content)
            self._notify_elo_observer("cassette_fts", query, rank=1)
            return RetrievalResult(
                symbol_id=symbol_id,
                content=content,
                source="cassette",
                retrieval_path=retrieval_path,
                content_hash=content_hash,
                retrieved_at=self._get_timestamp()
            )

        # 3. Try Local Index (symbol resolver for @SYMBOL references)
        if symbol_id.startswith("@"):
            retrieval_path.append("symbol_registry")
            content = self._try_symbol_resolver(symbol_id)
            if content:
                self._stats["symbol_hits"] += 1
                content_hash = self._compute_hash(content)
                self._notify_elo_observer("local_index", symbol_id, rank=1)
                return RetrievalResult(
                    symbol_id=symbol_id,
                    content=content,
                    source="symbol_registry",
                    retrieval_path=retrieval_path,
                    content_hash=content_hash,
                    retrieved_at=self._get_timestamp()
                )

        # 4. Try CAS (exact hash lookup) - Phase E.1
        if self._looks_like_hash(symbol_id):
            retrieval_path.append("cas_lookup")
            result = self.cas_resolver.lookup(symbol_id)
            if result and result.verified:
                self._stats["cas_hits"] += 1
                self._notify_elo_observer("cas", result.content_hash, rank=1)
                return RetrievalResult(
                    symbol_id=symbol_id,
                    content=result.content,
                    source="cas",
                    retrieval_path=retrieval_path,
                    content_hash=result.content_hash,
                    retrieved_at=self._get_timestamp()
                )

        # 5. Try Vector Fallback (governed semantic search) - Phase E.2
        # Only if budget is available and vector fallback is enabled
        if self._enable_vector_fallback and remaining_budget and remaining_budget > 0:
            retrieval_path.append("vector_fallback")
            fallback = self.vector_fallback
            if fallback:
                vector_results = fallback.search(query, remaining_budget)
                if vector_results:
                    self._stats["vector_fallback_hits"] += 1
                    # Notify ELO for each result
                    for i, vr in enumerate(vector_results):
                        self._notify_elo_observer(
                            "vector_fallback",
                            vr.content_hash,
                            rank=i + 1,
                            was_used=True
                        )
                    # Format vector results
                    content = self._format_vector_results(vector_results)
                    content_hash = self._compute_hash(content)
                    return RetrievalResult(
                        symbol_id=symbol_id,
                        content=content,
                        source="vector_fallback",
                        retrieval_path=retrieval_path,
                        content_hash=content_hash,
                        retrieved_at=self._get_timestamp()
                    )

        # 6. Fail-closed
        self._stats["failures"] += 1
        if self.fail_on_unresolved:
            raise CortexRetrievalError(
                f"Failed to resolve '{symbol_id}' via governed retrieval chain. "
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

    def _format_vector_results(self, results: List[Any]) -> str:
        """Format vector search results into content string (Phase E)."""
        if not results:
            return ""

        formatted_parts = []
        for r in results:
            if hasattr(r, 'content'):
                formatted_parts.append(r.content)
            elif isinstance(r, dict) and 'content' in r:
                formatted_parts.append(r['content'])
            else:
                formatted_parts.append(str(r))

        return "\n\n---\n\n".join(formatted_parts)

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
        """Get resolver statistics (Phase E updated)."""
        total_hits = (
            self._stats["spc_hits"] +
            self._stats["cortex_hits"] +
            self._stats["cassette_hits"] +
            self._stats["symbol_hits"] +
            self._stats["cas_hits"] +
            self._stats["vector_fallback_hits"]
        )
        return {
            **self._stats,
            "hit_rate": total_hits / max(1, self._stats["total_queries"])
        }
