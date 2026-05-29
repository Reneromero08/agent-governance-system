"""
CAS Resolver (Phase E.1)

Content-Addressable Storage lookup by exact SHA-256 hash.
Third priority in the retrieval chain (after FTS and Local Index).

This module provides exact hash lookup as an alternative to fuzzy search.
If you have the exact content hash, CAS is faster and more precise.

Retrieval Order:
1. SPC (existing)
2. Main Cassette FTS
3. Local Index
4. CAS (exact hash) <- THIS MODULE
5. Vector Fallback
6. Fail-closed
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from catalytic_chat.mcp_integration import ChatToolExecutor, McpAccessError


@dataclass
class CASResult:
    """Result from CAS lookup."""
    content: str
    content_hash: str
    verified: bool
    source: str


class CASResolverError(Exception):
    """Raised when CAS operations fail."""
    pass


class CASResolver:
    """
    Content-Addressable Storage resolver.

    Provides exact hash lookup for content retrieval.
    All results are verified - content must match the requested hash.

    NO TRUST BYPASS: Even if the source claims the content matches,
    we verify the hash ourselves before returning.
    """

    # Valid SHA-256 hex pattern (64 hex chars)
    HASH_PATTERN = re.compile(r'^[0-9a-fA-F]{64}$')

    # Also accept sha256: prefix
    PREFIXED_PATTERN = re.compile(r'^sha256:([0-9a-fA-F]{64})$')

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        tool_executor: Optional[ChatToolExecutor] = None
    ):
        """
        Initialize CAS resolver.

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

    def normalize_hash(self, hash_input: str) -> Optional[str]:
        """
        Normalize hash input to lowercase 64-char hex.

        Accepts:
        - Raw 64-char hex: "abc123..."
        - Prefixed: "sha256:abc123..."

        Returns:
            Normalized lowercase hex hash, or None if invalid
        """
        if not hash_input:
            return None

        hash_input = hash_input.strip()

        # Check for sha256: prefix
        prefixed_match = self.PREFIXED_PATTERN.match(hash_input)
        if prefixed_match:
            return prefixed_match.group(1).lower()

        # Check for raw hash
        if self.HASH_PATTERN.match(hash_input):
            return hash_input.lower()

        return None

    def is_valid_hash(self, hash_input: str) -> bool:
        """Check if input is a valid SHA-256 hash format."""
        return self.normalize_hash(hash_input) is not None

    def compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def lookup(self, content_hash: str) -> Optional[CASResult]:
        """
        Lookup content by exact SHA-256 hash.

        Args:
            content_hash: SHA-256 hash (with or without sha256: prefix)

        Returns:
            CASResult if found and verified, None otherwise

        Note:
            This method verifies the content hash before returning.
            NO TRUST BYPASS - we don't trust the source's claimed hash.
        """
        normalized = self.normalize_hash(content_hash)
        if not normalized:
            return None

        # Try to find content by hash via cassette network
        result = self._query_by_hash(normalized)
        if not result:
            return None

        # CRITICAL: Verify content matches hash (NO TRUST BYPASS)
        actual_hash = self.compute_hash(result['content'])
        if actual_hash != normalized:
            # Hash mismatch - fail closed
            return None

        return CASResult(
            content=result['content'],
            content_hash=normalized,
            verified=True,
            source=result.get('source', 'cassette_network')
        )

    def _query_by_hash(self, hash_hex: str) -> Optional[Dict[str, Any]]:
        """
        Query cassette network for content by hash.

        Uses cassette_network_query with hash filter.
        """
        try:
            # Query using the hash as the query term
            # The cassette network can find content by hash in chunk metadata
            result = self.tool_executor.execute_tool(
                "cassette_network_query",
                {
                    "query": f"hash:{hash_hex}",
                    "limit": 1
                }
            )

            if not result:
                return None

            # Parse MCP result
            content_items = result.get('content', [])
            if not content_items:
                return None

            # Get first result
            first = content_items[0]
            if isinstance(first, dict) and 'text' in first:
                # MCP returns {"type": "text", "text": "..."}
                return {'content': first['text'], 'source': 'cassette_network'}
            elif isinstance(first, str):
                return {'content': first, 'source': 'cassette_network'}

            return None

        except McpAccessError:
            # MCP query failed, return None (not found)
            return None
        except Exception:
            # Other errors, fail closed
            return None

    def verify_content(self, content: str, expected_hash: str) -> bool:
        """
        Verify that content matches expected hash.

        Args:
            content: Content to verify
            expected_hash: Expected SHA-256 hash

        Returns:
            True if hash matches, False otherwise
        """
        normalized = self.normalize_hash(expected_hash)
        if not normalized:
            return False

        actual_hash = self.compute_hash(content)
        return actual_hash == normalized
