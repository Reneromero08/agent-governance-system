#!/usr/bin/env python3
"""
Session Cache (L4) - Per-session symbol expansion cache.

Layer 4 of the compression stack. Caches symbol expansions per session
to achieve 90%+ compression on warm queries.

Key insight: Same symbols expand repeatedly within a session.
- Query 1 (cold): Full expansion (~50 tokens)
- Query 2-N (warm): Hash confirmation (~1 token)

Usage:
    from session_cache import SessionCache

    cache = SessionCache(session_id="agent-001", codebook_hash="abc123...")

    # Cold query - cache miss
    entry = cache.get("C3")  # None

    # Store expansion
    cache.put("C3", "Contract rule 3: ...", tokens=52)

    # Warm query - cache hit
    entry = cache.get("C3")  # SessionCacheEntry with expansion

    # Session end - get stats
    stats = cache.get_stats()
    print(f"Hit rate: {stats.hit_rate:.1f}%")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass
class SessionCacheEntry:
    """Single cache entry for symbol expansion."""

    pointer: str              # SPC pointer (e.g., "C3", "I5", "sha256:...")
    expansion_hash: str       # SHA256 of expansion text
    expansion_text: str       # Full expansion (in-memory only)
    codebook_hash: str        # Codebook hash for invalidation check
    tokens_cold: int          # Token count of full expansion
    created_at: str           # ISO8601 timestamp
    access_count: int = 0     # Number of cache hits
    last_accessed: str = ""   # ISO8601 timestamp of last access

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry (excluding expansion_text for persistence)."""
        return {
            "pointer": self.pointer,
            "expansion_hash": self.expansion_hash,
            "codebook_hash": self.codebook_hash,
            "tokens_cold": self.tokens_cold,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], expansion_text: str = "") -> "SessionCacheEntry":
        """Deserialize entry."""
        return cls(
            pointer=data["pointer"],
            expansion_hash=data["expansion_hash"],
            expansion_text=expansion_text,
            codebook_hash=data["codebook_hash"],
            tokens_cold=data["tokens_cold"],
            created_at=data["created_at"],
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", ""),
        )


@dataclass
class SessionCacheStats:
    """Statistics for session cache performance."""

    total: int = 0           # Total cache lookups
    hits: int = 0            # Cache hits
    misses: int = 0          # Cache misses
    puts: int = 0            # Cache insertions
    invalidations: int = 0   # Entries invalidated
    tokens_saved: int = 0    # Total tokens saved from hits

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        return (self.hits / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stats."""
        return {
            "total": self.total,
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "invalidations": self.invalidations,
            "tokens_saved": self.tokens_saved,
            "hit_rate": round(self.hit_rate, 2),
        }


# ==============================================================================
# SESSION CACHE
# ==============================================================================


class SessionCache:
    """Per-session symbol expansion cache (L4 compression layer).

    Caches symbol expansions during a session to avoid redundant expansion.
    Achieves 90%+ compression on warm queries by returning hash confirmation
    instead of full expansion text.

    Fail-closed semantics:
    - Codebook change invalidates entire cache
    - Invalid entries are never returned

    Persistence:
    - In-memory during session for <1ms lookup
    - Snapshot to working_set JSON for session resume
    - Only hashes persisted, not full expansion text
    """

    TOKENS_WARM = 1  # Cost of hash confirmation

    def __init__(
        self,
        session_id: str,
        codebook_hash: str,
        max_entries: int = 1000
    ):
        """Initialize session cache.

        Args:
            session_id: Unique session identifier
            codebook_hash: SHA256 of codebook for invalidation
            max_entries: Maximum cache entries (LRU eviction)
        """
        self.session_id = session_id
        self.codebook_hash = codebook_hash
        self.max_entries = max_entries

        self._cache: Dict[str, SessionCacheEntry] = {}
        self._stats = SessionCacheStats()
        self._created_at = datetime.now(timezone.utc).isoformat()

    def get(self, pointer: str) -> Optional[SessionCacheEntry]:
        """Look up cached expansion by pointer.

        Args:
            pointer: SPC pointer string

        Returns:
            SessionCacheEntry on hit, None on miss
        """
        self._stats.total += 1

        entry = self._cache.get(pointer)
        if entry is None:
            self._stats.misses += 1
            return None

        # Cache hit - update access tracking
        self._stats.hits += 1
        self._stats.tokens_saved += (entry.tokens_cold - self.TOKENS_WARM)
        entry.access_count += 1
        entry.last_accessed = datetime.now(timezone.utc).isoformat()

        return entry

    def put(
        self,
        pointer: str,
        expansion: str,
        tokens: int
    ) -> SessionCacheEntry:
        """Store expansion in cache.

        Args:
            pointer: SPC pointer string
            expansion: Full expansion text
            tokens: Token count of expansion

        Returns:
            Created SessionCacheEntry
        """
        now = datetime.now(timezone.utc).isoformat()

        # Compute expansion hash
        expansion_hash = hashlib.sha256(expansion.encode("utf-8")).hexdigest()

        entry = SessionCacheEntry(
            pointer=pointer,
            expansion_hash=expansion_hash,
            expansion_text=expansion,
            codebook_hash=self.codebook_hash,
            tokens_cold=tokens,
            created_at=now,
            access_count=0,
            last_accessed=now,
        )

        # LRU eviction if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_lru()

        self._cache[pointer] = entry
        self._stats.puts += 1

        return entry

    def invalidate(self, pointer: Optional[str] = None) -> int:
        """Invalidate cache entry or all entries.

        Args:
            pointer: Specific pointer to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        if pointer is not None:
            if pointer in self._cache:
                del self._cache[pointer]
                self._stats.invalidations += 1
                return 1
            return 0

        # Invalidate all
        count = len(self._cache)
        self._cache.clear()
        self._stats.invalidations += count
        return count

    def invalidate_all(self) -> int:
        """Clear entire cache (e.g., on codebook change).

        Returns:
            Number of entries cleared
        """
        return self.invalidate(None)

    def validate_codebook(self, current_hash: str) -> bool:
        """Check if codebook matches cache state.

        Args:
            current_hash: Current codebook SHA256

        Returns:
            True if valid, False if invalidation needed
        """
        if current_hash != self.codebook_hash:
            # Fail-closed: invalidate on codebook change
            self.invalidate_all()
            self.codebook_hash = current_hash
            return False
        return True

    def snapshot(self) -> Dict[str, Any]:
        """Create serializable snapshot for working_set persistence.

        Note: Only stores hashes, not full expansion text.

        Returns:
            Dict suitable for JSON serialization
        """
        entries = [entry.to_dict() for entry in self._cache.values()]

        # Compute merkle root of entries for integrity
        entry_hashes = sorted([e["expansion_hash"] for e in entries])
        merkle_data = "|".join(entry_hashes).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_data).hexdigest()

        return {
            "schema_version": "1.0.0",
            "session_id": self.session_id,
            "codebook_hash": self.codebook_hash,
            "entry_count": len(entries),
            "merkle_root": merkle_root,
            "entries": entries,
            "stats": self._stats.to_dict(),
            "created_at": self._created_at,
        }

    def restore(self, snapshot: Dict[str, Any]) -> int:
        """Restore cache state from snapshot.

        Note: Restores metadata only. Expansion text must be re-fetched
        on first access (lazy restoration).

        Args:
            snapshot: Snapshot from snapshot()

        Returns:
            Number of entries restored
        """
        if not snapshot:
            return 0

        # Validate schema
        if snapshot.get("schema_version") != "1.0.0":
            return 0

        # Validate codebook hash
        if snapshot.get("codebook_hash") != self.codebook_hash:
            # Codebook changed - cannot restore
            return 0

        # Validate merkle root
        entries = snapshot.get("entries", [])
        entry_hashes = sorted([e["expansion_hash"] for e in entries])
        merkle_data = "|".join(entry_hashes).encode("utf-8")
        expected_root = hashlib.sha256(merkle_data).hexdigest()

        if snapshot.get("merkle_root") != expected_root:
            # Integrity check failed
            return 0

        # Restore entries (without expansion_text)
        for entry_data in entries:
            entry = SessionCacheEntry.from_dict(entry_data, expansion_text="")
            self._cache[entry.pointer] = entry

        return len(entries)

    def get_stats(self) -> SessionCacheStats:
        """Return cache statistics.

        Returns:
            SessionCacheStats with current metrics
        """
        return self._stats

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __contains__(self, pointer: str) -> bool:
        """Check if pointer is cached."""
        return pointer in self._cache

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with lowest access count (LRU approximation)
        # Ties broken by oldest last_accessed timestamp
        def lru_key(pointer: str) -> tuple:
            entry = self._cache[pointer]
            # Primary: access count (lower = less used)
            # Secondary: last accessed time (older = less recent)
            timestamp = entry.last_accessed or entry.created_at
            return (entry.access_count, timestamp)

        lru_pointer = min(self._cache.keys(), key=lru_key)
        del self._cache[lru_pointer]


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def compute_expansion_hash(expansion: str) -> str:
    """Compute SHA256 hash of expansion text.

    Args:
        expansion: Full expansion text

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(expansion.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses simple heuristic: ~4 characters per token.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return max(1, len(text) // 4)
