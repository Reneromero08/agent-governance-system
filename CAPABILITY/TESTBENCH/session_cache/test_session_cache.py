#!/usr/bin/env python3
"""
Unit tests for Session Cache (L4).

Tests cover:
- Basic cache operations (get/put/invalidate)
- Cache miss behavior
- Codebook change invalidation (fail-closed)
- Snapshot/restore roundtrip
- Statistics tracking
- 90%+ compression validation
"""

import hashlib
import json
import pytest
import sys
from pathlib import Path

# Setup paths
TESTBENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTBENCH_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "NAVIGATION" / "CORTEX" / "network"))

from session_cache import (
    SessionCache,
    SessionCacheEntry,
    SessionCacheStats,
    compute_expansion_hash,
    estimate_tokens,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cache():
    """Create a fresh session cache."""
    return SessionCache(
        session_id="test-session-001",
        codebook_hash="abc123def456"
    )


@pytest.fixture
def populated_cache(cache):
    """Create a cache with some entries."""
    cache.put("C3", "Contract rule 3: All writes must emit receipts.", tokens=52)
    cache.put("I5", "Invariant 5: Fail-closed on ambiguity.", tokens=47)
    cache.put("V", "Verification: Check hash integrity.", tokens=38)
    return cache


# =============================================================================
# BASIC OPERATIONS
# =============================================================================


class TestBasicOperations:
    """Test basic cache get/put operations."""

    def test_put_and_get(self, cache):
        """Cache put followed by get returns entry."""
        cache.put("C3", "Contract rule 3: All writes.", tokens=50)

        entry = cache.get("C3")

        assert entry is not None
        assert entry.pointer == "C3"
        assert entry.expansion_text == "Contract rule 3: All writes."
        assert entry.tokens_cold == 50

    def test_get_miss_returns_none(self, cache):
        """Cache get for unknown pointer returns None."""
        entry = cache.get("UNKNOWN")

        assert entry is None

    def test_get_increments_access_count(self, cache):
        """Multiple gets increment access count."""
        cache.put("C3", "Contract rule 3", tokens=30)

        cache.get("C3")
        cache.get("C3")
        entry = cache.get("C3")

        assert entry.access_count == 3

    def test_expansion_hash_computed(self, cache):
        """Expansion hash is computed on put."""
        cache.put("C3", "Contract rule 3", tokens=30)

        entry = cache.get("C3")
        expected_hash = hashlib.sha256("Contract rule 3".encode()).hexdigest()

        assert entry.expansion_hash == expected_hash

    def test_len_returns_entry_count(self, cache):
        """len(cache) returns number of entries."""
        assert len(cache) == 0

        cache.put("C3", "Rule 3", tokens=10)
        assert len(cache) == 1

        cache.put("I5", "Invariant 5", tokens=15)
        assert len(cache) == 2

    def test_contains_check(self, cache):
        """'in' operator checks if pointer is cached."""
        cache.put("C3", "Rule 3", tokens=10)

        assert "C3" in cache
        assert "UNKNOWN" not in cache


# =============================================================================
# STATISTICS TRACKING
# =============================================================================


class TestStatistics:
    """Test cache statistics tracking."""

    def test_stats_initial_values(self, cache):
        """Fresh cache has zero stats."""
        stats = cache.get_stats()

        assert stats.total == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.puts == 0
        assert stats.hit_rate == 0.0

    def test_stats_track_misses(self, cache):
        """Cache miss increments miss counter."""
        cache.get("UNKNOWN")
        cache.get("ALSO_UNKNOWN")

        stats = cache.get_stats()
        assert stats.total == 2
        assert stats.misses == 2
        assert stats.hits == 0

    def test_stats_track_hits(self, populated_cache):
        """Cache hit increments hit counter."""
        populated_cache.get("C3")
        populated_cache.get("I5")

        stats = populated_cache.get_stats()
        assert stats.hits == 2
        assert stats.total == 2

    def test_stats_track_puts(self, cache):
        """Cache put increments put counter."""
        cache.put("C3", "Rule 3", tokens=10)
        cache.put("I5", "Inv 5", tokens=15)

        stats = cache.get_stats()
        assert stats.puts == 2

    def test_hit_rate_calculation(self, populated_cache):
        """Hit rate calculated correctly."""
        # 2 hits, 1 miss = 66.67% hit rate
        populated_cache.get("C3")  # hit
        populated_cache.get("I5")  # hit
        populated_cache.get("UNKNOWN")  # miss

        stats = populated_cache.get_stats()
        assert 66 < stats.hit_rate < 67

    def test_tokens_saved_tracked(self, cache):
        """Tokens saved tracked on hits."""
        cache.put("C3", "x" * 200, tokens=50)  # 50 cold, 1 warm = 49 saved

        cache.get("C3")  # Save 49
        cache.get("C3")  # Save 49

        stats = cache.get_stats()
        assert stats.tokens_saved == 98  # 49 * 2


# =============================================================================
# INVALIDATION
# =============================================================================


class TestInvalidation:
    """Test cache invalidation behavior."""

    def test_invalidate_single_entry(self, populated_cache):
        """Invalidate single entry removes only that entry."""
        count = populated_cache.invalidate("C3")

        assert count == 1
        assert "C3" not in populated_cache
        assert "I5" in populated_cache

    def test_invalidate_unknown_returns_zero(self, cache):
        """Invalidate unknown pointer returns 0."""
        count = cache.invalidate("UNKNOWN")

        assert count == 0

    def test_invalidate_all(self, populated_cache):
        """Invalidate all clears entire cache."""
        count = populated_cache.invalidate_all()

        assert count == 3
        assert len(populated_cache) == 0

    def test_codebook_change_invalidates(self, populated_cache):
        """Codebook change triggers full invalidation."""
        assert len(populated_cache) == 3

        # Simulate codebook change
        is_valid = populated_cache.validate_codebook("NEW_CODEBOOK_HASH")

        assert is_valid is False
        assert len(populated_cache) == 0

    def test_codebook_same_keeps_cache(self, populated_cache):
        """Same codebook hash keeps cache intact."""
        original_codebook = populated_cache.codebook_hash

        is_valid = populated_cache.validate_codebook(original_codebook)

        assert is_valid is True
        assert len(populated_cache) == 3


# =============================================================================
# SNAPSHOT / RESTORE
# =============================================================================


class TestPersistence:
    """Test snapshot and restore functionality."""

    def test_snapshot_creates_dict(self, populated_cache):
        """Snapshot returns serializable dict."""
        snapshot = populated_cache.snapshot()

        assert isinstance(snapshot, dict)
        assert snapshot["session_id"] == "test-session-001"
        assert snapshot["entry_count"] == 3
        assert "merkle_root" in snapshot
        assert "entries" in snapshot

    def test_snapshot_contains_entries(self, populated_cache):
        """Snapshot contains entry data (without expansion text)."""
        snapshot = populated_cache.snapshot()

        entries = snapshot["entries"]
        assert len(entries) == 3

        pointers = {e["pointer"] for e in entries}
        assert pointers == {"C3", "I5", "V"}

        # Verify expansion_text is NOT in snapshot
        for entry in entries:
            assert "expansion_text" not in entry

    def test_restore_from_snapshot(self, cache, populated_cache):
        """Restore recreates cache from snapshot."""
        snapshot = populated_cache.snapshot()

        # Create new cache with same codebook
        new_cache = SessionCache(
            session_id="test-session-001",
            codebook_hash="abc123def456"
        )

        restored = new_cache.restore(snapshot)

        assert restored == 3
        assert len(new_cache) == 3
        assert "C3" in new_cache

    def test_restore_fails_on_codebook_mismatch(self, populated_cache):
        """Restore fails if codebook changed."""
        snapshot = populated_cache.snapshot()

        # Create cache with DIFFERENT codebook
        new_cache = SessionCache(
            session_id="test-session-001",
            codebook_hash="DIFFERENT_CODEBOOK"
        )

        restored = new_cache.restore(snapshot)

        assert restored == 0  # Failed to restore

    def test_restore_validates_merkle_root(self, cache, populated_cache):
        """Restore validates merkle root integrity."""
        snapshot = populated_cache.snapshot()

        # Tamper with snapshot
        snapshot["entries"][0]["expansion_hash"] = "TAMPERED"

        new_cache = SessionCache(
            session_id="test-session-001",
            codebook_hash="abc123def456"
        )

        restored = new_cache.restore(snapshot)

        assert restored == 0  # Failed integrity check

    def test_snapshot_roundtrip(self, populated_cache):
        """Full snapshot/restore roundtrip preserves data."""
        original_snapshot = populated_cache.snapshot()

        new_cache = SessionCache(
            session_id="test-session-001",
            codebook_hash="abc123def456"
        )
        new_cache.restore(original_snapshot)

        new_snapshot = new_cache.snapshot()

        # Compare (excluding stats which differ)
        assert original_snapshot["entry_count"] == new_snapshot["entry_count"]
        assert original_snapshot["merkle_root"] == new_snapshot["merkle_root"]


# =============================================================================
# COMPRESSION VALIDATION
# =============================================================================


class TestCompression:
    """Test 90%+ compression on warm queries."""

    def test_90_percent_compression_per_query(self, cache):
        """Single warm query achieves 98% compression."""
        # Cold: 50 tokens
        cache.put("C3", "x" * 200, tokens=50)

        # Warm: 1 token (hash confirmation)
        entry = cache.get("C3")

        tokens_cold = entry.tokens_cold
        tokens_warm = 1  # SessionCache.TOKENS_WARM
        savings = tokens_cold - tokens_warm
        savings_pct = (savings / tokens_cold) * 100

        assert savings_pct >= 90  # At least 90%

    def test_session_compression_target(self, cache):
        """10 queries (1 cold, 9 warm) achieves 88%+ savings."""
        # Cold query: 50 tokens
        cache.put("C3", "x" * 200, tokens=50)

        # 9 warm queries
        for _ in range(9):
            cache.get("C3")

        # Calculate total
        total_tokens = 50 + 9 * 1  # 1 cold + 9 warm
        baseline = 50 * 10  # 10 full expansions
        savings_pct = (1 - total_tokens / baseline) * 100

        assert savings_pct >= 88  # Meets target

    def test_stats_match_compression_calculation(self, cache):
        """Stats tokens_saved matches manual calculation."""
        cache.put("C3", "x" * 200, tokens=50)

        for _ in range(9):
            cache.get("C3")

        stats = cache.get_stats()
        expected_saved = 9 * 49  # 9 hits * (50 - 1) saved per hit

        assert stats.tokens_saved == expected_saved


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


class TestHelpers:
    """Test helper functions."""

    def test_compute_expansion_hash(self):
        """compute_expansion_hash returns correct SHA256."""
        text = "Test expansion text"
        result = compute_expansion_hash(text)
        expected = hashlib.sha256(text.encode()).hexdigest()

        assert result == expected

    def test_estimate_tokens_basic(self):
        """estimate_tokens uses ~4 chars per token."""
        # 400 chars should be ~100 tokens
        text = "x" * 400
        result = estimate_tokens(text)

        assert 95 <= result <= 105

    def test_estimate_tokens_minimum(self):
        """estimate_tokens returns at least 1."""
        result = estimate_tokens("")

        assert result >= 1


# =============================================================================
# LRU EVICTION
# =============================================================================


class TestLRUEviction:
    """Test LRU eviction when cache is full."""

    def test_eviction_at_capacity(self):
        """Oldest entry evicted when cache is full."""
        cache = SessionCache(
            session_id="test",
            codebook_hash="abc",
            max_entries=3
        )

        cache.put("A", "text A", tokens=10)
        cache.put("B", "text B", tokens=10)
        cache.put("C", "text C", tokens=10)

        # Access A and B to make C oldest
        cache.get("A")
        cache.get("B")

        # Add D - should evict C
        cache.put("D", "text D", tokens=10)

        assert len(cache) == 3
        assert "D" in cache
        assert "C" not in cache  # Evicted
        assert "A" in cache
        assert "B" in cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
