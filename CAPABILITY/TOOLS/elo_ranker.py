#!/usr/bin/env python3
"""
ELO Ranker - Search result ranker that boosts results by ELO score.

Combines similarity scores with ELO ratings to produce final rankings.
High-ELO entities get boosted in search results, while still respecting
similarity as the primary factor.

Part of the agent-governance-system CORTEX infrastructure.
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import EloDatabase and EloEngine from sibling module
try:
    from ..PRIMITIVES.elo_db import EloDatabase
    from ..PRIMITIVES.elo_engine import EloEngine
except ImportError:
    # Allow direct script execution
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from PRIMITIVES.elo_db import EloDatabase
    from PRIMITIVES.elo_engine import EloEngine


@dataclass
class RankedResult:
    """A search result with ELO-boosted ranking."""
    file_path: str
    content_hash: str
    similarity: float
    elo_score: float
    elo_tier: str
    final_score: float
    rank: int


class EloRanker:
    """Rank search results with ELO boost."""

    SIMILARITY_WEIGHT = 0.7
    ELO_WEIGHT = 0.3
    ELO_NORMALIZER = 2000.0  # Normalize ELO to 0-1 range (assuming max ~2000)

    def __init__(self, db: EloDatabase, engine: EloEngine):
        """
        Initialize with database and engine.

        Args:
            db: EloDatabase instance for score retrieval
            engine: EloEngine instance for tier classification
        """
        self.db = db
        self.engine = engine

    def compute_final_score(self, similarity: float, elo: float) -> float:
        """
        Compute ELO-boosted final score.

        Args:
            similarity: Raw similarity score (0-1)
            elo: ELO score (typically 800-2000)

        Returns:
            Final score (0-1 range, clamped)
        """
        elo_normalized = min(elo / self.ELO_NORMALIZER, 1.0)
        final = similarity * self.SIMILARITY_WEIGHT + elo_normalized * self.ELO_WEIGHT
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, final))

    def _get_entity_id(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Extract entity ID from a search result.

        Args:
            result: Search result dictionary

        Returns:
            Entity ID (file_path or hash) or None if not found
        """
        # Try file_path first
        if "file_path" in result:
            return result["file_path"]
        # Fall back to hash
        if "hash" in result:
            return result["hash"]
        # Try content_hash
        if "content_hash" in result:
            return result["content_hash"]
        return None

    def _get_entity_type(self, result: Dict[str, Any]) -> str:
        """
        Determine entity type from a search result.

        Args:
            result: Search result dictionary

        Returns:
            Entity type string ("file" or "vector")
        """
        if "file_path" in result:
            return "file"
        return "vector"

    def rank_results(
        self,
        results: List[Dict[str, Any]],
        entity_type: str = "file"
    ) -> List[RankedResult]:
        """
        Re-rank search results with ELO boost.

        Args:
            results: List of search results with keys:
                     - file_path or hash (entity_id)
                     - similarity (0-1 score)
            entity_type: "file" or "vector"

        Returns:
            List of RankedResult sorted by final_score descending
        """
        ranked = []

        for result in results:
            # Extract identifiers
            entity_id = self._get_entity_id(result)
            if entity_id is None:
                continue

            # Get similarity score
            similarity = result.get("similarity", 0.0)

            # Get ELO score (default 1000.0 for unknown entities)
            elo_score = self.db.get_elo(entity_type, entity_id)

            # Get tier
            elo_tier = self.engine.get_tier(elo_score)

            # Compute final score
            final_score = self.compute_final_score(similarity, elo_score)

            # Extract file_path and content_hash
            file_path = result.get("file_path", "")
            content_hash = result.get("hash", result.get("content_hash", ""))

            ranked.append(RankedResult(
                file_path=file_path,
                content_hash=content_hash,
                similarity=similarity,
                elo_score=elo_score,
                elo_tier=elo_tier,
                final_score=final_score,
                rank=0  # Will be set after sorting
            ))

        # Sort by final_score descending
        ranked.sort(key=lambda r: r.final_score, reverse=True)

        # Assign ranks
        for i, item in enumerate(ranked):
            item.rank = i + 1

        return ranked

    def boost_semantic_search(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Boost semantic search results and return in original format.

        Adds elo_score, elo_tier, and final_score to each result.
        Re-sorts by final_score.

        Args:
            results: List of search results

        Returns:
            List of results with ELO fields added, sorted by final_score
        """
        boosted = []

        for result in results:
            # Create a copy to avoid modifying original
            boosted_result = dict(result)

            # Get entity info
            entity_id = self._get_entity_id(result)
            entity_type = self._get_entity_type(result)

            if entity_id is None:
                # No valid entity ID, use defaults
                boosted_result["elo_score"] = self.db.DEFAULT_ELO
                boosted_result["elo_tier"] = self.engine.get_tier(self.db.DEFAULT_ELO)
                boosted_result["final_score"] = result.get("similarity", 0.0)
            else:
                # Get ELO score
                elo_score = self.db.get_elo(entity_type, entity_id)
                boosted_result["elo_score"] = elo_score
                boosted_result["elo_tier"] = self.engine.get_tier(elo_score)

                # Compute final score
                similarity = result.get("similarity", 0.0)
                boosted_result["final_score"] = self.compute_final_score(similarity, elo_score)

            boosted.append(boosted_result)

        # Sort by final_score descending
        boosted.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)

        return boosted

    def boost_cortex_query(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Boost cortex query results with ELO as secondary sort.

        Primary: original relevance (preserves original order for ties)
        Secondary: ELO score (for breaking ties)

        Args:
            results: List of cortex query results

        Returns:
            List of results with ELO fields added, sorted by relevance then ELO
        """
        boosted = []

        for i, result in enumerate(results):
            # Create a copy to avoid modifying original
            boosted_result = dict(result)

            # Get entity info
            entity_id = self._get_entity_id(result)
            entity_type = self._get_entity_type(result)

            if entity_id is None:
                elo_score = self.db.DEFAULT_ELO
            else:
                elo_score = self.db.get_elo(entity_type, entity_id)

            boosted_result["elo_score"] = elo_score
            boosted_result["elo_tier"] = self.engine.get_tier(elo_score)

            # Store original rank for tie-breaking
            boosted_result["_original_rank"] = i

            # Get relevance score (use similarity or score field)
            relevance = result.get("relevance", result.get("similarity", result.get("score", 0.0)))
            boosted_result["_relevance"] = relevance

            boosted.append(boosted_result)

        # Sort by relevance descending, then by ELO descending for ties
        boosted.sort(key=lambda r: (r.get("_relevance", 0.0), r.get("elo_score", 0.0)), reverse=True)

        # Clean up internal fields
        for result in boosted:
            result.pop("_original_rank", None)
            result.pop("_relevance", None)

        return boosted

    def get_quality_stats(self, results: List[RankedResult]) -> Dict[str, Any]:
        """
        Calculate quality statistics for ranked results.

        Args:
            results: List of RankedResult objects

        Returns:
            Dictionary with quality statistics
        """
        if not results:
            return {
                "total": 0,
                "high_elo_in_top5": 0,
                "high_elo_pct_top5": 0.0,
                "avg_elo": 0.0,
                "avg_final_score": 0.0,
                "tier_distribution": {}
            }

        total = len(results)

        # Count high ELO in top 5
        top5 = results[:5]
        high_elo_in_top5 = sum(1 for r in top5 if r.elo_tier == "HIGH")
        high_elo_pct_top5 = (high_elo_in_top5 / min(5, len(top5))) * 100.0 if top5 else 0.0

        # Calculate averages
        avg_elo = sum(r.elo_score for r in results) / total
        avg_final_score = sum(r.final_score for r in results) / total

        # Tier distribution
        tier_distribution = {}
        for result in results:
            tier = result.elo_tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        return {
            "total": total,
            "high_elo_in_top5": high_elo_in_top5,
            "high_elo_pct_top5": round(high_elo_pct_top5, 1),
            "avg_elo": round(avg_elo, 1),
            "avg_final_score": round(avg_final_score, 3),
            "tier_distribution": tier_distribution
        }


def run_self_test() -> bool:
    """
    Run comprehensive self-test of EloRanker functionality.

    Returns:
        True if all tests pass, False otherwise.
    """
    import tempfile
    import shutil

    print("=" * 60)
    print("ELO Ranker Self-Test")
    print("=" * 60)

    # Create temporary directories for testing
    test_dir = tempfile.mkdtemp(prefix="elo_ranker_test_")
    test_db_path = os.path.join(test_dir, "test_elo.db")
    test_log_path = os.path.join(test_dir, "elo_updates.jsonl")

    all_passed = True

    try:
        # Setup: Create database and engine
        print("\n[Setup] Creating EloDatabase and EloEngine...")
        db = EloDatabase(test_db_path)
        engine = EloEngine(db, test_log_path)
        ranker = EloRanker(db, engine)
        print("  PASSED: EloRanker created successfully")

        # Setup: Insert test entities at different ELO levels
        print("\n[Setup] Inserting test entities at different ELO levels...")
        test_entities = [
            ("file", "src/high_elo.py", 1600.0),
            ("file", "src/medium_elo.py", 1200.0),
            ("file", "src/low_elo.py", 800.0),
            ("file", "src/default_elo.py", 1000.0),
            ("vector", "hash_high", 1700.0),
            ("vector", "hash_low", 850.0),
        ]
        for entity_type, entity_id, score in test_entities:
            db.set_elo(entity_type, entity_id, score)
        print(f"  Inserted {len(test_entities)} test entities")

        # Test 1: Compute final score formula
        print("\n[Test 1] Test compute_final_score formula...")
        # High similarity (0.9) + Low ELO (800):  0.9*0.7 + 0.4*0.3 = 0.63 + 0.12 = 0.75
        score1 = ranker.compute_final_score(0.9, 800.0)
        expected1 = 0.9 * 0.7 + (800.0 / 2000.0) * 0.3
        assert abs(score1 - expected1) < 0.001, f"Expected {expected1}, got {score1}"

        # Med similarity (0.7) + High ELO (1600): 0.7*0.7 + 0.8*0.3 = 0.49 + 0.24 = 0.73
        score2 = ranker.compute_final_score(0.7, 1600.0)
        expected2 = 0.7 * 0.7 + (1600.0 / 2000.0) * 0.3
        assert abs(score2 - expected2) < 0.001, f"Expected {expected2}, got {score2}"

        # Low similarity (0.5) + High ELO (1600): 0.5*0.7 + 0.8*0.3 = 0.35 + 0.24 = 0.59
        score3 = ranker.compute_final_score(0.5, 1600.0)
        expected3 = 0.5 * 0.7 + (1600.0 / 2000.0) * 0.3
        assert abs(score3 - expected3) < 0.001, f"Expected {expected3}, got {score3}"

        print(f"  High sim (0.9) + Low ELO (800):  {score1:.3f} (expected {expected1:.3f})")
        print(f"  Med sim (0.7) + High ELO (1600): {score2:.3f} (expected {expected2:.3f})")
        print(f"  Low sim (0.5) + High ELO (1600): {score3:.3f} (expected {expected3:.3f})")
        print("  PASSED: Formula computes correctly")

        # Test 2: High ELO boosts ranking
        print("\n[Test 2] Test that high ELO boosts ranking...")
        test_results = [
            {"file_path": "src/high_elo.py", "similarity": 0.7},  # ELO 1600
            {"file_path": "src/low_elo.py", "similarity": 0.7},   # ELO 800
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        assert ranked[0].file_path == "src/high_elo.py", "High ELO should rank first at same similarity"
        assert ranked[1].file_path == "src/low_elo.py", "Low ELO should rank second at same similarity"
        print(f"  Rank 1: {ranked[0].file_path} (final_score={ranked[0].final_score:.3f}, elo={ranked[0].elo_score})")
        print(f"  Rank 2: {ranked[1].file_path} (final_score={ranked[1].final_score:.3f}, elo={ranked[1].elo_score})")
        print("  PASSED: High ELO entity ranks above low ELO at same similarity")

        # Test 3: High similarity still wins over low similarity
        print("\n[Test 3] Test that high similarity wins over low similarity...")
        test_results = [
            {"file_path": "src/low_elo.py", "similarity": 0.95},   # ELO 800, high sim
            {"file_path": "src/high_elo.py", "similarity": 0.5},   # ELO 1600, low sim
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        assert ranked[0].file_path == "src/low_elo.py", "High similarity should still win over high ELO"
        print(f"  Rank 1: {ranked[0].file_path} (sim={ranked[0].similarity}, elo={ranked[0].elo_score}, final={ranked[0].final_score:.3f})")
        print(f"  Rank 2: {ranked[1].file_path} (sim={ranked[1].similarity}, elo={ranked[1].elo_score}, final={ranked[1].final_score:.3f})")
        print("  PASSED: High similarity still wins over low similarity with high ELO")

        # Test 4: Unknown entities get default ELO
        print("\n[Test 4] Test that unknown entities get default ELO (1000.0)...")
        test_results = [
            {"file_path": "unknown/file.py", "similarity": 0.8},
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        assert ranked[0].elo_score == 1000.0, f"Expected default ELO 1000.0, got {ranked[0].elo_score}"
        print(f"  Unknown file ELO: {ranked[0].elo_score}")
        print("  PASSED: Unknown entities use default ELO 1000.0")

        # Test 5: Max ELO handling (clamp to 1.0)
        print("\n[Test 5] Test max ELO handling...")
        # Set an extremely high ELO
        db.set_elo("file", "src/max_elo.py", 2500.0)
        test_results = [
            {"file_path": "src/max_elo.py", "similarity": 1.0},
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        # final_score should be clamped: 1.0*0.7 + min(2500/2000, 1.0)*0.3 = 0.7 + 0.3 = 1.0
        assert ranked[0].final_score <= 1.0, f"Final score should be <= 1.0, got {ranked[0].final_score}"
        print(f"  Max ELO (2500) final_score: {ranked[0].final_score:.3f}")
        print("  PASSED: Final score properly clamped to [0, 1]")

        # Test 6: boost_semantic_search preserves original fields
        print("\n[Test 6] Test boost_semantic_search preserves original fields...")
        test_results = [
            {"file_path": "src/medium_elo.py", "similarity": 0.8, "custom_field": "preserved"},
            {"file_path": "src/high_elo.py", "similarity": 0.75, "another_field": 42},
        ]
        boosted = ranker.boost_semantic_search(test_results)

        assert boosted[0].get("custom_field") == "preserved" or boosted[1].get("custom_field") == "preserved", \
            "Custom field should be preserved"
        assert "elo_score" in boosted[0], "elo_score should be added"
        assert "elo_tier" in boosted[0], "elo_tier should be added"
        assert "final_score" in boosted[0], "final_score should be added"
        print(f"  Result 1: elo_score={boosted[0].get('elo_score')}, elo_tier={boosted[0].get('elo_tier')}")
        print(f"  Result 2: elo_score={boosted[1].get('elo_score')}, elo_tier={boosted[1].get('elo_tier')}")
        print("  PASSED: Original fields preserved, ELO fields added")

        # Test 7: boost_cortex_query uses ELO for tie-breaking
        print("\n[Test 7] Test boost_cortex_query uses ELO for tie-breaking...")
        test_results = [
            {"file_path": "src/low_elo.py", "relevance": 0.9},   # ELO 800
            {"file_path": "src/high_elo.py", "relevance": 0.9},  # ELO 1600
        ]
        boosted = ranker.boost_cortex_query(test_results)

        # With same relevance, high ELO should come first
        assert boosted[0].get("file_path") == "src/high_elo.py", "High ELO should win tie-breaker"
        print(f"  Result 1: {boosted[0].get('file_path')} (elo={boosted[0].get('elo_score')})")
        print(f"  Result 2: {boosted[1].get('file_path')} (elo={boosted[1].get('elo_score')})")
        print("  PASSED: ELO used as tie-breaker")

        # Test 8: get_quality_stats calculates correctly
        print("\n[Test 8] Test get_quality_stats calculates correctly...")
        test_results = [
            {"file_path": "src/high_elo.py", "similarity": 0.9},    # ELO 1600 (HIGH)
            {"file_path": "src/medium_elo.py", "similarity": 0.85}, # ELO 1200 (MEDIUM)
            {"file_path": "src/default_elo.py", "similarity": 0.8}, # ELO 1000 (LOW)
            {"file_path": "src/low_elo.py", "similarity": 0.75},    # ELO 800 (LOW)
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")
        stats = ranker.get_quality_stats(ranked)

        assert stats["total"] == 4, f"Expected total=4, got {stats['total']}"
        assert stats["high_elo_in_top5"] == 1, f"Expected 1 HIGH in top5, got {stats['high_elo_in_top5']}"
        assert "HIGH" in stats["tier_distribution"], "Tier distribution should include HIGH"
        assert "LOW" in stats["tier_distribution"], "Tier distribution should include LOW"
        print(f"  Stats: {stats}")
        print("  PASSED: Quality stats calculated correctly")

        # Test 9: Handle both file paths and content hashes
        print("\n[Test 9] Test handling of both file paths and content hashes...")
        test_results = [
            {"hash": "hash_high", "similarity": 0.8},  # ELO 1700
            {"hash": "hash_low", "similarity": 0.8},   # ELO 850
        ]
        ranked = ranker.rank_results(test_results, entity_type="vector")

        assert ranked[0].content_hash == "hash_high", "High ELO hash should rank first"
        print(f"  Rank 1: hash={ranked[0].content_hash} (elo={ranked[0].elo_score})")
        print(f"  Rank 2: hash={ranked[1].content_hash} (elo={ranked[1].elo_score})")
        print("  PASSED: Content hashes handled correctly")

        # Test 10: Ranking comparison (before/after)
        print("\n[Test 10] Ranking comparison (before/after ELO boost)...")
        test_results = [
            {"file_path": "src/low_elo.py", "similarity": 0.82},    # ELO 800
            {"file_path": "src/medium_elo.py", "similarity": 0.80}, # ELO 1200
            {"file_path": "src/high_elo.py", "similarity": 0.78},   # ELO 1600
            {"file_path": "src/default_elo.py", "similarity": 0.76}, # ELO 1000
        ]

        print("\n  Before ELO boost (by similarity only):")
        for i, r in enumerate(sorted(test_results, key=lambda x: x["similarity"], reverse=True), 1):
            elo = db.get_elo("file", r["file_path"])
            print(f"    {i}. {r['file_path']}: sim={r['similarity']:.2f}, elo={elo:.0f}")

        ranked = ranker.rank_results(test_results, entity_type="file")
        print("\n  After ELO boost (by final_score):")
        for r in ranked:
            print(f"    {r.rank}. {r.file_path}: sim={r.similarity:.2f}, elo={r.elo_score:.0f}, final={r.final_score:.3f}")

        print("\n  PASSED: Ranking comparison shows ELO influence")

        # Cleanup
        db.close()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n[Cleanup] Removed test directory: {test_dir}")

    return all_passed


if __name__ == "__main__":
    # Run self-test
    if not run_self_test():
        print("\nSelf-test failed!")
        sys.exit(1)

    sys.exit(0)
