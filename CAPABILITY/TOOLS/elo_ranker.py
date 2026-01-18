#!/usr/bin/env python3
"""
ELO Ranker - Attaches ELO metadata to search results WITHOUT modifying ranking.

DESIGN DECISION: ELO as Suggestion, Not Modification
=====================================================

ELO scores are attached as metadata only. They do NOT affect search ranking.
Results remain sorted by semantic similarity (relevance).

WHY THIS DESIGN:
1. Prevents echo chambers - popular content can't bury relevant content
2. Avoids "lost treasures" - valuable but undiscovered content still surfaces
3. Relevance always wins - similarity is the only ranking factor
4. ELO provides context - "this file is frequently accessed" is useful info
5. Simpler to reason about - no hidden ranking manipulation

WHERE ELO STILL CONTROLS (not just suggests):
- LITE packs: ELO determines compression level (token budget is real)
- Memory pruning: ELO determines what gets archived (storage is real)
- Dashboard: ELO shows usage patterns (visibility)

FORMULA HISTORY:
- Original spec: final_score = similarity * 0.7 + (elo / 2000) * 0.3
- Current: final_score = similarity (ELO has zero weight on ranking)

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
    """
    A search result with ELO metadata attached.

    Note: Ranking is by similarity only. ELO is informational metadata.
    """
    file_path: str
    content_hash: str
    similarity: float      # The ONLY factor that determines ranking
    elo_score: float       # Informational: usage frequency metric
    elo_tier: str          # Informational: HIGH/MEDIUM/LOW/VERY_LOW
    final_score: float     # Equal to similarity (ELO does not modify)
    rank: int


class EloRanker:
    """
    Attach ELO metadata to search results.

    IMPORTANT: This class does NOT modify search ranking.
    Results remain sorted by similarity (semantic relevance).
    ELO is attached as informational metadata only.

    Design rationale:
    - Similarity determines what's relevant to the query
    - ELO tells you how frequently something has been accessed
    - These are independent signals - ELO should inform, not override relevance

    Example output:
        1. docs/auth/config.md  (sim: 0.92)  [ELO: 1847 - frequently accessed]
        2. src/auth/handler.py  (sim: 0.88)  [ELO: 623 - rarely accessed]

    Result #2 surfaces despite low ELO because it's RELEVANT.
    The ELO metadata provides useful context without burying content.
    """

    # DEPRECATED: These weights are kept for reference but no longer used
    # The original formula was: similarity * 0.7 + elo_normalized * 0.3
    # Now: similarity * 1.0 + elo_normalized * 0.0 (ELO is metadata only)
    _LEGACY_SIMILARITY_WEIGHT = 0.7  # Historical reference
    _LEGACY_ELO_WEIGHT = 0.3         # Historical reference

    ELO_NORMALIZER = 2000.0  # For display purposes (normalize to 0-1 range)

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
        Compute final score (equal to similarity - ELO does not modify ranking).

        This method exists for API compatibility but ELO has zero weight.

        Args:
            similarity: Raw similarity score (0-1)
            elo: ELO score (ignored for ranking, kept as metadata)

        Returns:
            Final score equal to similarity (clamped to [0, 1])
        """
        # ELO does NOT modify ranking - similarity is the only factor
        # The elo parameter is accepted but not used in the score
        _ = elo  # Explicitly ignore (kept for API compatibility)
        return max(0.0, min(1.0, similarity))

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
        # Try path (used by cassette network)
        if "path" in result:
            return result["path"]
        return None

    def _get_entity_type(self, result: Dict[str, Any]) -> str:
        """
        Determine entity type from a search result.

        Args:
            result: Search result dictionary

        Returns:
            Entity type string ("file" or "vector")
        """
        if "file_path" in result or "path" in result:
            return "file"
        return "vector"

    def rank_results(
        self,
        results: List[Dict[str, Any]],
        entity_type: str = "file"
    ) -> List[RankedResult]:
        """
        Attach ELO metadata to results and sort by SIMILARITY ONLY.

        IMPORTANT: Ranking is by similarity, not by ELO.
        ELO is attached as informational metadata.

        Args:
            results: List of search results with keys:
                     - file_path or hash (entity_id)
                     - similarity (0-1 score)
            entity_type: "file" or "vector"

        Returns:
            List of RankedResult sorted by similarity descending
            (ELO metadata attached but does not affect order)
        """
        ranked = []

        for result in results:
            # Extract identifiers
            entity_id = self._get_entity_id(result)
            if entity_id is None:
                continue

            # Get similarity score (the ONLY ranking factor)
            similarity = result.get("similarity", result.get("score", 0.0))

            # Get ELO score as metadata (does NOT affect ranking)
            elo_score = self.db.get_elo(entity_type, entity_id)

            # Get tier (informational)
            elo_tier = self.engine.get_tier(elo_score)

            # Final score equals similarity (ELO is metadata only)
            final_score = self.compute_final_score(similarity, elo_score)

            # Extract file_path and content_hash
            file_path = result.get("file_path", result.get("path", ""))
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

        # Sort by SIMILARITY only (ELO does not affect ranking)
        ranked.sort(key=lambda r: r.similarity, reverse=True)

        # Assign ranks
        for i, item in enumerate(ranked):
            item.rank = i + 1

        return ranked

    def annotate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Attach ELO metadata to results WITHOUT changing order.

        This is the primary method for MCP integration.
        Results stay in their original order (by similarity).
        ELO fields are added as informational metadata.

        Args:
            results: List of search results (already sorted by similarity)

        Returns:
            Same list with elo_score, elo_tier added to each result
        """
        annotated = []

        for result in results:
            # Create a copy to avoid modifying original
            annotated_result = dict(result)

            # Get entity info
            entity_id = self._get_entity_id(result)
            entity_type = self._get_entity_type(result)

            if entity_id is None:
                # No valid entity ID, use defaults
                elo_score = self.db.DEFAULT_ELO
            else:
                # Get ELO score
                elo_score = self.db.get_elo(entity_type, entity_id)

            # Attach ELO metadata (informational only)
            annotated_result["elo_score"] = elo_score
            annotated_result["elo_tier"] = self.engine.get_tier(elo_score)

            # Note: We do NOT add final_score or re-sort
            # The original similarity-based order is preserved

            annotated.append(annotated_result)

        return annotated

    def boost_semantic_search(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Attach ELO metadata to semantic search results.

        IMPORTANT: Does NOT re-sort results. Order is preserved.
        This method exists for backward compatibility.
        Prefer annotate_results() for clarity.

        Args:
            results: List of search results (sorted by similarity)

        Returns:
            Same list with ELO fields added, ORDER UNCHANGED
        """
        # Just annotate - do not re-sort
        return self.annotate_results(results)

    def boost_cortex_query(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Attach ELO metadata to cortex query results.

        IMPORTANT: Does NOT re-sort results. Order is preserved.
        ELO is informational metadata only.

        Args:
            results: List of cortex query results

        Returns:
            Same list with ELO fields added, ORDER UNCHANGED
        """
        # Just annotate - do not re-sort
        return self.annotate_results(results)

    def get_quality_stats(self, results: List[RankedResult]) -> Dict[str, Any]:
        """
        Calculate quality statistics for ranked results.

        Useful for monitoring the distribution of ELO in search results.

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
                "avg_similarity": 0.0,
                "tier_distribution": {},
                "note": "ELO is metadata only - does not affect ranking"
            }

        total = len(results)

        # Count high ELO in top 5 (informational - not a ranking factor)
        top5 = results[:5]
        high_elo_in_top5 = sum(1 for r in top5 if r.elo_tier == "HIGH")
        high_elo_pct_top5 = (high_elo_in_top5 / min(5, len(top5))) * 100.0 if top5 else 0.0

        # Calculate averages
        avg_elo = sum(r.elo_score for r in results) / total
        avg_similarity = sum(r.similarity for r in results) / total

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
            "avg_similarity": round(avg_similarity, 3),
            "tier_distribution": tier_distribution,
            "note": "ELO is metadata only - does not affect ranking"
        }


def run_self_test() -> bool:
    """
    Run comprehensive self-test of EloRanker functionality.

    Tests verify that:
    1. ELO is attached as metadata
    2. Ranking is by SIMILARITY only (ELO does not change order)
    3. High-similarity results always beat low-similarity regardless of ELO

    Returns:
        True if all tests pass, False otherwise.
    """
    import tempfile
    import shutil

    print("=" * 60)
    print("ELO Ranker Self-Test")
    print("=" * 60)
    print("\nDESIGN: ELO is metadata only - does NOT affect ranking")
    print("        Results are sorted by SIMILARITY only")

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

        # Test 1: Final score equals similarity (ELO has no weight)
        print("\n[Test 1] Verify final_score equals similarity (ELO ignored)...")
        score1 = ranker.compute_final_score(0.9, 800.0)
        assert abs(score1 - 0.9) < 0.001, f"Expected 0.9, got {score1}"

        score2 = ranker.compute_final_score(0.7, 1600.0)
        assert abs(score2 - 0.7) < 0.001, f"Expected 0.7, got {score2}"

        score3 = ranker.compute_final_score(0.5, 2000.0)
        assert abs(score3 - 0.5) < 0.001, f"Expected 0.5, got {score3}"

        print(f"  sim=0.9, elo=800:  final={score1:.3f} (expected 0.9)")
        print(f"  sim=0.7, elo=1600: final={score2:.3f} (expected 0.7)")
        print(f"  sim=0.5, elo=2000: final={score3:.3f} (expected 0.5)")
        print("  PASSED: ELO does not affect final_score")

        # Test 2: Ranking is by similarity, NOT by ELO
        print("\n[Test 2] Verify ranking is by SIMILARITY only...")
        test_results = [
            {"file_path": "src/high_elo.py", "similarity": 0.7},   # ELO 1600
            {"file_path": "src/low_elo.py", "similarity": 0.8},    # ELO 800
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        # Low ELO file should rank FIRST because it has higher similarity
        assert ranked[0].file_path == "src/low_elo.py", \
            "Higher similarity should rank first regardless of ELO"
        assert ranked[1].file_path == "src/high_elo.py", \
            "Lower similarity should rank second regardless of ELO"

        print(f"  Rank 1: {ranked[0].file_path} (sim={ranked[0].similarity}, elo={ranked[0].elo_score})")
        print(f"  Rank 2: {ranked[1].file_path} (sim={ranked[1].similarity}, elo={ranked[1].elo_score})")
        print("  PASSED: Higher similarity wins regardless of ELO")

        # Test 3: ELO metadata is attached correctly
        print("\n[Test 3] Verify ELO metadata is attached...")
        test_results = [
            {"file_path": "src/high_elo.py", "similarity": 0.9},
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        assert ranked[0].elo_score == 1600.0, f"Expected ELO 1600, got {ranked[0].elo_score}"
        assert ranked[0].elo_tier == "HIGH", f"Expected tier HIGH, got {ranked[0].elo_tier}"
        print(f"  ELO metadata: score={ranked[0].elo_score}, tier={ranked[0].elo_tier}")
        print("  PASSED: ELO metadata attached correctly")

        # Test 4: Unknown entities get default ELO
        print("\n[Test 4] Verify unknown entities get default ELO (1000.0)...")
        test_results = [
            {"file_path": "unknown/file.py", "similarity": 0.8},
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        assert ranked[0].elo_score == 1000.0, f"Expected default ELO 1000.0, got {ranked[0].elo_score}"
        print(f"  Unknown file ELO: {ranked[0].elo_score}")
        print("  PASSED: Unknown entities use default ELO 1000.0")

        # Test 5: annotate_results preserves order
        print("\n[Test 5] Verify annotate_results preserves original order...")
        test_results = [
            {"file_path": "src/low_elo.py", "similarity": 0.95},    # ELO 800
            {"file_path": "src/high_elo.py", "similarity": 0.90},   # ELO 1600
            {"file_path": "src/medium_elo.py", "similarity": 0.85}, # ELO 1200
        ]
        annotated = ranker.annotate_results(test_results)

        # Order should be exactly preserved (not sorted by ELO)
        assert annotated[0]["file_path"] == "src/low_elo.py", "Order should be preserved"
        assert annotated[1]["file_path"] == "src/high_elo.py", "Order should be preserved"
        assert annotated[2]["file_path"] == "src/medium_elo.py", "Order should be preserved"

        # ELO metadata should be attached
        assert "elo_score" in annotated[0], "elo_score should be added"
        assert "elo_tier" in annotated[0], "elo_tier should be added"

        print("  Order preserved:")
        for i, r in enumerate(annotated, 1):
            print(f"    {i}. {r['file_path']} (sim={r['similarity']}, elo={r['elo_score']})")
        print("  PASSED: Order preserved, ELO metadata attached")

        # Test 6: boost_semantic_search does NOT re-sort
        print("\n[Test 6] Verify boost_semantic_search does NOT re-sort...")
        test_results = [
            {"file_path": "src/low_elo.py", "similarity": 0.9},    # ELO 800
            {"file_path": "src/high_elo.py", "similarity": 0.8},   # ELO 1600
        ]
        boosted = ranker.boost_semantic_search(test_results)

        # Order should be preserved (low ELO file stays first due to higher similarity)
        assert boosted[0]["file_path"] == "src/low_elo.py", \
            "boost_semantic_search should NOT re-sort"
        print(f"  First result: {boosted[0]['file_path']} (elo={boosted[0]['elo_score']})")
        print("  PASSED: boost_semantic_search preserves order")

        # Test 7: boost_cortex_query does NOT re-sort
        print("\n[Test 7] Verify boost_cortex_query does NOT re-sort...")
        test_results = [
            {"file_path": "src/low_elo.py", "relevance": 0.9},     # ELO 800
            {"file_path": "src/high_elo.py", "relevance": 0.9},    # ELO 1600
        ]
        boosted = ranker.boost_cortex_query(test_results)

        # Order should be preserved (not sorted by ELO)
        assert boosted[0]["file_path"] == "src/low_elo.py", \
            "boost_cortex_query should NOT re-sort"
        print(f"  First result: {boosted[0]['file_path']} (elo={boosted[0]['elo_score']})")
        print("  PASSED: boost_cortex_query preserves order")

        # Test 8: get_quality_stats works correctly
        print("\n[Test 8] Verify get_quality_stats works correctly...")
        test_results = [
            {"file_path": "src/high_elo.py", "similarity": 0.9},    # ELO 1600 (HIGH)
            {"file_path": "src/medium_elo.py", "similarity": 0.85}, # ELO 1200 (MEDIUM)
            {"file_path": "src/default_elo.py", "similarity": 0.8}, # ELO 1000 (LOW)
            {"file_path": "src/low_elo.py", "similarity": 0.75},    # ELO 800 (LOW)
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")
        stats = ranker.get_quality_stats(ranked)

        assert stats["total"] == 4, f"Expected total=4, got {stats['total']}"
        assert "note" in stats, "Stats should include note about ELO being metadata only"
        print(f"  Stats: {stats}")
        print("  PASSED: Quality stats calculated correctly")

        # Test 9: Demonstrate similarity always wins
        print("\n[Test 9] Demonstrate: Similarity ALWAYS wins over ELO...")
        test_results = [
            {"file_path": "src/low_elo.py", "similarity": 0.95},    # ELO 800, BEST SIM
            {"file_path": "src/high_elo.py", "similarity": 0.50},   # ELO 1600, WORST SIM
        ]
        ranked = ranker.rank_results(test_results, entity_type="file")

        print("\n  Results (sorted by SIMILARITY):")
        for r in ranked:
            print(f"    Rank {r.rank}: {r.file_path}")
            print(f"             similarity={r.similarity:.2f} (RANKING FACTOR)")
            print(f"             elo={r.elo_score:.0f} tier={r.elo_tier} (METADATA ONLY)")

        assert ranked[0].file_path == "src/low_elo.py", \
            "Low ELO with high similarity should rank first"
        print("\n  PASSED: Low ELO file ranks first due to higher similarity")

        # Cleanup
        db.close()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nKey takeaway: ELO is METADATA, not a ranking modifier.")
        print("Similarity determines relevance. ELO provides context.")

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
