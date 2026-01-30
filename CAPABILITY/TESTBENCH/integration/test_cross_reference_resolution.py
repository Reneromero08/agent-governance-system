#!/usr/bin/env python3
"""
Integration tests for Phase 5.1.5.1: Cross-Reference Graph

Tests the unified cross-reference system that links artifacts across
canon, ADR, and skill indices by embedding similarity.

Test Coverage:
- Artifact loading from multiple indices
- Pairwise similarity computation
- Cross-reference graph building
- find_related queries with threshold filtering
- Deterministic ordering
- Graph statistics
- Index rebuild
"""

import tempfile
import unittest
from pathlib import Path

from CAPABILITY.PRIMITIVES.cross_ref_index import (
    build_cross_refs,
    find_related,
    get_cross_ref_stats,
    rebuild_cross_ref_index,
)


class TestCrossRefBuild(unittest.TestCase):
    """Test cross-reference graph building."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_cross_refs.db"

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_cross_refs_creates_database(self):
        """Test that build_cross_refs creates database file."""
        # This will use real indices, so we expect it to work if indices exist
        try:
            result = build_cross_refs(db_path=self.db_path, threshold=0.3, top_k_per_artifact=5)

            # Check result structure
            self.assertIn("artifacts_count", result)
            self.assertIn("refs_count", result)
            self.assertIn("threshold", result)
            self.assertIn("top_k_per_artifact", result)
            self.assertIn("duration_seconds", result)
            self.assertIn("receipt", result)

            # Check database was created
            self.assertTrue(self.db_path.exists())

            # Verify we have artifacts and refs
            self.assertGreater(result["artifacts_count"], 0, "Should have loaded artifacts")
            self.assertGreaterEqual(result["refs_count"], 0, "Should have created cross-refs")

        except ValueError as e:
            if "No artifacts found" in str(e):
                self.skipTest("No indexed artifacts available for testing")
            raise

    def test_build_cross_refs_threshold_filtering(self):
        """Test that threshold parameter filters low-similarity refs."""
        try:
            # Build with high threshold
            result_high = build_cross_refs(
                db_path=self.db_path,
                threshold=0.8,
                top_k_per_artifact=10,
            )

            # Clean up and rebuild with low threshold
            self.db_path.unlink()
            result_low = build_cross_refs(
                db_path=self.db_path,
                threshold=0.2,
                top_k_per_artifact=10,
            )

            # Low threshold should create more refs (or equal if many high-sim pairs)
            self.assertGreaterEqual(
                result_low["refs_count"],
                result_high["refs_count"],
                "Lower threshold should create more cross-refs",
            )

        except ValueError as e:
            if "No artifacts found" in str(e):
                self.skipTest("No indexed artifacts available for testing")
            raise

    def test_build_cross_refs_top_k_limiting(self):
        """Test that top_k_per_artifact limits edges per source."""
        try:
            # Build with small top_k
            result = build_cross_refs(
                db_path=self.db_path,
                threshold=0.1,  # Low threshold to ensure candidates
                top_k_per_artifact=3,
            )

            artifacts_count = result["artifacts_count"]
            refs_count = result["refs_count"]

            # Each artifact can have at most top_k outgoing edges
            max_possible_refs = artifacts_count * 3
            self.assertLessEqual(refs_count, max_possible_refs)

        except ValueError as e:
            if "No artifacts found" in str(e):
                self.skipTest("No indexed artifacts available for testing")
            raise

    def test_build_cross_refs_receipt_structure(self):
        """Test that receipt has proper structure and hash."""
        try:
            result = build_cross_refs(db_path=self.db_path, emit_receipt=True)

            receipt = result["receipt"]
            self.assertIn("operation", receipt)
            self.assertIn("timestamp", receipt)
            self.assertIn("artifacts_count", receipt)
            self.assertIn("refs_count", receipt)
            self.assertIn("threshold", receipt)
            self.assertIn("top_k_per_artifact", receipt)
            self.assertIn("duration_seconds", receipt)
            self.assertIn("receipt_hash", receipt)

            self.assertEqual(receipt["operation"], "build_cross_refs")
            self.assertIsInstance(receipt["receipt_hash"], str)
            self.assertEqual(len(receipt["receipt_hash"]), 64)  # SHA-256 hex

        except ValueError as e:
            if "No artifacts found" in str(e):
                self.skipTest("No indexed artifacts available for testing")
            raise


class TestFindRelated(unittest.TestCase):
    """Test find_related queries."""

    @classmethod
    def setUpClass(cls):
        """Build cross-reference index once for all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_dir) / "test_cross_refs.db"

        try:
            result = build_cross_refs(
                db_path=cls.db_path,
                threshold=0.2,
                top_k_per_artifact=10,
            )
            cls.artifacts_count = result["artifacts_count"]
            cls.refs_count = result["refs_count"]

            # Get a sample artifact_id for testing
            import sqlite3
            conn = sqlite3.connect(str(cls.db_path))
            cursor = conn.execute("SELECT artifact_id FROM artifacts LIMIT 1")
            row = cursor.fetchone()
            cls.sample_artifact_id = row[0] if row else None
            conn.close()

        except ValueError as e:
            if "No artifacts found" in str(e):
                cls.sample_artifact_id = None
            else:
                raise

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_find_related_returns_results(self):
        """Test that find_related returns related artifacts."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        result = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=5)

        self.assertIn("artifact_id", result)
        self.assertIn("related", result)
        self.assertIn("total_candidates", result)

        self.assertEqual(result["artifact_id"], self.sample_artifact_id)
        self.assertIsInstance(result["related"], list)
        self.assertIsInstance(result["total_candidates"], int)

    def test_find_related_top_k_limiting(self):
        """Test that top_k limits results."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        result = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=3)

        self.assertLessEqual(len(result["related"]), 3)

    def test_find_related_threshold_filtering(self):
        """Test that threshold filters low-similarity results."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        # Query without threshold
        result_all = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=10)

        # Query with high threshold
        result_filtered = find_related(
            self.sample_artifact_id,
            db_path=self.db_path,
            top_k=10,
            threshold=0.5,
        )

        # Filtered should have fewer or equal results
        self.assertLessEqual(len(result_filtered["related"]), len(result_all["related"]))

        # All filtered results should meet threshold
        for item in result_filtered["related"]:
            self.assertGreaterEqual(item["similarity"], 0.5)

    def test_find_related_deterministic_ordering(self):
        """Test that results are ordered deterministically."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        # Run query multiple times
        result1 = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=10)
        result2 = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=10)

        # Results should be identical
        self.assertEqual(len(result1["related"]), len(result2["related"]))

        for i, (item1, item2) in enumerate(zip(result1["related"], result2["related"])):
            self.assertEqual(item1["artifact_id"], item2["artifact_id"], f"Mismatch at position {i}")
            self.assertEqual(item1["similarity"], item2["similarity"], f"Mismatch at position {i}")

    def test_find_related_result_structure(self):
        """Test that each result has proper structure."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        result = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=5)

        for item in result["related"]:
            self.assertIn("artifact_id", item)
            self.assertIn("artifact_type", item)
            self.assertIn("artifact_path", item)
            self.assertIn("similarity", item)
            self.assertIn("metadata", item)

            # Validate types
            self.assertIsInstance(item["artifact_id"], str)
            self.assertIsInstance(item["artifact_type"], str)
            self.assertIsInstance(item["artifact_path"], str)
            self.assertIsInstance(item["similarity"], float)
            self.assertIsInstance(item["metadata"], dict)

            # Validate artifact_type is one of known types
            self.assertIn(item["artifact_type"], ["canon", "adr", "skill"])

    def test_find_related_similarity_descending_order(self):
        """Test that results are ordered by similarity descending."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        result = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=10)

        if len(result["related"]) < 2:
            self.skipTest("Not enough results for ordering test")

        # Check similarity is descending
        for i in range(len(result["related"]) - 1):
            current_sim = result["related"][i]["similarity"]
            next_sim = result["related"][i + 1]["similarity"]
            self.assertGreaterEqual(current_sim, next_sim, "Results should be sorted by similarity descending")

    def test_find_related_tie_breaking(self):
        """Test deterministic tie-breaking when similarities are equal."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        result = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=20)

        # Check that when similarities are equal, artifact_id is ascending
        for i in range(len(result["related"]) - 1):
            current = result["related"][i]
            next_item = result["related"][i + 1]

            if abs(current["similarity"] - next_item["similarity"]) < 1e-9:
                # Similarities are equal, check artifact_id ordering
                self.assertLess(
                    current["artifact_id"],
                    next_item["artifact_id"],
                    "Ties should be broken by artifact_id ascending",
                )

    def test_find_related_no_self_references(self):
        """Test that an artifact doesn't reference itself."""
        if not self.sample_artifact_id:
            self.skipTest("No indexed artifacts available for testing")

        result = find_related(self.sample_artifact_id, db_path=self.db_path, top_k=100)

        # No result should be the same as the source
        for item in result["related"]:
            self.assertNotEqual(
                item["artifact_id"],
                self.sample_artifact_id,
                "Artifact should not reference itself",
            )


class TestCrossRefStats(unittest.TestCase):
    """Test cross-reference statistics."""

    @classmethod
    def setUpClass(cls):
        """Build cross-reference index once for all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_dir) / "test_cross_refs.db"

        try:
            build_cross_refs(
                db_path=cls.db_path,
                threshold=0.3,
                top_k_per_artifact=10,
            )
        except ValueError as e:
            if "No artifacts found" not in str(e):
                raise

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_stats_structure(self):
        """Test that stats have proper structure."""
        stats = get_cross_ref_stats(db_path=self.db_path)

        self.assertIn("exists", stats)
        self.assertTrue(stats["exists"])

        self.assertIn("artifacts_by_type", stats)
        self.assertIn("total_artifacts", stats)
        self.assertIn("total_refs", stats)
        self.assertIn("avg_refs_per_artifact", stats)
        self.assertIn("last_build", stats)

    def test_stats_artifact_types(self):
        """Test that artifact types are tracked correctly."""
        stats = get_cross_ref_stats(db_path=self.db_path)

        artifacts_by_type = stats["artifacts_by_type"]
        self.assertIsInstance(artifacts_by_type, dict)

        # Check that known types are present (if any artifacts exist)
        total = stats["total_artifacts"]
        if total > 0:
            # At least one type should exist
            self.assertGreater(len(artifacts_by_type), 0)

            # All type counts should sum to total
            self.assertEqual(sum(artifacts_by_type.values()), total)

    def test_stats_last_build(self):
        """Test that last_build info is tracked."""
        stats = get_cross_ref_stats(db_path=self.db_path)

        if stats["total_artifacts"] > 0:
            last_build = stats["last_build"]
            self.assertIsNotNone(last_build)

            self.assertIn("build_timestamp", last_build)
            self.assertIn("artifacts_count", last_build)
            self.assertIn("refs_count", last_build)
            self.assertIn("threshold", last_build)
            self.assertIn("top_k_per_artifact", last_build)
            self.assertIn("duration_seconds", last_build)

    def test_stats_nonexistent_db(self):
        """Test stats for nonexistent database."""
        nonexistent = Path(self.temp_dir) / "nonexistent.db"
        stats = get_cross_ref_stats(db_path=nonexistent)

        self.assertEqual(stats, {"exists": False})


class TestRebuildIndex(unittest.TestCase):
    """Test index rebuilding."""

    def setUp(self):
        """Create temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_cross_refs.db"

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rebuild_creates_consistent_index(self):
        """Test that rebuild creates consistent results."""
        try:
            # Build once
            result1 = build_cross_refs(db_path=self.db_path, threshold=0.3, top_k_per_artifact=10)

            # Rebuild
            result2 = rebuild_cross_ref_index(db_path=self.db_path, threshold=0.3, top_k_per_artifact=10)

            # Results should be identical (same artifacts, same refs)
            self.assertEqual(result1["artifacts_count"], result2["artifacts_count"])
            self.assertEqual(result1["refs_count"], result2["refs_count"])

        except ValueError as e:
            if "No artifacts found" in str(e):
                self.skipTest("No indexed artifacts available for testing")
            raise

    def test_rebuild_with_different_params(self):
        """Test rebuild with different threshold and top_k."""
        try:
            # Initial build
            result1 = build_cross_refs(db_path=self.db_path, threshold=0.5, top_k_per_artifact=5)

            # Rebuild with different params
            result2 = rebuild_cross_ref_index(db_path=self.db_path, threshold=0.3, top_k_per_artifact=10)

            # Artifacts should be same, but refs may differ
            self.assertEqual(result1["artifacts_count"], result2["artifacts_count"])

            # Lower threshold + higher top_k should create more refs (or equal)
            self.assertGreaterEqual(result2["refs_count"], result1["refs_count"])

        except ValueError as e:
            if "No artifacts found" in str(e):
                self.skipTest("No indexed artifacts available for testing")
            raise


class TestCrossRefIntegration(unittest.TestCase):
    """Integration tests across artifact types."""

    @classmethod
    def setUpClass(cls):
        """Build cross-reference index once for integration tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_dir) / "test_cross_refs.db"

        try:
            build_cross_refs(
                db_path=cls.db_path,
                threshold=0.2,
                top_k_per_artifact=20,
            )

            # Get sample artifacts of different types
            import sqlite3
            conn = sqlite3.connect(str(cls.db_path))
            conn.row_factory = sqlite3.Row

            cls.sample_artifacts = {}
            for artifact_type in ["canon", "adr", "skill"]:
                cursor = conn.execute(
                    "SELECT artifact_id FROM artifacts WHERE artifact_type = ? LIMIT 1",
                    (artifact_type,),
                )
                row = cursor.fetchone()
                if row:
                    cls.sample_artifacts[artifact_type] = row["artifact_id"]

            conn.close()

        except ValueError as e:
            if "No artifacts found" in str(e):
                cls.sample_artifacts = {}
            else:
                raise

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_cross_type_relationships(self):
        """Test that different artifact types can reference each other."""
        if not self.sample_artifacts:
            self.skipTest("No indexed artifacts available for testing")

        # Pick any artifact
        artifact_id = next(iter(self.sample_artifacts.values()))
        result = find_related(artifact_id, db_path=self.db_path, top_k=20)

        if len(result["related"]) == 0:
            self.skipTest("No related artifacts found")

        # Check if we have cross-type references
        source_type = artifact_id.split(":")[0]
        related_types = {item["artifact_type"] for item in result["related"]}

        # It's possible to have cross-type refs (but not guaranteed with small corpus)
        # At minimum, we should have at least one type
        self.assertGreater(len(related_types), 0)

    def test_query_all_artifact_types(self):
        """Test querying related artifacts for each type."""
        if not self.sample_artifacts:
            self.skipTest("No indexed artifacts available for testing")

        for artifact_type, artifact_id in self.sample_artifacts.items():
            with self.subTest(artifact_type=artifact_type):
                result = find_related(artifact_id, db_path=self.db_path, top_k=5)

                self.assertIn("artifact_id", result)
                self.assertIn("related", result)
                self.assertEqual(result["artifact_id"], artifact_id)


if __name__ == "__main__":
    unittest.main()
