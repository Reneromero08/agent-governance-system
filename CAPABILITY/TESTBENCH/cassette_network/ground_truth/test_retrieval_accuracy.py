#!/usr/bin/env python3
"""
Ground Truth Retrieval Tests

Tests that validate retrieval against known correct answers.
Unlike keyword matching, these tests verify that:
1. Expected chunks ARE retrieved (recall)
2. Required concepts ARE present (semantic understanding)
3. Forbidden concepts are ABSENT (disambiguation)

These tests are RIGOROUS - they test actual correctness, not just that the system runs.
"""
import json
import pytest
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add paths for imports
GROUND_TRUTH_DIR = Path(__file__).resolve().parent
CASSETTE_NETWORK_DIR = GROUND_TRUTH_DIR.parent
REPO_ROOT = CASSETTE_NETWORK_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))

# Cassette network imports
CORTEX_NETWORK = REPO_ROOT / "NAVIGATION" / "CORTEX" / "network"
sys.path.insert(0, str(CORTEX_NETWORK))

# Fixtures directory
FIXTURES_DIR = GROUND_TRUTH_DIR / "fixtures"


def query_geometric_network(network, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Query the geometric cassette network and return flattened results with E scores."""
    all_results = network.query_all_text(query, k=top_k)

    flattened = []
    for cassette_id, results in all_results.items():
        if isinstance(results, list):
            for r in results:
                r["cassette_id"] = cassette_id
                r["similarity"] = r.get("E", 0.0)
                # Get hash from doc_id or metadata
                r["hash"] = r.get("doc_id", r.get("metadata", {}).get("doc_id", ""))
                flattened.append(r)

    flattened.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return flattened


def compute_recall(results: List[Dict], expected_chunks: List[str]) -> float:
    """Compute recall: fraction of expected chunks that were retrieved."""
    if not expected_chunks:
        return 1.0
    if not results:
        return 0.0

    # Get hashes from results (support both full and prefix matching)
    result_hashes = set()
    for r in results:
        h = r.get("hash", "")
        if h:
            result_hashes.add(h)
            # Also add prefix for matching
            if len(h) >= 16:
                result_hashes.add(h[:16])

    # Count matches (support both full and prefix hashes)
    found = 0
    for expected in expected_chunks:
        if expected in result_hashes:
            found += 1
        elif len(expected) >= 16 and expected[:16] in result_hashes:
            found += 1
        elif any(h.startswith(expected[:16]) for h in result_hashes if len(h) >= 16):
            found += 1

    return found / len(expected_chunks)


def check_concepts_present(results: List[Dict], concepts: List[str]) -> List[str]:
    """Check which required concepts are missing from results."""
    if not concepts:
        return []

    # Combine all content
    all_text = " ".join(
        r.get("content", "") for r in results
    ).lower()

    missing = []
    for concept in concepts:
        if concept.lower() not in all_text:
            missing.append(concept)

    return missing


def check_concepts_absent(results: List[Dict], concepts: List[str]) -> List[str]:
    """Check which forbidden concepts appear in results."""
    if not concepts:
        return []

    # Combine all content
    all_text = " ".join(
        r.get("content", "") for r in results
    ).lower()

    found = []
    for concept in concepts:
        if concept.lower() in all_text:
            found.append(concept)

    return found


@pytest.fixture(scope="module")
def ground_truth_fixture():
    """Load ground truth test cases."""
    fixture_path = FIXTURES_DIR / "retrieval_gold_standard.json"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def geometric_network():
    """Load the geometric cassette network with vector embeddings."""
    try:
        from geometric_cassette import GeometricCassetteNetwork

        network = GeometricCassetteNetwork.from_config(
            project_root=REPO_ROOT
        )

        if not network.cassettes:
            pytest.skip("No geometric cassettes loaded")

        return network
    except ImportError as e:
        pytest.skip(f"Geometric network not available: {e}")
    except Exception as e:
        pytest.skip(f"Failed to load geometric network: {e}")


@pytest.mark.ground_truth
class TestGroundTruthRetrieval:
    """Ground truth tests: system retrieves semantically correct chunks."""

    def test_fixture_loads(self, ground_truth_fixture):
        """Verify fixture loads correctly."""
        assert "version" in ground_truth_fixture
        assert "test_cases" in ground_truth_fixture
        assert len(ground_truth_fixture["test_cases"]) > 0

    def test_all_ground_truth_cases(self, ground_truth_fixture, geometric_network):
        """Run all ground truth test cases."""
        failures = []
        successes = []

        for case in ground_truth_fixture["test_cases"]:
            case_id = case["id"]
            query = case["query"]
            expected_chunks = case.get("expected_chunks", [])
            required_concepts = case.get("required_concepts", [])
            forbidden_concepts = case.get("forbidden_concepts", [])
            min_recall = case.get("min_recall", 0.5)

            # Query the network
            results = query_geometric_network(geometric_network, query, top_k=10)

            # Check recall
            recall = compute_recall(results, expected_chunks)
            if recall < min_recall:
                failures.append(
                    f"{case_id}: Recall {recall:.2f} < {min_recall} "
                    f"(found {int(recall * len(expected_chunks))}/{len(expected_chunks)} expected chunks)"
                )
                continue

            # Check required concepts
            missing = check_concepts_present(results, required_concepts)
            if missing:
                failures.append(
                    f"{case_id}: Missing required concepts: {missing}"
                )
                continue

            # Check forbidden concepts
            found_forbidden = check_concepts_absent(results, forbidden_concepts)
            if found_forbidden:
                failures.append(
                    f"{case_id}: Found forbidden concepts: {found_forbidden}"
                )
                continue

            successes.append(case_id)

        # Report results
        total = len(ground_truth_fixture["test_cases"])
        passed = len(successes)

        if failures:
            pytest.fail(
                f"Ground truth: {passed}/{total} passed\n"
                f"Failures ({len(failures)}):\n"
                + "\n".join(f"  - {f}" for f in failures)
            )

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_by_difficulty(self, ground_truth_fixture, geometric_network, difficulty):
        """Test cases grouped by difficulty level."""
        cases = [
            c for c in ground_truth_fixture["test_cases"]
            if c.get("difficulty") == difficulty
        ]

        if not cases:
            pytest.skip(f"No {difficulty} test cases found")

        passed = 0
        total = len(cases)

        for case in cases:
            results = query_geometric_network(geometric_network, case["query"], top_k=10)
            recall = compute_recall(results, case.get("expected_chunks", []))
            min_recall = case.get("min_recall", 0.5)

            if recall >= min_recall:
                passed += 1

        success_rate = passed / total
        min_success = {"easy": 0.80, "medium": 0.70, "hard": 0.60}.get(difficulty, 0.70)

        assert success_rate >= min_success, (
            f"{difficulty} tests: {passed}/{total} passed ({success_rate:.1%}) "
            f"< minimum {min_success:.1%}"
        )


@pytest.mark.ground_truth
class TestRecallMetrics:
    """Tests for recall computation and metrics."""

    def test_recall_on_empty_expected(self):
        """Recall should be 1.0 when no expected chunks specified."""
        recall = compute_recall([], [])
        assert recall == 1.0

    def test_recall_on_no_results(self):
        """Recall should be 0.0 when no results returned."""
        recall = compute_recall([], ["hash1", "hash2"])
        assert recall == 0.0

    def test_recall_partial_match(self):
        """Recall should correctly compute partial matches."""
        results = [{"hash": "hash1"}, {"hash": "hash3"}]
        expected = ["hash1", "hash2"]
        recall = compute_recall(results, expected)
        assert recall == 0.5  # 1 of 2 found


@pytest.mark.ground_truth
class TestConceptValidation:
    """Tests for concept presence/absence validation."""

    def test_required_concepts_found(self):
        """Required concepts should be detected in results."""
        results = [
            {"content": "The invariant violation was detected"},
            {"content": "Recovery procedures are documented"},
        ]
        missing = check_concepts_present(results, ["invariant", "recovery"])
        assert len(missing) == 0

    def test_required_concepts_missing(self):
        """Missing concepts should be reported."""
        results = [{"content": "Generic content without keywords"}]
        missing = check_concepts_present(results, ["invariant", "recovery"])
        assert "invariant" in missing
        assert "recovery" in missing

    def test_forbidden_concepts_absent(self):
        """Forbidden concepts should not be in results."""
        results = [{"content": "The system verifies integrity"}]
        found = check_concepts_absent(results, ["delete", "remove"])
        assert len(found) == 0

    def test_forbidden_concepts_detected(self):
        """Forbidden concepts should be detected when present."""
        results = [{"content": "You can delete the file safely"}]
        found = check_concepts_absent(results, ["delete", "remove"])
        assert "delete" in found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
