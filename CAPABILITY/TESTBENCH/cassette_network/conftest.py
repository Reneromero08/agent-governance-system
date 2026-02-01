#!/usr/bin/env python3
"""
Pytest configuration for Cassette Network rigorous test suite.

Provides shared fixtures for:
- Cassette network initialization
- Ground truth test data loading
- Negative control fixtures
- Embedding engine access
- Semantic search validation
"""
import json
import pytest
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root detection
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Cassette network imports
CORTEX_NETWORK = REPO_ROOT / "NAVIGATION" / "CORTEX" / "network"
CORTEX_SEMANTIC = REPO_ROOT / "NAVIGATION" / "CORTEX" / "semantic"
sys.path.insert(0, str(CORTEX_NETWORK))
sys.path.insert(0, str(CORTEX_SEMANTIC))

# Fixture directory
FIXTURES_DIR = Path(__file__).parent


# =============================================================================
# CASSETTE NETWORK FIXTURES
# =============================================================================

class EmbeddingEngineWrapper:
    """Wrapper around GeometricReasoner for tests."""

    def __init__(self, reasoner):
        self.reasoner = reasoner

    def embed(self, text: str):
        """Generate embedding for text."""
        state = self.reasoner.initialize(text)
        return state.vector

    def embed_batch(self, texts: list):
        """Generate embeddings for multiple texts."""
        return [self.embed(t) for t in texts]

    def serialize(self, embedding) -> bytes:
        """Serialize embedding to bytes."""
        return embedding.tobytes()

    def deserialize(self, data: bytes):
        """Deserialize bytes to embedding."""
        import numpy as np
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, emb_a, emb_b) -> float:
        """Compute cosine similarity (embeddings are L2-normalized)."""
        import numpy as np
        return float(np.dot(emb_a, emb_b))


@pytest.fixture(scope="session")
def embedding_engine():
    """Load embedding engine for semantic similarity computation."""
    try:
        from geometric_cassette import GeometricCassetteNetwork
        network = GeometricCassetteNetwork.from_config(project_root=REPO_ROOT)
        return EmbeddingEngineWrapper(network.reasoner)
    except ImportError as e:
        pytest.skip(f"Embedding engine not available: {e}")
    except Exception as e:
        pytest.skip(f"Failed to initialize embedding engine: {e}")


@pytest.fixture(scope="session")
def network_hub():
    """Load the semantic network hub with all cassettes."""
    try:
        from network_hub import SemanticNetworkHub
        hub = SemanticNetworkHub(verbose=False, enforce_sync=False)

        # Load cassettes from configuration
        cassettes_json = CORTEX_NETWORK / "cassettes.json"
        if cassettes_json.exists():
            config = json.loads(cassettes_json.read_text(encoding="utf-8"))
            for cassette_config in config.get("cassettes", []):
                try:
                    from generic_cassette import GenericCassette
                    cassette = GenericCassette.from_config(cassette_config)
                    hub.register_cassette(cassette)
                except Exception as e:
                    print(f"Warning: Failed to load cassette {cassette_config.get('id')}: {e}")

        return hub
    except ImportError:
        pytest.skip("Network hub not available")


@pytest.fixture(scope="session")
def memory_cassette(tmp_path_factory):
    """Create a memory cassette for write tests."""
    try:
        from memory_cassette import MemoryCassette
        tmp_dir = tmp_path_factory.mktemp("cassette_test")
        db_path = tmp_dir / "test_memory.db"
        return MemoryCassette(db_path=db_path, agent_id="test_agent")
    except ImportError:
        pytest.skip("Memory cassette not available")


@pytest.fixture(scope="session")
def generic_cassette():
    """Load a generic read-only cassette for query tests."""
    try:
        from generic_cassette import GenericCassette

        # Use canon cassette as primary test target
        cassettes_dir = REPO_ROOT / "NAVIGATION" / "CORTEX" / "cassettes"
        canon_db = cassettes_dir / "canon.db"

        if not canon_db.exists():
            pytest.skip(f"Canon cassette not found: {canon_db}")

        return GenericCassette(
            db_path=canon_db,
            cassette_id="canon",
            capabilities=["fts", "semantic_search"],
        )
    except ImportError:
        pytest.skip("Generic cassette not available")


# =============================================================================
# GROUND TRUTH FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def ground_truth_fixture():
    """Load ground truth test cases from JSON fixture."""
    fixture_path = FIXTURES_DIR / "ground_truth" / "fixtures" / "retrieval_gold_standard.json"

    if not fixture_path.exists():
        pytest.skip(f"Ground truth fixture not found: {fixture_path}")

    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def negative_controls_fixture():
    """Load negative control test cases from JSON fixture."""
    fixture_path = FIXTURES_DIR / "adversarial" / "fixtures" / "negative_controls.json"

    if not fixture_path.exists():
        pytest.skip(f"Negative controls fixture not found: {fixture_path}")

    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def semantic_confusers_fixture():
    """Load semantic confuser test cases from JSON fixture."""
    fixture_path = FIXTURES_DIR / "adversarial" / "fixtures" / "semantic_confusers.json"

    if not fixture_path.exists():
        pytest.skip(f"Semantic confusers fixture not found: {fixture_path}")

    return json.loads(fixture_path.read_text(encoding="utf-8"))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def query_cassette_network(hub, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Query the cassette network and return flattened results.

    Args:
        hub: SemanticNetworkHub instance
        query: Search query string
        top_k: Maximum results per cassette

    Returns:
        List of result dicts with cassette_id added
    """
    all_results = hub.query_all(query, top_k=top_k)

    flattened = []
    for cassette_id, results in all_results.items():
        if isinstance(results, list):
            for r in results:
                r["cassette_id"] = cassette_id
                flattened.append(r)
        elif isinstance(results, dict) and "error" in results:
            # Skip error results
            continue

    # Sort by similarity descending
    flattened.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return flattened


def compute_recall(results: List[Dict], expected_hashes: List[str]) -> float:
    """
    Compute recall: what fraction of expected hashes were retrieved?

    Args:
        results: List of query results (must have 'hash' key)
        expected_hashes: List of expected content hashes

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not expected_hashes:
        return 1.0  # No expectations = trivially satisfied

    result_hashes = {r.get("hash", r.get("chunk_hash", "")) for r in results}
    expected_set = set(expected_hashes)

    found = result_hashes & expected_set
    return len(found) / len(expected_set)


def check_concepts_present(results: List[Dict], required_concepts: List[str]) -> List[str]:
    """
    Check which required concepts are present in results.

    Args:
        results: List of query results (must have 'content' or 'text' key)
        required_concepts: List of concepts that should appear

    Returns:
        List of missing concepts
    """
    # Combine all result text
    combined_text = ""
    for r in results:
        text = r.get("content", r.get("text", r.get("text_preview", "")))
        combined_text += " " + text.lower()

    missing = []
    for concept in required_concepts:
        if concept.lower() not in combined_text:
            missing.append(concept)

    return missing


def check_concepts_absent(results: List[Dict], forbidden_concepts: List[str]) -> List[str]:
    """
    Check which forbidden concepts are (incorrectly) present in results.

    Args:
        results: List of query results
        forbidden_concepts: List of concepts that should NOT appear

    Returns:
        List of forbidden concepts that were found
    """
    combined_text = ""
    for r in results:
        text = r.get("content", r.get("text", r.get("text_preview", "")))
        combined_text += " " + text.lower()

    found_forbidden = []
    for concept in forbidden_concepts:
        if concept.lower() in combined_text:
            found_forbidden.append(concept)

    return found_forbidden


def get_max_similarity(results: List[Dict]) -> float:
    """Get maximum similarity score from results."""
    if not results:
        return 0.0
    return max(r.get("similarity", 0.0) for r in results)


# =============================================================================
# PYTEST MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "ground_truth: marks tests that validate against known correct answers"
    )
    config.addinivalue_line(
        "markers", "adversarial: marks tests that verify rejection of invalid inputs"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests that verify identical outputs for identical inputs"
    )
    config.addinivalue_line(
        "markers", "compression: marks tests that validate compression claims"
    )
    config.addinivalue_line(
        "markers", "coverage: marks tests that measure corpus reachability"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "cross_model: marks tests for cross-model retrieval"
    )
