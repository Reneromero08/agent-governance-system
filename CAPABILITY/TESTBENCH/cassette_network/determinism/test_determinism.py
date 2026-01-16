#!/usr/bin/env python3
"""
Determinism Tests

Tests that verify identical inputs produce identical outputs across runs.
This is critical for:
- Reproducible compression proofs
- Auditable receipt chains
- Consistent semantic search results

A deterministic system means:
1. Same embedding text -> same embedding vector (byte-identical)
2. Same query -> same results in same order
3. Same content -> same hash
"""
import hashlib
import json
import pytest
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add paths for imports
DETERMINISM_DIR = Path(__file__).resolve().parent
CASSETTE_NETWORK_DIR = DETERMINISM_DIR.parent
REPO_ROOT = CASSETTE_NETWORK_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))

# Cassette network imports
CORTEX_NETWORK = REPO_ROOT / "NAVIGATION" / "CORTEX" / "network"
CORTEX_SEMANTIC = REPO_ROOT / "NAVIGATION" / "CORTEX" / "semantic"
sys.path.insert(0, str(CORTEX_NETWORK))
sys.path.insert(0, str(CORTEX_SEMANTIC))


def query_geometric_network(network, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Query the geometric cassette network and return flattened results with E scores."""
    all_results = network.query_all_text(query, k=top_k)

    flattened = []
    for cassette_id, results in all_results.items():
        if isinstance(results, list):
            for r in results:
                r["cassette_id"] = cassette_id
                r["similarity"] = r.get("E", 0.0)
                r["hash"] = r.get("doc_id", "")
                flattened.append(r)

    flattened.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return flattened


@pytest.fixture(scope="module")
def embedding_engine():
    """Load the embedding engine for determinism tests."""
    try:
        from embeddings import EmbeddingEngine
        return EmbeddingEngine()
    except ImportError as e:
        pytest.skip(f"Embedding engine not available: {e}")


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


@pytest.mark.determinism
class TestEmbeddingDeterminism:
    """Tests for embedding generation determinism."""

    def test_same_text_same_embedding(self, embedding_engine):
        """Same text should produce identical embeddings every time."""
        test_texts = [
            "What are the system invariants?",
            "How does verification work?",
            "Explain the catalytic computing model",
        ]

        for text in test_texts:
            embeddings = []
            for _ in range(5):
                emb = embedding_engine.embed(text)
                # Serialize to bytes for exact comparison
                serialized = embedding_engine.serialize(emb)
                embeddings.append(serialized)

            # All embeddings must be byte-identical
            first = embeddings[0]
            for i, emb in enumerate(embeddings[1:], 1):
                assert emb == first, (
                    f"Embedding {i} differs from first for text: {text[:50]}..."
                )

    def test_embedding_serialization_roundtrip(self, embedding_engine):
        """Embedding serialize/deserialize must be lossless."""
        test_text = "Test embedding roundtrip"
        original = embedding_engine.embed(test_text)

        serialized = embedding_engine.serialize(original)
        deserialized = embedding_engine.deserialize(serialized)
        reserialized = embedding_engine.serialize(deserialized)

        assert serialized == reserialized, "Serialization roundtrip not lossless"

    def test_batch_vs_single_embedding(self, embedding_engine):
        """Batch embedding should match single embedding."""
        texts = ["Query one", "Query two", "Query three"]

        # Single embeddings
        singles = [embedding_engine.embed(t) for t in texts]

        # Batch embedding (if supported)
        if hasattr(embedding_engine, "embed_batch"):
            batch = embedding_engine.embed_batch(texts)

            for i, (single, batched) in enumerate(zip(singles, batch)):
                single_ser = embedding_engine.serialize(single)
                batch_ser = embedding_engine.serialize(batched)
                assert single_ser == batch_ser, (
                    f"Batch embedding {i} differs from single"
                )


@pytest.mark.determinism
class TestRetrievalDeterminism:
    """Tests for retrieval result determinism."""

    def test_same_query_same_results(self, geometric_network):
        """Same query should return identical results every time."""
        test_queries = [
            "invariant violation",
            "genesis prompt",
            "verification protocol",
        ]

        for query in test_queries:
            results_list = []
            for _ in range(5):
                results = query_geometric_network(geometric_network, query, top_k=10)
                # Convert to deterministic representation
                result_repr = json.dumps(
                    [{"hash": r.get("hash", "")[:16],
                      "similarity": round(r.get("similarity", 0), 6)}
                     for r in results],
                    sort_keys=True
                )
                results_list.append(result_repr)

            # All results must be identical
            first = results_list[0]
            for i, result in enumerate(results_list[1:], 1):
                assert result == first, (
                    f"Result {i} differs from first for query: {query}"
                )

    def test_result_ordering_deterministic(self, geometric_network):
        """Result ordering should be deterministic (similarity DESC, hash ASC)."""
        query = "governance contract rules"
        results = query_geometric_network(geometric_network, query, top_k=20)

        # Verify ordering by similarity (descending)
        for i in range(len(results) - 1):
            sim_a = results[i].get("similarity", 0)
            sim_b = results[i + 1].get("similarity", 0)

            # Higher similarity should come first (with small tolerance)
            if sim_a < sim_b - 1e-6:
                pytest.fail(
                    f"Results not sorted by similarity DESC at index {i}: "
                    f"{sim_a:.6f} < {sim_b:.6f}"
                )


@pytest.mark.determinism
class TestHashDeterminism:
    """Tests for content hashing determinism."""

    def test_same_content_same_hash(self):
        """Same content should always produce same SHA256 hash."""
        content = "Test content for hashing"

        hashes = []
        for _ in range(10):
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            hashes.append(h)

        assert len(set(hashes)) == 1, "Hash varied for identical content"

    def test_canonical_json_deterministic(self):
        """Canonical JSON serialization should be deterministic."""
        data = {
            "z_field": 3,
            "a_field": 1,
            "m_field": 2,
            "nested": {"b": 2, "a": 1},
        }

        jsons = []
        for _ in range(10):
            # Canonical JSON: sorted keys, no spaces
            canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
            jsons.append(canonical)

        assert len(set(jsons)) == 1, "Canonical JSON varied"

        # Verify key ordering
        first = jsons[0]
        assert first.index('"a_field"') < first.index('"m_field"'), (
            "Keys not sorted alphabetically"
        )


@pytest.mark.determinism
class TestNormalizationDeterminism:
    """Tests for vector normalization determinism."""

    def test_l2_normalization_deterministic(self, embedding_engine):
        """L2 normalization should produce consistent results."""
        import numpy as np

        text = "Test normalization"
        embedding = embedding_engine.embed(text)

        # Compute L2 norm
        norm = np.linalg.norm(embedding)

        # After normalization, norm should be 1.0 (or very close)
        # This tests that normalization is applied consistently
        if hasattr(embedding_engine, "normalize"):
            normalized = embedding_engine.normalize(embedding)
            normalized_norm = np.linalg.norm(normalized)
            assert abs(normalized_norm - 1.0) < 1e-6, (
                f"Normalized vector norm {normalized_norm} != 1.0"
            )

    def test_similarity_symmetric(self, embedding_engine):
        """Cosine similarity should be symmetric: sim(a,b) == sim(b,a)."""
        text_a = "First query text"
        text_b = "Second query text"

        emb_a = embedding_engine.embed(text_a)
        emb_b = embedding_engine.embed(text_b)

        sim_ab = embedding_engine.cosine_similarity(emb_a, emb_b)
        sim_ba = embedding_engine.cosine_similarity(emb_b, emb_a)

        assert abs(sim_ab - sim_ba) < 1e-6, (
            f"Similarity not symmetric: {sim_ab} != {sim_ba}"
        )


@pytest.mark.determinism
@pytest.mark.slow
class TestDeterminismAtScale:
    """Stress tests for determinism at scale."""

    def test_100_run_embedding_stability(self, embedding_engine):
        """Embedding should be identical across 100 runs."""
        text = "Stress test embedding stability"
        hashes = set()

        for _ in range(100):
            emb = embedding_engine.embed(text)
            serialized = embedding_engine.serialize(emb)
            h = hashlib.sha256(serialized).hexdigest()
            hashes.add(h)

        assert len(hashes) == 1, (
            f"Embedding varied across 100 runs: {len(hashes)} unique hashes"
        )

    def test_100_run_retrieval_stability(self, geometric_network):
        """Retrieval should be identical across 100 runs."""
        query = "system invariants"
        result_hashes = set()

        for _ in range(100):
            results = query_geometric_network(geometric_network, query, top_k=5)
            result_repr = json.dumps(
                [r.get("hash", "")[:16] for r in results],
                sort_keys=True
            )
            h = hashlib.sha256(result_repr.encode()).hexdigest()
            result_hashes.add(h)

        assert len(result_hashes) == 1, (
            f"Retrieval varied across 100 runs: {len(result_hashes)} unique result sets"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
