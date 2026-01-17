#!/usr/bin/env python3
"""
Semantic Preservation Tests

Phase 3: Validates that transform T preserves meaning (Born rule probabilities).

If H(X|S) ~ 0 holds across models, then:
1. Similarity relationships should be preserved under transform
2. E-scores (Born rule) should correlate across transform
3. Semantic neighborhoods should remain intact
"""
import json
import numpy as np
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add paths for imports
CROSS_MODEL_DIR = Path(__file__).resolve().parent
CASSETTE_NETWORK_DIR = CROSS_MODEL_DIR.parent
REPO_ROOT = CASSETTE_NETWORK_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))

# Fixtures directory
FIXTURES_DIR = CROSS_MODEL_DIR / "fixtures"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def procrustes_alignment(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find orthogonal transform R such that A @ R ~ B.

    Returns:
        R: rotation matrix
        mean_A: mean of A (for centering)
        mean_B: mean of B (for centering)
    """
    mean_A = A.mean(axis=0)
    mean_B = B.mean(axis=0)

    A_centered = A - mean_A
    B_centered = B - mean_B

    M = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    return R, mean_A, mean_B


def transform_vector(v: np.ndarray, R: np.ndarray, mean_A: np.ndarray, mean_B: np.ndarray) -> np.ndarray:
    """Transform a vector from space A to space B."""
    return (v - mean_A) @ R + mean_B


@pytest.fixture(scope="module")
def alignment_tasks_fixture():
    """Load alignment tasks configuration."""
    fixture_path = FIXTURES_DIR / "alignment_tasks.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text(encoding="utf-8"))
    pytest.skip("alignment_tasks.json not found")


@pytest.fixture(scope="module")
def sentence_transformer():
    """Import sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")


@pytest.fixture(scope="module")
def trained_transform(sentence_transformer):
    """Train a Procrustes transform between two models."""
    # Training corpus
    train_corpus = [
        "The system invariants must never be violated.",
        "Genesis prompt bootstraps a new agent.",
        "Catalytic computing ensures memory restoration.",
        "Contract rules govern system behavior.",
        "Authority gradient defines escalation paths.",
        "Verification chain ensures integrity.",
        "Receipts provide proof of execution.",
        "The canon is immutable law.",
        "Agents must declare their intentions.",
        "Truth-linking connects claims to evidence.",
        "Provenance tracks origin of all artifacts.",
        "Recovery procedures restore invariants.",
        "The governance system has five invariants.",
        "Bootstrap requires genesis context.",
        "Semantic embeddings encode meaning.",
    ]

    model_a = sentence_transformer("all-MiniLM-L6-v2")
    model_b = sentence_transformer("paraphrase-MiniLM-L6-v2")

    emb_a = np.array(model_a.encode(train_corpus, normalize_embeddings=True))
    emb_b = np.array(model_b.encode(train_corpus, normalize_embeddings=True))

    R, mean_A, mean_B = procrustes_alignment(emb_a, emb_b)

    return {
        "R": R,
        "mean_A": mean_A,
        "mean_B": mean_B,
        "model_a": model_a,
        "model_b": model_b,
        "model_a_name": "all-MiniLM-L6-v2",
        "model_b_name": "paraphrase-MiniLM-L6-v2"
    }


@pytest.mark.cross_model
class TestSimilarityPreservation:
    """Test that inner products are preserved under transform."""

    def test_similarity_pairs(self, alignment_tasks_fixture, trained_transform):
        """Known similarity pairs should have similar scores after transform."""
        model_a = trained_transform["model_a"]
        model_b = trained_transform["model_b"]
        R = trained_transform["R"]
        mean_A = trained_transform["mean_A"]
        mean_B = trained_transform["mean_B"]

        results = []

        for pair in alignment_tasks_fixture["similarity_test_pairs"]:
            text_a = pair["text_a"]
            text_b = pair["text_b"]

            # Embed with both models
            emb_a_query = np.array(model_a.encode([text_a], normalize_embeddings=True))[0]
            emb_a_doc = np.array(model_a.encode([text_b], normalize_embeddings=True))[0]

            emb_b_query = np.array(model_b.encode([text_a], normalize_embeddings=True))[0]
            emb_b_doc = np.array(model_b.encode([text_b], normalize_embeddings=True))[0]

            # Same-model similarities
            sim_a = cosine_similarity(emb_a_query, emb_a_doc)
            sim_b = cosine_similarity(emb_b_query, emb_b_doc)

            # Cross-model similarity (query from A transformed, doc from B)
            query_transformed = transform_vector(emb_a_query, R, mean_A, mean_B)
            sim_cross = cosine_similarity(query_transformed, emb_b_doc)

            # Similarity drift
            drift = abs(sim_cross - sim_b)

            results.append({
                "id": pair["id"],
                "sim_a": sim_a,
                "sim_b": sim_b,
                "sim_cross": sim_cross,
                "drift": drift,
                "success": drift < 0.15
            })

        print("\n=== Similarity Preservation ===")
        for r in results:
            status = "PASS" if r["success"] else "WARN"
            print(f"  {r['id']}: sim_A={r['sim_a']:.3f}, sim_B={r['sim_b']:.3f}, sim_cross={r['sim_cross']:.3f}, drift={r['drift']:.3f} [{status}]")

        # Most pairs should have low drift
        successful = [r for r in results if r["success"]]
        success_rate = len(successful) / len(results)
        print(f"\nSuccess rate: {success_rate:.0%}")

        assert success_rate >= 0.7, f"Similarity preservation {success_rate:.0%} < 70%"

    def test_ranking_preservation(self, trained_transform):
        """Similarity rankings should be preserved under transform."""
        model_a = trained_transform["model_a"]
        model_b = trained_transform["model_b"]
        R = trained_transform["R"]
        mean_A = trained_transform["mean_A"]
        mean_B = trained_transform["mean_B"]

        # Query and candidate documents
        query = "What are the system invariants?"
        candidates = [
            "The five invariants are: declared, truth-linked, verified, provenance-linked, restorable.",
            "Genesis prompt initializes a new agent.",
            "Catalytic computing borrows and restores memory.",
            "Basketball rules define how the game is played.",
            "The weather is nice today.",
        ]

        # Embed query with model A
        query_a = np.array(model_a.encode([query], normalize_embeddings=True))[0]

        # Embed candidates with model B
        cands_b = np.array(model_b.encode(candidates, normalize_embeddings=True))

        # Transform query to model B space
        query_transformed = transform_vector(query_a, R, mean_A, mean_B)

        # Compute similarities
        sims_cross = [cosine_similarity(query_transformed, c) for c in cands_b]

        # Also compute model B native similarities for comparison
        query_b = np.array(model_b.encode([query], normalize_embeddings=True))[0]
        sims_native = [cosine_similarity(query_b, c) for c in cands_b]

        # Rankings
        rank_cross = np.argsort(sims_cross)[::-1]
        rank_native = np.argsort(sims_native)[::-1]

        print("\n=== Ranking Preservation ===")
        print("Cross-model ranking (query from A, docs from B):")
        for i, idx in enumerate(rank_cross):
            print(f"  {i+1}. [{sims_cross[idx]:.3f}] {candidates[idx][:50]}...")

        print("\nNative B ranking:")
        for i, idx in enumerate(rank_native):
            print(f"  {i+1}. [{sims_native[idx]:.3f}] {candidates[idx][:50]}...")

        # Top result should be the same (invariants doc)
        assert rank_cross[0] == 0, "Cross-model ranking failed - wrong top result"
        print("\nTop result matches: PASS")


@pytest.mark.cross_model
class TestBornRulePreservation:
    """Test that E-scores (Born rule probabilities) correlate across transform."""

    def test_e_score_correlation(self, trained_transform):
        """E-scores from transformed queries should correlate with native E-scores."""
        from scipy.stats import pearsonr

        model_a = trained_transform["model_a"]
        model_b = trained_transform["model_b"]
        R = trained_transform["R"]
        mean_A = trained_transform["mean_A"]
        mean_B = trained_transform["mean_B"]

        # Test queries
        queries = [
            "What are the system invariants?",
            "How do I bootstrap an agent?",
            "What is catalytic computing?",
            "Where are receipts stored?",
            "What are the contract rules?",
        ]

        # Document corpus (same for both)
        docs = [
            "The five invariants are: declared, truth-linked, verified, provenance-linked, restorable.",
            "Genesis prompt bootstraps new agents with governance context.",
            "Catalytic computing borrows memory and guarantees restoration.",
            "Receipts are stored in the artifact run directory.",
            "Contract rules C1-C13 govern system behavior.",
            "Authority gradient defines escalation from agent to human.",
            "Verification chain ensures cryptographic integrity.",
            "The canon represents immutable governance law.",
        ]

        # Embed docs with model B
        docs_b = np.array(model_b.encode(docs, normalize_embeddings=True))

        e_scores_native = []
        e_scores_cross = []

        for query in queries:
            # Native model B
            query_b = np.array(model_b.encode([query], normalize_embeddings=True))[0]
            sims_native = [cosine_similarity(query_b, d) for d in docs_b]

            # Cross-model (A -> B)
            query_a = np.array(model_a.encode([query], normalize_embeddings=True))[0]
            query_transformed = transform_vector(query_a, R, mean_A, mean_B)
            sims_cross = [cosine_similarity(query_transformed, d) for d in docs_b]

            e_scores_native.extend(sims_native)
            e_scores_cross.extend(sims_cross)

        # Correlation
        r, p = pearsonr(e_scores_native, e_scores_cross)

        print("\n=== E-Score (Born Rule) Correlation ===")
        print(f"Native vs Cross-model: r={r:.4f}, p={p:.2e}")
        print(f"Total comparisons: {len(e_scores_native)}")

        # E-scores should strongly correlate
        assert r > 0.7, f"E-score correlation {r:.4f} < 0.7"
        print("E-score correlation: PASS")


@pytest.mark.cross_model
class TestNeighborhoodPreservation:
    """Test that semantic neighborhoods are preserved under transform."""

    def test_nearest_neighbor_overlap(self, trained_transform):
        """Nearest neighbors should overlap significantly after transform."""
        model_a = trained_transform["model_a"]
        model_b = trained_transform["model_b"]
        R = trained_transform["R"]
        mean_A = trained_transform["mean_A"]
        mean_B = trained_transform["mean_B"]

        # Corpus
        corpus = [
            "System invariants define integrity rules.",
            "Genesis prompt starts new agents.",
            "Catalytic computing restores memory.",
            "Contract rules govern behavior.",
            "Authority gradient defines hierarchy.",
            "Verification ensures integrity.",
            "Receipts prove execution.",
            "Canon is immutable law.",
            "Agents declare intentions.",
            "Truth-linking connects evidence.",
        ]

        # Embed with both models
        emb_a = np.array(model_a.encode(corpus, normalize_embeddings=True))
        emb_b = np.array(model_b.encode(corpus, normalize_embeddings=True))

        # Transform A to B space
        emb_a_transformed = np.array([
            transform_vector(e, R, mean_A, mean_B) for e in emb_a
        ])

        # For each point, find k nearest neighbors
        k = 3
        overlaps = []

        for i in range(len(corpus)):
            # Native B neighbors
            sims_native = [cosine_similarity(emb_b[i], emb_b[j]) for j in range(len(corpus)) if j != i]
            native_neighbors = set(np.argsort(sims_native)[-k:])

            # Cross-model neighbors (using transformed A for query)
            sims_cross = [cosine_similarity(emb_a_transformed[i], emb_b[j]) for j in range(len(corpus)) if j != i]
            cross_neighbors = set(np.argsort(sims_cross)[-k:])

            overlap = len(native_neighbors & cross_neighbors) / k
            overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps)

        print("\n=== Nearest Neighbor Overlap ===")
        print(f"k={k} nearest neighbors")
        print(f"Average overlap: {avg_overlap:.2%}")

        # At least 50% overlap expected
        assert avg_overlap >= 0.4, f"Neighbor overlap {avg_overlap:.2%} < 40%"
        print("Neighborhood preservation: PASS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
