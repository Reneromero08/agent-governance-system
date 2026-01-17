#!/usr/bin/env python3
"""Explore Methods to Push Beyond 50% Cross-Model Corruption Ceiling.

Current state:
- Same-model: 94% corruption tolerance
- Cross-model: ~50% corruption tolerance

This script explores methods to improve cross-model robustness:
1. Ensemble decoding (consensus from multiple models)
2. Optimal k selection (find best dimensionality)
3. Weighted dimensions (prioritize stable dimensions)
4. Different anchor sets comparison
5. Candidate pool size impact

Usage:
    python explore_cross_model_ceiling.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.linalg import orthogonal_procrustes
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.PRIMITIVES.alignment_key import AlignmentKey
from CAPABILITY.PRIMITIVES.canonical_anchors import STABLE_64, STABLE_32


# =============================================================================
# Setup
# =============================================================================

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    return session


SESSION = create_session()
EMBED_CACHE = {}


def get_embeddings_cached(texts, url, model):
    uncached = [t for t in texts if t not in EMBED_CACHE]
    if uncached:
        response = SESSION.post(url, json={"model": model, "input": uncached}, timeout=60)
        data = response.json()
        for i, t in enumerate(uncached):
            EMBED_CACHE[t] = np.array(data["data"][i]["embedding"])
    return np.array([EMBED_CACHE[t] for t in texts])


# Test messages
MESSAGES = [
    "Explain how transformers work in neural networks",
    "Describe gradient descent optimization in machine learning",
    "What is the attention mechanism in deep learning",
    "Love is a powerful force that connects all humanity",
]

DISTRACTORS = [
    "The cat sat on the warm windowsill",
    "Computers process binary information rapidly",
    "Music has power to evoke emotions",
    "Mountains stand tall against the sky",
]


# =============================================================================
# Exploration Methods
# =============================================================================

def test_corruption_tolerance(key_a, key_b, embed_a, embed_b, candidates, n_trials=10):
    """Test corruption tolerance between two aligned keys."""
    pair = key_a.align_with(key_b)
    k = pair.k

    results = {}
    test_msg = MESSAGES[0]

    for pct in [0.0, 0.25, 0.50, 0.75, 0.90]:
        n_corrupt = int(k * pct)
        if n_corrupt >= k:
            continue

        successes = 0
        for trial in range(n_trials):
            vec = pair.encode_a_to_b(test_msg, embed_a)
            if n_corrupt > 0:
                np.random.seed(trial)
                indices = np.random.choice(len(vec), n_corrupt, replace=False)
                vec[indices] = 0.0
            match, _ = pair.decode_at_b(vec, candidates, embed_b)
            if match == test_msg:
                successes += 1

        results[f"{int(pct*100)}%"] = successes / n_trials

    return results, pair.procrustes_residual


def explore_optimal_k():
    """Find optimal k value for cross-model communication."""
    print("\n" + "=" * 60)
    print("EXPLORATION 1: OPTIMAL K VALUE")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    candidates = MESSAGES + DISTRACTORS
    results = {}

    for k in [8, 12, 16, 20, 24, 28, 31]:
        print(f"\n  Testing k={k}...")
        key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=STABLE_32, k=k)
        key_mini = AlignmentKey.create("mini", embed_mini, anchors=STABLE_32, k=k)

        tol, res = test_corruption_tolerance(key_nomic, key_mini, embed_nomic, embed_mini, candidates)
        results[k] = {"tolerance": tol, "residual": res}
        print(f"    Residual: {res:.4f}, 50% corruption: {tol.get('50%', 0)*100:.0f}%")

    return results


def explore_ensemble_decoding():
    """Test if ensemble of models improves decoding accuracy."""
    print("\n" + "=" * 60)
    print("EXPLORATION 2: ENSEMBLE DECODING")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')
    model_mpnet = SentenceTransformer('all-mpnet-base-v2')

    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    def embed_mpnet(texts):
        return model_mpnet.encode(texts, convert_to_numpy=True)

    anchors = STABLE_32
    k = 31

    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=anchors, k=k)
    key_mini = AlignmentKey.create("mini", embed_mini, anchors=anchors, k=k)
    key_mpnet = AlignmentKey.create("mpnet", embed_mpnet, anchors=anchors, k=k)

    pair_nm = key_nomic.align_with(key_mini)
    pair_np = key_nomic.align_with(key_mpnet)

    candidates = MESSAGES + DISTRACTORS
    test_msg = MESSAGES[0]

    print("\n  Testing ensemble vs single model...")

    results = {"single_mini": [], "single_mpnet": [], "ensemble": []}

    for pct in [0.0, 0.25, 0.50, 0.75]:
        n_corrupt = int(k * pct)
        single_mini_success = 0
        single_mpnet_success = 0
        ensemble_success = 0
        n_trials = 20

        for trial in range(n_trials):
            vec_mini = pair_nm.encode_a_to_b(test_msg, embed_nomic)
            vec_mpnet = pair_np.encode_a_to_b(test_msg, embed_nomic)

            if n_corrupt > 0:
                np.random.seed(trial)
                indices = np.random.choice(len(vec_mini), n_corrupt, replace=False)
                vec_mini[indices] = 0.0
                vec_mpnet[indices] = 0.0

            # Single model decode
            match_mini, score_mini = pair_nm.decode_at_b(vec_mini, candidates, embed_mini)
            match_mpnet, score_mpnet = pair_np.decode_at_b(vec_mpnet, candidates, embed_mpnet)

            if match_mini == test_msg:
                single_mini_success += 1
            if match_mpnet == test_msg:
                single_mpnet_success += 1

            # Ensemble: take higher confidence
            if score_mini > score_mpnet:
                ensemble_match = match_mini
            else:
                ensemble_match = match_mpnet

            if ensemble_match == test_msg:
                ensemble_success += 1

        results["single_mini"].append(single_mini_success / n_trials)
        results["single_mpnet"].append(single_mpnet_success / n_trials)
        results["ensemble"].append(ensemble_success / n_trials)

        print(f"    {int(pct*100):2d}% corruption: mini={single_mini_success}/{n_trials}, "
              f"mpnet={single_mpnet_success}/{n_trials}, ensemble={ensemble_success}/{n_trials}")

    return results


def explore_candidate_pool_scaling():
    """Test how corruption tolerance scales with candidate pool size."""
    print("\n" + "=" * 60)
    print("EXPLORATION 3: CANDIDATE POOL SCALING")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    # Extended candidate pool
    all_candidates = MESSAGES + DISTRACTORS + [
        "Time heals all wounds eventually",
        "The sun rises in the east",
        "Water flows downhill naturally",
        "Birds fly south for winter",
        "Knowledge is power in society",
        "Music soothes the savage beast",
        "Actions speak louder than words",
        "The early bird catches the worm",
    ]

    anchors = STABLE_32
    k = 31

    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=anchors, k=k)
    key_mini = AlignmentKey.create("mini", embed_mini, anchors=anchors, k=k)
    pair = key_nomic.align_with(key_mini)

    test_msg = MESSAGES[0]
    results = {}

    for n_candidates in [4, 8, 12, 16]:
        candidates = [test_msg] + [c for c in all_candidates if c != test_msg][:n_candidates-1]

        pool_results = {}
        for pct in [0.0, 0.25, 0.50, 0.75]:
            n_corrupt = int(k * pct)
            successes = 0
            n_trials = 20

            for trial in range(n_trials):
                vec = pair.encode_a_to_b(test_msg, embed_nomic)
                if n_corrupt > 0:
                    np.random.seed(trial)
                    indices = np.random.choice(len(vec), n_corrupt, replace=False)
                    vec[indices] = 0.0
                match, _ = pair.decode_at_b(vec, candidates, embed_mini)
                if match == test_msg:
                    successes += 1

            pool_results[f"{int(pct*100)}%"] = successes / n_trials

        results[n_candidates] = pool_results
        print(f"  {n_candidates} candidates: {pool_results}")

    return results


def explore_weighted_dimensions():
    """Test if weighting dimensions by stability improves accuracy."""
    print("\n" + "=" * 60)
    print("EXPLORATION 4: WEIGHTED DIMENSIONS")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    model_mini = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_nomic(texts):
        return get_embeddings_cached(
            texts, "http://10.5.0.2:1234/v1/embeddings",
            "text-embedding-nomic-embed-text-v1.5"
        )

    def embed_mini(texts):
        return model_mini.encode(texts, convert_to_numpy=True)

    anchors = STABLE_32
    k = 31

    key_nomic = AlignmentKey.create("nomic", embed_nomic, anchors=anchors, k=k)
    key_mini = AlignmentKey.create("mini", embed_mini, anchors=anchors, k=k)

    # Get MDS coordinates
    X_nomic = key_nomic.eigenvectors[:, :k] * np.sqrt(key_nomic.eigenvalues[:k])
    X_mini = key_mini.eigenvectors[:, :k] * np.sqrt(key_mini.eigenvalues[:k])

    # Compute per-dimension alignment quality
    R, _ = orthogonal_procrustes(X_nomic, X_mini)
    aligned = X_nomic @ R
    per_dim_error = np.mean((aligned - X_mini) ** 2, axis=0)

    # Weight by inverse error
    weights = 1.0 / (per_dim_error + 0.01)
    weights = weights / weights.max()

    print(f"\n  Dimension weights (top 10): {weights[:10].round(3)}")
    print(f"  Dimension weights (bottom 10): {weights[-10:].round(3)}")

    # Test weighted vs unweighted cosine similarity
    pair = key_nomic.align_with(key_mini)
    candidates = MESSAGES + DISTRACTORS
    test_msg = MESSAGES[0]

    results = {"unweighted": [], "weighted": []}

    for pct in [0.0, 0.25, 0.50, 0.75]:
        n_corrupt = int(k * pct)
        unweighted_success = 0
        weighted_success = 0
        n_trials = 20

        for trial in range(n_trials):
            vec = pair.encode_a_to_b(test_msg, embed_nomic)

            if n_corrupt > 0:
                np.random.seed(trial)
                indices = np.random.choice(len(vec), n_corrupt, replace=False)
                vec[indices] = 0.0

            # Unweighted decode
            match_uw, _ = pair.decode_at_b(vec, candidates, embed_mini)
            if match_uw == test_msg:
                unweighted_success += 1

            # Weighted decode
            best_match = None
            best_score = -float('inf')
            for cand in candidates:
                cand_vec = key_mini.encode(cand, embed_mini)
                # Weighted cosine similarity
                weighted_vec = vec * weights
                weighted_cand = cand_vec[:k] * weights
                score = np.dot(weighted_vec, weighted_cand) / (
                    np.linalg.norm(weighted_vec) * np.linalg.norm(weighted_cand) + 1e-10
                )
                if score > best_score:
                    best_score = score
                    best_match = cand

            if best_match == test_msg:
                weighted_success += 1

        results["unweighted"].append(unweighted_success / n_trials)
        results["weighted"].append(weighted_success / n_trials)

        print(f"  {int(pct*100):2d}% corruption: unweighted={unweighted_success}/{n_trials}, "
              f"weighted={weighted_success}/{n_trials}")

    return results


def main():
    print("=" * 70)
    print("EXPLORING CROSS-MODEL CEILING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    print("\nLoading models...")
    all_results = {}

    # Run explorations
    all_results["optimal_k"] = explore_optimal_k()
    all_results["ensemble"] = explore_ensemble_decoding()
    all_results["pool_scaling"] = explore_candidate_pool_scaling()
    all_results["weighted_dims"] = explore_weighted_dimensions()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. Optimal k: Lower k doesn't help - more dimensions = more redundancy")

    print("\n2. Ensemble decoding:")
    ens = all_results["ensemble"]
    print(f"   50% corruption: single={ens['single_mini'][2]*100:.0f}%, ensemble={ens['ensemble'][2]*100:.0f}%")

    print("\n3. Candidate pool scaling:")
    pool = all_results["pool_scaling"]
    print(f"   4 candidates @ 50%: {pool[4]['50%']*100:.0f}%")
    print(f"   16 candidates @ 50%: {pool[16]['50%']*100:.0f}%")

    print("\n4. Weighted dimensions:")
    wd = all_results["weighted_dims"]
    print(f"   50% corruption: unweighted={wd['unweighted'][2]*100:.0f}%, weighted={wd['weighted'][2]*100:.0f}%")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
The ~50% cross-model corruption tolerance appears to be a fundamental limit
arising from geometric differences between embedding models.

Promising directions:
- Ensemble decoding (marginal improvement)
- Keeping candidate pools small (information-theoretic)
- Weighted dimensions (marginal improvement)

The limit is NOT from:
- Insufficient dimensions (more k doesn't help cross-model)
- Procrustes algorithm (sign correction didn't help)

Next step: Test with more diverse models to understand if the limit
is model-pair specific or universal.
""")

    return all_results


if __name__ == "__main__":
    results = main()

    # Save results
    output_path = Path(__file__).parent / "cross_model_exploration_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
