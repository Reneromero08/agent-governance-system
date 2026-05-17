#!/usr/bin/env python3
"""E.X.3.2b: Real Model Alignment Resistance Test

The synthetic test (alignment_resistance.py) showed NO gap.
This means the gap we saw (random +0.96 vs trained +0.43) is NOT from
simple semantic similarity - it's something else in trained models.

This test uses real sentence-transformers models to:
1. Confirm the gap exists with real models
2. Compare real models vs random
3. Identify what property causes the resistance

Usage:
    python -m benchmarks.validation.real_model_resistance
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.mds import squared_distance_matrix, classical_mds
from lib.procrustes import procrustes_align, cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("WARNING: sentence-transformers not available")


# =============================================================================
# ANCHOR WORDS (same as used in E.X.1-2)
# =============================================================================

ANCHOR_WORDS = [
    # Core concepts
    "time", "space", "energy", "matter", "light",
    "force", "motion", "wave", "particle", "field",
    # Abstract
    "truth", "beauty", "justice", "freedom", "power",
    "knowledge", "wisdom", "love", "fear", "hope",
    # Concrete
    "water", "fire", "earth", "air", "stone",
    "tree", "river", "mountain", "ocean", "star",
    # Human
    "mind", "body", "soul", "heart", "brain",
    "thought", "feeling", "memory", "dream", "vision",
    # Actions
    "create", "destroy", "build", "break", "grow",
    "change", "move", "stop", "begin", "end",
    # Relations
    "cause", "effect", "order", "chaos", "balance",
    "unity", "division", "connection", "separation", "flow",
    # Additional for 64 total
    "language", "meaning", "symbol", "pattern", "structure",
]


def generate_random_embeddings(words: list, dim: int, seed: int) -> dict:
    """Generate random L2-normalized embeddings."""
    rng = np.random.default_rng(seed)
    embeddings = {}
    for word in words:
        vec = rng.standard_normal(dim)
        vec = vec / np.linalg.norm(vec)
        embeddings[word] = vec
    return embeddings


def get_model_embeddings(model_name: str, words: list) -> dict:
    """Get embeddings from a real sentence-transformers model."""
    model = SentenceTransformer(model_name)
    vectors = model.encode(words, convert_to_numpy=True)

    embeddings = {}
    for i, word in enumerate(words):
        vec = vectors[i]
        vec = vec / np.linalg.norm(vec)  # L2 normalize
        embeddings[word] = vec

    return embeddings, model.get_sentence_embedding_dimension()


def compute_alignment_metrics(emb_a: dict, emb_b: dict, words: list) -> dict:
    """Compute alignment improvement between two embedding sets."""
    common_words = [w for w in words if w in emb_a and w in emb_b]
    n = len(common_words)

    if n < 3:
        return {'improvement': 0.0, 'raw_similarity': 0.0, 'aligned_similarity': 0.0}

    X_a = np.array([emb_a[w] for w in common_words])
    X_b = np.array([emb_b[w] for w in common_words])

    # MDS on each
    D2_a = squared_distance_matrix(X_a)
    D2_b = squared_distance_matrix(X_b)

    coords_a, _, _ = classical_mds(D2_a)
    coords_b, _, _ = classical_mds(D2_b)

    # Match dimensions
    k = min(coords_a.shape[1], coords_b.shape[1])
    coords_a = coords_a[:, :k]
    coords_b = coords_b[:, :k]

    # Procrustes alignment
    R, residual = procrustes_align(coords_a, coords_b)
    coords_a_aligned = coords_a @ R

    # Measure similarities
    raw_sims = []
    aligned_sims = []
    for i in range(n):
        raw_sims.append(cosine_similarity(coords_a[i], coords_b[i]))
        aligned_sims.append(cosine_similarity(coords_a_aligned[i], coords_b[i]))

    return {
        'raw_similarity': float(np.mean(raw_sims)),
        'aligned_similarity': float(np.mean(aligned_sims)),
        'improvement': float(np.mean(aligned_sims) - np.mean(raw_sims)),
        'residual': float(residual),
        'n_words': n,
    }


def main():
    parser = argparse.ArgumentParser(description='E.X.3.2b: Real Model Resistance Test')
    parser.add_argument('--n-random', type=int, default=5,
                        help='Number of random model pairs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    print("=" * 70)
    print("E.X.3.2b: REAL MODEL ALIGNMENT RESISTANCE TEST")
    print("=" * 70)
    print()

    if not ST_AVAILABLE:
        print("ERROR: sentence-transformers not available")
        return 1

    words = ANCHOR_WORDS[:64]
    print(f"Using {len(words)} anchor words")
    print()

    # Models to test
    models = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'paraphrase-MiniLM-L6-v2',
    ]

    # Get embeddings from real models
    print("Loading real models...")
    model_embeddings = {}
    model_dims = {}
    for model_name in models:
        print(f"  Loading {model_name}...")
        emb, dim = get_model_embeddings(model_name, words)
        model_embeddings[model_name] = emb
        model_dims[model_name] = dim
    print()

    # Test 1: Random vs Random (baseline)
    print("-" * 70)
    print("Test 1: Random vs Random (10 pairs)")
    print("-" * 70)

    random_improvements = []
    dim = 384  # Match typical model dimension

    for i in range(args.n_random):
        rand_a = generate_random_embeddings(words, dim, args.seed + i * 2)
        rand_b = generate_random_embeddings(words, dim, args.seed + i * 2 + 1)
        result = compute_alignment_metrics(rand_a, rand_b, words)
        random_improvements.append(result['improvement'])
        print(f"  Pair {i+1}: improvement={result['improvement']:+.4f}")

    mean_random = float(np.mean(random_improvements))
    print(f"\nMean random improvement: {mean_random:+.4f}")
    print()

    # Test 2: Real Model vs Real Model
    print("-" * 70)
    print("Test 2: Real Model vs Real Model")
    print("-" * 70)

    model_pairs = list(combinations(models, 2))
    trained_improvements = []

    for m1, m2 in model_pairs:
        result = compute_alignment_metrics(model_embeddings[m1], model_embeddings[m2], words)
        trained_improvements.append(result['improvement'])
        print(f"  {m1} vs {m2}:")
        print(f"    raw={result['raw_similarity']:.4f}, aligned={result['aligned_similarity']:.4f}, "
              f"improvement={result['improvement']:+.4f}")

    mean_trained = float(np.mean(trained_improvements))
    print(f"\nMean trained improvement: {mean_trained:+.4f}")
    print()

    # Test 3: Random vs Real Model
    print("-" * 70)
    print("Test 3: Random vs Real Model")
    print("-" * 70)

    cross_improvements = []
    for model_name in models:
        for i in range(3):  # 3 random instances per model
            rand = generate_random_embeddings(words, model_dims[model_name], args.seed + i * 100)
            result = compute_alignment_metrics(rand, model_embeddings[model_name], words)
            cross_improvements.append(result['improvement'])
            print(f"  Random vs {model_name} [{i+1}]: improvement={result['improvement']:+.4f}")

    mean_cross = float(np.mean(cross_improvements))
    print(f"\nMean random-vs-trained improvement: {mean_cross:+.4f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Random vs Random:      {mean_random:+.4f}")
    print(f"Trained vs Trained:    {mean_trained:+.4f}")
    print(f"Random vs Trained:     {mean_cross:+.4f}")
    print()

    gap = mean_random - mean_trained
    print(f"GAP (Random - Trained): {gap:+.4f}")
    print()

    if gap > 0.3:
        verdict = "CONFIRMED"
        explanation = (
            f"Real trained models show {gap:.2f} LESS alignment improvement than random. "
            "Trained models have structure that resists arbitrary rotation."
        )
    elif gap > 0.1:
        verdict = "PARTIAL"
        explanation = (
            f"Moderate gap ({gap:.2f}). Some resistance detected but less than expected."
        )
    else:
        verdict = "NOT CONFIRMED"
        explanation = (
            f"Minimal gap ({gap:.2f}). Real models behave like random for alignment."
        )

    print(f"VERDICT: {verdict}")
    print()
    print(explanation)
    print()

    # Save results
    result = {
        'test_id': 'real-model-resistance-E.X.3.2b',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'n_random': args.n_random,
            'n_words': len(words),
            'seed': args.seed,
            'models': models,
        },
        'results': {
            'random_vs_random': {
                'improvements': random_improvements,
                'mean': mean_random,
            },
            'trained_vs_trained': {
                'improvements': trained_improvements,
                'mean': mean_trained,
            },
            'random_vs_trained': {
                'improvements': cross_improvements,
                'mean': mean_cross,
            },
            'gap': gap,
        },
        'interpretation': {
            'verdict': verdict,
            'explanation': explanation,
        },
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'real_model_resistance.json'

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0 if verdict == "CONFIRMED" else 1


if __name__ == '__main__':
    sys.exit(main())
