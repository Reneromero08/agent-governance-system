#!/usr/bin/env python3
"""E.X.3.2c: Held-Out Alignment Resistance Test

KEY INSIGHT: The original benchmark measured alignment on HELD-OUT words,
not the same words used for Procrustes fitting.

- Fitting on 64 anchors, testing on 218 held-out words
- Trained models: 0.38-0.43 aligned similarity on held-out
- Random on fitting words: 0.96

The question: Do random embeddings also generalize to held-out?
If random shows low held-out alignment, that's the signal.

Usage:
    python -m benchmarks.validation.held_out_resistance
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.mds import squared_distance_matrix, classical_mds
from lib.procrustes import procrustes_align, out_of_sample_mds, cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# Anchor words for fitting
ANCHOR_WORDS = [
    "time", "space", "energy", "matter", "light",
    "force", "motion", "wave", "particle", "field",
    "truth", "beauty", "justice", "freedom", "power",
    "knowledge", "wisdom", "love", "fear", "hope",
    "water", "fire", "earth", "air", "stone",
    "tree", "river", "mountain", "ocean", "star",
    "mind", "body", "soul", "heart", "brain",
    "thought", "feeling", "memory", "dream", "vision",
    "create", "destroy", "build", "break", "grow",
    "change", "move", "stop", "begin", "end",
    "cause", "effect", "order", "chaos", "balance",
    "unity", "division", "connection", "separation", "flow",
    "language", "meaning", "symbol", "pattern", "structure",
]

# Held-out words for testing
HELD_OUT_WORDS = [
    "sun", "moon", "cloud", "rain", "wind",
    "forest", "desert", "island", "valley", "cave",
    "animal", "plant", "human", "child", "adult",
    "friend", "enemy", "family", "stranger", "leader",
    "word", "sentence", "story", "poem", "song",
    "color", "sound", "taste", "touch", "smell",
    "happy", "sad", "angry", "calm", "excited",
    "fast", "slow", "big", "small", "old",
    "new", "young", "ancient", "modern", "future",
    "past", "present", "moment", "eternity", "instant",
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


def get_model_embeddings(model_name: str, words: list) -> tuple:
    """Get embeddings from a real model."""
    model = SentenceTransformer(model_name)
    vectors = model.encode(words, convert_to_numpy=True)

    embeddings = {}
    for i, word in enumerate(words):
        vec = vectors[i]
        vec = vec / np.linalg.norm(vec)
        embeddings[word] = vec

    return embeddings, model.get_sentence_embedding_dimension()


def compute_held_out_alignment(
    emb_a: dict,
    emb_b: dict,
    anchor_words: list,
    held_out_words: list
) -> dict:
    """Compute alignment on held-out words after fitting on anchors.

    1. Build MDS coordinates for anchors
    2. Procrustes align model A to model B on anchors
    3. Project held-out words using out-of-sample MDS
    4. Apply rotation to held-out projections
    5. Measure similarity on held-out words
    """
    # Filter to available words
    anchors_a = [w for w in anchor_words if w in emb_a]
    anchors_b = [w for w in anchor_words if w in emb_b]
    common_anchors = [w for w in anchors_a if w in anchors_b]

    held_out_a = [w for w in held_out_words if w in emb_a]
    held_out_b = [w for w in held_out_words if w in emb_b]
    common_held_out = [w for w in held_out_a if w in held_out_b]

    n_anchors = len(common_anchors)
    n_held_out = len(common_held_out)

    if n_anchors < 10 or n_held_out < 5:
        return {'error': 'Not enough words'}

    # Build anchor matrices
    X_anchor_a = np.array([emb_a[w] for w in common_anchors])
    X_anchor_b = np.array([emb_b[w] for w in common_anchors])

    # Build held-out matrices
    X_held_a = np.array([emb_a[w] for w in common_held_out])
    X_held_b = np.array([emb_b[w] for w in common_held_out])

    # MDS on anchors
    D2_anchor_a = squared_distance_matrix(X_anchor_a)
    D2_anchor_b = squared_distance_matrix(X_anchor_b)

    coords_anchor_a, eigenvalues_a, eigenvectors_a = classical_mds(D2_anchor_a)
    coords_anchor_b, eigenvalues_b, eigenvectors_b = classical_mds(D2_anchor_b)

    # Match dimensions
    k = min(coords_anchor_a.shape[1], coords_anchor_b.shape[1])
    coords_anchor_a = coords_anchor_a[:, :k]
    coords_anchor_b = coords_anchor_b[:, :k]
    eigenvectors_a = eigenvectors_a[:, :k]
    eigenvectors_b = eigenvectors_b[:, :k]
    eigenvalues_a = eigenvalues_a[:k]
    eigenvalues_b = eigenvalues_b[:k]

    # Procrustes on anchors
    R, residual = procrustes_align(coords_anchor_a, coords_anchor_b)

    # Project held-out words using out-of-sample MDS
    # Compute squared distances from held-out to anchors
    d2_held_to_anchor_a = np.zeros((n_held_out, n_anchors))
    d2_held_to_anchor_b = np.zeros((n_held_out, n_anchors))

    for i, w in enumerate(common_held_out):
        for j, aw in enumerate(common_anchors):
            d2_held_to_anchor_a[i, j] = np.sum((emb_a[w] - emb_a[aw])**2)
            d2_held_to_anchor_b[i, j] = np.sum((emb_b[w] - emb_b[aw])**2)

    # Out-of-sample projection
    coords_held_a = out_of_sample_mds(d2_held_to_anchor_a, D2_anchor_a, eigenvectors_a, eigenvalues_a)
    coords_held_b = out_of_sample_mds(d2_held_to_anchor_b, D2_anchor_b, eigenvectors_b, eigenvalues_b)

    # Apply rotation to held-out A
    coords_held_a_aligned = coords_held_a @ R

    # Measure similarities on held-out
    raw_sims = []
    aligned_sims = []
    for i in range(n_held_out):
        raw_sims.append(cosine_similarity(coords_held_a[i], coords_held_b[i]))
        aligned_sims.append(cosine_similarity(coords_held_a_aligned[i], coords_held_b[i]))

    # Also measure on anchors for comparison
    anchor_raw_sims = []
    anchor_aligned_sims = []
    coords_anchor_a_aligned = coords_anchor_a @ R
    for i in range(n_anchors):
        anchor_raw_sims.append(cosine_similarity(coords_anchor_a[i], coords_anchor_b[i]))
        anchor_aligned_sims.append(cosine_similarity(coords_anchor_a_aligned[i], coords_anchor_b[i]))

    return {
        'n_anchors': n_anchors,
        'n_held_out': n_held_out,
        'anchor_raw_similarity': float(np.mean(anchor_raw_sims)),
        'anchor_aligned_similarity': float(np.mean(anchor_aligned_sims)),
        'anchor_improvement': float(np.mean(anchor_aligned_sims) - np.mean(anchor_raw_sims)),
        'held_out_raw_similarity': float(np.mean(raw_sims)),
        'held_out_aligned_similarity': float(np.mean(aligned_sims)),
        'held_out_improvement': float(np.mean(aligned_sims) - np.mean(raw_sims)),
        'residual': float(residual),
    }


def main():
    parser = argparse.ArgumentParser(description='E.X.3.2c: Held-Out Resistance Test')
    parser.add_argument('--n-random', type=int, default=10,
                        help='Number of random pairs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    print("=" * 70)
    print("E.X.3.2c: HELD-OUT ALIGNMENT RESISTANCE TEST")
    print("=" * 70)
    print()
    print("Key question: Does alignment generalize to held-out words?")
    print()

    if not ST_AVAILABLE:
        print("ERROR: sentence-transformers not available")
        return 1

    all_words = list(set(ANCHOR_WORDS + HELD_OUT_WORDS))
    print(f"Using {len(ANCHOR_WORDS)} anchor words, {len(HELD_OUT_WORDS)} held-out words")
    print()

    # Load real models (use locally cached ones)
    models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
    print("Loading real models...")
    model_embeddings = {}
    model_dims = {}
    for m in models:
        print(f"  {m}...")
        emb, dim = get_model_embeddings(m, all_words)
        model_embeddings[m] = emb
        model_dims[m] = dim
    print()

    # Test 1: Random vs Random
    print("-" * 70)
    print("Test 1: Random vs Random (held-out generalization)")
    print("-" * 70)

    random_results = []
    dim = 384

    for i in range(args.n_random):
        rand_a = generate_random_embeddings(all_words, dim, args.seed + i * 2)
        rand_b = generate_random_embeddings(all_words, dim, args.seed + i * 2 + 1)

        result = compute_held_out_alignment(rand_a, rand_b, ANCHOR_WORDS, HELD_OUT_WORDS)
        random_results.append(result)

        print(f"  Pair {i+1}: anchor_aligned={result['anchor_aligned_similarity']:.4f}, "
              f"held_out_aligned={result['held_out_aligned_similarity']:.4f}")

    mean_random_anchor = float(np.mean([r['anchor_aligned_similarity'] for r in random_results]))
    mean_random_held_out = float(np.mean([r['held_out_aligned_similarity'] for r in random_results]))

    print(f"\nRandom mean:")
    print(f"  Anchor aligned:   {mean_random_anchor:.4f}")
    print(f"  Held-out aligned: {mean_random_held_out:.4f}")
    print(f"  Generalization:   {mean_random_held_out - mean_random_anchor:+.4f}")
    print()

    # Test 2: Real model pair
    print("-" * 70)
    print("Test 2: Real Model Pair (held-out generalization)")
    print("-" * 70)

    trained_result = compute_held_out_alignment(
        model_embeddings['all-mpnet-base-v2'],
        model_embeddings['all-MiniLM-L6-v2'],
        ANCHOR_WORDS,
        HELD_OUT_WORDS
    )

    print(f"  all-mpnet-base-v2 vs all-MiniLM-L6-v2:")
    print(f"    Anchor raw:       {trained_result['anchor_raw_similarity']:.4f}")
    print(f"    Anchor aligned:   {trained_result['anchor_aligned_similarity']:.4f}")
    print(f"    Held-out raw:     {trained_result['held_out_raw_similarity']:.4f}")
    print(f"    Held-out aligned: {trained_result['held_out_aligned_similarity']:.4f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"                    | Random      | Trained")
    print(f"--------------------|-------------|----------")
    print(f"Anchor aligned      | {mean_random_anchor:.4f}      | {trained_result['anchor_aligned_similarity']:.4f}")
    print(f"Held-out aligned    | {mean_random_held_out:.4f}      | {trained_result['held_out_aligned_similarity']:.4f}")
    print()

    # The key comparison
    gap_anchor = mean_random_anchor - trained_result['anchor_aligned_similarity']
    gap_held_out = mean_random_held_out - trained_result['held_out_aligned_similarity']

    print(f"GAP (Random - Trained):")
    print(f"  On anchors:   {gap_anchor:+.4f}")
    print(f"  On held-out:  {gap_held_out:+.4f}")
    print()

    if gap_held_out > 0.3:
        verdict = "CONFIRMED"
        explanation = (
            f"Random embeddings don't generalize to held-out (gap={gap_held_out:.3f}). "
            "Trained models have structure that transfers beyond fitting set."
        )
    elif gap_held_out > 0.1:
        verdict = "PARTIAL"
        explanation = f"Moderate gap on held-out ({gap_held_out:.3f}). Some structure detected."
    elif gap_held_out < -0.1:
        verdict = "REVERSED"
        explanation = (
            f"Trained models show WORSE held-out alignment than random ({gap_held_out:.3f}). "
            "This is unexpected - needs investigation."
        )
    else:
        verdict = "NOT CONFIRMED"
        explanation = f"Similar held-out performance ({gap_held_out:.3f}). No clear signal."

    print(f"VERDICT: {verdict}")
    print()
    print(explanation)
    print()

    # Save
    result = {
        'test_id': 'held-out-resistance-E.X.3.2c',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'random': {
            'mean_anchor_aligned': mean_random_anchor,
            'mean_held_out_aligned': mean_random_held_out,
        },
        'trained': trained_result,
        'gaps': {
            'anchor': gap_anchor,
            'held_out': gap_held_out,
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
        output_path = output_dir / 'held_out_resistance.json'

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
