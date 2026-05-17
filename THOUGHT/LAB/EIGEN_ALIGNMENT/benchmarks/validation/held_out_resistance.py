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

EPS = 1e-10

from lib.mds import squared_distance_matrix, classical_mds
from lib.procrustes import procrustes_align, out_of_sample_mds, cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


def _mutual_information_continuous(x: list, y: list, *, n_bins: int = 8) -> float:
    """
    Phi-style proxy: mutual information I(X;Y) for continuous values via histogram binning.

    This measures the coupling/integration between two sequences.
    Adapted from Q32 benchmarks.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.size == 0 or yy.size == 0:
        return 0.0
    n = int(min(xx.size, yy.size))
    xx = xx[:n]
    yy = yy[:n]

    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 5:
        return 0.0

    n_bins_i = max(2, int(n_bins))
    x_min = float(np.min(xx))
    x_max = float(np.max(xx))
    y_min = float(np.min(yy))
    y_max = float(np.max(yy))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    x_edges = np.linspace(x_min - EPS, x_max + EPS, n_bins_i + 1)
    y_edges = np.linspace(y_min - EPS, y_max + EPS, n_bins_i + 1)
    joint, _, _ = np.histogram2d(xx, yy, bins=(x_edges, y_edges))
    pxy = joint / float(np.sum(joint) + EPS)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px @ py + EPS)
        mi = float(np.nansum(pxy * np.log2(ratio + EPS)))
    return float(max(0.0, mi))


def compute_neighbor_coupling(held_out_embs: np.ndarray, anchor_embs: np.ndarray, k: int = 5) -> float:
    """
    J-style metric: Mean cosine similarity between held-out words and their k nearest anchors.

    This measures how well held-out words "couple" to the anchor manifold.
    Higher J = held-out words are semantically close to anchors (good interpolation)
    Lower J = held-out words are in semantic voids (poor coverage)
    """
    # Compute similarity matrix: held_out x anchors
    # Embeddings should be L2 normalized, so dot product = cosine similarity
    sim_matrix = held_out_embs @ anchor_embs.T  # (n_held_out, n_anchors)

    # For each held-out word, find k nearest anchors and average their similarities
    neighbor_sims = []
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i]
        top_k_indices = np.argsort(-row)[:k]
        top_k_sims = row[top_k_indices]
        neighbor_sims.append(float(np.mean(top_k_sims)))

    return float(np.mean(neighbor_sims))


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

    # === PHI: Integration between anchor and held-out performance ===
    # Measures if knowing anchor alignment predicts held-out alignment
    phi_proxy = _mutual_information_continuous(anchor_aligned_sims, aligned_sims)

    # === J: Neighbor coupling in original embedding space ===
    # Measures how well held-out words couple to anchor manifold
    # Higher J = held-out near anchors, lower J = held-out in semantic voids
    j_coupling_a = compute_neighbor_coupling(X_held_a, X_anchor_a, k=5)
    j_coupling_b = compute_neighbor_coupling(X_held_b, X_anchor_b, k=5)
    j_coupling_mean = (j_coupling_a + j_coupling_b) / 2

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
        # New metrics
        'phi_proxy': float(phi_proxy),
        'j_coupling_a': float(j_coupling_a),
        'j_coupling_b': float(j_coupling_b),
        'j_coupling_mean': float(j_coupling_mean),
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
    mean_random_phi = float(np.mean([r['phi_proxy'] for r in random_results]))
    mean_random_j = float(np.mean([r['j_coupling_mean'] for r in random_results]))

    print(f"\nRandom mean:")
    print(f"  Anchor aligned:   {mean_random_anchor:.4f}")
    print(f"  Held-out aligned: {mean_random_held_out:.4f}")
    print(f"  Generalization:   {mean_random_held_out - mean_random_anchor:+.4f}")
    print(f"  [Phi] I(anchor;held_out): {mean_random_phi:.4f} bits")
    print(f"  [J]   Neighbor coupling:  {mean_random_j:.4f}")
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
    print(f"    [Phi] I(anchor;held_out): {trained_result['phi_proxy']:.4f} bits")
    print(f"    [J]   Neighbor coupling:  {trained_result['j_coupling_mean']:.4f}")
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
    print(f"[Phi] integration   | {mean_random_phi:.4f}      | {trained_result['phi_proxy']:.4f}")
    print(f"[J] coupling        | {mean_random_j:.4f}      | {trained_result['j_coupling_mean']:.4f}")
    print()

    # The key comparison
    gap_anchor = mean_random_anchor - trained_result['anchor_aligned_similarity']
    gap_held_out = mean_random_held_out - trained_result['held_out_aligned_similarity']

    print(f"GAP (Random - Trained):")
    print(f"  On anchors:   {gap_anchor:+.4f}")
    print(f"  On held-out:  {gap_held_out:+.4f}")
    print()

    # Note: gap = random - trained, so NEGATIVE gap means trained is BETTER
    if gap_held_out < -0.3:
        verdict = "CONFIRMED"
        explanation = (
            f"Trained models generalize to held-out, random doesn't (trained={trained_result['held_out_aligned_similarity']:.3f}, "
            f"random={mean_random_held_out:.3f}). "
            f"J coupling explains this: trained J={trained_result['j_coupling_mean']:.3f}, random J={mean_random_j:.3f}."
        )
    elif gap_held_out < -0.1:
        verdict = "PARTIAL"
        explanation = f"Moderate generalization gap ({-gap_held_out:.3f}). Some structure detected."
    elif gap_held_out > 0.3:
        verdict = "ANOMALY"
        explanation = (
            f"Random shows BETTER held-out alignment than trained ({gap_held_out:.3f}). "
            "This is unexpected - needs investigation."
        )
    else:
        verdict = "NOT CONFIRMED"
        explanation = f"Similar held-out performance (gap={gap_held_out:.3f}). No clear signal."

    print(f"VERDICT: {verdict}")
    print()
    print(explanation)
    print()

    # Phi/J gaps
    gap_phi = mean_random_phi - trained_result['phi_proxy']
    gap_j = mean_random_j - trained_result['j_coupling_mean']

    print(f"  [Phi] gap:    {gap_phi:+.4f}")
    print(f"  [J] gap:      {gap_j:+.4f}")
    print()

    # Save
    result = {
        'test_id': 'held-out-resistance-E.X.3.2c',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'random': {
            'mean_anchor_aligned': mean_random_anchor,
            'mean_held_out_aligned': mean_random_held_out,
            'mean_phi': mean_random_phi,
            'mean_j': mean_random_j,
        },
        'trained': trained_result,
        'gaps': {
            'anchor': gap_anchor,
            'held_out': gap_held_out,
            'phi': gap_phi,
            'j': gap_j,
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
