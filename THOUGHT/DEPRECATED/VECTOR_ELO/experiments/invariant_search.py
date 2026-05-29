#!/usr/bin/env python3
"""
Invariant Search: Finding the Platonic Constant

If all embedding models converge to the same underlying semantic manifold
(Platonic Representation Hypothesis), there must be an INVARIANT - something
that remains constant across all models despite different coordinate systems.

Candidate Invariants Tested:
1. Eigenvalue spectrum ratio - shape of distance matrix
2. Cross-ratio - projective invariant for 4 points
3. Triangle area ratios - geometric invariant
4. Angle structure - angles in triangles formed by triplets
5. Rank order preservation - Spearman correlation
6. Distance ratios - ratios between pairs of distances

The invariant is THE KEY to universal semantic alignment.

Usage:
    python invariant_search.py
"""

import json
import sys
import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Models to test
MODELS = {
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
    'e5-large-v2': 'intfloat/e5-large-v2',
    'bge-large-en-v1.5': 'BAAI/bge-large-en-v1.5',
    'gte-large': 'thenlper/gte-large',
}

E5_PREFIX = 'query: '

ANCHOR_WORDS = ['dog', 'love', 'up', 'true', 'king']

# Cache
_model_cache = {}


def load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    from sentence_transformers import SentenceTransformer
    print(f"  Loading {name}...", end=" ", flush=True)
    start = time.time()
    model = SentenceTransformer(MODELS[name])
    print(f"({time.time()-start:.1f}s)")
    _model_cache[name] = model
    return model


def get_embeddings(name: str, words: List[str]) -> np.ndarray:
    model = load_model(name)
    if name == 'e5-large-v2':
        words = [E5_PREFIX + w for w in words]
    return model.encode(words, normalize_embeddings=True)


def get_distance_matrix(name: str, words: List[str]) -> np.ndarray:
    emb = get_embeddings(name, words)
    return emb @ emb.T  # cosine similarity (normalized)


# ============================================================
# INVARIANT CANDIDATES
# ============================================================

def eigenvalue_spectrum(matrix: np.ndarray) -> np.ndarray:
    """Get normalized eigenvalue spectrum."""
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.sort(eigvals)[::-1]  # descending
    return eigvals / eigvals[0]  # normalize by largest


def cross_ratio(d_ac: float, d_bd: float, d_ad: float, d_bc: float) -> float:
    """
    Cross-ratio: (AC * BD) / (AD * BC)
    Projective invariant - preserved under projective transformations.
    """
    if d_ad * d_bc == 0:
        return float('inf')
    return (d_ac * d_bd) / (d_ad * d_bc)


def all_cross_ratios(matrix: np.ndarray) -> List[float]:
    """Compute cross-ratios for all 4-point combinations."""
    n = matrix.shape[0]
    ratios = []
    for a, b, c, d in combinations(range(n), 4):
        # Use 1 - similarity as distance
        d_ac = 1 - matrix[a, c]
        d_bd = 1 - matrix[b, d]
        d_ad = 1 - matrix[a, d]
        d_bc = 1 - matrix[b, c]
        if d_ad * d_bc > 0.001:  # avoid division issues
            cr = cross_ratio(d_ac, d_bd, d_ad, d_bc)
            if not np.isinf(cr) and cr < 100:  # filter outliers
                ratios.append(cr)
    return ratios


def triangle_areas(matrix: np.ndarray) -> List[float]:
    """
    Compute areas of triangles formed by all triplets.
    Using Heron's formula with distances.
    """
    n = matrix.shape[0]
    areas = []
    for i, j, k in combinations(range(n), 3):
        # Convert similarity to distance
        a = 1 - matrix[i, j]
        b = 1 - matrix[j, k]
        c = 1 - matrix[i, k]
        # Heron's formula
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq > 0:
            areas.append(np.sqrt(area_sq))
        else:
            areas.append(0.0)
    return areas


def triangle_angles(matrix: np.ndarray) -> List[float]:
    """
    Compute angles of triangles formed by all triplets.
    Returns all angles (3 per triangle).
    """
    n = matrix.shape[0]
    angles = []
    for i, j, k in combinations(range(n), 3):
        a = 1 - matrix[j, k]  # opposite to i
        b = 1 - matrix[i, k]  # opposite to j
        c = 1 - matrix[i, j]  # opposite to k

        # Law of cosines: cos(A) = (b² + c² - a²) / (2bc)
        for side_a, side_b, side_c in [(a, b, c), (b, c, a), (c, a, b)]:
            if side_b * side_c > 0.001:
                cos_angle = (side_b**2 + side_c**2 - side_a**2) / (2 * side_b * side_c)
                cos_angle = np.clip(cos_angle, -1, 1)
                angles.append(np.arccos(cos_angle))
    return angles


def distance_ratios(matrix: np.ndarray) -> List[float]:
    """
    Compute ratios between all pairs of distances.
    If the manifold structure is preserved, ratios should be constant.
    """
    n = matrix.shape[0]
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(1 - matrix[i, j])

    ratios = []
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[j] > 0.001:
                ratios.append(distances[i] / distances[j])
    return sorted(ratios)


def rank_order(matrix: np.ndarray) -> List[int]:
    """Get rank order of off-diagonal elements."""
    n = matrix.shape[0]
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(matrix[i, j])
    return list(np.argsort(values))


# ============================================================
# COMPARISON
# ============================================================

def compare_invariant(inv1: List[float], inv2: List[float], name: str) -> Dict:
    """Compare two invariant vectors."""
    if len(inv1) != len(inv2):
        return {'name': name, 'error': 'length mismatch'}

    arr1 = np.array(inv1)
    arr2 = np.array(inv2)

    # Multiple correlation measures
    pearson_r, _ = pearsonr(arr1, arr2)
    spearman_r, _ = spearmanr(arr1, arr2)

    # Relative error
    rel_error = np.mean(np.abs(arr1 - arr2) / (np.abs(arr1) + 1e-10))

    return {
        'name': name,
        'pearson': float(pearson_r),
        'spearman': float(spearman_r),
        'rel_error': float(rel_error),
        'mean_diff': float(np.mean(np.abs(arr1 - arr2))),
    }


def test_invariants():
    """Test all invariant candidates across models."""
    print("=" * 70)
    print("INVARIANT SEARCH: Finding the Platonic Constant")
    print("=" * 70)
    print(f"\nAnchor words: {ANCHOR_WORDS}")
    print(f"Models: {list(MODELS.keys())}")

    # Get all distance matrices
    print("\nLoading models and computing distance matrices...")
    matrices = {}
    for name in MODELS:
        matrices[name] = get_distance_matrix(name, ANCHOR_WORDS)

    ref_model = 'all-MiniLM-L6-v2'
    ref_matrix = matrices[ref_model]

    # Compute invariants for reference
    print(f"\nComputing invariants for reference model: {ref_model}")
    ref_invariants = {
        'eigenvalue_spectrum': eigenvalue_spectrum(ref_matrix).tolist(),
        'cross_ratios': all_cross_ratios(ref_matrix),
        'triangle_areas': triangle_areas(ref_matrix),
        'triangle_angles': triangle_angles(ref_matrix),
        'distance_ratios': distance_ratios(ref_matrix),
        'rank_order': rank_order(ref_matrix),
    }

    # Compare each model
    print("\n" + "=" * 70)
    print("INVARIANT COMPARISON RESULTS")
    print("=" * 70)

    results = {}
    for name, matrix in matrices.items():
        if name == ref_model:
            continue

        print(f"\n{name} vs {ref_model}:")
        print("-" * 50)

        invariants = {
            'eigenvalue_spectrum': eigenvalue_spectrum(matrix).tolist(),
            'cross_ratios': all_cross_ratios(matrix),
            'triangle_areas': triangle_areas(matrix),
            'triangle_angles': triangle_angles(matrix),
            'distance_ratios': distance_ratios(matrix),
            'rank_order': rank_order(matrix),
        }

        model_results = {}
        for inv_name in ref_invariants:
            comp = compare_invariant(
                ref_invariants[inv_name],
                invariants[inv_name],
                inv_name
            )
            model_results[inv_name] = comp

            if 'error' not in comp:
                print(f"  {inv_name:20s}: pearson={comp['pearson']:.4f}, spearman={comp['spearman']:.4f}, rel_err={comp['rel_error']:.4f}")

        results[name] = model_results

    # Find best invariant
    print("\n" + "=" * 70)
    print("INVARIANT RANKINGS (by mean Pearson correlation across models)")
    print("=" * 70)

    inv_scores = {}
    for inv_name in ref_invariants:
        scores = []
        for model_name, model_results in results.items():
            if 'error' not in model_results[inv_name]:
                scores.append(model_results[inv_name]['pearson'])
        if scores:
            inv_scores[inv_name] = {
                'mean': np.mean(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'std': np.std(scores),
            }

    # Sort by mean correlation
    sorted_invs = sorted(inv_scores.items(), key=lambda x: x[1]['mean'], reverse=True)

    print(f"\n{'Invariant':<25} {'Mean r':>10} {'Min r':>10} {'Max r':>10} {'Std':>10}")
    print("-" * 65)
    for inv_name, scores in sorted_invs:
        print(f"{inv_name:<25} {scores['mean']:>10.4f} {scores['min']:>10.4f} {scores['max']:>10.4f} {scores['std']:>10.4f}")

    # THE WINNER
    best_inv = sorted_invs[0]
    print("\n" + "=" * 70)
    print(f"BEST INVARIANT CANDIDATE: {best_inv[0]}")
    print(f"Mean correlation: {best_inv[1]['mean']:.4f}")
    print(f"Min correlation: {best_inv[1]['min']:.4f}")
    print("=" * 70)

    if best_inv[1]['mean'] > 0.9:
        print("\n*** STRONG INVARIANT FOUND! ***")
        print("This could be the Platonic constant for cross-model alignment.")
    elif best_inv[1]['mean'] > 0.7:
        print("\n*** MODERATE INVARIANT FOUND ***")
        print("Promising but may need refinement.")
    else:
        print("\n*** WEAK INVARIANTS ***")
        print("Need to search for other candidates.")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    return convert({
        'reference_model': ref_model,
        'reference_invariants': ref_invariants,
        'comparisons': results,
        'rankings': {k: v for k, v in sorted_invs},
        'best_invariant': best_inv[0],
    })


if __name__ == "__main__":
    results = test_invariants()

    # Save results
    output_path = 'invariant_search_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
