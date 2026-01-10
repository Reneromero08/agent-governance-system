#!/usr/bin/env python3
"""
E.X.3.4: Geometry Analysis with Advanced Tools

Uses UMAP, POT (Wasserstein), geomstats, and HDBSCAN to analyze
the geometric differences between random/untrained/trained embeddings.

Key questions:
1. Do trained models have distinct manifold structure (UMAP)?
2. What is the Wasserstein distance between embedding distributions?
3. Do trained embeddings cluster semantically (HDBSCAN)?
4. How do embeddings relate to hypersphere geometry (geomstats)?
"""

import argparse
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import geometry packages
import umap
import ot  # POT - Python Optimal Transport
import hdbscan
from geomstats.geometry.hypersphere import Hypersphere

# Import from existing tests
from untrained_transformer import (
    generate_random_embeddings,
    get_untrained_bert_embeddings,
    get_trained_bert_embeddings,
    compute_effective_dimensionality,
    ANCHOR_WORDS,
    HELD_OUT_WORDS,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_wasserstein_distance(emb_a: dict, emb_b: dict) -> float:
    """
    Compute Wasserstein-2 distance between two embedding distributions.

    This measures how "far apart" two embedding spaces are geometrically.
    Lower = more similar distributions.
    """
    words_a = list(emb_a.keys())
    words_b = list(emb_b.keys())

    # Use shared words
    shared = set(words_a) & set(words_b)
    if len(shared) < 10:
        return float('nan')

    vecs_a = np.array([emb_a[w] for w in shared])
    vecs_b = np.array([emb_b[w] for w in shared])

    n = len(shared)

    # Uniform weights
    a_weights = np.ones(n) / n
    b_weights = np.ones(n) / n

    # Cost matrix: pairwise Euclidean distances
    M = ot.dist(vecs_a, vecs_b, metric='euclidean')

    # Wasserstein distance
    W = ot.emd2(a_weights, b_weights, M)

    return float(W)


def compute_umap_projection(embeddings: dict, n_components: int = 3,
                            n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Project embeddings to low dimensions using UMAP.

    Returns coordinates that preserve local structure.
    """
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False
    )

    coords = reducer.fit_transform(vecs)
    return coords, words


def compute_hdbscan_clusters(embeddings: dict, min_cluster_size: int = 5) -> dict:
    """
    Find density-based clusters in embedding space.

    Returns cluster statistics:
    - n_clusters: number of clusters found
    - n_noise: points not in any cluster
    - cluster_sizes: list of cluster sizes
    """
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3)
    labels = clusterer.fit_predict(vecs)

    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = np.sum(labels == -1)

    cluster_sizes = []
    for label in unique_labels:
        if label >= 0:
            cluster_sizes.append(int(np.sum(labels == label)))

    return {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'noise_ratio': float(n_noise / len(words)),
        'cluster_sizes': sorted(cluster_sizes, reverse=True),
        'labels': labels.tolist(),
    }


def compute_hypersphere_metrics(embeddings: dict) -> dict:
    """
    Analyze embeddings on hypersphere using geomstats.

    Measures:
    - Mean distance to origin (should be ~1 for normalized)
    - Variance of norms
    - Frechet mean on sphere
    """
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])

    # Check norms
    norms = np.linalg.norm(vecs, axis=1)

    # Create hypersphere
    dim = vecs.shape[1] - 1  # Hypersphere dim = ambient dim - 1
    sphere = Hypersphere(dim=dim)

    # Project to unit sphere if needed
    vecs_normalized = vecs / norms[:, np.newaxis]

    # Check if points are on sphere
    on_sphere = np.allclose(np.linalg.norm(vecs_normalized, axis=1), 1.0)

    # Compute pairwise geodesic distances (sample)
    n_sample = min(100, len(words))
    indices = np.random.choice(len(words), n_sample, replace=False)
    sample_vecs = vecs_normalized[indices]

    geodesic_dists = []
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            # Geodesic distance on sphere = arccos(dot product) for unit vectors
            dot = np.clip(np.dot(sample_vecs[i], sample_vecs[j]), -1, 1)
            dist = np.arccos(dot)
            geodesic_dists.append(dist)

    return {
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'on_unit_sphere': bool(on_sphere),
        'mean_geodesic_dist': float(np.mean(geodesic_dists)),
        'std_geodesic_dist': float(np.std(geodesic_dists)),
        'max_geodesic_dist': float(np.max(geodesic_dists)),
    }


def compute_umap_spread(coords: np.ndarray) -> dict:
    """
    Analyze the spread/structure of UMAP projection.
    """
    # Convex hull volume proxy: std in each dimension
    stds = np.std(coords, axis=0)

    # Sparsity: mean pairwise distance
    n = len(coords)
    sample_size = min(500, n)
    indices = np.random.choice(n, sample_size, replace=False)
    sample = coords[indices]

    dists = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            dists.append(np.linalg.norm(sample[i] - sample[j]))

    return {
        'std_per_dim': stds.tolist(),
        'total_spread': float(np.prod(stds)),
        'mean_pairwise_dist': float(np.mean(dists)),
        'std_pairwise_dist': float(np.std(dists)),
    }


def main():
    parser = argparse.ArgumentParser(description='E.X.3.4: Geometry Analysis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', '-o', type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)

    all_words = list(set(ANCHOR_WORDS + HELD_OUT_WORDS))
    dim = 768

    print("=" * 70)
    print("E.X.3.4: GEOMETRY ANALYSIS")
    print("=" * 70)
    print()
    print(f"Words: {len(all_words)}")
    print(f"Dimensions: {dim}")
    print()

    # Generate embeddings
    print("Generating embeddings...")

    print("  Random embeddings...")
    random_emb = generate_random_embeddings(all_words, dim, args.seed)

    print("  Untrained BERT...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    untrained_emb, _ = get_untrained_bert_embeddings(all_words)

    print("  Trained BERT...")
    trained_emb, _ = get_trained_bert_embeddings(all_words)

    print()

    results = {
        'test_id': 'geometry-analysis-E.X.3.4',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    # === Test 1: Effective Dimensionality (recap) ===
    print("-" * 70)
    print("Test 1: Effective Dimensionality (recap)")
    print("-" * 70)

    rand_eff = compute_effective_dimensionality(random_emb)
    untrained_eff = compute_effective_dimensionality(untrained_emb)
    trained_eff = compute_effective_dimensionality(trained_emb)

    print(f"  Participation Ratio:")
    print(f"    Random:    {rand_eff['participation_ratio']:.1f} / {dim}")
    print(f"    Untrained: {untrained_eff['participation_ratio']:.1f} / {dim}")
    print(f"    Trained:   {trained_eff['participation_ratio']:.1f} / {dim}")
    print()

    results['effective_dimensionality'] = {
        'random': rand_eff,
        'untrained': untrained_eff,
        'trained': trained_eff,
    }

    # === Test 2: Wasserstein Distances ===
    print("-" * 70)
    print("Test 2: Wasserstein Distances (Optimal Transport)")
    print("-" * 70)

    # Generate a second random for comparison
    random_emb_2 = generate_random_embeddings(all_words, dim, args.seed + 1)

    w_rand_rand = compute_wasserstein_distance(random_emb, random_emb_2)
    w_rand_untrained = compute_wasserstein_distance(random_emb, untrained_emb)
    w_rand_trained = compute_wasserstein_distance(random_emb, trained_emb)
    w_untrained_trained = compute_wasserstein_distance(untrained_emb, trained_emb)

    print(f"  Random <-> Random:      {w_rand_rand:.4f}")
    print(f"  Random <-> Untrained:   {w_rand_untrained:.4f}")
    print(f"  Random <-> Trained:     {w_rand_trained:.4f}")
    print(f"  Untrained <-> Trained:  {w_untrained_trained:.4f}")
    print()

    results['wasserstein'] = {
        'random_to_random': w_rand_rand,
        'random_to_untrained': w_rand_untrained,
        'random_to_trained': w_rand_trained,
        'untrained_to_trained': w_untrained_trained,
    }

    # === Test 3: HDBSCAN Clustering ===
    print("-" * 70)
    print("Test 3: HDBSCAN Clustering")
    print("-" * 70)

    rand_clusters = compute_hdbscan_clusters(random_emb)
    untrained_clusters = compute_hdbscan_clusters(untrained_emb)
    trained_clusters = compute_hdbscan_clusters(trained_emb)

    print(f"  Random:    {rand_clusters['n_clusters']} clusters, {rand_clusters['noise_ratio']:.1%} noise")
    print(f"  Untrained: {untrained_clusters['n_clusters']} clusters, {untrained_clusters['noise_ratio']:.1%} noise")
    print(f"  Trained:   {trained_clusters['n_clusters']} clusters, {trained_clusters['noise_ratio']:.1%} noise")

    if trained_clusters['n_clusters'] > 0:
        print(f"    Trained cluster sizes: {trained_clusters['cluster_sizes'][:5]}...")
    print()

    # Remove labels from saved results (too large)
    results['hdbscan'] = {
        'random': {k: v for k, v in rand_clusters.items() if k != 'labels'},
        'untrained': {k: v for k, v in untrained_clusters.items() if k != 'labels'},
        'trained': {k: v for k, v in trained_clusters.items() if k != 'labels'},
    }

    # === Test 4: Hypersphere Metrics ===
    print("-" * 70)
    print("Test 4: Hypersphere Geometry (geomstats)")
    print("-" * 70)

    rand_sphere = compute_hypersphere_metrics(random_emb)
    untrained_sphere = compute_hypersphere_metrics(untrained_emb)
    trained_sphere = compute_hypersphere_metrics(trained_emb)

    print(f"  Mean geodesic distance (on unit sphere):")
    print(f"    Random:    {rand_sphere['mean_geodesic_dist']:.4f} rad")
    print(f"    Untrained: {untrained_sphere['mean_geodesic_dist']:.4f} rad")
    print(f"    Trained:   {trained_sphere['mean_geodesic_dist']:.4f} rad")
    print()
    print(f"  Geodesic distance std (spread on sphere):")
    print(f"    Random:    {rand_sphere['std_geodesic_dist']:.4f}")
    print(f"    Untrained: {untrained_sphere['std_geodesic_dist']:.4f}")
    print(f"    Trained:   {trained_sphere['std_geodesic_dist']:.4f}")
    print()

    results['hypersphere'] = {
        'random': rand_sphere,
        'untrained': untrained_sphere,
        'trained': trained_sphere,
    }

    # === Test 5: UMAP Projection ===
    print("-" * 70)
    print("Test 5: UMAP Projection (3D)")
    print("-" * 70)

    print("  Projecting random...")
    rand_coords, _ = compute_umap_projection(random_emb, n_components=3)
    rand_spread = compute_umap_spread(rand_coords)

    print("  Projecting untrained...")
    untrained_coords, _ = compute_umap_projection(untrained_emb, n_components=3)
    untrained_spread = compute_umap_spread(untrained_coords)

    print("  Projecting trained...")
    trained_coords, _ = compute_umap_projection(trained_emb, n_components=3)
    trained_spread = compute_umap_spread(trained_coords)

    print()
    print(f"  UMAP spread (total volume proxy):")
    print(f"    Random:    {rand_spread['total_spread']:.4f}")
    print(f"    Untrained: {untrained_spread['total_spread']:.4f}")
    print(f"    Trained:   {trained_spread['total_spread']:.4f}")
    print()
    print(f"  Mean pairwise distance in UMAP space:")
    print(f"    Random:    {rand_spread['mean_pairwise_dist']:.4f}")
    print(f"    Untrained: {untrained_spread['mean_pairwise_dist']:.4f}")
    print(f"    Trained:   {trained_spread['mean_pairwise_dist']:.4f}")
    print()

    results['umap'] = {
        'random': rand_spread,
        'untrained': untrained_spread,
        'trained': trained_spread,
    }

    # === Summary ===
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("| Metric                    | Random  | Untrained | Trained |")
    print("|---------------------------|---------|-----------|---------|")
    print(f"| Participation Ratio       | {rand_eff['participation_ratio']:7.1f} | {untrained_eff['participation_ratio']:9.1f} | {trained_eff['participation_ratio']:7.1f} |")
    print(f"| HDBSCAN clusters          | {rand_clusters['n_clusters']:7d} | {untrained_clusters['n_clusters']:9d} | {trained_clusters['n_clusters']:7d} |")
    print(f"| Noise ratio               | {rand_clusters['noise_ratio']:6.1%} | {untrained_clusters['noise_ratio']:8.1%} | {trained_clusters['noise_ratio']:6.1%} |")
    print(f"| Geodesic dist (mean)      | {rand_sphere['mean_geodesic_dist']:7.4f} | {untrained_sphere['mean_geodesic_dist']:9.4f} | {trained_sphere['mean_geodesic_dist']:7.4f} |")
    print(f"| UMAP spread               | {rand_spread['total_spread']:7.2f} | {untrained_spread['total_spread']:9.2f} | {trained_spread['total_spread']:7.2f} |")
    print()

    print("Wasserstein distances:")
    print(f"  Random <-> Trained:     {w_rand_trained:.4f}")
    print(f"  Untrained <-> Trained:  {w_untrained_trained:.4f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if trained_clusters['n_clusters'] > rand_clusters['n_clusters']:
        print("[OK] Trained embeddings form MORE clusters than random")
        print("  -> Semantic grouping exists")
    else:
        print("[X] Trained embeddings don't form more clusters")

    if trained_eff['participation_ratio'] < untrained_eff['participation_ratio']:
        print("[OK] Training REDUCES effective dimensionality")
        print(f"  -> {untrained_eff['participation_ratio']:.1f} -> {trained_eff['participation_ratio']:.1f} effective dims")

    if trained_sphere['mean_geodesic_dist'] < rand_sphere['mean_geodesic_dist']:
        print("[OK] Trained embeddings are MORE concentrated on sphere")
    else:
        print("* Trained embeddings are spread similar to random on sphere")

    print()

    # Save results
    output_path = args.output or str(Path(__file__).parent / 'results' / 'geometry_analysis.json')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
