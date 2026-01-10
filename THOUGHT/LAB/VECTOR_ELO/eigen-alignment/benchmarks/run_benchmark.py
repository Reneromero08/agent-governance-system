#!/usr/bin/env python3
"""Benchmark harness for Eigen-Spectrum Alignment.

Runs cross-model alignment benchmarks and produces receipts.

Usage:
    python run_benchmark.py [--models MODEL1,MODEL2,...] [--anchor-sizes 8,16,32,64]
    python run_benchmark.py --quick  # Quick test with 2 models, 8 anchors
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import mds, procrustes, protocol
from lib.adapters import SentenceTransformersAdapter


def load_word_list(path: Path) -> list[str]:
    """Load word list from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def compute_neighborhood_overlap(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    k: int = 10
) -> float:
    """Compute neighborhood overlap@k between two coordinate sets.

    For each point, computes the overlap between its k nearest
    neighbors in coords_a vs coords_b.

    Args:
        coords_a: (n, d) first coordinate set
        coords_b: (n, d) second coordinate set (aligned)
        k: Number of neighbors to consider

    Returns:
        Mean overlap ratio [0, 1]
    """
    n = coords_a.shape[0]
    k = min(k, n - 1)

    # Compute pairwise distances
    dist_a = cdist(coords_a, coords_a)
    dist_b = cdist(coords_b, coords_b)

    overlaps = []
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        nn_a = set(np.argsort(dist_a[i])[1:k+1])
        nn_b = set(np.argsort(dist_b[i])[1:k+1])

        overlap = len(nn_a & nn_b) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def run_model_pair_benchmark(
    model_a: str,
    model_b: str,
    anchors: list[str],
    held_out: list[str]
) -> dict[str, Any]:
    """Run benchmark for a pair of models.

    Args:
        model_a: First model name
        model_b: Second model name (reference)
        anchors: List of anchor words
        held_out: List of held-out evaluation words

    Returns:
        Benchmark results dictionary
    """
    print(f"\n  {model_a} -> {model_b}")

    # Load models
    adapter_a = SentenceTransformersAdapter(model_a)
    adapter_b = SentenceTransformersAdapter(model_b)

    # Compute anchor embeddings
    print(f"    Computing anchor embeddings...")
    emb_a_anchors = adapter_a.embed(anchors)
    emb_b_anchors = adapter_b.embed(anchors)

    # Compute MDS
    D2_a = mds.squared_distance_matrix(emb_a_anchors)
    D2_b = mds.squared_distance_matrix(emb_b_anchors)

    X_a, ev_a, V_a = mds.classical_mds(D2_a)
    X_b, ev_b, V_b = mds.classical_mds(D2_b)

    # Eigenvalue correlation
    k = min(len(ev_a), len(ev_b))
    spearman_r, _ = spearmanr(ev_a[:k], ev_b[:k])
    pearson_r, _ = pearsonr(ev_a[:k], ev_b[:k])

    print(f"    Eigenvalue correlation: Spearman={spearman_r:.4f}, Pearson={pearson_r:.4f}")

    # Procrustes alignment
    R, residual = procrustes.procrustes_align(X_a, X_b)
    print(f"    Procrustes residual: {residual:.4f}")

    # Held-out evaluation
    print(f"    Evaluating on {len(held_out)} held-out words...")
    emb_a_held = adapter_a.embed(held_out)
    emb_b_held = adapter_b.embed(held_out)

    # Compute held-out distances to anchors
    d2_a_held = cdist(emb_a_held, emb_a_anchors, 'sqeuclidean')
    d2_b_held = cdist(emb_b_held, emb_b_anchors, 'sqeuclidean')

    # Out-of-sample MDS projection
    Y_a = procrustes.out_of_sample_mds(d2_a_held, D2_a, V_a, ev_a)
    Y_b = procrustes.out_of_sample_mds(d2_b_held, D2_b, V_b, ev_b)

    # Align
    k_min = min(Y_a.shape[1], R.shape[0])
    Y_a_aligned = Y_a[:, :k_min] @ R[:k_min, :k_min]

    # Compute similarities before/after alignment
    raw_sims = []
    aligned_sims = []
    for i in range(len(held_out)):
        k_dim = min(Y_a.shape[1], Y_b.shape[1])
        raw_sim = procrustes.cosine_similarity(Y_a[i, :k_dim], Y_b[i, :k_dim])
        aligned_sim = procrustes.cosine_similarity(Y_a_aligned[i, :min(k_min, Y_b.shape[1])],
                                                    Y_b[i, :min(k_min, Y_b.shape[1])])
        raw_sims.append(raw_sim)
        aligned_sims.append(aligned_sim)

    mean_raw = float(np.mean(raw_sims))
    mean_aligned = float(np.mean(aligned_sims))
    improvement = mean_aligned - mean_raw

    print(f"    Mean similarity: raw={mean_raw:.4f}, aligned={mean_aligned:.4f}, Δ={improvement:+.4f}")

    # Neighborhood overlap
    overlap_10 = compute_neighborhood_overlap(Y_a_aligned, Y_b[:, :k_min], k=10)
    overlap_50 = compute_neighborhood_overlap(Y_a_aligned, Y_b[:, :k_min], k=50)
    print(f"    Neighborhood overlap: @10={overlap_10:.4f}, @50={overlap_50:.4f}")

    return {
        'model_a': model_a,
        'model_b': model_b,
        'n_anchors': len(anchors),
        'n_held_out': len(held_out),
        'eigenvalue_spearman': float(spearman_r),
        'eigenvalue_pearson': float(pearson_r),
        'procrustes_residual': float(residual),
        'mean_raw_similarity': mean_raw,
        'mean_aligned_similarity': mean_aligned,
        'improvement': improvement,
        'neighborhood_overlap_10': overlap_10,
        'neighborhood_overlap_50': overlap_50,
    }


def run_benchmark(
    models: list[str],
    anchor_sizes: list[int],
    output_dir: Path
) -> dict[str, Any]:
    """Run full benchmark suite.

    Args:
        models: List of model names
        anchor_sizes: List of anchor set sizes to test
        output_dir: Directory for output files

    Returns:
        Complete benchmark results
    """
    start_time = time.time()

    # Load held-out set
    held_out_path = Path(__file__).parent / 'held_out' / 'eval_set.txt'
    held_out = load_word_list(held_out_path)
    print(f"Loaded {len(held_out)} held-out words")

    results = {
        'benchmark_id': f"esap-bench-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'models': models,
        'anchor_sizes': anchor_sizes,
        'n_held_out': len(held_out),
        'model_pairs': [],
    }

    for anchor_size in anchor_sizes:
        print(f"\n{'='*60}")
        print(f"ANCHOR SIZE: {anchor_size}")
        print(f"{'='*60}")

        # Load anchors
        anchor_path = Path(__file__).parent / 'anchor_sets' / f'anchors_{anchor_size}.txt'
        anchors = load_word_list(anchor_path)
        print(f"Loaded {len(anchors)} anchors")

        # Test all pairs (reference is first model)
        reference = models[0]
        for model in models[1:]:
            pair_result = run_model_pair_benchmark(
                model_a=model,
                model_b=reference,
                anchors=anchors,
                held_out=held_out
            )
            pair_result['anchor_size'] = anchor_size
            results['model_pairs'].append(pair_result)

    # Summary statistics
    results['duration_seconds'] = time.time() - start_time

    all_improvements = [p['improvement'] for p in results['model_pairs']]
    all_eigenvalue_corr = [p['eigenvalue_spearman'] for p in results['model_pairs']]

    results['summary'] = {
        'mean_improvement': float(np.mean(all_improvements)),
        'max_improvement': float(np.max(all_improvements)),
        'min_improvement': float(np.min(all_improvements)),
        'mean_eigenvalue_correlation': float(np.mean(all_eigenvalue_corr)),
        'min_eigenvalue_correlation': float(np.min(all_eigenvalue_corr)),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='ESAP Benchmark Suite')
    parser.add_argument('--models', '-m', type=str,
                        default='all-MiniLM-L6-v2,all-mpnet-base-v2',
                        help='Comma-separated model names')
    parser.add_argument('--anchor-sizes', '-a', type=str,
                        default='8,16',
                        help='Comma-separated anchor sizes')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Quick test (2 models, 8 anchors)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory')

    args = parser.parse_args()

    if args.quick:
        models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
        anchor_sizes = [8]
    else:
        models = args.models.split(',')
        anchor_sizes = [int(x) for x in args.anchor_sizes.split(',')]

    output_dir = Path(args.output) if args.output else Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EIGEN-SPECTRUM ALIGNMENT BENCHMARK")
    print("="*60)
    print(f"Models: {models}")
    print(f"Anchor sizes: {anchor_sizes}")
    print(f"Output: {output_dir}")

    results = run_benchmark(models, anchor_sizes, output_dir)

    # Save results
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Generate report
    report_path = output_dir / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# ESAP Benchmark Report\n\n")
        f.write(f"**Benchmark ID:** {results['benchmark_id']}\n")
        f.write(f"**Timestamp:** {results['timestamp']}\n")
        f.write(f"**Duration:** {results['duration_seconds']:.1f}s\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Mean improvement | {results['summary']['mean_improvement']:+.4f} |\n")
        f.write(f"| Mean eigenvalue correlation | {results['summary']['mean_eigenvalue_correlation']:.4f} |\n")
        f.write(f"\n## Model Pairs\n\n")
        f.write(f"| Source | Target | Anchors | Eigenvalue r | Improvement | Overlap@10 |\n")
        f.write(f"|--------|--------|---------|--------------|-------------|------------|\n")
        for p in results['model_pairs']:
            f.write(f"| {p['model_a'][:15]} | {p['model_b'][:15]} | {p['anchor_size']} | ")
            f.write(f"{p['eigenvalue_spearman']:.4f} | {p['improvement']:+.4f} | {p['neighborhood_overlap_10']:.4f} |\n")

    print(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Mean improvement: {results['summary']['mean_improvement']:+.4f}")
    print(f"Mean eigenvalue correlation: {results['summary']['mean_eigenvalue_correlation']:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
