#!/usr/bin/env python3
"""E.X.3.1: Null Hypothesis Test - Random Embedding Baseline.

Tests whether eigenvalue spectrum invariance is a mathematical artifact
or reflects learned semantic structure.

Hypothesis:
    If random embeddings show Spearman ~ 1.0 -> invariance is trivial (FAIL)
    If random embeddings show Spearman << 1.0 -> invariance is learned (PASS)

Usage:
    python -m benchmarks.validation.null_hypothesis [--n-models 5] [--n-anchors 64] [--seeds 42,43,44]
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


def generate_random_embeddings(n_words: int, dim: int, seed: int) -> np.ndarray:
    """Generate random L2-normalized embeddings.

    Args:
        n_words: Number of words/vectors
        dim: Embedding dimension
        seed: Random seed

    Returns:
        (n_words, dim) L2-normalized random vectors
    """
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n_words, dim))
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def compute_eigenvalue_spectrum(embeddings: np.ndarray) -> np.ndarray:
    """Compute eigenvalue spectrum from embeddings.

    Args:
        embeddings: (n, d) L2-normalized embeddings

    Returns:
        Sorted eigenvalues (descending)
    """
    D2 = squared_distance_matrix(embeddings)
    _, eigenvalues, _ = classical_mds(D2)
    return eigenvalues


def run_random_baseline(
    n_models: int = 5,
    n_anchors: int = 64,
    dim: int = 384,
    base_seed: int = 42
) -> dict:
    """Run random embedding baseline test.

    Args:
        n_models: Number of random "models" to generate
        n_anchors: Number of anchor words
        dim: Embedding dimension
        base_seed: Base random seed

    Returns:
        Test results dict
    """
    print(f"Generating {n_models} random models ({n_anchors} anchors, {dim}d)")

    # Generate random embeddings for each "model"
    spectra = []
    for i in range(n_models):
        seed = base_seed + i
        embeddings = generate_random_embeddings(n_anchors, dim, seed)
        eigenvalues = compute_eigenvalue_spectrum(embeddings)
        spectra.append(eigenvalues)
        print(f"  Model {i+1}: {len(eigenvalues)} eigenvalues")

    # Compute pairwise Spearman correlations
    pairs = list(combinations(range(n_models), 2))
    spearman_values = []

    print(f"\nComputing Spearman correlation for {len(pairs)} pairs...")
    for i, j in pairs:
        # Use min length (should be same, but be safe)
        min_len = min(len(spectra[i]), len(spectra[j]))
        rho, pval = spearmanr(spectra[i][:min_len], spectra[j][:min_len])
        spearman_values.append(rho)
        print(f"  Pair ({i+1}, {j+1}): Spearman = {rho:.4f}")

    mean_spearman = float(np.mean(spearman_values))
    std_spearman = float(np.std(spearman_values))
    min_spearman = float(np.min(spearman_values))
    max_spearman = float(np.max(spearman_values))

    return {
        'n_models': n_models,
        'n_anchors': n_anchors,
        'dim': dim,
        'base_seed': base_seed,
        'n_pairs': len(pairs),
        'spearman_values': [float(v) for v in spearman_values],
        'mean_spearman': mean_spearman,
        'std_spearman': std_spearman,
        'min_spearman': min_spearman,
        'max_spearman': max_spearman,
    }


def run_permutation_test(
    n_permutations: int = 10,
    n_anchors: int = 64,
    dim: int = 384,
    seed: int = 42
) -> dict:
    """Run permutation test - shuffle word associations.

    Generates one "base" embedding set, then creates permuted versions
    by shuffling which row corresponds to which word.

    Args:
        n_permutations: Number of permuted versions
        n_anchors: Number of anchor words
        dim: Embedding dimension
        seed: Random seed

    Returns:
        Test results dict
    """
    print(f"\nPermutation test ({n_permutations} permutations)")

    rng = np.random.default_rng(seed)

    # Generate base embeddings
    base = generate_random_embeddings(n_anchors, dim, seed)
    base_spectrum = compute_eigenvalue_spectrum(base)

    # Create permuted versions and compare to base
    spearman_values = []
    for i in range(n_permutations):
        # Permute row order (shuffle word associations)
        perm = rng.permutation(n_anchors)
        permuted = base[perm]
        perm_spectrum = compute_eigenvalue_spectrum(permuted)

        min_len = min(len(base_spectrum), len(perm_spectrum))
        rho, _ = spearmanr(base_spectrum[:min_len], perm_spectrum[:min_len])
        spearman_values.append(rho)
        print(f"  Permutation {i+1}: Spearman = {rho:.4f}")

    return {
        'n_permutations': n_permutations,
        'n_anchors': n_anchors,
        'dim': dim,
        'seed': seed,
        'spearman_values': [float(v) for v in spearman_values],
        'mean_spearman': float(np.mean(spearman_values)),
        'std_spearman': float(np.std(spearman_values)),
    }


def run_alignment_improvement_test(
    n_pairs: int = 10,
    n_anchors: int = 64,
    dim: int = 384,
    base_seed: int = 42
) -> dict:
    """Test if Procrustes alignment improves similarity for random embeddings.

    This is the critical test: if random embeddings also show alignment improvement,
    then even that metric is trivial (geometric artifact).

    Args:
        n_pairs: Number of random model pairs to test
        n_anchors: Number of anchor words
        dim: Embedding dimension
        base_seed: Base random seed

    Returns:
        Test results dict with improvement metrics
    """
    print(f"\nAlignment improvement test ({n_pairs} random pairs)")

    improvements = []
    raw_sims = []
    aligned_sims = []
    residuals = []

    for pair_idx in range(n_pairs):
        # Generate two different random "models"
        seed_a = base_seed + pair_idx * 2
        seed_b = base_seed + pair_idx * 2 + 1

        emb_a = generate_random_embeddings(n_anchors, dim, seed_a)
        emb_b = generate_random_embeddings(n_anchors, dim, seed_b)

        # Compute MDS coordinates for each
        D2_a = squared_distance_matrix(emb_a)
        D2_b = squared_distance_matrix(emb_b)

        X_a, eigenvalues_a, _ = classical_mds(D2_a)
        X_b, eigenvalues_b, _ = classical_mds(D2_b)

        # Use minimum dimensionality
        k_min = min(X_a.shape[1], X_b.shape[1])
        X_a = X_a[:, :k_min]
        X_b = X_b[:, :k_min]

        # Procrustes alignment: rotate X_a to match X_b
        R, residual = procrustes_align(X_a, X_b)
        X_a_aligned = X_a @ R

        # Measure raw and aligned similarities
        pair_raw_sims = []
        pair_aligned_sims = []
        for i in range(n_anchors):
            raw_sim = cosine_similarity(X_a[i], X_b[i])
            aligned_sim = cosine_similarity(X_a_aligned[i], X_b[i])
            pair_raw_sims.append(raw_sim)
            pair_aligned_sims.append(aligned_sim)

        mean_raw = float(np.mean(pair_raw_sims))
        mean_aligned = float(np.mean(pair_aligned_sims))
        improvement = mean_aligned - mean_raw

        improvements.append(improvement)
        raw_sims.append(mean_raw)
        aligned_sims.append(mean_aligned)
        residuals.append(float(residual))

        print(f"  Pair {pair_idx+1}: raw={mean_raw:.4f}, aligned={mean_aligned:.4f}, "
              f"improvement={improvement:+.4f}, residual={residual:.4f}")

    return {
        'n_pairs': n_pairs,
        'n_anchors': n_anchors,
        'dim': dim,
        'base_seed': base_seed,
        'improvements': improvements,
        'mean_improvement': float(np.mean(improvements)),
        'std_improvement': float(np.std(improvements)),
        'mean_raw_similarity': float(np.mean(raw_sims)),
        'mean_aligned_similarity': float(np.mean(aligned_sims)),
        'mean_residual': float(np.mean(residuals)),
    }


def interpret_results(random_result: dict, trained_baseline: float = 1.0) -> dict:
    """Interpret test results.

    Args:
        random_result: Results from random baseline test
        trained_baseline: Expected Spearman for trained models (default: 1.0)

    Returns:
        Interpretation dict
    """
    mean_random = random_result['mean_spearman']

    # Decision threshold
    if mean_random > 0.9:
        verdict = "FAIL"
        explanation = (
            f"Random embeddings show high correlation (mean={mean_random:.4f}). "
            "The eigenvalue invariance is likely a mathematical artifact, "
            "not a property of learned semantic structure."
        )
    elif mean_random > 0.5:
        verdict = "INCONCLUSIVE"
        explanation = (
            f"Random embeddings show moderate correlation (mean={mean_random:.4f}). "
            "Partial invariance may be due to geometry of high-dimensional spaces. "
            "Further investigation needed."
        )
    else:
        verdict = "PASS"
        explanation = (
            f"Random embeddings show low correlation (mean={mean_random:.4f}). "
            f"Trained models show {trained_baseline:.4f}. "
            "The invariance appears to be a property of learned representations, "
            "supporting the Platonic hypothesis."
        )

    effect_size = trained_baseline - mean_random

    return {
        'verdict': verdict,
        'explanation': explanation,
        'effect_size': effect_size,
        'random_mean': mean_random,
        'trained_baseline': trained_baseline,
    }


def main():
    parser = argparse.ArgumentParser(description='E.X.3.1: Null Hypothesis Test')
    parser.add_argument('--n-models', type=int, default=5,
                        help='Number of random models to generate')
    parser.add_argument('--n-anchors', type=int, default=64,
                        help='Number of anchor words')
    parser.add_argument('--dim', type=int, default=384,
                        help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    print("=" * 60)
    print("E.X.3.1: NULL HYPOTHESIS TEST")
    print("=" * 60)
    print()
    print("Testing if eigenvalue spectrum invariance is a mathematical artifact")
    print("or a property of learned semantic representations.")
    print()

    # Run random baseline
    random_result = run_random_baseline(
        n_models=args.n_models,
        n_anchors=args.n_anchors,
        dim=args.dim,
        base_seed=args.seed
    )

    # Run permutation test
    perm_result = run_permutation_test(
        n_permutations=10,
        n_anchors=args.n_anchors,
        dim=args.dim,
        seed=args.seed
    )

    # Run alignment improvement test
    alignment_result = run_alignment_improvement_test(
        n_pairs=10,
        n_anchors=args.n_anchors,
        dim=args.dim,
        base_seed=args.seed
    )

    # Interpret results
    interpretation = interpret_results(random_result)

    # Interpret alignment results
    trained_improvement = 0.43  # From trained model benchmarks
    random_improvement = alignment_result['mean_improvement']
    if random_improvement > 0.3:
        alignment_verdict = "FAIL"
        alignment_explanation = (
            f"Random embeddings show high alignment improvement ({random_improvement:+.4f}). "
            "Even the alignment improvement is a geometric artifact."
        )
    elif random_improvement > 0.1:
        alignment_verdict = "INCONCLUSIVE"
        alignment_explanation = (
            f"Random embeddings show moderate alignment improvement ({random_improvement:+.4f}). "
            "Partial improvement may be geometric. Trained improvement ({trained_improvement:+.4f}) "
            "may still be meaningful if significantly higher."
        )
    else:
        alignment_verdict = "PASS"
        alignment_explanation = (
            f"Random embeddings show low alignment improvement ({random_improvement:+.4f}). "
            f"Trained models show {trained_improvement:+.4f}. "
            "The alignment improvement reflects learned semantic structure."
        )

    alignment_interpretation = {
        'verdict': alignment_verdict,
        'explanation': alignment_explanation,
        'random_improvement': random_improvement,
        'trained_baseline': trained_improvement,
        'effect_size': trained_improvement - random_improvement,
    }

    # Assemble full result
    result = {
        'test_id': 'null-hypothesis-E.X.3.1',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'n_models': args.n_models,
            'n_anchors': args.n_anchors,
            'dim': args.dim,
            'base_seed': args.seed,
        },
        'random_baseline': random_result,
        'permutation_test': perm_result,
        'alignment_improvement': alignment_result,
        'eigenvalue_interpretation': interpretation,
        'alignment_interpretation': alignment_interpretation,
    }

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("--- Eigenvalue Spearman (already known to be trivial) ---")
    print(f"Random baseline mean Spearman: {random_result['mean_spearman']:.4f}")
    print(f"Trained model baseline:        1.0000")
    print(f"VERDICT: {interpretation['verdict']}")
    print()
    print("--- Alignment Improvement (the real test) ---")
    print(f"Random mean improvement:  {alignment_result['mean_improvement']:+.4f}")
    print(f"Random raw similarity:    {alignment_result['mean_raw_similarity']:.4f}")
    print(f"Random aligned similarity: {alignment_result['mean_aligned_similarity']:.4f}")
    print(f"Trained improvement:      +0.4300 (from benchmarks)")
    print(f"Effect size:              {alignment_interpretation['effect_size']:+.4f}")
    print()
    print(f"ALIGNMENT VERDICT: {alignment_interpretation['verdict']}")
    print()
    print(alignment_interpretation['explanation'])
    print()

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'null_hypothesis.json'

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Exit code based on verdict
    if interpretation['verdict'] == 'FAIL':
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
