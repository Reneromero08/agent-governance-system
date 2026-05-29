#!/usr/bin/env python3
"""E.X.3.2: Alignment Resistance Test - Does structure resist rotation?

Key insight from E.X.3.1:
    Random embeddings: +0.96 alignment improvement (infinitely malleable)
    Trained embeddings: +0.43 alignment improvement (something resists)

    The GAP (0.53) might BE the signal of learned semantic structure.

Hypothesis:
    Models with MORE semantic structure should show LESS alignment improvement
    because real relationships resist arbitrary rotation.

Test:
    1. Measure alignment resistance (1 - improvement) for each model
    2. Measure semantic quality via SimLex-999 correlation
    3. If resistance correlates with semantic quality -> PASS

Usage:
    python -m benchmarks.validation.alignment_resistance
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr, pearsonr

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.mds import squared_distance_matrix, classical_mds
from lib.procrustes import procrustes_align, cosine_similarity


# =============================================================================
# SIMLEX-999 GROUND TRUTH
# =============================================================================

# Subset of SimLex-999 word pairs with human similarity scores (0-10 scale)
# These are high-confidence pairs spanning the similarity spectrum
SIMLEX_SUBSET = [
    # High similarity (7-10)
    ("old", "new", 1.58),  # antonyms - should be LOW
    ("smart", "intelligent", 9.20),
    ("hard", "difficult", 8.77),
    ("happy", "cheerful", 9.55),
    ("fast", "quick", 8.75),

    # Medium similarity (4-7)
    ("cup", "mug", 8.02),
    ("car", "automobile", 8.94),
    ("big", "large", 8.47),
    ("small", "little", 8.75),
    ("dog", "cat", 1.75),  # related but different

    # Low similarity (0-4)
    ("king", "queen", 2.48),  # related but different gender
    ("man", "woman", 2.47),
    ("hot", "cold", 1.08),  # antonyms
    ("good", "bad", 0.65),  # antonyms
    ("up", "down", 0.54),  # antonyms

    # Unrelated (0-2)
    ("book", "tree", 0.28),
    ("car", "apple", 0.12),
    ("computer", "banana", 0.22),
    ("dog", "computer", 1.04),
    ("house", "water", 0.50),
]


def load_simlex_words() -> list:
    """Get unique words from SimLex subset."""
    words = set()
    for w1, w2, _ in SIMLEX_SUBSET:
        words.add(w1)
        words.add(w2)
    return sorted(list(words))


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

def generate_random_embeddings(words: list, dim: int, seed: int) -> dict:
    """Generate random L2-normalized embeddings for words."""
    rng = np.random.default_rng(seed)
    embeddings = {}
    for word in words:
        vec = rng.standard_normal(dim)
        vec = vec / np.linalg.norm(vec)
        embeddings[word] = vec
    return embeddings


def generate_semantic_embeddings(words: list, dim: int, seed: int,
                                  semantic_strength: float = 1.0) -> dict:
    """Generate embeddings with injected semantic structure.

    Args:
        words: List of words
        dim: Embedding dimension
        seed: Random seed
        semantic_strength: 0.0 = random, 1.0 = full semantic structure

    Returns:
        Dict mapping words to embeddings
    """
    rng = np.random.default_rng(seed)

    # Start with random base
    embeddings = {}
    for word in words:
        vec = rng.standard_normal(dim)
        embeddings[word] = vec

    if semantic_strength > 0:
        # Inject semantic structure: pull similar words together
        for w1, w2, sim in SIMLEX_SUBSET:
            if w1 in embeddings and w2 in embeddings:
                # Normalize similarity to [0, 1]
                sim_norm = sim / 10.0

                # Blend embeddings based on similarity
                blend = semantic_strength * sim_norm * 0.5
                embeddings[w1] = (1 - blend) * embeddings[w1] + blend * embeddings[w2]
                embeddings[w2] = (1 - blend) * embeddings[w2] + blend * embeddings[w1]

    # L2 normalize all
    for word in embeddings:
        embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])

    return embeddings


# =============================================================================
# METRICS
# =============================================================================

def compute_simlex_correlation(embeddings: dict) -> float:
    """Compute Spearman correlation with SimLex human judgments."""
    model_sims = []
    human_sims = []

    for w1, w2, human_sim in SIMLEX_SUBSET:
        if w1 in embeddings and w2 in embeddings:
            # Cosine similarity
            model_sim = np.dot(embeddings[w1], embeddings[w2])
            model_sims.append(model_sim)
            human_sims.append(human_sim)

    if len(model_sims) < 3:
        return 0.0

    rho, _ = spearmanr(model_sims, human_sims)
    return float(rho) if np.isfinite(rho) else 0.0


def compute_alignment_improvement(emb_a: dict, emb_b: dict, words: list) -> dict:
    """Compute alignment improvement between two embedding sets.

    Returns dict with raw_sim, aligned_sim, improvement, resistance.
    """
    # Build embedding matrices (same word order)
    common_words = [w for w in words if w in emb_a and w in emb_b]
    n = len(common_words)

    if n < 3:
        return {'improvement': 0.0, 'resistance': 1.0}

    dim = len(emb_a[common_words[0]])
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

    mean_raw = float(np.mean(raw_sims))
    mean_aligned = float(np.mean(aligned_sims))
    improvement = mean_aligned - mean_raw

    # Resistance = how much the model resists perfect alignment
    # Random gets ~0.96 aligned similarity, so resistance = 1 - (aligned / 0.96)
    # Or simpler: resistance = 1 - improvement (relative to random's 0.96)
    resistance = 1.0 - (improvement / 0.96) if improvement > 0 else 1.0

    return {
        'raw_similarity': mean_raw,
        'aligned_similarity': mean_aligned,
        'improvement': improvement,
        'resistance': resistance,
        'residual': float(residual),
        'n_words': n,
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_semantic_strength_sweep(
    n_trials: int = 5,
    dim: int = 100,
    base_seed: int = 42
) -> list:
    """Sweep semantic strength and measure resistance vs quality.

    This is the key test: as we inject more semantic structure,
    does resistance increase AND does SimLex correlation increase?
    """
    print("=" * 60)
    print("E.X.3.2: ALIGNMENT RESISTANCE TEST")
    print("=" * 60)
    print()
    print("Testing: Does semantic structure cause alignment resistance?")
    print()

    words = load_simlex_words()
    print(f"Using {len(words)} words from SimLex-999 subset")
    print()

    # Sweep semantic strength from 0 (random) to 1 (full structure)
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    print("-" * 60)
    print(f"{'Strength':>10} | {'Resistance':>10} | {'SimLex r':>10} | {'Improvement':>12}")
    print("-" * 60)

    for strength in strengths:
        trial_resistances = []
        trial_simlex = []
        trial_improvements = []

        for trial in range(n_trials):
            seed_a = base_seed + trial * 100
            seed_b = base_seed + trial * 100 + 1

            # Generate embeddings with specified semantic strength
            emb_a = generate_semantic_embeddings(words, dim, seed_a, strength)
            emb_b = generate_semantic_embeddings(words, dim, seed_b, strength)

            # Measure alignment resistance
            align_result = compute_alignment_improvement(emb_a, emb_b, words)
            trial_resistances.append(align_result['resistance'])
            trial_improvements.append(align_result['improvement'])

            # Measure semantic quality (average of both)
            simlex_a = compute_simlex_correlation(emb_a)
            simlex_b = compute_simlex_correlation(emb_b)
            trial_simlex.append((simlex_a + simlex_b) / 2)

        mean_resistance = float(np.mean(trial_resistances))
        mean_simlex = float(np.mean(trial_simlex))
        mean_improvement = float(np.mean(trial_improvements))

        results.append({
            'semantic_strength': strength,
            'mean_resistance': mean_resistance,
            'mean_simlex_correlation': mean_simlex,
            'mean_improvement': mean_improvement,
            'std_resistance': float(np.std(trial_resistances)),
            'std_simlex': float(np.std(trial_simlex)),
        })

        print(f"{strength:>10.2f} | {mean_resistance:>10.4f} | {mean_simlex:>10.4f} | {mean_improvement:>+12.4f}")

    return results


def run_random_vs_trained_comparison(
    n_pairs: int = 10,
    dim: int = 100,
    base_seed: int = 42
) -> dict:
    """Direct comparison: random vs semantically-structured embeddings."""
    print()
    print("-" * 60)
    print("Random vs Semantic comparison")
    print("-" * 60)
    print()

    words = load_simlex_words()

    random_improvements = []
    semantic_improvements = []
    random_simlex = []
    semantic_simlex = []

    for i in range(n_pairs):
        seed_a = base_seed + i * 2
        seed_b = base_seed + i * 2 + 1

        # Random embeddings
        rand_a = generate_random_embeddings(words, dim, seed_a)
        rand_b = generate_random_embeddings(words, dim, seed_b)
        rand_result = compute_alignment_improvement(rand_a, rand_b, words)
        random_improvements.append(rand_result['improvement'])
        random_simlex.append((compute_simlex_correlation(rand_a) +
                              compute_simlex_correlation(rand_b)) / 2)

        # Semantic embeddings (full strength)
        sem_a = generate_semantic_embeddings(words, dim, seed_a, 1.0)
        sem_b = generate_semantic_embeddings(words, dim, seed_b, 1.0)
        sem_result = compute_alignment_improvement(sem_a, sem_b, words)
        semantic_improvements.append(sem_result['improvement'])
        semantic_simlex.append((compute_simlex_correlation(sem_a) +
                                compute_simlex_correlation(sem_b)) / 2)

    result = {
        'random': {
            'mean_improvement': float(np.mean(random_improvements)),
            'std_improvement': float(np.std(random_improvements)),
            'mean_simlex': float(np.mean(random_simlex)),
        },
        'semantic': {
            'mean_improvement': float(np.mean(semantic_improvements)),
            'std_improvement': float(np.std(semantic_improvements)),
            'mean_simlex': float(np.mean(semantic_simlex)),
        },
        'gap': float(np.mean(random_improvements) - np.mean(semantic_improvements)),
    }

    print(f"Random:   improvement={result['random']['mean_improvement']:+.4f}, SimLex={result['random']['mean_simlex']:.4f}")
    print(f"Semantic: improvement={result['semantic']['mean_improvement']:+.4f}, SimLex={result['semantic']['mean_simlex']:.4f}")
    print(f"Gap:      {result['gap']:+.4f}")

    return result


def interpret_results(sweep_results: list, comparison: dict) -> dict:
    """Interpret the test results."""

    # Extract arrays for correlation
    strengths = [r['semantic_strength'] for r in sweep_results]
    resistances = [r['mean_resistance'] for r in sweep_results]
    simlex_corrs = [r['mean_simlex_correlation'] for r in sweep_results]

    # Key test: does resistance correlate with semantic quality?
    if len(strengths) >= 3:
        # Resistance vs strength
        r_resist_strength, _ = spearmanr(strengths, resistances)
        # SimLex vs strength
        r_simlex_strength, _ = spearmanr(strengths, simlex_corrs)
        # Resistance vs SimLex
        r_resist_simlex, _ = spearmanr(resistances, simlex_corrs)
    else:
        r_resist_strength = 0.0
        r_simlex_strength = 0.0
        r_resist_simlex = 0.0

    # Decision logic
    gap = comparison['gap']

    if gap > 0.3 and r_resist_simlex > 0.7:
        verdict = "PASS"
        explanation = (
            f"Semantic structure causes alignment resistance (gap={gap:.3f}). "
            f"Resistance correlates with SimLex (r={r_resist_simlex:.3f}). "
            "The 'resistance to alignment' IS the signal of learned structure."
        )
    elif gap > 0.1 and r_resist_simlex > 0.5:
        verdict = "PARTIAL"
        explanation = (
            f"Moderate evidence: gap={gap:.3f}, resistance-SimLex r={r_resist_simlex:.3f}. "
            "More investigation needed with real trained models."
        )
    else:
        verdict = "INCONCLUSIVE"
        explanation = (
            f"Weak signal: gap={gap:.3f}, resistance-SimLex r={r_resist_simlex:.3f}. "
            "Synthetic semantic injection may not capture real model behavior."
        )

    return {
        'verdict': verdict,
        'explanation': explanation,
        'correlations': {
            'resistance_vs_strength': float(r_resist_strength) if np.isfinite(r_resist_strength) else 0.0,
            'simlex_vs_strength': float(r_simlex_strength) if np.isfinite(r_simlex_strength) else 0.0,
            'resistance_vs_simlex': float(r_resist_simlex) if np.isfinite(r_resist_simlex) else 0.0,
        },
        'gap': gap,
    }


def main():
    parser = argparse.ArgumentParser(description='E.X.3.2: Alignment Resistance Test')
    parser.add_argument('--n-trials', type=int, default=5,
                        help='Trials per semantic strength level')
    parser.add_argument('--n-pairs', type=int, default=10,
                        help='Pairs for random vs semantic comparison')
    parser.add_argument('--dim', type=int, default=100,
                        help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args()

    # Run sweep
    sweep_results = run_semantic_strength_sweep(
        n_trials=args.n_trials,
        dim=args.dim,
        base_seed=args.seed
    )

    # Run comparison
    comparison = run_random_vs_trained_comparison(
        n_pairs=args.n_pairs,
        dim=args.dim,
        base_seed=args.seed
    )

    # Interpret
    interpretation = interpret_results(sweep_results, comparison)

    # Print summary
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"VERDICT: {interpretation['verdict']}")
    print()
    print(interpretation['explanation'])
    print()
    print("Correlations:")
    for k, v in interpretation['correlations'].items():
        print(f"  {k}: {v:.4f}")
    print()

    # Assemble result
    result = {
        'test_id': 'alignment-resistance-E.X.3.2',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'n_trials': args.n_trials,
            'n_pairs': args.n_pairs,
            'dim': args.dim,
            'base_seed': args.seed,
        },
        'sweep_results': sweep_results,
        'comparison': comparison,
        'interpretation': interpretation,
    }

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'alignment_resistance.json'

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0 if interpretation['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
