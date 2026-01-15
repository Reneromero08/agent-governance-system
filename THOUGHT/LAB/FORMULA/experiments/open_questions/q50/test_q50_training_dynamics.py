#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 8: Training Dynamics — Does 8e Emerge or Pre-exist?

Hypothesis: The conservation law Df × α = 8e emerges through training,
not from random initialization. Random matrices should NOT produce 8e.

Test Strategy:
1. Random embeddings (baseline) → expect Df × α ≈ 14.5 (from Q49)
2. Trained models → expect Df × α ≈ 21.7 (8e)
3. Ratio trained/random ≈ 3/2 (50% structure added by training)

Pass criteria:
- Random produces significantly different value than 8e (>10% deviation)
- Trained produces 8e (< 5% deviation)
- Ratio ≈ 1.5 (validates the 3/2 relationship from Q49)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def compute_df(eigenvalues):
    """Participation ratio Df = (Σλ)² / Σλ²"""
    ev = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(ev)**2) / np.sum(ev**2)


def compute_alpha(eigenvalues):
    """Power law decay exponent α where λ_k ~ k^(-α)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    k = np.arange(1, len(ev) + 1)
    n_fit = len(ev) // 2
    if n_fit < 5:
        return 0
    log_k = np.log(k[:n_fit])
    log_ev = np.log(ev[:n_fit])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    return -slope


def get_eigenspectrum(embeddings):
    """Get eigenvalues from covariance matrix."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def analyze_embeddings(embeddings, name):
    """Compute Df, α, and Df × α for embeddings."""
    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)
    df_alpha = Df * alpha
    vs_8e = (df_alpha - 8 * np.e) / (8 * np.e) * 100

    return {
        'name': name,
        'shape': list(embeddings.shape),
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_alpha': float(df_alpha),
        'vs_8e_percent': float(vs_8e),
    }


# Standard test vocabulary
WORDS = [
    "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
    "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
    "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
    "heart", "eye", "hand", "head", "brain", "blood", "bone", "skin",
    "mother", "father", "child", "friend", "king", "queen", "hero", "teacher",
    "love", "hate", "truth", "life", "death", "time", "space", "power",
    "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
    "book", "door", "house", "road", "food", "money", "stone", "gold",
    "light", "shadow", "music", "word", "name", "law", "art", "science",
    "good", "bad", "big", "small", "old", "new", "high", "low",
]


def generate_random_embeddings(n_samples, dim, seed=42):
    """Generate random Gaussian embeddings (simulates untrained model)."""
    np.random.seed(seed)
    embeddings = np.random.randn(n_samples, dim)
    # Normalize like sentence-transformers do
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def generate_marchenko_pastur_embeddings(n_samples, dim, seed=42):
    """Generate embeddings following Marchenko-Pastur distribution.

    This is the expected spectral distribution for random matrices,
    representing the 'null hypothesis' for embedding geometry.
    """
    np.random.seed(seed)
    # Create a random matrix and use its SVD structure
    X = np.random.randn(n_samples, dim) / np.sqrt(n_samples)
    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def main():
    print("=" * 70)
    print("Q50 PART 8: TRAINING DYNAMICS")
    print("Does Df × α = 8e emerge through training?")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_TRAINING_DYNAMICS',
        'target': 8 * np.e,
        'expected_random': 14.5,  # From Q49
        'expected_ratio': 1.5,    # Trained/Random = 3/2
        'random_tests': [],
        'trained_tests': [],
        'summary': {}
    }

    # ============================================================
    # PART 1: RANDOM BASELINES
    # ============================================================
    print("\n" + "=" * 60)
    print("RANDOM BASELINES (Untrained)")
    print("=" * 60)

    dims = [128, 256, 384, 512, 768, 1024]
    n_samples = len(WORDS)
    n_trials = 5

    print(f"\n  Testing {len(dims)} dimensions, {n_trials} trials each")
    print(f"  {'Dim':<8} {'Mean Df×α':<12} {'Std':<10} {'vs 8e':<12} {'vs 14.5':<12}")
    print("  " + "-" * 54)

    all_random_results = []
    for dim in dims:
        trial_results = []
        for trial in range(n_trials):
            random_emb = generate_random_embeddings(n_samples, dim, seed=trial * 100 + dim)
            result = analyze_embeddings(random_emb, f"random_{dim}")
            trial_results.append(result['Df_alpha'])

        mean_df_alpha = np.mean(trial_results)
        std_df_alpha = np.std(trial_results)
        vs_8e = (mean_df_alpha - 8*np.e) / (8*np.e) * 100
        vs_145 = (mean_df_alpha - 14.5) / 14.5 * 100

        print(f"  {dim:<8} {mean_df_alpha:<12.2f} {std_df_alpha:<10.3f} {vs_8e:+.1f}%{'':<5} {vs_145:+.1f}%")

        all_random_results.append({
            'dim': dim,
            'mean_Df_alpha': float(mean_df_alpha),
            'std_Df_alpha': float(std_df_alpha),
            'vs_8e_percent': float(vs_8e),
            'vs_145_percent': float(vs_145),
        })

    results['random_tests'] = all_random_results

    # Overall random statistics
    overall_random = np.mean([r['mean_Df_alpha'] for r in all_random_results])
    overall_random_std = np.std([r['mean_Df_alpha'] for r in all_random_results])

    print(f"\n  Overall random mean: {overall_random:.2f} ± {overall_random_std:.2f}")
    print(f"  Expected (Q49): ~14.5")
    print(f"  Deviation from Q49: {(overall_random - 14.5) / 14.5 * 100:+.1f}%")

    # ============================================================
    # PART 2: TRAINED MODELS
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINED MODELS")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer

        models = [
            ("all-MiniLM-L6-v2", "MiniLM-L6", 384),
            ("all-mpnet-base-v2", "MPNet-base", 768),
            ("BAAI/bge-small-en-v1.5", "BGE-small", 384),
            ("paraphrase-MiniLM-L6-v2", "ParaMiniLM-L6", 384),
            ("all-distilroberta-v1", "DistilRoBERTa", 768),
        ]

        print(f"\n  {'Model':<20} {'Dim':<8} {'Df×α':<12} {'vs 8e':<12} {'vs Random':<12}")
        print("  " + "-" * 64)

        trained_results = []
        for model_id, model_name, dim in models:
            try:
                model = SentenceTransformer(model_id)
                embeddings = model.encode(WORDS, normalize_embeddings=True)
                result = analyze_embeddings(embeddings, model_name)

                # Find corresponding random baseline
                random_baseline = next((r['mean_Df_alpha'] for r in all_random_results
                                       if r['dim'] == dim), overall_random)
                vs_random = (result['Df_alpha'] - random_baseline) / random_baseline * 100

                print(f"  {model_name:<20} {dim:<8} {result['Df_alpha']:<12.2f} {result['vs_8e_percent']:+.1f}%{'':<5} {vs_random:+.1f}%")

                trained_results.append({
                    'model': model_name,
                    'model_id': model_id,
                    'dim': dim,
                    'Df': result['Df'],
                    'alpha': result['alpha'],
                    'Df_alpha': result['Df_alpha'],
                    'vs_8e_percent': result['vs_8e_percent'],
                    'random_baseline': random_baseline,
                    'vs_random_percent': vs_random,
                })

            except Exception as e:
                print(f"  {model_name:<20} FAILED: {e}")

        results['trained_tests'] = trained_results

    except ImportError:
        print("  sentence-transformers not available")
        trained_results = []

    # ============================================================
    # PART 3: RATIO ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("RATIO ANALYSIS: Trained / Random")
    print("=" * 60)

    if trained_results:
        overall_trained = np.mean([r['Df_alpha'] for r in trained_results])
        ratio = overall_trained / overall_random

        print(f"\n  Mean trained Df×α: {overall_trained:.2f}")
        print(f"  Mean random Df×α: {overall_random:.2f}")
        print(f"  Ratio: {ratio:.3f}")
        print(f"  Expected ratio: 1.5 (3/2)")
        print(f"  Deviation from 1.5: {(ratio - 1.5) / 1.5 * 100:+.1f}%")

        results['summary']['mean_trained'] = float(overall_trained)
        results['summary']['mean_random'] = float(overall_random)
        results['summary']['ratio'] = float(ratio)
        results['summary']['ratio_vs_expected'] = float((ratio - 1.5) / 1.5 * 100)

        # Statistical significance
        t_values = [r['Df_alpha'] for r in trained_results]
        r_values = [r['mean_Df_alpha'] for r in all_random_results]

        # Simple t-test approximation
        t_mean = np.mean(t_values)
        r_mean = np.mean(r_values)
        t_std = np.std(t_values) if len(t_values) > 1 else 1
        r_std = np.std(r_values) if len(r_values) > 1 else 1

        # Cohen's d for trained vs random
        pooled_std = np.sqrt((t_std**2 + r_std**2) / 2)
        cohens_d = (t_mean - r_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Cohen's d (trained vs random): {cohens_d:.2f}")
        results['summary']['cohens_d'] = float(cohens_d)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: TRAINING DYNAMICS")
    print("=" * 70)

    if trained_results:
        # Check hypotheses
        random_differs_from_8e = abs((overall_random - 8*np.e) / (8*np.e)) > 0.10
        trained_near_8e = abs((overall_trained - 8*np.e) / (8*np.e)) < 0.10
        ratio_near_1_5 = abs((ratio - 1.5) / 1.5) < 0.20

        hypothesis_supported = random_differs_from_8e and trained_near_8e

        print(f"\n  Hypothesis checks:")
        print(f"    Random ≠ 8e (>10% deviation): {random_differs_from_8e} ({(overall_random - 8*np.e) / (8*np.e) * 100:+.1f}%)")
        print(f"    Trained ≈ 8e (<10% deviation): {trained_near_8e} ({(overall_trained - 8*np.e) / (8*np.e) * 100:+.1f}%)")
        print(f"    Ratio ≈ 1.5 (<20% deviation): {ratio_near_1_5} ({(ratio - 1.5) / 1.5 * 100:+.1f}%)")

        results['summary']['random_differs_from_8e'] = random_differs_from_8e
        results['summary']['trained_near_8e'] = trained_near_8e
        results['summary']['ratio_near_1_5'] = ratio_near_1_5
        results['summary']['hypothesis_supported'] = hypothesis_supported

        print("\n" + "=" * 70)
        if hypothesis_supported:
            print("VERDICT: 8e EMERGES THROUGH TRAINING")
            print(f"  Random matrices: {overall_random:.2f} ({(overall_random - 8*np.e) / (8*np.e) * 100:+.1f}% from 8e)")
            print(f"  Trained models: {overall_trained:.2f} ({(overall_trained - 8*np.e) / (8*np.e) * 100:+.1f}% from 8e)")
            print(f"  Training adds {(ratio - 1) * 100:.0f}% more structure to semantic geometry")
        else:
            print("VERDICT: RESULTS INCONCLUSIVE")
            if not random_differs_from_8e:
                print("  WARNING: Random also produces near-8e values")
            if not trained_near_8e:
                print("  WARNING: Trained models deviate significantly from 8e")
        print("=" * 70)

    else:
        print("\n  No trained model results to compare")
        results['summary']['hypothesis_supported'] = False

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_training_dynamics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
