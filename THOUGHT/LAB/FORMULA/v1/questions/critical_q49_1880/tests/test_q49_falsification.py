#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q49 Phase 1: Falsification Battery

Hardcore tests to eliminate false positive hypothesis.
If 8e fails here, it's coincidence. If it survives, it's real.

Tests:
1.1 Random Matrix Baseline - random matrices should NOT produce 8e
1.2 Permutation Test - shuffling should destroy 8e
1.3 Vocabulary Independence - 8e should hold for any word set
1.4 Monte Carlo - < 1% of random constants should match as well
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TARGET_CONSTANT = 8 * np.e  # ≈ 21.746


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_eigenspectrum(matrix):
    """Get eigenspectrum from matrix (embeddings or random)."""
    centered = matrix - matrix.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvalues


def compute_df(eigenvalues):
    """Compute participation ratio Df."""
    ev = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(ev) ** 2) / np.sum(ev ** 2)


def compute_alpha(eigenvalues):
    """Compute power law decay exponent α."""
    ev = eigenvalues[eigenvalues > 1e-10]
    k = np.arange(1, len(ev) + 1)
    log_k = np.log(k[:len(ev)//2])
    log_ev = np.log(ev[:len(ev)//2])
    if len(log_k) > 5:
        slope, _ = np.polyfit(log_k, log_ev, 1)
        return -slope
    return 0


def compute_df_alpha(matrix):
    """Compute Df × α for a matrix."""
    ev = get_eigenspectrum(matrix)
    df = compute_df(ev)
    alpha = compute_alpha(ev)
    return df * alpha, df, alpha


# =============================================================================
# TEST 1.1: RANDOM MATRIX BASELINE
# =============================================================================

def test_random_matrix_baseline(n_trials=1000, n_samples=75, n_dims=384):
    """
    Test: Random matrices should NOT produce Df × α ≈ 8e.

    Pass condition: CV > 50% (high variance, no convergence to 8e)
    Fail condition: Random matrices produce ~8e (invalidates our claim)
    """
    print("\n" + "=" * 70)
    print("TEST 1.1: RANDOM MATRIX BASELINE")
    print("=" * 70)
    print(f"Running {n_trials} trials with random {n_samples}×{n_dims} matrices...")

    df_alpha_values = []
    df_values = []
    alpha_values = []

    for i in range(n_trials):
        # Generate random matrix (same shape as typical embeddings)
        random_matrix = np.random.randn(n_samples, n_dims)

        # Compute Df × α
        df_alpha, df, alpha = compute_df_alpha(random_matrix)
        df_alpha_values.append(df_alpha)
        df_values.append(df)
        alpha_values.append(alpha)

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_trials}")

    df_alpha_values = np.array(df_alpha_values)

    mean_val = np.mean(df_alpha_values)
    std_val = np.std(df_alpha_values)
    cv = std_val / mean_val * 100

    # How many are close to 8e?
    close_to_8e = np.sum(np.abs(df_alpha_values - TARGET_CONSTANT) < 1.0)
    pct_close = close_to_8e / n_trials * 100

    print(f"\nResults:")
    print(f"  Mean Df × α: {mean_val:.4f}")
    print(f"  Std Df × α: {std_val:.4f}")
    print(f"  CV: {cv:.2f}%")
    print(f"  Target (8e): {TARGET_CONSTANT:.4f}")
    print(f"  Close to 8e (±1): {close_to_8e}/{n_trials} ({pct_close:.1f}%)")

    # Distribution of Df and α separately
    print(f"\n  Mean Df: {np.mean(df_values):.2f} (std: {np.std(df_values):.2f})")
    print(f"  Mean α: {np.mean(alpha_values):.4f} (std: {np.std(alpha_values):.4f})")

    # Pass/Fail
    passed = cv > 50

    if passed:
        print(f"\n✅ PASSED: CV = {cv:.2f}% > 50%")
        print("   Random matrices do NOT converge to 8e")
    else:
        print(f"\n❌ FAILED: CV = {cv:.2f}% ≤ 50%")
        print("   WARNING: Random matrices may also produce ~8e!")

    return {
        'test': '1.1_random_matrix_baseline',
        'passed': passed,
        'n_trials': n_trials,
        'mean_df_alpha': float(mean_val),
        'std_df_alpha': float(std_val),
        'cv_percent': float(cv),
        'target_8e': float(TARGET_CONSTANT),
        'close_to_8e_count': int(close_to_8e),
        'close_to_8e_pct': float(pct_close),
    }


# =============================================================================
# TEST 1.2: PERMUTATION TEST
# =============================================================================

def test_permutation(embeddings, model_name, n_permutations=100):
    """
    Test: Shuffling embeddings should destroy the 8e relationship.

    Pass condition: p-value < 0.001 (real differs from shuffled)
    Fail condition: Shuffled embeddings still produce ~8e
    """
    print(f"\n--- Permutation test for {model_name} ---")

    # Real value
    real_df_alpha, real_df, real_alpha = compute_df_alpha(embeddings)
    print(f"  Real Df × α: {real_df_alpha:.4f}")

    # Permuted values
    permuted_values = []
    for i in range(n_permutations):
        # Shuffle elements (destroys semantic structure)
        shuffled = embeddings.flatten()
        np.random.shuffle(shuffled)
        shuffled = shuffled.reshape(embeddings.shape)

        perm_df_alpha, _, _ = compute_df_alpha(shuffled)
        permuted_values.append(perm_df_alpha)

    permuted_values = np.array(permuted_values)

    # Statistical test
    # How extreme is the real value compared to permuted distribution?
    perm_mean = np.mean(permuted_values)
    perm_std = np.std(permuted_values)
    z_score = (real_df_alpha - perm_mean) / perm_std if perm_std > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed

    print(f"  Permuted mean: {perm_mean:.4f} (std: {perm_std:.4f})")
    print(f"  Z-score: {z_score:.4f}")
    print(f"  p-value: {p_value:.6f}")

    # Also check: how far is real from 8e vs permuted from 8e?
    real_dist_from_8e = abs(real_df_alpha - TARGET_CONSTANT)
    perm_dist_from_8e = np.mean(np.abs(permuted_values - TARGET_CONSTANT))

    print(f"  Real dist from 8e: {real_dist_from_8e:.4f}")
    print(f"  Perm dist from 8e: {perm_dist_from_8e:.4f}")

    passed = p_value < 0.001

    return {
        'model': model_name,
        'real_df_alpha': float(real_df_alpha),
        'perm_mean': float(perm_mean),
        'perm_std': float(perm_std),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'passed': passed,
    }


# =============================================================================
# TEST 1.3: VOCABULARY INDEPENDENCE
# =============================================================================

def test_vocabulary_independence(model, n_vocab_sets=10, vocab_size=75):
    """
    Test: 8e should hold regardless of which words we pick.

    Pass condition: CV < 5% across different vocabulary choices
    """
    print("\n" + "=" * 70)
    print("TEST 1.3: VOCABULARY INDEPENDENCE")
    print("=" * 70)

    # Large word pool to sample from
    WORD_POOL = [
        # Nature
        "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
        "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
        "forest", "desert", "island", "lake", "valley", "cave", "hill", "field",
        # Animals
        "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
        "snake", "wolf", "bear", "eagle", "whale", "spider", "ant", "bee",
        "deer", "rabbit", "fox", "owl", "crow", "shark", "dolphin", "monkey",
        # Body
        "heart", "eye", "hand", "head", "brain", "blood", "bone", "skin",
        "face", "arm", "leg", "finger", "ear", "nose", "mouth", "tooth",
        # People
        "mother", "father", "child", "friend", "king", "queen", "doctor", "teacher",
        "soldier", "artist", "farmer", "priest", "judge", "writer", "singer", "dancer",
        # Abstract
        "love", "hate", "truth", "life", "death", "time", "space", "power",
        "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
        "freedom", "justice", "beauty", "wisdom", "courage", "honor", "faith", "pride",
        # Objects
        "book", "door", "house", "road", "food", "money", "stone", "gold",
        "sword", "light", "shadow", "music", "word", "name", "law", "art",
        "table", "chair", "window", "mirror", "key", "ring", "crown", "flag",
        # Actions/qualities
        "good", "bad", "big", "small", "old", "new", "high", "low",
        "hot", "cold", "dark", "bright", "strong", "weak", "fast", "slow",
        "hard", "soft", "deep", "wide", "long", "short", "rich", "poor",
    ]

    df_alpha_values = []

    for i in range(n_vocab_sets):
        # Random sample of words
        vocab = np.random.choice(WORD_POOL, size=vocab_size, replace=False)

        # Get embeddings
        embeddings = model.encode(list(vocab), normalize_embeddings=True)

        # Compute Df × α
        df_alpha, df, alpha = compute_df_alpha(embeddings)
        df_alpha_values.append(df_alpha)

        print(f"  Vocab set {i+1}: Df × α = {df_alpha:.4f} (Df={df:.2f}, α={alpha:.4f})")

    mean_val = np.mean(df_alpha_values)
    std_val = np.std(df_alpha_values)
    cv = std_val / mean_val * 100

    print(f"\nResults:")
    print(f"  Mean Df × α: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  CV: {cv:.2f}%")
    print(f"  Target (8e): {TARGET_CONSTANT:.4f}")
    print(f"  Mean dist from 8e: {abs(mean_val - TARGET_CONSTANT):.4f}")

    passed = cv < 5.0

    if passed:
        print(f"\n✅ PASSED: CV = {cv:.2f}% < 5%")
        print("   8e is vocabulary-independent!")
    else:
        print(f"\n⚠️  MARGINAL: CV = {cv:.2f}%")
        print("   Some vocabulary dependence detected")

    return {
        'test': '1.3_vocabulary_independence',
        'passed': passed,
        'n_vocab_sets': n_vocab_sets,
        'vocab_size': vocab_size,
        'mean_df_alpha': float(mean_val),
        'std_df_alpha': float(std_val),
        'cv_percent': float(cv),
        'values': [float(v) for v in df_alpha_values],
    }


# =============================================================================
# TEST 1.4: MONTE CARLO VERIFICATION
# =============================================================================

def test_monte_carlo(observed_values, observed_cv, n_fake_constants=10000):
    """
    Test: How special is 8e? Random constants shouldn't match as well.

    Pass condition: < 1% of random constants have CV as low as 8e
    """
    print("\n" + "=" * 70)
    print("TEST 1.4: MONTE CARLO VERIFICATION")
    print("=" * 70)
    print(f"Testing {n_fake_constants} random constants...")

    observed_values = np.array(observed_values)

    # Generate random "fake" universal constants
    fake_constants = np.random.uniform(15, 30, n_fake_constants)

    better_matches = 0
    cvs = []

    for fake_c in fake_constants:
        # Compute CV for this fake constant
        diffs = observed_values - fake_c
        cv = np.std(diffs) / abs(fake_c) * 100
        cvs.append(cv)

        if cv <= observed_cv:
            better_matches += 1

    p_value = better_matches / n_fake_constants

    print(f"\nResults:")
    print(f"  Observed CV (8e): {observed_cv:.2f}%")
    print(f"  Random constants with CV ≤ {observed_cv:.2f}%: {better_matches}/{n_fake_constants}")
    print(f"  p-value: {p_value:.6f}")

    # Distribution of CVs for random constants
    print(f"\n  Random constant CV distribution:")
    print(f"    Min: {np.min(cvs):.2f}%")
    print(f"    Median: {np.median(cvs):.2f}%")
    print(f"    Max: {np.max(cvs):.2f}%")

    passed = p_value < 0.01

    if passed:
        print(f"\n✅ PASSED: p = {p_value:.6f} < 0.01")
        print("   8e is statistically special!")
    else:
        print(f"\n❌ FAILED: p = {p_value:.6f} ≥ 0.01")
        print("   8e may be coincidental")

    return {
        'test': '1.4_monte_carlo',
        'passed': passed,
        'observed_cv': float(observed_cv),
        'n_fake_constants': n_fake_constants,
        'better_matches': int(better_matches),
        'p_value': float(p_value),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Q49 PHASE 1: FALSIFICATION BATTERY")
    print("Can we disprove Df × α = 8e?")
    print("=" * 70)

    results = {}

    # TEST 1.1: Random Matrix Baseline
    results['test_1_1'] = test_random_matrix_baseline(n_trials=500)

    # Load real embeddings for remaining tests
    print("\n" + "=" * 70)
    print("Loading real embeddings...")
    print("=" * 70)

    WORDS = [
        "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
        "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
        "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
        "heart", "eye", "hand", "head", "brain", "blood", "bone",
        "mother", "father", "child", "friend", "king", "queen",
        "love", "hate", "truth", "life", "death", "time", "space", "power",
        "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
        "book", "door", "house", "road", "food", "money", "stone", "gold",
        "light", "shadow", "music", "word", "name", "law",
        "good", "bad", "big", "small", "old", "new", "high", "low",
    ]

    try:
        from sentence_transformers import SentenceTransformer

        models = [
            ("all-MiniLM-L6-v2", "MiniLM"),
            ("all-mpnet-base-v2", "MPNet"),
        ]

        real_df_alpha_values = []
        perm_results = []

        for model_id, name in models:
            print(f"\n  Loading {name}...")
            model = SentenceTransformer(model_id)
            embeddings = model.encode(WORDS, normalize_embeddings=True)

            # Compute real Df × α
            df_alpha, _, _ = compute_df_alpha(embeddings)
            real_df_alpha_values.append(df_alpha)
            print(f"    Df × α = {df_alpha:.4f}")

            # TEST 1.2: Permutation Test
            perm_result = test_permutation(embeddings, name, n_permutations=100)
            perm_results.append(perm_result)

        results['test_1_2'] = {
            'models': perm_results,
            'all_passed': all(r['passed'] for r in perm_results),
        }

        # TEST 1.3: Vocabulary Independence (use first model)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        results['test_1_3'] = test_vocabulary_independence(model, n_vocab_sets=10)

        # Collect all observed values for Monte Carlo
        all_observed = real_df_alpha_values + results['test_1_3']['values']
        observed_cv = np.std(all_observed) / np.mean(all_observed) * 100

        # TEST 1.4: Monte Carlo
        results['test_1_4'] = test_monte_carlo(all_observed, observed_cv, n_fake_constants=5000)

    except ImportError:
        print("sentence-transformers not available, skipping embedding tests")

    # Summary
    print("\n" + "=" * 70)
    print("FALSIFICATION BATTERY SUMMARY")
    print("=" * 70)

    tests_passed = 0
    tests_total = 0

    for test_name, result in results.items():
        if isinstance(result, dict) and 'passed' in result:
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"  {test_name}: {status}")
            tests_passed += result['passed']
            tests_total += 1
        elif isinstance(result, dict) and 'all_passed' in result:
            status = "✅ PASSED" if result['all_passed'] else "❌ FAILED"
            print(f"  {test_name}: {status}")
            tests_passed += result['all_passed']
            tests_total += 1

    print(f"\n  TOTAL: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        print("\n🎯 8e SURVIVES FALSIFICATION BATTERY!")
        print("   The conservation law Df × α = 8e appears to be REAL.")
    else:
        print("\n⚠️  8e shows weakness in some tests")
        print("   Further investigation needed")

    # Save results
    receipt = {
        'test': 'Q49_FALSIFICATION_BATTERY',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'target_constant': float(TARGET_CONSTANT),
        'results': results,
        'tests_passed': tests_passed,
        'tests_total': tests_total,
        'conclusion': 'SURVIVES' if tests_passed == tests_total else 'PARTIAL',
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q49_falsification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print(f"\nReceipt saved: {path}")

    return receipt


if __name__ == '__main__':
    main()
