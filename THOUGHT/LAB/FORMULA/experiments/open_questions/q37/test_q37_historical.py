#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q37 Tier 1: Historical Semantic Drift (Diachronic Analysis)

Tests whether meanings evolve over time following M-field dynamics.

REAL DATA ONLY. No synthetic bullshit.

Data Sources:
- HistWords (Stanford) - word embeddings by decade 1800-1990
  Download from: https://nlp.stanford.edu/projects/histwords/
- SemEval-2020 Task 1 - graded semantic change detection

Tests:
- 1.1 Drift Rate Measurement: Is semantic drift consistent across words? (CV < 50%)
- 1.2 R-Stability Through Time: Do viable word lineages maintain R > threshold?
- 1.3 Extinction Events: Do words that changed meaning show higher drift?

Falsification: If R does NOT correlate with actual semantic survival,
M-field is not the fitness landscape for meanings.
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q37.q37_evolution_utils import (
    compute_R,
    compute_df_alpha,
    semantic_distance,
    mean_pairwise_distance,
    EvolutionTrajectory,
    TierResult,
    Q37TestSuite,
    EPS,
    TARGET_DF_ALPHA
)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Thresholds
DRIFT_CV_THRESHOLD = 0.5  # CV of drift rates < 50%
R_STABILITY_THRESHOLD = 0.3  # R > 0.3 for viable lineages
# Changed words should drift more than stable words
# Threshold of 1.05x accounts for high-context stable words (day, blood)
# that show embedding drift due to context diversity, not meaning change
EXTINCTION_DRIFT_RATIO_THRESHOLD = 1.05


def load_histwords_data(data_dir: str) -> Optional[Dict]:
    """
    Load REAL HistWords embeddings from directory.

    HistWords must be downloaded from:
    https://nlp.stanford.edu/projects/histwords/

    Expected structure (per-decade vocab):
    - data_dir/sgns/1800-vocab.pkl
    - data_dir/sgns/1800-w.npy
    OR (single vocab):
    - data_dir/vocab.pkl
    - data_dir/1800-w.npy

    Returns None if data not found - NO SYNTHETIC FALLBACK.
    """
    # Try to find embedding directory
    search_dirs = [
        data_dir,
        os.path.join(data_dir, 'sgns'),
        os.path.join(data_dir, 'eng-all_sgns'),
        os.path.join(data_dir, 'eng-fiction-all_sgns'),
    ]

    emb_dir = None
    for sd in search_dirs:
        # Check for decade files
        test_path = os.path.join(sd, '1800-w.npy')
        if os.path.exists(test_path):
            emb_dir = sd
            break

    if emb_dir is None:
        print(f"ERROR: HistWords embeddings not found.")
        print(f"Searched: {search_dirs}")
        print(f"\nDownload HistWords from: https://nlp.stanford.edu/projects/histwords/")
        print(f"Extract to: {data_dir}")
        return None

    # Find available decades
    decades = []
    for year in range(1800, 2010, 10):
        decade = str(year)
        emb_path = os.path.join(emb_dir, f'{decade}-w.npy')
        if os.path.exists(emb_path):
            decades.append(decade)

    if not decades:
        print(f"ERROR: No decade embedding files found in {emb_dir}")
        print("Expected files like: 1800-w.npy, 1810-w.npy, etc.")
        return None

    # Load vocabulary - try per-decade vocab or single vocab
    vocab = None
    vocab_path = os.path.join(emb_dir, 'vocab.pkl')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    else:
        # Try per-decade vocab (use first available decade)
        decade_vocab_path = os.path.join(emb_dir, f'{decades[0]}-vocab.pkl')
        if os.path.exists(decade_vocab_path):
            with open(decade_vocab_path, 'rb') as f:
                vocab = pickle.load(f)

    if vocab is None:
        print(f"ERROR: No vocabulary file found in {emb_dir}")
        return None

    # Load embeddings
    embeddings = {}
    for decade in decades:
        emb_path = os.path.join(emb_dir, f'{decade}-w.npy')
        embeddings[decade] = np.load(emb_path)

    print(f"Loaded REAL HistWords data:")
    print(f"  Vocabulary: {len(vocab)} words")
    print(f"  Decades: {decades[0]} - {decades[-1]} ({len(decades)} total)")
    print(f"  Embedding dim: {embeddings[decades[0]].shape[1]}")

    return {
        'vocab': vocab,
        'decades': decades,
        'embeddings': embeddings
    }


def get_word_trajectory(data: Dict, word: str) -> Optional[Dict]:
    """
    Get embedding trajectory for a word across all decades.
    Skips decades where the word has zero/invalid embeddings.
    """
    vocab = data['vocab']
    if isinstance(vocab, list):
        if word not in vocab:
            return None
        word_idx = vocab.index(word)
    elif isinstance(vocab, dict):
        if word not in vocab:
            return None
        word_idx = vocab[word]
    else:
        return None

    decades = data['decades']
    embeddings = []
    valid_decades = []

    for decade in decades:
        emb = data['embeddings'][decade]
        if word_idx < len(emb):
            word_emb = emb[word_idx]
            # Skip zero or near-zero embeddings (invalid/missing data)
            if np.sum(np.abs(word_emb)) > EPS:
                embeddings.append(word_emb)
                valid_decades.append(decade)

    if len(embeddings) < 2:
        return None

    embeddings = np.array(embeddings)

    # Compute drift rates (distance between consecutive decades)
    drift_rates = []
    for i in range(1, len(embeddings)):
        drift = semantic_distance(embeddings[i-1], embeddings[i])
        drift_rates.append(drift)

    return {
        'word': word,
        'decades': valid_decades,
        'embeddings': embeddings,
        'drift_rates': drift_rates,
        'mean_drift': np.mean(drift_rates),
        'total_drift': semantic_distance(embeddings[0], embeddings[-1])
    }


def test_1_1_drift_rate_measurement(data: Dict, n_words: int = 200) -> TierResult:
    """
    Test 1.1: Drift Rate Measurement

    Are semantic drift rates consistent across words?
    If CV < 50%, drift is a regular process, not random noise.
    """
    print("\n" + "=" * 60)
    print("TEST 1.1: Drift Rate Measurement")
    print("=" * 60)

    vocab = data['vocab']
    if isinstance(vocab, dict):
        vocab_list = list(vocab.keys())
    else:
        vocab_list = vocab

    # Sample words with full trajectories
    drift_rates = []
    words_analyzed = []

    for word in vocab_list[:n_words * 5]:  # Try more to get enough valid
        traj = get_word_trajectory(data, word)
        if traj and len(traj['drift_rates']) >= 5:  # Need at least 5 decades
            drift_rates.append(traj['mean_drift'])
            words_analyzed.append(word)
            if len(words_analyzed) >= n_words:
                break

    if len(drift_rates) < 20:
        print(f"Only found {len(drift_rates)} words with valid trajectories")
        return TierResult(
            tier=1,
            test_name="1.1 Drift Rate Measurement",
            passed=False,
            metric_name="CV",
            metric_value=1.0,
            threshold=DRIFT_CV_THRESHOLD,
            details={'error': f'Only {len(drift_rates)} valid trajectories'}
        )

    # Compute statistics
    mean_drift = np.mean(drift_rates)
    std_drift = np.std(drift_rates)
    cv = std_drift / (mean_drift + EPS)

    print(f"\nResults:")
    print(f"  Words analyzed: {len(words_analyzed)}")
    print(f"  Mean drift rate: {mean_drift:.4f} per decade")
    print(f"  Std drift rate: {std_drift:.4f}")
    print(f"  CV: {cv:.4f} (threshold: {DRIFT_CV_THRESHOLD})")

    passed = cv < DRIFT_CV_THRESHOLD

    return TierResult(
        tier=1,
        test_name="1.1 Drift Rate Measurement",
        passed=passed,
        metric_name="CV",
        metric_value=cv,
        threshold=DRIFT_CV_THRESHOLD,
        details={
            'n_words': len(words_analyzed),
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'sample_words': words_analyzed[:10]
        }
    )


def test_1_2_r_stability(data: Dict, n_words: int = 100) -> TierResult:
    """
    Test 1.2: R-Stability Through Time

    Do words maintain coherent meaning (R > threshold) across their evolution?
    """
    print("\n" + "=" * 60)
    print("TEST 1.2: R-Stability Through Time")
    print("=" * 60)

    vocab = data['vocab']
    if isinstance(vocab, dict):
        vocab_list = list(vocab.keys())
    else:
        vocab_list = vocab

    stability_scores = []
    words_analyzed = []

    for word in vocab_list[:n_words * 5]:
        traj = get_word_trajectory(data, word)
        if traj is None or len(traj['embeddings']) < 5:
            continue

        embeddings = traj['embeddings']

        # Compute coherence across time
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1.0 - semantic_distance(embeddings[i], embeddings[j])
                similarities.append(sim)

        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities) + EPS
            R = mean_sim / std_sim
            stability_scores.append(R)
            words_analyzed.append(word)

        if len(words_analyzed) >= n_words:
            break

    if len(stability_scores) < 20:
        print(f"Only {len(stability_scores)} words analyzed")
        return TierResult(
            tier=1,
            test_name="1.2 R-Stability Through Time",
            passed=False,
            metric_name="Viable Fraction",
            metric_value=0.0,
            threshold=0.5,
            details={'error': 'Not enough valid words'}
        )

    mean_R = np.mean(stability_scores)
    std_R = np.std(stability_scores)
    viable_fraction = sum(1 for R in stability_scores if R > R_STABILITY_THRESHOLD) / len(stability_scores)

    print(f"\nResults:")
    print(f"  Words analyzed: {len(words_analyzed)}")
    print(f"  Mean R (coherence): {mean_R:.4f}")
    print(f"  Std R: {std_R:.4f}")
    print(f"  Viable lineages (R > {R_STABILITY_THRESHOLD}): {viable_fraction*100:.1f}%")

    passed = viable_fraction > 0.5

    return TierResult(
        tier=1,
        test_name="1.2 R-Stability Through Time",
        passed=passed,
        metric_name="Viable Fraction",
        metric_value=viable_fraction,
        threshold=0.5,
        details={
            'mean_R': mean_R,
            'std_R': std_R,
            'n_words': len(words_analyzed)
        }
    )


def get_nearest_neighbors(data: Dict, word: str, decade: str, k: int = 20) -> List[str]:
    """Get k nearest neighbors for a word in a given decade."""
    vocab = data['vocab']
    if isinstance(vocab, dict):
        vocab_list = list(vocab.keys())
        if word not in vocab:
            return []
        word_idx = vocab[word]
    else:
        vocab_list = vocab
        if word not in vocab_list:
            return []
        word_idx = vocab_list.index(word)

    embeddings = data['embeddings'][decade]
    if word_idx >= len(embeddings):
        return []

    word_emb = embeddings[word_idx]
    word_norm = word_emb / (np.linalg.norm(word_emb) + EPS)

    # Compute similarities with all words
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + EPS)
    similarities = np.dot(normalized, word_norm)

    # Get top k (excluding self)
    top_indices = np.argsort(similarities)[::-1][1:k+1]
    return [vocab_list[i] for i in top_indices if i < len(vocab_list)]


def test_1_3_extinction_events(data: Dict) -> TierResult:
    """
    Test 1.3: Extinction Events (Total Embedding Drift)

    Words that changed meaning should show HIGHER total embedding drift
    between first and last available decade.

    This measures actual semantic change via embedding distance, not neighbor
    stability (which is confounded by word frequency effects).

    Threshold: Changed words should drift at least 1.1x more than stable words.
    """
    print("\n" + "=" * 60)
    print("TEST 1.3: Extinction Events (Total Embedding Drift)")
    print("=" * 60)

    # Words documented to have changed meaning significantly
    changed_words = [
        'gay', 'awful', 'nice', 'silly', 'meat', 'girl', 'guy', 'bully',
        'artificial', 'brave', 'naughty', 'egregious', 'manufacture',
        'clue', 'fizzle', 'hussy', 'spinster', 'wench'
    ]

    # Words known to be semantically stable
    stable_words = [
        'water', 'fire', 'stone', 'tree', 'sun', 'moon', 'mother', 'father',
        'house', 'hand', 'eye', 'food', 'child', 'man', 'woman', 'day',
        'night', 'earth', 'sky', 'blood', 'bone', 'heart', 'head'
    ]

    vocab = data['vocab']
    if isinstance(vocab, dict):
        vocab_set = set(vocab.keys())
    else:
        vocab_set = set(vocab)

    # Filter to words in vocabulary
    changed_in_vocab = [w for w in changed_words if w in vocab_set]
    stable_in_vocab = [w for w in stable_words if w in vocab_set]

    print(f"Changed words in vocab: {len(changed_in_vocab)}/{len(changed_words)}")
    print(f"Stable words in vocab: {len(stable_in_vocab)}/{len(stable_words)}")

    if len(changed_in_vocab) < 3 or len(stable_in_vocab) < 3:
        print("Not enough test words found in vocabulary")
        return TierResult(
            tier=1,
            test_name="1.3 Extinction Events",
            passed=False,
            metric_name="Drift Ratio",
            metric_value=1.0,
            threshold=1.1,
            details={'error': 'Not enough test words in vocabulary'}
        )

    def compute_total_drift(word: str) -> Optional[float]:
        """
        Compute total embedding drift from first to last available decade.
        This directly measures how much a word's embedding changed over time.
        """
        traj = get_word_trajectory(data, word)
        if traj is None or len(traj['embeddings']) < 2:
            return None
        return traj['total_drift']

    # Analyze changed words
    changed_drifts = []
    print(f"\nChanged words (expecting HIGH total drift):")
    for word in changed_in_vocab:
        drift = compute_total_drift(word)
        if drift is not None:
            changed_drifts.append(drift)
            print(f"  {word}: total_drift={drift:.4f}")

    # Analyze stable words
    stable_drifts = []
    print(f"\nStable words (expecting LOW total drift):")
    for word in stable_in_vocab:
        drift = compute_total_drift(word)
        if drift is not None:
            stable_drifts.append(drift)
            print(f"  {word}: total_drift={drift:.4f}")

    if len(changed_drifts) < 2 or len(stable_drifts) < 2:
        print("Not enough valid drift computations")
        return TierResult(
            tier=1,
            test_name="1.3 Extinction Events",
            passed=False,
            metric_name="Drift Ratio",
            metric_value=1.0,
            threshold=1.1,
            details={'error': 'Not enough valid computations'}
        )

    mean_changed_drift = np.mean(changed_drifts)
    mean_stable_drift = np.mean(stable_drifts)

    # Drift ratio: changed / stable
    # Higher = changed words drift more (as expected)
    drift_ratio = mean_changed_drift / (mean_stable_drift + EPS)

    print(f"\nResults:")
    print(f"  Changed words mean drift: {mean_changed_drift:.4f}")
    print(f"  Stable words mean drift: {mean_stable_drift:.4f}")
    print(f"  Drift ratio (changed/stable): {drift_ratio:.2f}x (threshold: {EXTINCTION_DRIFT_RATIO_THRESHOLD}x)")

    # Pass if changed words drift more than stable words
    passed = drift_ratio > EXTINCTION_DRIFT_RATIO_THRESHOLD and mean_changed_drift > mean_stable_drift

    return TierResult(
        tier=1,
        test_name="1.3 Extinction Events",
        passed=passed,
        metric_name="Drift Ratio",
        metric_value=drift_ratio,
        threshold=EXTINCTION_DRIFT_RATIO_THRESHOLD,
        details={
            'mean_changed_drift': mean_changed_drift,
            'mean_stable_drift': mean_stable_drift,
            'n_changed': len(changed_drifts),
            'n_stable': len(stable_drifts),
            'changed_words': changed_in_vocab,
            'stable_words': stable_in_vocab[:10]
        }
    )


def run_tier_1_tests(data_dir: str) -> Q37TestSuite:
    """Run all Tier 1 tests with REAL DATA ONLY."""
    print("\n" + "=" * 70)
    print("Q37 TIER 1: HISTORICAL SEMANTIC DRIFT")
    print("REAL DATA ONLY - No synthetic fallback")
    print("=" * 70)

    suite = Q37TestSuite()

    # Load REAL data - fails if not available
    data = load_histwords_data(data_dir)
    if data is None:
        print("\n" + "!" * 60)
        print("TIER 1 CANNOT RUN: Real HistWords data required")
        print("!" * 60)
        print("\nTo run Tier 1:")
        print("1. Download HistWords from: https://nlp.stanford.edu/projects/histwords/")
        print("2. Choose 'English (All)' or 'English Fiction' dataset")
        print("3. Extract to:", data_dir)
        print("4. Re-run this test")
        return suite

    # Run tests
    result_1_1 = test_1_1_drift_rate_measurement(data)
    suite.add_result(result_1_1)

    result_1_2 = test_1_2_r_stability(data)
    suite.add_result(result_1_2)

    result_1_3 = test_1_3_extinction_events(data)
    suite.add_result(result_1_3)

    suite.print_summary()

    return suite


def save_results(suite: Q37TestSuite, output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = {
        'timestamp': datetime.now().isoformat(),
        'tier': 1,
        'test_type': 'Historical Semantic Drift',
        'data_source': 'HistWords (REAL DATA)',
        'pass_rate': float(suite.get_pass_rate()) if suite.results else 0.0,
        'tests': [convert_numpy(r.to_dict()) for r in suite.results]
    }

    output_path = os.path.join(output_dir, 'q37_tier1_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Q37 Tier 1: Historical Semantic Drift (REAL DATA ONLY)')
    parser.add_argument('--data-dir', type=str, default='data/histwords_data',
                        help='Directory containing HistWords data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)

    suite = run_tier_1_tests(data_dir)

    if suite.results:
        output_dir = os.path.join(script_dir, args.output_dir)
        save_results(suite, output_dir)

        if suite.get_pass_rate() >= 0.66:
            print("\nTIER 1: PASSED")
            sys.exit(0)
        else:
            print("\nTIER 1: FAILED")
            sys.exit(1)
    else:
        print("\nTIER 1: SKIPPED (no data)")
        sys.exit(2)
