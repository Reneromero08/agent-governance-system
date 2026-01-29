#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q48: Riemann-Spectral Bridge

Does the eigenvalue spectrum of semantic embeddings follow the same
universal statistics as Riemann zeta zeros?

Tests:
1. Marchenko-Pastur comparison (random matrix baseline)
2. GUE spacing statistics (quantum chaos / Riemann connection)
3. Cumulative variance shape analysis

If semantic spacings match GUE → meaning and primes follow same law.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.special import gamma as gamma_func

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# THEORETICAL DISTRIBUTIONS
# =============================================================================

def marchenko_pastur_density(lam, gamma):
    """
    Marchenko-Pastur distribution for eigenvalues of random covariance matrices.

    Parameters:
        lam: eigenvalue
        gamma: aspect ratio n/p (samples/features)

    Returns:
        density at lam
    """
    lambda_max = (1 + np.sqrt(gamma))**2
    lambda_min = (1 - np.sqrt(gamma))**2

    if gamma > 1:
        lambda_min = 0

    if lam < lambda_min or lam > lambda_max:
        return 0.0

    numerator = np.sqrt((lambda_max - lam) * (lam - lambda_min))
    denominator = 2 * np.pi * gamma * lam

    return numerator / denominator


def gue_spacing_pdf(s):
    """
    GUE (Wigner surmise) for nearest-neighbor spacing.

    This is what Riemann zeros follow (Montgomery-Odlyzko).

    p(s) = (32/π²) s² exp(-4s²/π)
    """
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def poisson_spacing_pdf(s):
    """
    Poisson distribution for uncorrelated spacings.

    p(s) = exp(-s)

    If spacings match this → no structure, random.
    """
    return np.exp(-s)


def goe_spacing_pdf(s):
    """
    GOE (Gaussian Orthogonal Ensemble) spacing.

    p(s) = (π/2) s exp(-πs²/4)

    Alternative universality class (time-reversal symmetric).
    """
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


# =============================================================================
# EIGENVALUE ANALYSIS
# =============================================================================

def get_eigenspectrum(embeddings_dict):
    """Get eigenspectrum from embedding dictionary."""
    words = sorted(embeddings_dict.keys())
    vecs = np.array([embeddings_dict[w] for w in words])
    vecs_centered = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
    return eigenvalues


def compute_spacing_statistics(eigenvalues):
    """
    Compute nearest-neighbor spacing statistics.

    This is the key test for Riemann connection.
    """
    # Sort ascending for spacing calculation
    sorted_ev = np.sort(eigenvalues)

    # Remove zeros/tiny values
    sorted_ev = sorted_ev[sorted_ev > 1e-8]

    # Compute spacings
    spacings = np.diff(sorted_ev)

    # Normalize by mean spacing (unfolding)
    mean_spacing = np.mean(spacings)
    if mean_spacing > 0:
        normalized_spacings = spacings / mean_spacing
    else:
        normalized_spacings = spacings

    return normalized_spacings


def fit_spacing_distribution(spacings, distribution='gue'):
    """
    Compute KL divergence between empirical spacings and theoretical distribution.

    Lower KL = better match.
    """
    if len(spacings) < 10:
        return float('inf')

    # Create histogram of empirical spacings
    bins = np.linspace(0, 4, 41)  # 0 to 4 in 40 bins
    hist, bin_edges = np.histogram(spacings, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Get theoretical distribution
    if distribution == 'gue':
        theoretical = np.array([gue_spacing_pdf(s) for s in bin_centers])
    elif distribution == 'goe':
        theoretical = np.array([goe_spacing_pdf(s) for s in bin_centers])
    elif distribution == 'poisson':
        theoretical = np.array([poisson_spacing_pdf(s) for s in bin_centers])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Normalize
    theoretical = theoretical / np.sum(theoretical)
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist = hist + eps
    theoretical = theoretical + eps

    # KL divergence
    kl = stats.entropy(hist, theoretical)

    return kl


def compute_level_repulsion(spacings):
    """
    Measure level repulsion strength.

    GUE shows strong level repulsion (small spacings are rare).
    Poisson shows no repulsion.

    Returns ratio: P(s < 0.5) / P(s > 0.5)
    Lower ratio = stronger repulsion = more GUE-like.
    """
    small = np.sum(spacings < 0.5)
    large = np.sum(spacings >= 0.5)

    if large == 0:
        return float('inf')

    return small / large


# =============================================================================
# EMBEDDING LOADERS
# =============================================================================

def load_embeddings():
    """Load embeddings from available models."""
    embeddings = {}

    # Standard test vocabulary
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

    # Try sentence-transformers (most likely available)
    try:
        from sentence_transformers import SentenceTransformer

        models = [
            ("all-MiniLM-L6-v2", "MiniLM"),
            ("all-mpnet-base-v2", "MPNet"),
        ]

        for model_id, name in models:
            try:
                model = SentenceTransformer(model_id)
                embs = model.encode(WORDS, normalize_embeddings=True)
                embeddings[name] = {word: embs[i] for i, word in enumerate(WORDS)}
                print(f"  Loaded {name}")
            except Exception as e:
                print(f"  Failed {name}: {e}")

    except ImportError:
        print("  sentence-transformers not available")

    # Try gensim
    try:
        import gensim.downloader as api

        models = [
            ("glove-wiki-gigaword-100", "GloVe-100"),
        ]

        for model_id, name in models:
            try:
                model = api.load(model_id)
                emb_dict = {w: model[w] for w in WORDS if w in model}
                if len(emb_dict) >= 50:
                    embeddings[name] = emb_dict
                    print(f"  Loaded {name} ({len(emb_dict)} words)")
            except Exception as e:
                print(f"  Failed {name}: {e}")

    except ImportError:
        print("  gensim not available")

    return embeddings


# =============================================================================
# MAIN TEST
# =============================================================================

def run_q48_tests():
    """Run all Q48 Riemann-Spectral Bridge tests."""

    print("=" * 70)
    print("Q48: RIEMANN-SPECTRAL BRIDGE")
    print("Testing if semantic eigenvalues follow GUE statistics (Riemann connection)")
    print("=" * 70)
    print()

    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings()

    if not embeddings:
        print("\nNo embeddings available. Install sentence-transformers:")
        print("  pip install sentence-transformers")
        return None

    print()

    results = {}

    for model_name, emb_dict in embeddings.items():
        print(f"\n{'='*50}")
        print(f"MODEL: {model_name}")
        print(f"{'='*50}")

        # Get eigenspectrum
        eigenvalues = get_eigenspectrum(emb_dict)
        n_eigenvalues = len(eigenvalues)

        print(f"\nEigenvalues: {n_eigenvalues}")
        print(f"  Top 5: {eigenvalues[:5]}")
        print(f"  Df (participation ratio): {(np.sum(eigenvalues)**2) / np.sum(eigenvalues**2):.1f}")

        # Compute spacings
        spacings = compute_spacing_statistics(eigenvalues)

        print(f"\nSpacing statistics:")
        print(f"  Mean spacing: {np.mean(spacings):.4f}")
        print(f"  Std spacing: {np.std(spacings):.4f}")

        # Test against distributions
        print(f"\nKL divergence (lower = better match):")

        kl_gue = fit_spacing_distribution(spacings, 'gue')
        kl_goe = fit_spacing_distribution(spacings, 'goe')
        kl_poisson = fit_spacing_distribution(spacings, 'poisson')

        print(f"  vs GUE (Riemann):  {kl_gue:.4f}")
        print(f"  vs GOE:            {kl_goe:.4f}")
        print(f"  vs Poisson (random): {kl_poisson:.4f}")

        # Level repulsion
        repulsion = compute_level_repulsion(spacings)
        print(f"\nLevel repulsion ratio: {repulsion:.4f}")
        print(f"  (GUE ≈ 0.3, Poisson ≈ 1.0)")

        # Determine best match
        kl_scores = {'GUE': kl_gue, 'GOE': kl_goe, 'Poisson': kl_poisson}
        best_match = min(kl_scores, key=kl_scores.get)

        print(f"\nBEST MATCH: {best_match}")

        if best_match == 'GUE':
            print("  → Semantic structure follows RIEMANN STATISTICS")
            print("  → Meaning and primes may share universal law")
        elif best_match == 'GOE':
            print("  → Semantic structure follows GOE (time-reversal symmetric)")
            print("  → Different universality class from Riemann")
        else:
            print("  → Semantic structure appears RANDOM")
            print("  → No Riemann connection detected")

        results[model_name] = {
            'n_eigenvalues': n_eigenvalues,
            'df': float((np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)),
            'kl_gue': float(kl_gue),
            'kl_goe': float(kl_goe),
            'kl_poisson': float(kl_poisson),
            'level_repulsion': float(repulsion),
            'best_match': best_match,
            'eigenvalues_top10': eigenvalues[:10].tolist(),
            'spacings_sample': spacings[:20].tolist() if len(spacings) >= 20 else spacings.tolist(),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    gue_matches = sum(1 for r in results.values() if r['best_match'] == 'GUE')
    total = len(results)

    print(f"\nModels matching GUE: {gue_matches}/{total}")

    if gue_matches == total:
        print("\n*** ALL MODELS MATCH GUE ***")
        print("Strong evidence for Riemann-Semantic connection!")
    elif gue_matches > 0:
        print(f"\n* Partial GUE match ({gue_matches}/{total})")
        print("Mixed evidence - investigate further")
    else:
        print("\n✗ No GUE matches")
        print("Semantic spacings do not follow Riemann statistics")

    # Save receipt
    receipt = {
        'test': 'Q48_RIEMANN_BRIDGE',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'models_tested': list(results.keys()),
        'results': results,
        'gue_match_count': gue_matches,
        'conclusion': 'GUE_MATCH' if gue_matches == total else 'PARTIAL' if gue_matches > 0 else 'NO_MATCH'
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    receipt_path = results_dir / f'q48_riemann_bridge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"\nReceipt saved: {receipt_path}")

    return receipt


if __name__ == '__main__':
    run_q48_tests()
