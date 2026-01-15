#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q48 Part 4: The Universal Constant Df × α

We found Df × α ≈ 22 with CV < 1% across models.

Is this:
- 7π ≈ 21.99?
- Something else?

Test across MORE models to validate universality.
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


def get_eigenspectrum(embeddings_dict):
    """Get eigenspectrum from embedding dictionary."""
    words = sorted(embeddings_dict.keys())
    vecs = np.array([embeddings_dict[w] for w in words])
    vecs_centered = vecs - vecs.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvalues


def compute_df_alpha(eigenvalues):
    """Compute Df × α for eigenvalue spectrum."""
    ev = eigenvalues[eigenvalues > 1e-10]

    # Participation ratio (Df)
    Df = (np.sum(ev) ** 2) / np.sum(ev ** 2)

    # Power law exponent (α) - fit to first half of spectrum
    k = np.arange(1, len(ev) + 1)
    log_k = np.log(k[:len(ev)//2])
    log_ev = np.log(ev[:len(ev)//2])

    if len(log_k) > 5:
        slope, _ = np.polyfit(log_k, log_ev, 1)
        alpha = -slope
    else:
        alpha = 0

    return {
        'Df': float(Df),
        'alpha': float(alpha),
        'Df_times_alpha': float(Df * alpha),
        'n_eigenvalues': len(ev),
    }


def main():
    print("=" * 70)
    print("Q48 PART 4: TESTING THE UNIVERSAL CONSTANT Df × α")
    print("=" * 70)

    # Mathematical constants to compare
    print("\nMathematical constants for comparison:")
    print(f"  7π      = {7 * np.pi:.6f}")
    print(f"  22      = 22.000000")
    print(f"  e³      = {np.e**3:.6f}")
    print(f"  8e      = {8 * np.e:.6f}")
    print(f"  π²×2    = {np.pi**2 * 2:.6f}")
    print(f"  √(2π)×8 = {np.sqrt(2*np.pi) * 8:.6f}")

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

    results = {}

    print("\nLoading models...")

    # Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer

        models = [
            ("all-MiniLM-L6-v2", "MiniLM"),
            ("all-mpnet-base-v2", "MPNet"),
            ("paraphrase-MiniLM-L6-v2", "ParaMiniLM"),
            ("all-distilroberta-v1", "DistilRoBERTa"),
        ]

        for model_id, name in models:
            try:
                model = SentenceTransformer(model_id)
                embs = model.encode(WORDS, normalize_embeddings=True)
                emb_dict = {word: embs[i] for i, word in enumerate(WORDS)}
                ev = get_eigenspectrum(emb_dict)
                results[name] = compute_df_alpha(ev)
                print(f"  {name}: Df×α = {results[name]['Df_times_alpha']:.4f}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

    except ImportError:
        print("  sentence-transformers not available")

    # Gensim models
    try:
        import gensim.downloader as api

        gensim_models = [
            ("glove-wiki-gigaword-100", "GloVe-100"),
            ("glove-wiki-gigaword-300", "GloVe-300"),
        ]

        for model_id, name in gensim_models:
            try:
                model = api.load(model_id)
                emb_dict = {w: model[w] for w in WORDS if w in model}
                if len(emb_dict) >= 50:
                    ev = get_eigenspectrum(emb_dict)
                    results[name] = compute_df_alpha(ev)
                    print(f"  {name}: Df×α = {results[name]['Df_times_alpha']:.4f}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

    except ImportError:
        print("  gensim not available")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if results:
        df_alpha_values = [r['Df_times_alpha'] for r in results.values()]

        print(f"\n{'Model':<20} {'Df':>10} {'α':>10} {'Df×α':>12}")
        print("-" * 55)
        for name, r in results.items():
            print(f"{name:<20} {r['Df']:>10.2f} {r['alpha']:>10.4f} {r['Df_times_alpha']:>12.4f}")

        mean_val = np.mean(df_alpha_values)
        std_val = np.std(df_alpha_values)
        cv = std_val / mean_val

        print("-" * 55)
        print(f"{'Mean':<20} {'':<10} {'':<10} {mean_val:>12.4f}")
        print(f"{'Std':<20} {'':<10} {'':<10} {std_val:>12.4f}")
        print(f"{'CV':<20} {'':<10} {'':<10} {cv*100:>11.2f}%")

        print("\n" + "=" * 70)
        print("COMPARISON TO CONSTANTS")
        print("=" * 70)

        constants = [
            ("7π", 7 * np.pi),
            ("22", 22.0),
            ("e³", np.e**3),
            ("8e", 8 * np.e),
            ("π²×2", np.pi**2 * 2),
        ]

        print(f"\nMeasured mean: {mean_val:.4f}")
        print(f"\nDifference from known constants:")
        for name, val in constants:
            diff = abs(mean_val - val)
            pct = diff / mean_val * 100
            print(f"  {name:<10} = {val:.4f}  →  diff = {diff:.4f} ({pct:.2f}%)")

        # Best match
        best_const = min(constants, key=lambda x: abs(mean_val - x[1]))
        print(f"\nBest match: {best_const[0]} = {best_const[1]:.4f}")

        if abs(mean_val - 7 * np.pi) / mean_val < 0.05:
            print("\n*** Df × α ≈ 7π ***")
            print("This would connect semantic spectra to circle geometry!")

        # Check if Df × α / n_dims is constant (size-independent)
        print("\n" + "=" * 70)
        print("SIZE-INDEPENDENCE CHECK")
        print("=" * 70)

        print(f"\n{'Model':<20} {'n_dims':>10} {'Df×α':>12} {'(Df×α)/n':>12}")
        print("-" * 55)
        normalized_values = []
        for name, r in results.items():
            norm = r['Df_times_alpha'] / r['n_eigenvalues']
            normalized_values.append(norm)
            print(f"{name:<20} {r['n_eigenvalues']:>10} {r['Df_times_alpha']:>12.4f} {norm:>12.6f}")

        norm_cv = np.std(normalized_values) / np.mean(normalized_values)
        print(f"\nNormalized CV: {norm_cv*100:.2f}%")

        if norm_cv < cv:
            print("→ Normalizing by dimension REDUCES variation")
            print("→ Df × α scales with dimensionality")
        else:
            print("→ Df × α is INDEPENDENT of dimensionality")
            print("→ This is a true universal constant!")

    # Save
    receipt = {
        'test': 'Q48_UNIVERSAL_CONSTANT',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'results': results,
        'mean_df_alpha': float(np.mean(df_alpha_values)) if results else None,
        'cv': float(cv) if results else None,
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q48_universal_constant_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"\nReceipt saved: {path}")


if __name__ == '__main__':
    main()
