#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q48 Part 2: Cumulative Variance Shape Analysis

The Q34 invariant is the CUMULATIVE variance curve (0.994 correlation).
This is different from raw eigenvalue spacings.

Here we analyze the SHAPE of cumulative variance:
1. What function describes it? (power law? exponential? log?)
2. Does this shape have mathematical significance?
3. Is there a connection to known spectral functions?
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# CURVE FITTING FUNCTIONS
# =============================================================================

def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def logarithmic(x, a, b):
    """Logarithmic: y = a * log(x) + b"""
    return a * np.log(x + 1) + b


def exponential_saturation(x, a, b, c):
    """Exponential saturation: y = a * (1 - exp(-b*x)) + c"""
    return a * (1 - np.exp(-b * x)) + c


def sigmoid(x, a, b, c, d):
    """Sigmoid: y = a / (1 + exp(-b*(x-c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d


def marchenko_pastur_cdf(x, gamma):
    """
    Approximate CDF of Marchenko-Pastur distribution.
    This is what random matrices follow.
    """
    lambda_max = (1 + np.sqrt(gamma))**2
    lambda_min = max(0, (1 - np.sqrt(gamma))**2)

    # Normalize x to [0, 1] range based on eigenvalue index
    x_norm = x / np.max(x)

    # Simple approximation of MP CDF shape
    return x_norm  # Linear for now - placeholder


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_cumulative_shape(eigenvalues, model_name):
    """Analyze the shape of cumulative variance curve."""

    # Sort descending
    ev = np.sort(eigenvalues)[::-1]

    # Normalize
    total = np.sum(ev)
    normalized = ev / total

    # Cumulative variance
    cumulative = np.cumsum(normalized)

    # X axis (dimension index, starting at 1)
    x = np.arange(1, len(cumulative) + 1)
    x_norm = x / len(x)  # Normalized to [0, 1]

    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"Dimensions: {len(ev)}")
    print(f"Variance explained at 50%: {np.searchsorted(cumulative, 0.5)} dims")
    print(f"Variance explained at 90%: {np.searchsorted(cumulative, 0.9)} dims")
    print(f"Variance explained at 99%: {np.searchsorted(cumulative, 0.99)} dims")

    # Fit different functions to the cumulative curve
    fits = {}

    # Power law fit
    try:
        popt, _ = curve_fit(power_law, x_norm, cumulative, p0=[1, 0.5], maxfev=5000)
        pred = power_law(x_norm, *popt)
        r2 = 1 - np.sum((cumulative - pred)**2) / np.sum((cumulative - np.mean(cumulative))**2)
        fits['Power Law'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'y = {popt[0]:.4f} * x^{popt[1]:.4f}'}
    except:
        fits['Power Law'] = {'params': [], 'r2': 0, 'formula': 'FAILED'}

    # Logarithmic fit
    try:
        popt, _ = curve_fit(logarithmic, x_norm * 100, cumulative, p0=[0.2, 0], maxfev=5000)
        pred = logarithmic(x_norm * 100, *popt)
        r2 = 1 - np.sum((cumulative - pred)**2) / np.sum((cumulative - np.mean(cumulative))**2)
        fits['Logarithmic'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'y = {popt[0]:.4f} * log(x) + {popt[1]:.4f}'}
    except:
        fits['Logarithmic'] = {'params': [], 'r2': 0, 'formula': 'FAILED'}

    # Exponential saturation fit
    try:
        popt, _ = curve_fit(exponential_saturation, x_norm, cumulative, p0=[1, 5, 0], maxfev=5000)
        pred = exponential_saturation(x_norm, *popt)
        r2 = 1 - np.sum((cumulative - pred)**2) / np.sum((cumulative - np.mean(cumulative))**2)
        fits['Exp Saturation'] = {'params': popt.tolist(), 'r2': r2, 'formula': f'y = {popt[0]:.4f} * (1 - exp(-{popt[1]:.4f}*x)) + {popt[2]:.4f}'}
    except:
        fits['Exp Saturation'] = {'params': [], 'r2': 0, 'formula': 'FAILED'}

    print(f"\nCURVE FITTING (R² scores):")
    for name, fit in sorted(fits.items(), key=lambda x: -x[1]['r2']):
        print(f"  {name:20s}: R² = {fit['r2']:.6f}")
        print(f"    {fit['formula']}")

    best_fit = max(fits.items(), key=lambda x: x[1]['r2'])
    print(f"\nBEST FIT: {best_fit[0]} (R² = {best_fit[1]['r2']:.6f})")

    # Analyze the decay rate (derivative of cumulative = eigenvalue distribution)
    decay = np.diff(cumulative)
    log_decay = np.log(decay[decay > 1e-10])

    if len(log_decay) > 10:
        # Fit line to log decay to find power law exponent
        x_log = np.arange(len(log_decay))
        slope, intercept, r_value, _, _ = stats.linregress(x_log, log_decay)
        print(f"\nDECAY ANALYSIS:")
        print(f"  Log-linear slope: {slope:.4f}")
        print(f"  Power law exponent: {-slope:.4f}")
        print(f"  R² of log fit: {r_value**2:.4f}")

        # This exponent is key - compare to known values
        # Zipf's law: exponent ≈ 1
        # Eigenvalue decay in NLP: typically 0.5-2.0
        print(f"\n  Interpretation:")
        if abs(slope) < 0.01:
            print(f"    → Flat decay (uniform eigenvalues)")
        elif abs(slope) < 0.5:
            print(f"    → Slow decay (distributed information)")
        elif abs(slope) < 1.5:
            print(f"    → Zipf-like decay (scale-free structure)")
        else:
            print(f"    → Fast decay (concentrated information)")
    else:
        slope = 0

    return {
        'model': model_name,
        'n_dims': len(ev),
        'dims_50pct': int(np.searchsorted(cumulative, 0.5)),
        'dims_90pct': int(np.searchsorted(cumulative, 0.9)),
        'dims_99pct': int(np.searchsorted(cumulative, 0.99)),
        'fits': {k: {'r2': v['r2'], 'formula': v['formula']} for k, v in fits.items()},
        'best_fit': best_fit[0],
        'best_r2': best_fit[1]['r2'],
        'decay_slope': float(slope) if slope else 0,
        'cumulative_sample': cumulative[::max(1, len(cumulative)//20)].tolist(),
    }


def load_embeddings():
    """Load embeddings from available models."""
    embeddings = {}

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

        for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
            try:
                model = SentenceTransformer(model_id)
                embs = model.encode(WORDS, normalize_embeddings=True)
                embeddings[name] = {word: embs[i] for i, word in enumerate(WORDS)}
                print(f"  Loaded {name}")
            except Exception as e:
                print(f"  Failed {name}: {e}")
    except ImportError:
        print("  sentence-transformers not available")

    try:
        import gensim.downloader as api
        for model_id, name in [("glove-wiki-gigaword-100", "GloVe-100")]:
            try:
                model = api.load(model_id)
                emb_dict = {w: model[w] for w in WORDS if w in model}
                if len(emb_dict) >= 50:
                    embeddings[name] = emb_dict
                    print(f"  Loaded {name}")
            except Exception as e:
                print(f"  Failed {name}: {e}")
    except ImportError:
        pass

    return embeddings


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


def main():
    print("=" * 70)
    print("Q48 PART 2: CUMULATIVE VARIANCE SHAPE ANALYSIS")
    print("What function describes the universal cumulative curve?")
    print("=" * 70)

    print("\nLoading embeddings...")
    embeddings = load_embeddings()

    if not embeddings:
        print("No embeddings available")
        return

    results = []
    for name, emb_dict in embeddings.items():
        eigenvalues = get_eigenspectrum(emb_dict)
        result = analyze_cumulative_shape(eigenvalues, name)
        results.append(result)

    # Cross-model comparison
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)

    if len(results) >= 2:
        # Compare cumulative curves
        curves = []
        for r in results:
            curves.append(np.array(r['cumulative_sample']))

        # Compute pairwise correlations
        print("\nCumulative curve correlations:")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                k = min(len(curves[i]), len(curves[j]))
                corr = np.corrcoef(curves[i][:k], curves[j][:k])[0, 1]
                print(f"  {results[i]['model']} vs {results[j]['model']}: {corr:.4f}")

        # Compare decay slopes
        slopes = [r['decay_slope'] for r in results]
        print(f"\nDecay slopes: {slopes}")
        print(f"  Mean: {np.mean(slopes):.4f}")
        print(f"  Std: {np.std(slopes):.4f}")
        print(f"  CV: {np.std(slopes)/abs(np.mean(slopes)):.2%}" if np.mean(slopes) != 0 else "  CV: N/A")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    best_fits = [r['best_fit'] for r in results]
    if len(set(best_fits)) == 1:
        print(f"\nAll models best fit by: {best_fits[0]}")
        print("This suggests a UNIVERSAL functional form for semantic eigenvalue decay.")

        if 'Power' in best_fits[0]:
            exponents = [r['fits'].get('Power Law', {}).get('params', [0, 0]) for r in results]
            print(f"\nPower law exponents would connect to:")
            print("  - Zipf's law (exponent ≈ 1)")
            print("  - Critical phenomena (scale-free)")
            print("  - Potentially to Riemann via spectral zeta functions")
    else:
        print(f"\nMixed best fits: {best_fits}")
        print("Different models follow different functional forms")

    # Save results
    receipt = {
        'test': 'Q48_CUMULATIVE_SHAPE',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'results': results,
        'best_fits': best_fits,
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q48_cumulative_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"\nReceipt saved: {path}")


if __name__ == '__main__':
    main()
