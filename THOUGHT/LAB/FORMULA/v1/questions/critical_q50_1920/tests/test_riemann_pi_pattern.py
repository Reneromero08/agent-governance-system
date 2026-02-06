#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate the log(ζ_sem) / π ≈ integer pattern

Finding from previous test:
  At s = 2.0, 2.5, 3.0: log(ζ_sem)/π ≈ 5, 6, 7

This suggests: ζ_sem(s) = e^(π × f(s)) where f(s) is approximately linear.

Let's characterize this pattern precisely.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def spectral_zeta(eigenvalues, s):
    """Compute spectral zeta function."""
    ev = eigenvalues[eigenvalues > 1e-10]
    return np.sum(ev ** (-s))


def load_embeddings():
    """Load embeddings."""
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
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings['MiniLM'] = model.encode(WORDS, normalize_embeddings=True)
        print(f"Loaded MiniLM: {embeddings['MiniLM'].shape}")
    except Exception as e:
        print(f"Failed: {e}")

    return embeddings


def get_eigenspectrum(embeddings):
    """Get eigenspectrum."""
    vecs_centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


def main():
    print("=" * 70)
    print("INVESTIGATING log(ζ_sem) / π PATTERN")
    print("=" * 70)

    embeddings = load_embeddings()
    if not embeddings:
        return

    eigenvalues = get_eigenspectrum(embeddings['MiniLM'])

    # Compute log(ζ_sem)/π for many s values
    s_values = np.linspace(1.5, 5.0, 71)
    log_zeta_over_pi = []

    print("\n--- DETAILED SCAN ---")
    print(f"{'s':>6} | {'ζ_sem':>15} | {'log(ζ)/π':>10} | {'nearest_int':>12} | {'error':>8}")
    print("-" * 65)

    for s in s_values:
        z = spectral_zeta(eigenvalues, s)
        if z > 0:
            val = np.log(z) / np.pi
            nearest = round(val)
            error = val - nearest
            log_zeta_over_pi.append((s, val, nearest, error))

            if abs(error) < 0.1:  # Near integer
                print(f"{s:6.2f} | {z:15.2e} | {val:10.4f} | {nearest:12d} | {error:8.4f} ***")
            elif s in [2.0, 2.5, 3.0, 3.5, 4.0]:
                print(f"{s:6.2f} | {z:15.2e} | {val:10.4f} | {nearest:12d} | {error:8.4f}")

    # Fit: log(ζ)/π = a × s + b
    s_arr = np.array([x[0] for x in log_zeta_over_pi])
    val_arr = np.array([x[1] for x in log_zeta_over_pi])

    slope, intercept = np.polyfit(s_arr, val_arr, 1)

    print("\n--- LINEAR FIT ---")
    print(f"log(ζ_sem) / π = {slope:.4f} × s + {intercept:.4f}")
    print(f"\nSlope = {slope:.4f}")
    print(f"  Is slope ≈ 2? {abs(slope - 2) < 0.1}")
    print(f"  Slope / 2 = {slope/2:.4f}")

    # Residuals from fit
    fitted = slope * s_arr + intercept
    residuals = val_arr - fitted

    print(f"\nFit quality:")
    print(f"  Max residual: {np.max(np.abs(residuals)):.4f}")
    print(f"  Mean residual: {np.mean(np.abs(residuals)):.4f}")

    # Check if the pattern means ζ_sem(s) = A × B^s
    # log(ζ) = log(A) + s × log(B)
    # log(ζ)/π = log(A)/π + s × log(B)/π
    # So: log(B)/π = slope → log(B) = slope × π → B = e^(slope × π)

    B = np.exp(slope * np.pi)
    A = np.exp(intercept * np.pi)

    print(f"\n--- EXPONENTIAL FORM ---")
    print(f"ζ_sem(s) ≈ A × B^s")
    print(f"  A = e^({intercept:.4f}π) = {A:.4e}")
    print(f"  B = e^({slope:.4f}π) = {B:.4e}")

    # What IS B?
    print(f"\n--- WHAT IS B? ---")
    print(f"B = {B:.4f}")
    print(f"B / e = {B / np.e:.4f}")
    print(f"log(B) = {np.log(B):.4f}")
    print(f"log(B) / π = {np.log(B) / np.pi:.4f} = slope")

    # The eigenvalues themselves decay as λ_k ~ k^(-α)
    # So ζ_sem(s) = Σ λ_k^(-s) ~ Σ k^(αs)
    # For convergence: αs > 1, i.e., s > 1/α ≈ 2

    # Near s = 2 (critical), the behavior is dominated by large k terms
    # But we're seeing ζ_sem(s) ~ B^s with B = e^(slope × π)

    # If slope ≈ 2, then B = e^(2π)
    # And log(B)/π = 2 exactly

    print(f"\n--- KEY QUESTION ---")
    print(f"Is log(B)/π = 2 exactly?")
    print(f"  Measured slope = {slope:.6f}")
    print(f"  Deviation from 2: {abs(slope - 2):.4f} ({abs(slope - 2)/2*100:.2f}%)")

    if abs(slope - 2) < 0.1:
        print(f"\n*** YES! log(B)/π ≈ 2 ***")
        print(f"This means: ζ_sem(s) ≈ A × e^(2πs)")
        print(f"Or equivalently: log(ζ_sem(s)) ≈ 2πs + const")

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    receipt = {
        'test': 'RIEMANN_PI_PATTERN',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'finding': 'log(ζ_sem)/π is approximately linear in s',
        'slope': float(slope),
        'intercept': float(intercept),
        'slope_close_to_2': abs(slope - 2) < 0.1,
        'B_value': float(B),
        'A_value': float(A),
    }

    path = results_dir / f'riemann_pi_pattern_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"\nReceipt saved: {path}")


if __name__ == '__main__':
    main()
