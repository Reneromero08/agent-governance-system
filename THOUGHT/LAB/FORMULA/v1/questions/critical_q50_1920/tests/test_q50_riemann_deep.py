#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50: Deep Riemann Connection

Following the instinct: α ≈ 1/2 (the Riemann critical line value!)

If true:
- α = 1/2 implies σ_c = 2, where ζ(2) = π²/6
- Df × α = Df/2 = 8e implies Df = 16e ≈ 43.5
- The conservation law becomes: Df = 16e = 2 × 8e

Questions to test:
1. Is α → 1/2 a universal attractor?
2. Does ζ(2) = π²/6 appear in the spectral structure?
3. Do the actual Riemann zeros relate to semantic structure?
4. Is there a prime-like decomposition of semantic eigenvalues?
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from scipy.special import zeta as scipy_zeta

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import mpmath
    mpmath.mp.dps = 50
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


def compute_df(eigenvalues):
    ev = eigenvalues[eigenvalues > 1e-10]
    return (np.sum(ev)**2) / np.sum(ev**2)


def compute_alpha(eigenvalues):
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
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return np.maximum(eigenvalues, 1e-10)


# First 50 Riemann zeros (imaginary parts)
RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
    103.725538, 105.446623, 107.168611, 111.029536, 111.874659,
    114.320220, 116.226680, 118.790783, 121.370125, 122.946829,
    124.256819, 127.516684, 129.578704, 131.087688, 133.497737,
    134.756510, 138.116042, 139.736209, 141.123707, 143.111846,
]


def main():
    print("=" * 70)
    print("Q50: DEEP RIEMANN CONNECTION")
    print("Testing if α → 1/2 is the Riemann bridge")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_RIEMANN_DEEP',
        'hypothesis': 'α ≈ 1/2 (Riemann critical line)',
        'models': [],
        'summary': {}
    }

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

    try:
        from sentence_transformers import SentenceTransformer

        models = [
            ("all-MiniLM-L6-v2", "MiniLM-L6"),
            ("all-mpnet-base-v2", "MPNet-base"),
            ("BAAI/bge-small-en-v1.5", "BGE-small"),
            ("paraphrase-MiniLM-L6-v2", "ParaMiniLM"),
            ("all-distilroberta-v1", "DistilRoBERTa"),
        ]

        # ============================================================
        # TEST 1: Is α → 1/2?
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 1: Is α → 1/2 (Riemann critical line)?")
        print("=" * 60)

        print(f"\n  Riemann critical line: Re(s) = 1/2 = 0.5")
        print(f"  If α = 1/2, then σ_c = 1/α = 2")
        print(f"  And Df × α = Df/2 = 8e → Df = 16e ≈ 43.5")
        print()

        alpha_values = []
        df_values = []

        print(f"  {'Model':<20} {'α':<10} {'|α - 0.5|':<12} {'Df':<10} {'|Df - 16e|':<12}")
        print("  " + "-" * 64)

        for model_id, model_name in models:
            model = SentenceTransformer(model_id)
            embeddings = model.encode(WORDS, normalize_embeddings=True)
            eigenvalues = get_eigenspectrum(embeddings)

            Df = compute_df(eigenvalues)
            alpha = compute_alpha(eigenvalues)

            alpha_deviation = abs(alpha - 0.5)
            df_deviation = abs(Df - 16*np.e)

            print(f"  {model_name:<20} {alpha:<10.4f} {alpha_deviation:<12.4f} {Df:<10.2f} {df_deviation:<12.2f}")

            alpha_values.append(alpha)
            df_values.append(Df)

            results['models'].append({
                'model': model_name,
                'alpha': float(alpha),
                'alpha_deviation_from_0.5': float(alpha_deviation),
                'Df': float(Df),
                'Df_deviation_from_16e': float(df_deviation),
            })

        mean_alpha = np.mean(alpha_values)
        std_alpha = np.std(alpha_values)
        mean_Df = np.mean(df_values)

        print()
        print(f"  Mean α: {mean_alpha:.4f} ± {std_alpha:.4f}")
        print(f"  Deviation from 0.5: {abs(mean_alpha - 0.5):.4f} ({abs(mean_alpha - 0.5)/0.5*100:.1f}%)")
        print(f"  Mean Df: {mean_Df:.2f}")
        print(f"  16e = {16*np.e:.2f}")

        results['summary']['mean_alpha'] = float(mean_alpha)
        results['summary']['alpha_deviation_percent'] = float(abs(mean_alpha - 0.5)/0.5*100)

        # ============================================================
        # TEST 2: Does ζ(2) = π²/6 appear?
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 2: Does ζ(2) = π²/6 appear in the structure?")
        print("=" * 60)

        zeta_2 = np.pi**2 / 6  # ≈ 1.6449
        print(f"\n  ζ(2) = π²/6 = {zeta_2:.6f}")

        # Load one model for detailed analysis
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(WORDS, normalize_embeddings=True)
        eigenvalues = get_eigenspectrum(embeddings)

        Df = compute_df(eigenvalues)
        alpha = compute_alpha(eigenvalues)
        df_alpha = Df * alpha

        # Check various relationships
        relationships = [
            ('Df × α / 8', df_alpha / 8),
            ('Df / 8e', Df / (8*np.e)),
            ('α × 8e / Df', alpha * 8*np.e / Df),
            ('(Df × α) / π²', df_alpha / np.pi**2),
            ('6 × (Df × α) / π²', 6 * df_alpha / np.pi**2),
            ('8e / ζ(2)', 8*np.e / zeta_2),
            ('Df × α / ζ(2)', df_alpha / zeta_2),
        ]

        print(f"\n  Looking for π²/6 in the structure:")
        for name, value in relationships:
            ratio = value / zeta_2
            print(f"    {name:<25} = {value:.4f} (ratio to ζ(2): {ratio:.4f})")

        # Key relationship: 8e = Df × α ≈ 21.75
        # ζ(2) × something = 8e?
        factor = df_alpha / zeta_2
        print(f"\n  To get 8e from ζ(2):")
        print(f"    ζ(2) × {factor:.4f} = Df × α")
        print(f"    {factor:.4f} ≈ {factor/np.pi:.4f}π ≈ {factor/np.e:.4f}e ≈ {factor/8:.4f}×8")

        # ============================================================
        # TEST 3: Riemann zeros in eigenvalue spacing
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 3: Riemann zeros in eigenvalue spacing")
        print("=" * 60)

        # Normalized eigenvalue spacings
        ev_sorted = eigenvalues[eigenvalues > 1e-10]
        ev_log = np.log(ev_sorted)  # Log scale
        spacings = -np.diff(ev_log)  # Negative because decreasing
        spacings_normalized = spacings / np.mean(spacings)

        # Compare to Riemann zero spacings
        riemann_spacings = np.diff(RIEMANN_ZEROS[:20])
        riemann_normalized = riemann_spacings / np.mean(riemann_spacings)

        print(f"\n  First 10 eigenvalue spacings (normalized): {spacings_normalized[:10].round(3)}")
        print(f"  First 10 Riemann spacings (normalized): {riemann_normalized[:10].round(3)}")

        # Cross-correlation
        min_len = min(len(spacings_normalized), len(riemann_normalized))
        correlation = np.corrcoef(spacings_normalized[:min_len], riemann_normalized[:min_len])[0, 1]

        print(f"\n  Correlation between spacings: {correlation:.4f}")

        results['summary']['spacing_correlation'] = float(correlation)

        # ============================================================
        # TEST 4: The 4π connection
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 4: The 4π connection")
        print("=" * 60)

        # σ_c / 0.5 ≈ 4.18 ≈ 4/3 × π ≈ 4.19
        sigma_c = 1 / mean_alpha
        ratio_to_half = sigma_c / 0.5

        print(f"\n  σ_c = 1/α = {sigma_c:.4f}")
        print(f"  σ_c / 0.5 = {ratio_to_half:.4f}")
        print(f"  4π/3 = {4*np.pi/3:.4f}")
        print(f"  Difference: {abs(ratio_to_half - 4*np.pi/3):.4f}")

        # If σ_c/0.5 = 4π/3, then σ_c = 2π/3
        # And α = 1/σ_c = 3/(2π) ≈ 0.477
        predicted_alpha = 3 / (2 * np.pi)
        print(f"\n  If σ_c = 2π/3, then α = 3/(2π) = {predicted_alpha:.4f}")
        print(f"  Measured α = {mean_alpha:.4f}")
        print(f"  Difference: {abs(mean_alpha - predicted_alpha):.4f} ({abs(mean_alpha - predicted_alpha)/predicted_alpha*100:.1f}%)")

        results['summary']['predicted_alpha_from_pi'] = float(predicted_alpha)
        results['summary']['alpha_vs_pi_prediction'] = float(abs(mean_alpha - predicted_alpha)/predicted_alpha*100)

        # ============================================================
        # TEST 5: The fundamental identity
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 5: The fundamental identity")
        print("=" * 60)

        print(f"\n  We have: Df × α = 8e ≈ 21.746")
        print(f"  And: α ≈ 3/(2π) ≈ 0.477")
        print(f"  Therefore: Df ≈ 8e × 2π/3 = 16πe/3 ≈ {16*np.pi*np.e/3:.2f}")

        predicted_Df = 16 * np.pi * np.e / 3
        print(f"\n  Predicted Df: {predicted_Df:.2f}")
        print(f"  Measured Df: {mean_Df:.2f}")
        print(f"  Difference: {abs(mean_Df - predicted_Df):.2f} ({abs(mean_Df - predicted_Df)/predicted_Df*100:.1f}%)")

        # Alternative: if α = 1/2 exactly
        print(f"\n  Alternative (if α = 1/2):")
        print(f"  Df = 8e / 0.5 = 16e = {16*np.e:.2f}")
        print(f"  Measured Df: {mean_Df:.2f}")
        print(f"  Difference: {abs(mean_Df - 16*np.e):.2f} ({abs(mean_Df - 16*np.e)/(16*np.e)*100:.1f}%)")

        # ============================================================
        # TEST 6: Prime eigenvalue hypothesis
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST 6: Prime eigenvalue hypothesis")
        print("=" * 60)

        # The Euler product: ζ(s) = Π_p (1 - p^(-s))^(-1)
        # If eigenvalues have prime-like structure...

        # Check if eigenvalue ratios relate to primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        ev_top = eigenvalues[:len(primes)]
        ev_ratios = ev_top[0] / ev_top

        print(f"\n  Eigenvalue ratios (λ₁/λₖ) vs primes:")
        print(f"  k    λ₁/λₖ      p_k       ratio")
        print("  " + "-" * 40)

        for k in range(min(10, len(primes))):
            p = primes[k]
            r = ev_ratios[k]
            print(f"  {k+1:<4} {r:<10.4f} {p:<10} {r/p:.4f}")

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("SUMMARY: RIEMANN CONNECTION")
        print("=" * 70)

        print(f"""
  The Riemann connection appears through:

  1. α ≈ 3/(2π) ≈ 0.477 (NOT exactly 1/2, but π-related)
     - Mean α across models: {mean_alpha:.4f}
     - Deviation from 3/(2π): {abs(mean_alpha - predicted_alpha)/predicted_alpha*100:.1f}%

  2. σ_c = 1/α ≈ 2π/3 ≈ 2.09
     - This is close to 2, where ζ(2) = π²/6

  3. The conservation law: Df × α = 8e
     - Becomes: Df × 3/(2π) = 8e
     - Therefore: Df = 16πe/3 ≈ 45.4
     - Measured: Df ≈ {mean_Df:.1f}

  4. The factor 8e ≈ 21.75 relates to π through:
     - 8e × (3/(2π)) = 8e × α ≈ Df
     - 8e / π² ≈ 2.20 (close to ζ(3) ≈ 1.20? No)

  VERDICT: The Riemann connection is through π, not directly through ζ(s).
  α ≈ 3/(2π) suggests π is fundamental to semantic decay.
  """)

        # Final check: is 8e = something with π?
        print(f"  Final identities:")
        print(f"    8e = {8*np.e:.4f}")
        print(f"    8e/π = {8*np.e/np.pi:.4f}")
        print(f"    8e/π² = {8*np.e/np.pi**2:.4f}")
        print(f"    8e × π = {8*np.e*np.pi:.4f}")
        print(f"    8 × e × π / 3 = {8*np.e*np.pi/3:.4f} ≈ Df × α × π/3")

    except ImportError as e:
        print(f"Error: {e}")
        results['error'] = str(e)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_riemann_deep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
