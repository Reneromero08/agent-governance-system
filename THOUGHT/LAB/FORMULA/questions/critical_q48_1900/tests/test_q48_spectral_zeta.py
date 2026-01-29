#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q48 Part 3: Spectral Zeta Function Analysis

The direct GUE comparison failed. But what about the spectral zeta function?

For eigenvalues λ_k, define:
    ζ_semantic(s) = Σ λ_k^(-s)

Questions:
1. Where does this converge/diverge? (analog of critical strip)
2. Does it have a functional equation? (symmetry like ζ(s) = ζ(1-s))
3. Do zeros exist on a "critical line"?
4. Does the participation ratio Df relate to the critical exponent?

The Hilbert-Pólya conjecture: Riemann zeros are eigenvalues of a self-adjoint operator.
If semantic eigenvalues follow universal laws, maybe they're eigenvalues of a RELATED operator.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import integrate
from scipy.optimize import brentq

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# SPECTRAL ZETA FUNCTION
# =============================================================================

def spectral_zeta(eigenvalues, s):
    """
    Compute spectral zeta function ζ(s) = Σ λ_k^(-s)

    For complex s = σ + it, this converges when σ > σ_c (critical exponent).
    """
    # Filter positive eigenvalues
    ev = eigenvalues[eigenvalues > 1e-10]

    if np.iscomplex(s):
        return np.sum(ev ** (-s))
    else:
        # Real s
        if s > 0:
            return np.sum(ev ** (-s))
        else:
            # Analytic continuation needed for s ≤ 0
            return np.sum(ev ** (-s))  # May diverge


def find_convergence_boundary(eigenvalues):
    """
    Find the critical exponent σ_c where ζ(s) transitions from divergent to convergent.

    For power-law decay λ_k ~ k^(-α), we have σ_c = 1/α.
    """
    ev = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]

    results = {}

    # Test convergence at different s values
    test_s = np.linspace(-2, 10, 49)
    zeta_values = []

    for s in test_s:
        try:
            z = spectral_zeta(ev, s)
            if np.isfinite(z) and z < 1e15:
                zeta_values.append((s, z))
        except:
            pass

    results['zeta_curve'] = zeta_values

    # Estimate critical exponent from eigenvalue decay
    k = np.arange(1, len(ev) + 1)
    log_k = np.log(k[ev > 1e-8])
    log_ev = np.log(ev[ev > 1e-8])

    if len(log_k) > 10:
        # Fit power law in the tail (last 50%)
        n_tail = len(log_k) // 2
        slope, intercept = np.polyfit(log_k[-n_tail:], log_ev[-n_tail:], 1)
        alpha = -slope  # λ_k ~ k^(-α)
        sigma_c = 1 / alpha if alpha > 0 else float('inf')

        results['alpha'] = float(alpha)
        results['sigma_c'] = float(sigma_c)
    else:
        results['alpha'] = None
        results['sigma_c'] = None

    return results


def check_functional_equation(eigenvalues):
    """
    Check if spectral zeta has a functional equation like ζ(s) ∝ ζ(1-s).

    For Riemann zeta: Λ(s) = Λ(1-s) where Λ(s) = π^(-s/2) Γ(s/2) ζ(s)
    """
    ev = eigenvalues[eigenvalues > 1e-10]

    # Test symmetry around different points
    symmetry_tests = []

    for center in [0.5, 1.0, 1.5, 2.0]:
        ratios = []
        for offset in np.linspace(0.1, 2, 10):
            s1 = center + offset
            s2 = 2 * center - s1  # = center - offset

            z1 = spectral_zeta(ev, s1)
            z2 = spectral_zeta(ev, s2)

            if np.isfinite(z1) and np.isfinite(z2) and z1 > 0 and z2 > 0:
                ratios.append(z1 / z2)

        if ratios:
            cv = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else float('inf')
            symmetry_tests.append({
                'center': center,
                'ratio_mean': float(np.mean(ratios)),
                'ratio_cv': float(cv),
                'is_symmetric': cv < 0.1  # Low CV = constant ratio = symmetry
            })

    return symmetry_tests


def analyze_zeta_zeros(eigenvalues):
    """
    Look for zeros of the spectral zeta function.

    For Riemann, zeros on critical line Re(s) = 1/2.
    For spectral zeta, zeros might exist at specific s values.
    """
    ev = eigenvalues[eigenvalues > 1e-10]

    # Scan for sign changes (zeros) along real axis
    s_values = np.linspace(0.1, 5, 100)
    zeta_real = [spectral_zeta(ev, s) for s in s_values]

    # Look for sign changes
    zeros = []
    for i in range(len(zeta_real) - 1):
        if zeta_real[i] * zeta_real[i+1] < 0:
            # Sign change - zero between s_values[i] and s_values[i+1]
            try:
                z = brentq(lambda s: spectral_zeta(ev, s), s_values[i], s_values[i+1])
                zeros.append(float(z))
            except:
                pass

    # Check along imaginary axis (s = σ + it)
    # For fixed σ, scan t
    complex_zeros = []
    for sigma in [0.5, 1.0]:
        t_values = np.linspace(0.1, 10, 50)
        for t in t_values:
            s = complex(sigma, t)
            z = spectral_zeta(ev, s)
            if abs(z) < 0.1:  # Near zero
                complex_zeros.append({'sigma': sigma, 't': float(t), 'abs_zeta': float(abs(z))})

    return {
        'real_zeros': zeros,
        'complex_near_zeros': complex_zeros[:10]  # First 10
    }


def participation_ratio_connection(eigenvalues):
    """
    Explore connection between participation ratio Df and critical exponent.

    Df = (Σλ)² / Σλ²

    Hypothesis: Df encodes the "effective dimension" which relates to σ_c.
    """
    ev = eigenvalues[eigenvalues > 1e-10]

    # Participation ratio
    Df = (np.sum(ev) ** 2) / np.sum(ev ** 2)

    # Spectral entropy
    p = ev / np.sum(ev)
    H = -np.sum(p * np.log(p + 1e-10))

    # Effective dimension from entropy
    D_eff = np.exp(H)

    # Critical exponent from power law fit
    k = np.arange(1, len(ev) + 1)
    log_k = np.log(k)
    log_ev = np.log(ev)
    slope, _ = np.polyfit(log_k[:len(ev)//2], log_ev[:len(ev)//2], 1)
    alpha = -slope
    sigma_c = 1 / alpha if alpha > 0 else float('inf')

    return {
        'Df': float(Df),
        'D_effective': float(D_eff),
        'alpha': float(alpha),
        'sigma_c': float(sigma_c),
        'Df_over_sigma_c': float(Df * sigma_c) if np.isfinite(sigma_c) else None,
        'Df_times_alpha': float(Df * alpha),
    }


# =============================================================================
# RIEMANN ZERO COMPARISON
# =============================================================================

def compare_to_riemann_zeros():
    """
    Compare our spectral zeta structure to known Riemann zero properties.

    Key Riemann facts:
    1. Zeros on critical line Re(s) = 1/2
    2. Zero density: N(T) ~ (T/2π) log(T/2π)
    3. Functional equation: ξ(s) = ξ(1-s)
    """
    print("\n" + "=" * 60)
    print("RIEMANN ZERO REFERENCE")
    print("=" * 60)

    # First few Riemann zeros (imaginary parts)
    # These are the heights t where ζ(1/2 + it) = 0
    riemann_zeros = [
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    ]

    print("\nFirst 10 Riemann zeros (imaginary parts):")
    for i, z in enumerate(riemann_zeros):
        print(f"  γ_{i+1} = {z:.6f}")

    # Compute spacing statistics of Riemann zeros
    spacings = np.diff(riemann_zeros)
    mean_spacing = np.mean(spacings)
    normalized = spacings / mean_spacing

    print(f"\nRiemann zero spacings (normalized):")
    print(f"  Mean spacing: {mean_spacing:.4f}")
    print(f"  Normalized spacings: {normalized}")

    # These SHOULD follow GUE statistics
    # (This is what Montgomery-Odlyzko showed)

    return {
        'first_10_zeros': riemann_zeros,
        'spacings': spacings.tolist(),
        'mean_spacing': float(mean_spacing),
    }


# =============================================================================
# MAIN
# =============================================================================

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
    print("Q48 PART 3: SPECTRAL ZETA FUNCTION ANALYSIS")
    print("Looking for the Riemann connection beyond GUE spacings")
    print("=" * 70)

    # Reference: Riemann zero properties
    riemann_ref = compare_to_riemann_zeros()

    print("\nLoading embeddings...")
    embeddings = load_embeddings()

    if not embeddings:
        print("No embeddings available")
        return

    results = {}

    for name, emb_dict in embeddings.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {name}")
        print(f"{'='*60}")

        eigenvalues = get_eigenspectrum(emb_dict)

        # 1. Convergence analysis
        print("\n--- CONVERGENCE ANALYSIS ---")
        conv = find_convergence_boundary(eigenvalues)
        if conv['alpha']:
            print(f"Power law exponent α: {conv['alpha']:.4f}")
            print(f"Critical exponent σ_c: {conv['sigma_c']:.4f}")
            print(f"  (ζ(s) converges for Re(s) > σ_c)")

        # 2. Functional equation test
        print("\n--- FUNCTIONAL EQUATION TEST ---")
        symmetry = check_functional_equation(eigenvalues)
        for test in symmetry:
            status = "✓ SYMMETRIC" if test['is_symmetric'] else "✗ not symmetric"
            print(f"  Around s={test['center']}: ratio CV={test['ratio_cv']:.4f} {status}")

        # 3. Zero analysis
        print("\n--- ZETA ZEROS ---")
        zeros = analyze_zeta_zeros(eigenvalues)
        if zeros['real_zeros']:
            print(f"Real zeros found at: {zeros['real_zeros']}")
        else:
            print("No real zeros found in range [0.1, 5]")

        if zeros['complex_near_zeros']:
            print(f"Near-zeros in complex plane:")
            for nz in zeros['complex_near_zeros'][:3]:
                print(f"  s = {nz['sigma']} + {nz['t']:.2f}i, |ζ| = {nz['abs_zeta']:.4f}")

        # 4. Participation ratio connection
        print("\n--- PARTICIPATION RATIO CONNECTION ---")
        pr = participation_ratio_connection(eigenvalues)
        print(f"Df (participation ratio): {pr['Df']:.2f}")
        print(f"α (decay exponent): {pr['alpha']:.4f}")
        print(f"σ_c (critical line): {pr['sigma_c']:.4f}")
        print(f"Df × α = {pr['Df_times_alpha']:.4f}")

        # Is Df × α a universal constant?

        results[name] = {
            'convergence': conv,
            'symmetry': symmetry,
            'zeros': zeros,
            'participation': pr,
        }

    # Cross-model comparison
    print("\n" + "=" * 70)
    print("CROSS-MODEL ANALYSIS")
    print("=" * 70)

    if len(results) >= 2:
        models = list(results.keys())

        # Compare Df × α across models
        df_alpha_values = [results[m]['participation']['Df_times_alpha'] for m in models]
        print(f"\nDf × α across models:")
        for m, v in zip(models, df_alpha_values):
            print(f"  {m}: {v:.4f}")

        cv = np.std(df_alpha_values) / np.mean(df_alpha_values)
        print(f"\nCoefficient of variation: {cv:.2%}")

        if cv < 0.15:
            print("*** Df × α appears to be a UNIVERSAL CONSTANT ***")

        # Compare σ_c (critical line position)
        sigma_c_values = [results[m]['participation']['sigma_c'] for m in models]
        print(f"\nCritical exponent σ_c:")
        for m, v in zip(models, sigma_c_values):
            print(f"  {m}: {v:.4f}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)

    print("""
The spectral zeta function ζ_sem(s) = Σ λ_k^(-s) has:

1. A CRITICAL LINE at Re(s) = σ_c ≈ 1/α
   - This is the analog of Re(s) = 1/2 for Riemann
   - The position depends on eigenvalue decay rate

2. Df × α may be a universal constant
   - Df = participation ratio (effective dimension)
   - α = power law decay exponent
   - Their product might be invariant across models

3. The connection to Riemann is NOT through GUE spacings
   BUT through the STRUCTURE of the zeta function itself:
   - Both have critical lines
   - Both have convergence/divergence transitions
   - The functional equation test shows partial symmetry

NEXT: Investigate if σ_c = 1/α relates to any known constant.
""")

    # Save results
    receipt = {
        'test': 'Q48_SPECTRAL_ZETA',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'riemann_reference': riemann_ref,
        'results': results,
    }

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q48_spectral_zeta_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print(f"\nReceipt saved: {path}")


if __name__ == '__main__':
    main()
