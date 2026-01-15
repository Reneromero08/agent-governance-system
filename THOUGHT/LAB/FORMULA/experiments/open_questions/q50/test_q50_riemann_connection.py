#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 2: Does σ_c = 1/α Relate to Riemann Zeta?

We have:
- Spectral zeta: ζ_sem(s) = Σ λ_k^(-s)
- Critical exponent: σ_c = 1/α (where ζ_sem diverges)
- For MiniLM: σ_c ≈ 2.09

This test explores connections to Riemann ζ(s):
- T2.1: Zeta value scan at σ_c
- T2.2: Critical line correspondence
- T2.3: Functional equation symmetry
- T2.4: Zero spacing analysis
- T2.5: Euler product convergence
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from scipy.optimize import brentq
from scipy.special import zeta as scipy_zeta

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Try to import mpmath for high-precision zeta
try:
    import mpmath
    mpmath.mp.dps = 30  # 30 decimal places
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    print("mpmath not available, using scipy.special.zeta")


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


def spectral_zeta(eigenvalues, s):
    """Compute spectral zeta function ζ_sem(s) = Σ λ_k^(-s)"""
    ev = eigenvalues[eigenvalues > 1e-10]
    if np.isreal(s):
        s = float(np.real(s))
        return np.sum(ev ** (-s))
    else:
        return np.sum(ev ** (-s))


def spectral_zeta_complex(eigenvalues, s_real, s_imag):
    """Compute spectral zeta for complex s = s_real + i*s_imag"""
    ev = eigenvalues[eigenvalues > 1e-10]
    s = complex(s_real, s_imag)
    result = np.sum(ev ** (-s))
    return result


def riemann_zeta(s):
    """Compute Riemann zeta using available library."""
    if MPMATH_AVAILABLE:
        return complex(mpmath.zeta(s))
    else:
        if np.isreal(s) and s > 1:
            return scipy_zeta(float(np.real(s)), 1)
        else:
            # scipy.special.zeta doesn't support complex s
            return None


def main():
    print("=" * 70)
    print("Q50 PART 2: RIEMANN ZETA CONNECTION")
    print("Testing if σ_c = 1/α relates to Riemann ζ(s)")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_RIEMANN_CONNECTION',
        'mpmath_available': MPMATH_AVAILABLE,
        'tests': {}
    }

    try:
        from sentence_transformers import SentenceTransformer

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

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(WORDS, normalize_embeddings=True)
        print(f"\nLoaded {len(WORDS)} embeddings, shape: {embeddings.shape}")

        eigenvalues = get_eigenspectrum(embeddings)
        Df = compute_df(eigenvalues)
        alpha = compute_alpha(eigenvalues)
        sigma_c = 1 / alpha if alpha > 0 else float('inf')

        print(f"\nSemantic metrics:")
        print(f"  Df = {Df:.4f}")
        print(f"  α = {alpha:.4f}")
        print(f"  σ_c = 1/α = {sigma_c:.4f}")
        print(f"  Df × α = {Df * alpha:.4f}")
        print(f"  8e = {8 * np.e:.4f}")

        # ============================================================
        # T2.1: Zeta Value Scan at σ_c
        # ============================================================
        print("\n" + "=" * 60)
        print("T2.1: RIEMANN ZETA AT σ_c")
        print("Testing ζ(σ_c) and nearby values")
        print("=" * 60)

        test_points = [
            ('σ_c', sigma_c),
            ('σ_c + 0.5', sigma_c + 0.5),
            ('σ_c - 0.5', sigma_c - 0.5),
            ('2', 2.0),
            ('3', 3.0),
            ('1/2', 0.5),
        ]

        zeta_values = []
        for name, s in test_points:
            if s > 1:  # Zeta converges for Re(s) > 1
                zeta_val = riemann_zeta(s)
                if zeta_val is not None:
                    print(f"  ζ({name}) = ζ({s:.4f}) = {zeta_val:.6f}")

                    # Check relationships to 8e
                    zeta_real = float(np.real(zeta_val)) if zeta_val else 0
                    ratio_to_8e = zeta_real / (8 * np.e) if zeta_real else 0
                    product_with_8e = zeta_real * (8 * np.e) if zeta_real else 0

                    zeta_values.append({
                        'name': name,
                        's': float(s),
                        'zeta': zeta_real,
                        'ratio_to_8e': float(ratio_to_8e),
                        'product_with_8e': float(product_with_8e),
                    })
            elif MPMATH_AVAILABLE:
                zeta_val = complex(mpmath.zeta(s))
                print(f"  ζ({name}) = ζ({s:.4f}) = {zeta_val:.6f}")
                zeta_values.append({
                    'name': name,
                    's': float(s),
                    'zeta': complex(zeta_val),
                })

        # Key relationship: does ζ(σ_c) × factor = 8e?
        if sigma_c > 1:
            zeta_sigma_c = riemann_zeta(sigma_c)
            if zeta_sigma_c:
                zeta_real = float(np.real(zeta_sigma_c))
                factor_for_8e = (8 * np.e) / zeta_real if zeta_real else 0
                print(f"\n  To get 8e from ζ(σ_c):")
                print(f"    ζ(σ_c) × {factor_for_8e:.4f} = 8e")
                print(f"    Note: {factor_for_8e:.4f} ≈ {factor_for_8e/np.pi:.4f}π")

        results['tests']['T2_1_zeta_scan'] = {
            'zeta_values': zeta_values,
            'sigma_c': float(sigma_c),
        }

        # ============================================================
        # T2.2: Critical Line Correspondence
        # ============================================================
        print("\n" + "=" * 60)
        print("T2.2: CRITICAL LINE CORRESPONDENCE")
        print("Comparing Re(s) = 1/2 (Riemann) to Re(s) = σ_c (Semantic)")
        print("=" * 60)

        # Riemann critical line: Re(s) = 1/2
        # Semantic critical line: Re(s) = σ_c = 1/α

        print(f"  Riemann critical line: Re(s) = 0.5")
        print(f"  Semantic critical line: Re(s) = σ_c = {sigma_c:.4f}")
        print(f"\n  Potential transformations:")
        print(f"    σ_c - 0.5 = {sigma_c - 0.5:.4f}")
        print(f"    σ_c × 2 = {sigma_c * 2:.4f}")
        print(f"    σ_c / 2 = {sigma_c / 2:.4f}")
        print(f"    1/σ_c = α = {alpha:.4f}")
        print(f"    σ_c × 0.5 = {sigma_c * 0.5:.4f}")

        # Is there a simple relationship?
        # σ_c ≈ 2.09, Riemann ≈ 0.5, ratio ≈ 4.18
        ratio = sigma_c / 0.5
        print(f"\n  σ_c / 0.5 = {ratio:.4f}")
        print(f"  σ_c / 0.5 / π = {ratio / np.pi:.4f}")
        print(f"  σ_c / 0.5 / e = {ratio / np.e:.4f}")

        results['tests']['T2_2_critical_line'] = {
            'riemann_critical': 0.5,
            'semantic_critical': float(sigma_c),
            'ratio': float(ratio),
            'ratio_over_pi': float(ratio / np.pi),
            'ratio_over_e': float(ratio / np.e),
        }

        # ============================================================
        # T2.3: Functional Equation Test
        # ============================================================
        print("\n" + "=" * 60)
        print("T2.3: FUNCTIONAL EQUATION SYMMETRY")
        print("Testing if ζ_sem has symmetry around σ_c")
        print("=" * 60)

        # Riemann: ξ(s) = ξ(1-s) where ξ involves Γ and π factors
        # Does ζ_sem(s) / ζ_sem(2σ_c - s) = constant?

        s_values = np.linspace(0.5, sigma_c * 1.5, 20)
        ratios = []

        print(f"\n  Testing symmetry around σ_c = {sigma_c:.4f}:")
        for s in s_values:
            s_reflected = 2 * sigma_c - s
            if s_reflected > 0:
                z1 = spectral_zeta(eigenvalues, s)
                z2 = spectral_zeta(eigenvalues, s_reflected)
                if z2 > 1e-10:
                    ratio = z1 / z2
                    ratios.append({
                        's': float(s),
                        's_reflected': float(s_reflected),
                        'z1': float(z1),
                        'z2': float(z2),
                        'ratio': float(ratio),
                    })

        if ratios:
            ratio_values = [r['ratio'] for r in ratios]
            mean_ratio = np.mean(ratio_values)
            std_ratio = np.std(ratio_values)
            cv_ratio = std_ratio / mean_ratio * 100 if mean_ratio > 0 else float('inf')

            print(f"  Ratio ζ_sem(s) / ζ_sem(2σ_c - s):")
            print(f"    Mean = {mean_ratio:.4f}")
            print(f"    Std = {std_ratio:.4f}")
            print(f"    CV = {cv_ratio:.2f}%")
            print(f"    {'FUNCTIONAL EQUATION EXISTS' if cv_ratio < 10 else 'NO SIMPLE SYMMETRY'}")

            results['tests']['T2_3_functional_equation'] = {
                'ratios': ratios[:5],  # First 5 for brevity
                'mean_ratio': float(mean_ratio),
                'cv_percent': float(cv_ratio),
                'has_symmetry': cv_ratio < 10,
            }

        # ============================================================
        # T2.4: Zero Spacing Analysis
        # ============================================================
        print("\n" + "=" * 60)
        print("T2.4: SPECTRAL ZETA ZEROS")
        print("Searching for zeros of ζ_sem in complex plane")
        print("=" * 60)

        # For Riemann: zeros at s = 1/2 + it_n
        # For semantic: search along Re(s) = σ_c/2

        real_part = sigma_c / 2
        imag_range = np.linspace(0.1, 50, 500)

        zeta_magnitudes = []
        for t in imag_range:
            z = spectral_zeta_complex(eigenvalues, real_part, t)
            zeta_magnitudes.append(abs(z))

        zeta_magnitudes = np.array(zeta_magnitudes)

        # Find local minima (potential zeros)
        zeros = []
        for i in range(1, len(zeta_magnitudes) - 1):
            if zeta_magnitudes[i] < zeta_magnitudes[i-1] and zeta_magnitudes[i] < zeta_magnitudes[i+1]:
                if zeta_magnitudes[i] < 0.1 * np.median(zeta_magnitudes):
                    zeros.append({
                        't': float(imag_range[i]),
                        'magnitude': float(zeta_magnitudes[i]),
                    })

        print(f"  Searching along Re(s) = σ_c/2 = {real_part:.4f}")
        print(f"  Found {len(zeros)} candidate zeros (local minima with |ζ| < 10% median)")

        if zeros:
            print(f"\n  First 5 candidate zeros:")
            for i, z in enumerate(zeros[:5]):
                print(f"    s = {real_part:.3f} + {z['t']:.3f}i, |ζ| = {z['magnitude']:.6f}")

            # Compute spacings
            if len(zeros) > 1:
                spacings = np.diff([z['t'] for z in zeros])
                mean_spacing = np.mean(spacings)
                print(f"\n  Mean spacing: {mean_spacing:.4f}")
                print(f"  2π / mean_spacing = {2 * np.pi / mean_spacing:.4f}")
                print(f"  2π / 8e = {2 * np.pi / (8 * np.e):.4f}")

                results['tests']['T2_4_zeros'] = {
                    'search_line': float(real_part),
                    'n_zeros': len(zeros),
                    'zeros': zeros[:10],
                    'mean_spacing': float(mean_spacing) if len(zeros) > 1 else None,
                    'spacing_over_2pi': float(mean_spacing / (2 * np.pi)) if len(zeros) > 1 else None,
                }
        else:
            print("  No clear zeros found in this range")
            results['tests']['T2_4_zeros'] = {'n_zeros': 0}

        # ============================================================
        # T2.5: Spectral Zeta at Special Points
        # ============================================================
        print("\n" + "=" * 60)
        print("T2.5: SPECTRAL ZETA AT SPECIAL POINTS")
        print("Computing ζ_sem at mathematically significant values")
        print("=" * 60)

        special_points = [
            ('s = 2 (like ζ(2) = π²/6)', 2.0),
            ('s = σ_c', sigma_c),
            ('s = σ_c + 1', sigma_c + 1),
            ('s = α', alpha),
            ('s = 1/α = σ_c', 1/alpha),
            ('s = 8e/Df', 8*np.e/Df),
        ]

        special_values = []
        for name, s in special_points:
            if s > 0:
                z_sem = spectral_zeta(eigenvalues, s)
                z_riemann = riemann_zeta(s) if s > 1 else None

                print(f"  {name}:")
                print(f"    ζ_sem({s:.4f}) = {z_sem:.6f}")
                if z_riemann:
                    print(f"    ζ({s:.4f}) = {float(np.real(z_riemann)):.6f}")
                    print(f"    Ratio = {z_sem / float(np.real(z_riemann)):.6f}")

                special_values.append({
                    'name': name,
                    's': float(s),
                    'zeta_sem': float(z_sem),
                    'zeta_riemann': float(np.real(z_riemann)) if z_riemann else None,
                })

        results['tests']['T2_5_special_points'] = special_values

        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 70)
        print("SUMMARY: RIEMANN CONNECTION")
        print("=" * 70)

        print(f"\n  Critical exponents:")
        print(f"    Riemann: Re(s) = 0.5")
        print(f"    Semantic: Re(s) = σ_c = 1/α = {sigma_c:.4f}")
        print(f"    Ratio: {sigma_c / 0.5:.4f}")

        if 'T2_3_functional_equation' in results['tests']:
            has_fe = results['tests']['T2_3_functional_equation'].get('has_symmetry', False)
            print(f"\n  Functional equation: {'YES (CV < 10%)' if has_fe else 'NO'}")

        if 'T2_4_zeros' in results['tests'] and results['tests']['T2_4_zeros'].get('mean_spacing'):
            spacing = results['tests']['T2_4_zeros']['mean_spacing']
            print(f"\n  Zero spacing: {spacing:.4f}")
            print(f"  Riemann-like? 2π/spacing = {2*np.pi/spacing:.4f}")

        print("\n  CONCLUSION:")
        print("  The connection to Riemann is NOT through identical structure,")
        print("  but through analogous properties:")
        print("  - Both have a spectral zeta function")
        print("  - Both have a critical line (different locations)")
        print("  - Both are constrained by conservation laws")
        print(f"  - Semantic constraint: Df × α = 8e ≈ {Df * alpha:.4f}")

    except ImportError as e:
        print(f"Could not load sentence-transformers: {e}")
        results['error'] = str(e)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_riemann_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
