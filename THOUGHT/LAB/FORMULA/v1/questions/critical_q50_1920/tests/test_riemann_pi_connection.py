#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riemann π Connection Test

Question: Does π appear in our spectral zeta at σ_c = 2?

We have:
- σ_c = 1/α ≈ 2 (our critical exponent)
- ζ(2) = π²/6 ≈ 1.6449 (Riemann zeta at 2)

Test: Compute ζ_sem(s) near s = 2 and look for π.
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


# =============================================================================
# SPECTRAL ZETA FUNCTION
# =============================================================================

def spectral_zeta(eigenvalues, s):
    """
    Compute spectral zeta function ζ_sem(s) = Σ λ_k^(-s)
    """
    ev = eigenvalues[eigenvalues > 1e-10]
    return np.sum(ev ** (-s))


def spectral_zeta_derivative(eigenvalues, s, h=1e-6):
    """Numerical derivative of spectral zeta."""
    return (spectral_zeta(eigenvalues, s + h) - spectral_zeta(eigenvalues, s - h)) / (2 * h)


def analyze_near_critical_point(eigenvalues, sigma_c):
    """
    Analyze ζ_sem(s) near s = σ_c ≈ 2

    For Riemann: ζ(s) has a simple pole at s = 1 with residue 1.
    Near s = 2, ζ(2) = π²/6.

    For our spectral zeta: What happens near σ_c?
    """
    results = {}

    # Sample points approaching σ_c from above
    epsilons = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    zeta_values = []
    for eps in epsilons:
        s = sigma_c + eps
        z = spectral_zeta(eigenvalues, s)
        zeta_values.append({
            's': s,
            'epsilon': eps,
            'zeta': float(z),
            'zeta_over_pi_sq': float(z / (np.pi ** 2)),
            'zeta_times_6_over_pi_sq': float(z * 6 / (np.pi ** 2)),  # = z / ζ(2)
        })

    results['approach_from_above'] = zeta_values

    # Check if ζ_sem(σ_c + ε) × ε → constant as ε → 0 (pole residue)
    residue_estimates = []
    for v in zeta_values:
        residue_estimates.append(v['zeta'] * v['epsilon'])

    results['residue_estimates'] = residue_estimates
    results['residue_stable'] = np.std(residue_estimates[-3:]) / np.mean(residue_estimates[-3:]) < 0.1

    # Compare to ζ(2) = π²/6
    zeta_2 = np.pi ** 2 / 6
    results['riemann_zeta_2'] = float(zeta_2)

    # At exactly σ_c (will be large/divergent)
    z_at_sigma_c = spectral_zeta(eigenvalues, sigma_c + 0.001)
    results['zeta_at_sigma_c'] = float(z_at_sigma_c)

    return results


def search_for_pi(eigenvalues, sigma_c):
    """
    Search for π in various combinations involving spectral zeta.
    """
    results = {}

    pi = np.pi
    e = np.e
    target_8e = 8 * e

    # Sample at several points
    test_points = [sigma_c + 0.1, sigma_c + 0.5, sigma_c + 1.0, 2.0, 2.5, 3.0]

    for s in test_points:
        z = spectral_zeta(eigenvalues, s)

        # Try various combinations
        combos = {
            'z': z,
            'z / pi': z / pi,
            'z / pi^2': z / (pi ** 2),
            'z * 6 / pi^2': z * 6 / (pi ** 2),  # Compare to ζ(2)
            'z / e': z / e,
            'z / 8e': z / target_8e,
            'sqrt(z) / pi': np.sqrt(z) / pi if z > 0 else None,
            'log(z) / pi': np.log(z) / pi if z > 0 else None,
        }

        # Check if any combo is close to a simple integer or simple fraction
        for name, val in combos.items():
            if val is not None:
                # Check closeness to integers 1-10
                for n in range(1, 11):
                    if abs(val - n) < 0.05 * n:
                        results[f's={s:.1f}, {name} ≈ {n}'] = float(val)
                    if abs(val - 1/n) < 0.05 / n:
                        results[f's={s:.1f}, {name} ≈ 1/{n}'] = float(val)

    return results


def compare_divergence_rates(eigenvalues, sigma_c):
    """
    Compare how ζ_sem diverges vs how Riemann ζ diverges.

    Riemann ζ(s) ~ 1/(s-1) near s=1 (simple pole)
    Does ζ_sem(s) ~ 1/(s-σ_c)^β near σ_c?
    """
    results = {}

    epsilons = np.logspace(-3, -0.5, 20)  # 0.001 to ~0.3

    zeta_vals = []
    for eps in epsilons:
        z = spectral_zeta(eigenvalues, sigma_c + eps)
        zeta_vals.append(z)

    zeta_vals = np.array(zeta_vals)

    # Fit: log(ζ) = -β * log(ε) + const
    # If β ≈ 1, it's a simple pole like Riemann at s=1
    log_eps = np.log(epsilons)
    log_zeta = np.log(zeta_vals)

    # Linear fit
    valid = np.isfinite(log_zeta)
    if valid.sum() > 5:
        slope, intercept = np.polyfit(log_eps[valid], log_zeta[valid], 1)
        beta = -slope

        results['divergence_exponent_beta'] = float(beta)
        results['beta_close_to_1'] = abs(beta - 1) < 0.1

        # Residue would be exp(intercept) if β = 1
        results['estimated_residue'] = float(np.exp(intercept))

        # Compare residue to π-related values
        residue = np.exp(intercept)
        results['residue / pi'] = float(residue / np.pi)
        results['residue / pi^2'] = float(residue / (np.pi ** 2))
        results['residue * 6 / pi^2'] = float(residue * 6 / (np.pi ** 2))

    return results


def test_functional_equation(eigenvalues, sigma_c):
    """
    Test if ζ_sem has a functional equation like Riemann.

    Riemann: ξ(s) = ξ(1-s) where ξ(s) = π^(-s/2) Γ(s/2) ζ(s)

    Test: Is there a center c such that f(c+x) ∝ f(c-x)?
    """
    from scipy.special import gamma

    results = {}

    # Test different potential symmetry centers
    for center in [sigma_c/2, 1.0, sigma_c - 0.5]:
        ratios = []
        for offset in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            s1 = center + offset
            s2 = center - offset

            if s2 > 0.1:  # Avoid divergence
                z1 = spectral_zeta(eigenvalues, s1)
                z2 = spectral_zeta(eigenvalues, s2)

                if z1 > 0 and z2 > 0:
                    ratios.append(z1 / z2)

        if len(ratios) > 3:
            cv = np.std(ratios) / np.mean(ratios)
            results[f'center={center:.2f}'] = {
                'mean_ratio': float(np.mean(ratios)),
                'cv': float(cv),
                'is_symmetric': cv < 0.15
            }

    return results


# =============================================================================
# MAIN
# =============================================================================

def load_embeddings():
    """Load embeddings from sentence transformers."""
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
                embeddings[name] = embs
                print(f"  Loaded {name}: {embs.shape}")
            except Exception as ex:
                print(f"  Failed {name}: {ex}")
    except ImportError:
        print("  sentence-transformers not available")

    return embeddings


def get_eigenspectrum(embeddings):
    """Get eigenspectrum from embeddings."""
    vecs_centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(vecs_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvalues


def compute_alpha(eigenvalues):
    """Compute power law decay exponent α."""
    ev = eigenvalues[eigenvalues > 1e-10]
    k = np.arange(1, len(ev) + 1)

    # Fit in log-log space
    log_k = np.log(k)
    log_ev = np.log(ev)

    # Use first half for stable fit
    n = len(ev) // 2
    slope, _ = np.polyfit(log_k[:n], log_ev[:n], 1)

    return -slope


def main():
    print("=" * 70)
    print("RIEMANN π CONNECTION TEST")
    print("Does π appear in our spectral zeta at σ_c ≈ 2?")
    print("=" * 70)

    print("\nLoading embeddings...")
    embeddings = load_embeddings()

    if not embeddings:
        print("No embeddings available")
        return

    all_results = {}

    for name, embs in embeddings.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {name}")
        print(f"{'='*60}")

        eigenvalues = get_eigenspectrum(embs)
        alpha = compute_alpha(eigenvalues)
        sigma_c = 1 / alpha

        print(f"\nα = {alpha:.4f}")
        print(f"σ_c = 1/α = {sigma_c:.4f}")
        print(f"Distance from 2: {abs(sigma_c - 2):.4f} ({abs(sigma_c - 2)/2*100:.1f}%)")

        # Test 1: Analyze near critical point
        print("\n--- NEAR CRITICAL POINT ANALYSIS ---")
        near_crit = analyze_near_critical_point(eigenvalues, sigma_c)

        print(f"\nAs s → σ_c from above:")
        for v in near_crit['approach_from_above'][:4]:
            print(f"  s = {v['s']:.3f}: ζ_sem = {v['zeta']:.4f}, ζ_sem/ζ(2) = {v['zeta_times_6_over_pi_sq']:.4f}")

        print(f"\nResidue stable: {near_crit['residue_stable']}")
        print(f"Residue estimates: {[f'{r:.4f}' for r in near_crit['residue_estimates'][-3:]]}")

        # Test 2: Search for π
        print("\n--- SEARCHING FOR π ---")
        pi_results = search_for_pi(eigenvalues, sigma_c)

        if pi_results:
            print("Found potential π connections:")
            for k, v in pi_results.items():
                print(f"  {k}: {v:.6f}")
        else:
            print("No obvious π connections found in simple ratios")

        # Test 3: Compare divergence rates
        print("\n--- DIVERGENCE RATE ANALYSIS ---")
        div_results = compare_divergence_rates(eigenvalues, sigma_c)

        if 'divergence_exponent_beta' in div_results:
            beta = div_results['divergence_exponent_beta']
            print(f"Divergence exponent β = {beta:.4f}")
            print(f"  (β = 1 would be simple pole like Riemann at s=1)")
            print(f"  β close to 1: {div_results['beta_close_to_1']}")

            if 'residue * 6 / pi^2' in div_results:
                print(f"\nResidue analysis:")
                print(f"  Residue / π = {div_results['residue / pi']:.4f}")
                print(f"  Residue / π² = {div_results['residue / pi^2']:.4f}")
                print(f"  Residue × 6/π² = {div_results['residue * 6 / pi^2']:.4f}")

        # Test 4: Functional equation
        print("\n--- FUNCTIONAL EQUATION TEST ---")
        func_eq = test_functional_equation(eigenvalues, sigma_c)

        for center, data in func_eq.items():
            status = "SYMMETRIC" if data['is_symmetric'] else "not symmetric"
            print(f"  {center}: CV = {data['cv']:.4f} ({status})")

        # Store results
        all_results[name] = {
            'alpha': float(alpha),
            'sigma_c': float(sigma_c),
            'near_critical': near_crit,
            'pi_search': pi_results,
            'divergence': div_results,
            'functional_equation': func_eq,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Does π appear?")
    print("=" * 70)

    pi_found = False
    pi_evidence = []

    for name, results in all_results.items():
        if results['pi_search']:
            pi_found = True
            pi_evidence.append(f"{name}: {list(results['pi_search'].keys())}")

        # Check if any ratio is close to 1 (exact match)
        if 'divergence' in results and 'residue * 6 / pi^2' in results['divergence']:
            val = results['divergence']['residue * 6 / pi^2']
            if 0.9 < val < 1.1:
                pi_found = True
                pi_evidence.append(f"{name}: Residue × 6/π² = {val:.4f} ≈ 1")

    if pi_found:
        print("\n*** π CONNECTIONS FOUND ***")
        for ev in pi_evidence:
            print(f"  {ev}")
    else:
        print("\n*** NO STRONG π CONNECTION ***")
        print("The spectral zeta diverges near σ_c ≈ 2, but")
        print("the coefficients don't obviously involve π.")
        print("\nThis suggests the connection is STRUCTURAL")
        print("(α ≈ 1/2) rather than NUMERICAL (coefficients with π).")

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    receipt = {
        'test': 'RIEMANN_PI_CONNECTION',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'question': 'Does π appear in spectral zeta at σ_c ≈ 2?',
        'answer': 'PARTIAL' if pi_found else 'NO',
        'pi_evidence': pi_evidence,
        'results': all_results,
    }

    path = results_dir / f'riemann_pi_connection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(receipt, f, indent=2, default=str)

    print(f"\nReceipt saved: {path}")


if __name__ == '__main__':
    main()
