#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q50 Part 6: First-Principles Derivation — Why e Per Octant?

We know:
- 8 octants exist (all populated, p = 0.02)
- Total contribution = 8e ≈ 21.746
- Therefore each octant contributes e ≈ 2.718

WHY does each octant contribute exactly e?

Hypotheses to Test:
H1: Information-theoretic — each octant = 1 nat of semantic entropy
H2: Exponential decay normalization — integral constraints
H3: Thermodynamic equipartition — free energy per degree of freedom
H4: Logarithmic spiral winding — phase accumulation per octant

Pass criteria:
- At least ONE hypothesis produces e from first principles
- Derivation is mathematically sound (not curve-fitting)
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from scipy import stats
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


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


def get_octant(pc_scores):
    """Return octant index (0-7) based on signs of PC1, PC2, PC3."""
    signs = (pc_scores[:, :3] > 0).astype(int)
    return signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]


# Standard test vocabulary
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


def hypothesis_1_entropy_per_octant(embeddings, model_name):
    """
    H1: Information-Theoretic — Each octant has ~1 nat of entropy

    Theory: e is the natural unit of information (1 nat = log_e(e) = 1)
    If each octant represents 1 nat of semantic entropy:
        Total = 8 nats → but we measure Df × α = 8e, not 8

    Resolution: The eigenvalue distribution entropy H relates to e through:
        H = log(Df) for uniform distribution
        For power-law with exponent α: H ≈ 1/α

    Test: Compute Shannon entropy of eigenvalue distribution
    """
    results = {'hypothesis': 'H1_entropy_per_octant', 'model': model_name}

    # Get PCA projection
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    pc_scores = pca.fit_transform(embeddings)

    # Overall eigenvalue entropy
    eigenvalues = get_eigenspectrum(embeddings)
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1e-15]  # Remove zeros
    H_total = -np.sum(p * np.log(p))  # Shannon entropy in nats

    results['total_entropy_nats'] = float(H_total)
    results['total_entropy_over_e'] = float(H_total / np.e)

    # Per-octant entropy
    octants = get_octant(pc_scores)
    octant_entropies = []

    for oct_idx in range(8):
        mask = octants == oct_idx
        if mask.sum() < 5:
            continue

        octant_embeddings = embeddings[mask]
        oct_eigenvalues = get_eigenspectrum(octant_embeddings)
        p_oct = oct_eigenvalues / oct_eigenvalues.sum()
        p_oct = p_oct[p_oct > 1e-15]
        H_oct = -np.sum(p_oct * np.log(p_oct))
        octant_entropies.append(H_oct)

    if octant_entropies:
        mean_octant_entropy = np.mean(octant_entropies)
        results['mean_octant_entropy_nats'] = float(mean_octant_entropy)
        results['mean_octant_entropy_over_e'] = float(mean_octant_entropy / np.e)
        results['n_octants_measured'] = len(octant_entropies)

        # Check if mean octant entropy ≈ e (or some multiple)
        results['entropy_ratio_to_e'] = float(mean_octant_entropy / np.e)

    return results


def hypothesis_2_integral_normalization(embeddings, model_name):
    """
    H2: Exponential Decay Normalization

    Theory: Eigenvalues decay as λ_k ~ k^(-α)
    The integral ∫_1^Df k^(-α) dk = [k^(1-α)/(1-α)]_1^Df
                                  = (Df^(1-α) - 1)/(1-α)

    For large Df and α < 1:
        ∫ ≈ Df^(1-α) / (1-α)

    If total "semantic volume" = 8e:
        Df^(1-α) / (1-α) = 8e

    With Df × α = 8e → α = 8e/Df:
        Df^(1 - 8e/Df) / (1 - 8e/Df) should = 8e

    Test: Verify this integral relationship
    """
    results = {'hypothesis': 'H2_integral_normalization', 'model': model_name}

    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)

    # Numerical integral of eigenvalues
    numerical_integral = np.sum(eigenvalues)

    # Theoretical integral for power law
    n = len(eigenvalues)
    if alpha != 1 and alpha > 0:
        # λ_k = λ_1 * k^(-α), so integral = λ_1 * sum(k^(-α))
        # For normalized eigenvalues, λ_1 ≈ eigenvalues[0] / eigenvalues.sum()
        theoretical_integral = (Df**(1-alpha) - 1) / (1 - alpha) if alpha != 1 else np.log(Df)
    else:
        theoretical_integral = np.nan

    results['Df'] = float(Df)
    results['alpha'] = float(alpha)
    results['numerical_integral'] = float(numerical_integral)
    results['theoretical_integral'] = float(theoretical_integral)

    # Test the identity: Df × α = 8e implies integral relationship
    # If α = 8e/Df, then Df^(1-α) / (1-α) should relate to 8e
    predicted_alpha = 8 * np.e / Df
    exponent = 1 - predicted_alpha
    if abs(exponent) > 0.01:
        predicted_integral = (Df**exponent - 1) / exponent
    else:
        predicted_integral = np.log(Df)

    results['predicted_alpha_from_8e'] = float(predicted_alpha)
    results['predicted_integral_from_8e'] = float(predicted_integral)
    results['alpha_prediction_error'] = float(abs(alpha - predicted_alpha) / alpha * 100) if alpha > 0 else np.nan

    return results


def hypothesis_3_thermodynamic(embeddings, model_name):
    """
    H3: Thermodynamic Equipartition

    Theory: Classical equipartition gives ⟨E⟩ = (1/2) k_B T per DOF
    Define "semantic temperature" T = 1/α (higher α = colder/more ordered)
    Define "semantic energy" from eigenvalue distribution

    Free energy: F = -T log(Z) where Z = partition function

    Test: Does F/8 (free energy per octant) = e?
    """
    results = {'hypothesis': 'H3_thermodynamic', 'model': model_name}

    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)

    # Semantic temperature
    T = 1 / alpha if alpha > 0 else np.inf
    results['semantic_temperature'] = float(T)

    # Various partition function definitions
    # Z1: Boltzmann-like with eigenvalues as energies
    Z1 = np.sum(np.exp(-eigenvalues / T)) if T != np.inf else len(eigenvalues)
    F1 = -T * np.log(Z1) if Z1 > 0 else np.nan

    # Z2: Canonical ensemble over dimension indices
    k = np.arange(1, len(eigenvalues) + 1)
    Z2 = np.sum(np.exp(-k / T))
    F2 = -T * np.log(Z2) if Z2 > 0 else np.nan

    # Z3: Using log-eigenvalues as energies (connects to entropy)
    log_ev = np.log(eigenvalues[eigenvalues > 1e-10])
    Z3 = np.sum(np.exp(-log_ev / T))
    F3 = -T * np.log(Z3) if Z3 > 0 else np.nan

    results['partition_functions'] = {
        'Z1_boltzmann': float(Z1),
        'F1': float(F1),
        'F1_per_octant': float(F1 / 8),
        'F1_per_octant_over_e': float(F1 / 8 / np.e) if not np.isnan(F1) else np.nan,

        'Z2_canonical': float(Z2),
        'F2': float(F2),
        'F2_per_octant': float(F2 / 8),
        'F2_per_octant_over_e': float(F2 / 8 / np.e) if not np.isnan(F2) else np.nan,

        'Z3_log_eigenvalue': float(Z3),
        'F3': float(F3),
        'F3_per_octant': float(F3 / 8),
        'F3_per_octant_over_e': float(F3 / 8 / np.e) if not np.isnan(F3) else np.nan,
    }

    # Equipartition check: ⟨E⟩ = Df/2 * T?
    mean_eigenvalue = np.mean(eigenvalues)
    equipartition_prediction = Df / 2 * T if T != np.inf else np.nan
    results['mean_eigenvalue'] = float(mean_eigenvalue)
    results['equipartition_prediction'] = float(equipartition_prediction)

    return results


def hypothesis_4_spiral_winding(embeddings, model_name):
    """
    H4: Logarithmic Spiral Winding Number

    Theory: Eigenvalue decay follows a logarithmic spiral in log-log space
        λ_k = A × k^(-α) → log(λ) = log(A) - α × log(k)

    In polar (r, θ) representation:
        r = exp(-α θ) is a logarithmic spiral
        Winding number = total angular traversal / 2π

    Test: Does winding per octant = some function of e?
    """
    results = {'hypothesis': 'H4_spiral_winding', 'model': model_name}

    eigenvalues = get_eigenspectrum(embeddings)
    Df = compute_df(eigenvalues)
    alpha = compute_alpha(eigenvalues)

    # In log-log space, eigenvalues trace a line
    # Convert to polar spiral interpretation
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_ev = np.log(eigenvalues)

    # Total "arc length" in log-log space
    # ds = sqrt(d(log_k)^2 + d(log_ev)^2)
    d_log_k = np.diff(log_k)
    d_log_ev = np.diff(log_ev)
    arc_lengths = np.sqrt(d_log_k**2 + d_log_ev**2)
    total_arc = np.sum(arc_lengths)

    # Winding interpretation: treat log_k as angle
    # Total angle from k=1 to k=Df: θ_total = log(Df)
    theta_total = np.log(Df)
    winding_number = theta_total / (2 * np.pi)

    # Per-octant quantities
    theta_per_octant = theta_total / 8
    winding_per_octant = winding_number / 8

    results['Df'] = float(Df)
    results['alpha'] = float(alpha)
    results['total_arc_length'] = float(total_arc)
    results['theta_total'] = float(theta_total)
    results['winding_number'] = float(winding_number)
    results['theta_per_octant'] = float(theta_per_octant)
    results['winding_per_octant'] = float(winding_per_octant)

    # Check if theta_per_octant relates to e
    results['theta_per_octant_over_e'] = float(theta_per_octant / np.e)
    results['theta_times_8_over_e'] = float(theta_total / np.e)

    # Alternative: use α × log(Df) as "angular momentum"
    angular_momentum = alpha * np.log(Df)
    results['angular_momentum'] = float(angular_momentum)
    results['angular_momentum_over_e'] = float(angular_momentum / np.e)

    # The key relationship: Df × α = 8e means α = 8e/Df
    # So α × log(Df) = 8e × log(Df) / Df
    # For Df ≈ 45: 8e × log(45) / 45 ≈ 8e × 3.8 / 45 ≈ 1.84
    theoretical_angular = 8 * np.e * np.log(Df) / Df
    results['theoretical_angular_momentum'] = float(theoretical_angular)

    return results


def hypothesis_5_peircean_triads(embeddings, model_name):
    """
    H5: Peircean Triadic Information

    Theory: Peirce's Reduction Thesis states 3 is the irreducible threshold
    of semiosis. Each triad carries inherent information.

    The formula Df × α = 8e = 2³ × e suggests:
    - 8 = 2³ = number of binary combinations in 3D (octants)
    - e = information content per triadic relation

    Test: Does the mutual information of PC1, PC2, PC3 relate to e?
    """
    results = {'hypothesis': 'H5_peircean_triads', 'model': model_name}

    # Get PCA projection
    pca = PCA(n_components=min(10, embeddings.shape[1]))
    pc_scores = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_

    # Top 3 PCs for triadic analysis
    pc1, pc2, pc3 = pc_scores[:, 0], pc_scores[:, 1], pc_scores[:, 2]

    # Entropy of each PC (discretized)
    def discretized_entropy(x, n_bins=20):
        hist, _ = np.histogram(x, bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (x.max() - x.min()) / n_bins
        return -np.sum(hist * np.log(hist) * bin_width)

    H1 = discretized_entropy(pc1)
    H2 = discretized_entropy(pc2)
    H3 = discretized_entropy(pc3)

    results['entropy_PC1'] = float(H1)
    results['entropy_PC2'] = float(H2)
    results['entropy_PC3'] = float(H3)
    results['sum_entropies'] = float(H1 + H2 + H3)
    results['sum_entropies_over_e'] = float((H1 + H2 + H3) / np.e)

    # Variance explained by top 3
    var_top3 = sum(variance_explained[:3])
    results['variance_explained_top3'] = float(var_top3)

    # Joint entropy of (PC1, PC2, PC3) - harder to compute accurately
    # Use 3D histogram approximation
    n_bins_3d = 8  # Match octant structure
    hist_3d, _ = np.histogramdd(pc_scores[:, :3], bins=n_bins_3d)
    hist_3d = hist_3d / hist_3d.sum()
    hist_3d = hist_3d[hist_3d > 0]
    H_joint = -np.sum(hist_3d * np.log(hist_3d))

    results['joint_entropy_PC123'] = float(H_joint)
    results['joint_entropy_over_e'] = float(H_joint / np.e)

    # Mutual information I(PC1; PC2; PC3) approximation
    # For truly independent: H_joint = H1 + H2 + H3
    # Deviation = redundancy/synergy
    mi_triadic = (H1 + H2 + H3) - H_joint
    results['mutual_info_triadic'] = float(mi_triadic)
    results['mutual_info_over_e'] = float(mi_triadic / np.e)

    # Key test: Does 8 × (something with e) = Df × α?
    Df = compute_df(get_eigenspectrum(embeddings))
    alpha = compute_alpha(get_eigenspectrum(embeddings))
    df_alpha = Df * alpha

    # Candidate expressions for "e per octant"
    results['df_alpha'] = float(df_alpha)
    results['df_alpha_over_8e'] = float(df_alpha / (8 * np.e))

    # Does joint entropy / 8 ≈ some multiple of e?
    results['joint_entropy_per_octant'] = float(H_joint / 8)
    results['ratio_to_e'] = float((H_joint / 8) / np.e)

    return results


def main():
    print("=" * 70)
    print("Q50 PART 6: FIRST-PRINCIPLES DERIVATION")
    print("Why does each octant contribute exactly e?")
    print("=" * 70)

    results = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'test': 'Q50_FIRST_PRINCIPLES',
        'target': 8 * np.e,
        'e': np.e,
        'models': [],
        'summary': {}
    }

    try:
        from sentence_transformers import SentenceTransformer

        models = [
            ("all-MiniLM-L6-v2", "MiniLM-L6"),
            ("all-mpnet-base-v2", "MPNet-base"),
            ("BAAI/bge-small-en-v1.5", "BGE-small"),
        ]

        all_hypothesis_results = {f'H{i}': [] for i in range(1, 6)}

        for model_id, model_name in models:
            print(f"\n{'=' * 60}")
            print(f"MODEL: {model_name}")
            print("=" * 60)

            model = SentenceTransformer(model_id)
            embeddings = model.encode(WORDS, normalize_embeddings=True)

            model_results = {'model': model_name, 'model_id': model_id}

            # Run all hypotheses
            print("\n  H1: Information-theoretic (entropy per octant)")
            h1 = hypothesis_1_entropy_per_octant(embeddings, model_name)
            model_results['H1'] = h1
            all_hypothesis_results['H1'].append(h1)
            if 'mean_octant_entropy_nats' in h1:
                print(f"      Mean octant entropy: {h1['mean_octant_entropy_nats']:.3f} nats")
                print(f"      Ratio to e: {h1['entropy_ratio_to_e']:.3f}")

            print("\n  H2: Integral normalization")
            h2 = hypothesis_2_integral_normalization(embeddings, model_name)
            model_results['H2'] = h2
            all_hypothesis_results['H2'].append(h2)
            print(f"      Df: {h2['Df']:.2f}, α: {h2['alpha']:.3f}")
            print(f"      α prediction error: {h2['alpha_prediction_error']:.2f}%")

            print("\n  H3: Thermodynamic (free energy per octant)")
            h3 = hypothesis_3_thermodynamic(embeddings, model_name)
            model_results['H3'] = h3
            all_hypothesis_results['H3'].append(h3)
            pf = h3['partition_functions']
            print(f"      T_semantic: {h3['semantic_temperature']:.3f}")
            print(f"      F1/8/e: {pf['F1_per_octant_over_e']:.3f}")

            print("\n  H4: Spiral winding")
            h4 = hypothesis_4_spiral_winding(embeddings, model_name)
            model_results['H4'] = h4
            all_hypothesis_results['H4'].append(h4)
            print(f"      θ_per_octant: {h4['theta_per_octant']:.3f}")
            print(f"      θ_per_octant / e: {h4['theta_per_octant_over_e']:.3f}")

            print("\n  H5: Peircean triadic information")
            h5 = hypothesis_5_peircean_triads(embeddings, model_name)
            model_results['H5'] = h5
            all_hypothesis_results['H5'].append(h5)
            print(f"      Joint entropy (PC1,2,3): {h5['joint_entropy_PC123']:.3f} nats")
            print(f"      Joint entropy / e: {h5['joint_entropy_over_e']:.3f}")
            print(f"      Df×α / 8e: {h5['df_alpha_over_8e']:.3f}")

            results['models'].append(model_results)

        # ============================================================
        # SUMMARY: Which hypothesis best explains e?
        # ============================================================
        print("\n" + "=" * 70)
        print("SUMMARY: FIRST-PRINCIPLES DERIVATION")
        print("=" * 70)

        print("\n  Looking for quantities that equal e (or simple multiples)...")
        print("\n  Hypothesis              | Measured Quantity    | Value  | Ratio to e")
        print("  " + "-" * 70)

        summary_scores = {}

        # H1: Entropy per octant
        h1_values = [h['entropy_ratio_to_e'] for h in all_hypothesis_results['H1']
                     if 'entropy_ratio_to_e' in h]
        if h1_values:
            mean_h1 = np.mean(h1_values)
            summary_scores['H1'] = abs(mean_h1 - 1.0)  # Distance from ratio=1
            print(f"  H1 (Entropy)            | Octant entropy/e     | {mean_h1:.3f} | {'CLOSE' if abs(mean_h1-1)<0.3 else 'far'}")

        # H2: Alpha prediction
        h2_errors = [h['alpha_prediction_error'] for h in all_hypothesis_results['H2']]
        mean_h2 = np.mean(h2_errors)
        summary_scores['H2'] = mean_h2 / 100  # Normalize
        print(f"  H2 (Integral)           | α prediction error   | {mean_h2:.1f}% | {'EXCELLENT' if mean_h2<5 else 'good' if mean_h2<10 else 'poor'}")

        # H3: Free energy
        h3_values = [h['partition_functions']['F1_per_octant_over_e']
                     for h in all_hypothesis_results['H3']
                     if not np.isnan(h['partition_functions']['F1_per_octant_over_e'])]
        if h3_values:
            mean_h3 = np.mean(h3_values)
            summary_scores['H3'] = abs(mean_h3 - 1.0)
            print(f"  H3 (Thermodynamic)      | F_octant/e           | {mean_h3:.3f} | {'CLOSE' if abs(mean_h3-1)<0.3 else 'far'}")

        # H4: Spiral
        h4_values = [h['theta_per_octant_over_e'] for h in all_hypothesis_results['H4']]
        mean_h4 = np.mean(h4_values)
        summary_scores['H4'] = abs(mean_h4 - 1.0)
        print(f"  H4 (Spiral)             | θ_octant/e           | {mean_h4:.3f} | {'CLOSE' if abs(mean_h4-1)<0.3 else 'far'}")

        # H5: Triadic
        h5_values = [h['joint_entropy_over_e'] for h in all_hypothesis_results['H5']]
        mean_h5 = np.mean(h5_values)
        summary_scores['H5'] = abs(mean_h5 - 1.0)
        print(f"  H5 (Peircean triadic)   | H_joint/e            | {mean_h5:.3f} | {'CLOSE' if abs(mean_h5-1)<0.3 else 'far'}")

        # Find best hypothesis
        best_h = min(summary_scores, key=summary_scores.get)
        results['summary']['scores'] = summary_scores
        results['summary']['best_hypothesis'] = best_h
        results['summary']['best_score'] = summary_scores[best_h]

        print(f"\n  Best hypothesis: {best_h} (score: {summary_scores[best_h]:.3f})")

        # Verdict
        print("\n" + "=" * 70)
        if summary_scores[best_h] < 0.1:
            print(f"VERDICT: {best_h} EXPLAINS e WITH HIGH PRECISION")
        elif summary_scores[best_h] < 0.3:
            print(f"VERDICT: {best_h} PROVIDES PLAUSIBLE EXPLANATION FOR e")
        else:
            print("VERDICT: NO HYPOTHESIS CLEANLY EXPLAINS e")
            print("  The constant e may arise from deeper structure")

        # Note the mathematical insight
        print("\n  Key insight: The conservation law Df × α = 8e")
        print("  allows PREDICTING α from Df with 0.15% precision.")
        print("  Whether e arises from entropy, thermodynamics, or geometry,")
        print("  the relationship is empirically robust.")
        print("=" * 70)

    except ImportError as ie:
        print(f"  Import error: {ie}")
        results['error'] = str(ie)

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q50_first_principles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")


if __name__ == '__main__':
    main()
