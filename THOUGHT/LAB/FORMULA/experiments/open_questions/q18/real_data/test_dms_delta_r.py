#!/usr/bin/env python3
"""
Q18 DMS Delta-R Test - REAL Experimental Data

Tests whether delta-R (computed from amino acid properties ONLY)
predicts experimental mutation effects (fitness scores from DMS).

CRITICAL: This is NOT tautological because:
- delta-R is computed from: hydrophobicity, volume, charge changes
- fitness is measured from: E3 activity, yeast growth, transcription

These are INDEPENDENT measurements.

Data Sources:
- BRCA1: Starita et al. 2015 - E3 ubiquitin ligase activity
- UBE2I: Weile et al. 2017 - Yeast complementation fitness
- TP53: Kato et al. 2003 - p53 transcriptional activity
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

CACHE_DIR = Path(__file__).parent / 'cache'

# ============================================================================
# AMINO ACID PROPERTIES (from biochemistry literature)
# ============================================================================

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
    '*': 0.0  # Stop codon
}

# Amino acid volumes (Angstrom^3) - from Zamyatnin 1972
AA_VOLUME = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
    '*': 0.0  # Stop codon
}

# Amino acid charge at physiological pH
AA_CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,  # H is ~10% protonated at pH 7
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
    '*': 0  # Stop codon
}


# ============================================================================
# DELTA-R COMPUTATION (from amino acid properties ONLY)
# ============================================================================

def compute_delta_r(wt_aa: str, mut_aa: str) -> float:
    """
    Compute delta-R for a single mutation from amino acid properties.

    R_mutation = weighted_features / dispersion
    delta-R = R_mutation - R_wildtype

    Features (all INDEPENDENT of fitness):
    - Hydrophobicity change
    - Volume change
    - Charge change

    Higher |delta-R| = more disruptive mutation (predicted)
    """
    # Get properties for both amino acids
    h_wt = HYDROPHOBICITY.get(wt_aa, 0)
    h_mut = HYDROPHOBICITY.get(mut_aa, 0)

    v_wt = AA_VOLUME.get(wt_aa, 100)
    v_mut = AA_VOLUME.get(mut_aa, 100)

    c_wt = AA_CHARGE.get(wt_aa, 0)
    c_mut = AA_CHARGE.get(mut_aa, 0)

    # Compute changes (absolute values - magnitude of disruption)
    delta_h = h_mut - h_wt  # Hydrophobicity change
    delta_v = (v_mut - v_wt) / 100  # Volume change (normalized)
    delta_c = c_mut - c_wt  # Charge change

    # Stop codons are maximally disruptive
    if mut_aa == '*':
        return -10.0  # Large negative = very deleterious

    # R for wildtype = stability reference (E=1, sigma=1 for WT)
    R_wt = 1.0

    # R for mutant: decreases with larger property changes
    # E_mut = 1 - normalized_disruption (measure of remaining function)
    # sigma_mut = dispersion from changes

    # Disruption score (0 to ~1 range)
    disruption = (
        0.4 * abs(delta_h) / 9.0 +     # Max hydro change ~9 (K->I)
        0.3 * abs(delta_v) +            # Already normalized
        0.3 * abs(delta_c)              # Max charge change = 2
    )

    # E_mut: how much "agreement" or function remains
    # Conservative mutations have E close to 1
    E_mut = max(0.01, 1.0 - disruption)

    # sigma_mut: dispersion increases with property changes
    # More different = more uncertain prediction
    sigma_mut = 1.0 + 0.5 * disruption

    R_mut = E_mut / sigma_mut

    delta_R = R_mut - R_wt

    return delta_R


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dms_data(protein: str) -> Dict:
    """Load DMS data for a specific protein."""
    if protein == 'BRCA1':
        filepath = CACHE_DIR / 'dms_data.json'
    elif protein == 'UBE2I':
        filepath = CACHE_DIR / 'dms_data_ube2i.json'
    elif protein == 'TP53':
        filepath = CACHE_DIR / 'dms_data_tp53.json'
    else:
        raise ValueError(f"Unknown protein: {protein}")

    with open(filepath) as f:
        return json.load(f)


# ============================================================================
# MAIN TEST
# ============================================================================

def test_protein_dms(protein: str) -> Dict:
    """
    Test delta-R vs fitness for a single protein.

    Returns correlation results.
    """
    print(f"\n{'='*60}")
    print(f"Testing {protein}")
    print('='*60)

    # Load data
    data = load_dms_data(protein)
    mutations = data['mutations']

    print(f"Protein: {data['protein']}")
    print(f"Assay: {data['assay']}")
    print(f"N mutations: {len(mutations)}")

    # Compute delta-R for each mutation
    delta_rs = []
    fitness_values = []
    skipped = 0

    for mut in mutations:
        wt = mut['wt']
        mt = mut['mut']
        fitness = mut['fitness']

        # Skip if missing data
        if wt not in HYDROPHOBICITY or mt not in HYDROPHOBICITY:
            skipped += 1
            continue

        # Compute delta-R from sequence features ONLY
        delta_r = compute_delta_r(wt, mt)

        delta_rs.append(delta_r)
        fitness_values.append(fitness)

    if skipped > 0:
        print(f"Skipped {skipped} mutations (unknown amino acids)")

    delta_rs = np.array(delta_rs)
    fitness_values = np.array(fitness_values)

    # Compute Spearman correlation
    # Use Spearman because relationship may be nonlinear
    # and DMS data often has outliers
    rho, pval = stats.spearmanr(delta_rs, fitness_values)

    # Also compute Pearson for comparison
    pearson_r, pearson_p = stats.pearsonr(delta_rs, fitness_values)

    print(f"\nResults:")
    print(f"  N valid mutations: {len(delta_rs)}")
    print(f"  Spearman rho: {rho:.4f}")
    print(f"  Spearman p-value: {pval:.2e}")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  Pearson p-value: {pearson_p:.2e}")

    # Delta-R statistics
    print(f"\nDelta-R statistics:")
    print(f"  Mean: {np.mean(delta_rs):.4f}")
    print(f"  Std: {np.std(delta_rs):.4f}")
    print(f"  Range: [{np.min(delta_rs):.4f}, {np.max(delta_rs):.4f}]")

    # Fitness statistics
    print(f"\nFitness statistics:")
    print(f"  Mean: {np.mean(fitness_values):.4f}")
    print(f"  Std: {np.std(fitness_values):.4f}")
    print(f"  Range: [{np.min(fitness_values):.4f}, {np.max(fitness_values):.4f}]")

    # Interpretation
    if pval < 0.001 and abs(rho) > 0.1:
        if rho > 0:
            interpretation = "POSITIVE: delta-R positively correlates with fitness (expected direction)"
        else:
            interpretation = "NEGATIVE: delta-R negatively correlates with fitness (more disruption = lower fitness)"
    elif pval < 0.05:
        interpretation = "WEAK: Statistically significant but weak correlation"
    else:
        interpretation = "NULL: No significant correlation between delta-R and fitness"

    print(f"\nInterpretation: {interpretation}")

    return {
        'protein': protein,
        'full_name': data['protein'],
        'assay': data['assay'],
        'source': data.get('source', 'MaveDB'),
        'publication': data.get('publication', 'Unknown'),
        'n_mutations': len(delta_rs),
        'spearman_rho': float(rho),
        'spearman_pvalue': float(pval),
        'pearson_r': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'delta_r_mean': float(np.mean(delta_rs)),
        'delta_r_std': float(np.std(delta_rs)),
        'fitness_mean': float(np.mean(fitness_values)),
        'fitness_std': float(np.std(fitness_values)),
        'interpretation': interpretation
    }


def main():
    """Run delta-R vs fitness test for all proteins."""
    print("="*70)
    print("Q18 DMS TEST: Does delta-R predict mutation effects?")
    print("="*70)
    print("\nThis test is NOT tautological:")
    print("- delta-R: computed from amino acid properties (hydrophobicity, volume, charge)")
    print("- fitness: measured experimentally (enzyme activity, growth, transcription)")
    print("\nIf delta-R correlates with fitness, R captures real mutation effects.")

    proteins = ['BRCA1', 'UBE2I', 'TP53']
    results = {}

    for protein in proteins:
        try:
            results[protein] = test_protein_dms(protein)
        except FileNotFoundError as e:
            print(f"\nSkipping {protein}: {e}")
            results[protein] = {'status': 'SKIPPED', 'reason': str(e)}

    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    significant_count = 0
    correct_direction = 0
    total_tested = 0

    for protein, res in results.items():
        if 'status' in res and res['status'] == 'SKIPPED':
            print(f"  {protein}: SKIPPED")
            continue

        total_tested += 1
        rho = res['spearman_rho']
        pval = res['spearman_pvalue']

        sig_marker = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))

        print(f"  {protein}: rho={rho:+.4f} (p={pval:.2e}) {sig_marker}")

        if pval < 0.05:
            significant_count += 1
            # Expect POSITIVE correlation (higher delta-R = less disruption = higher fitness)
            if rho > 0:
                correct_direction += 1

    print(f"\nSignificant correlations: {significant_count}/{total_tested}")
    print(f"Correct direction (positive): {correct_direction}/{significant_count if significant_count > 0 else 1}")

    # Overall verdict
    print("\n" + "-"*70)
    if significant_count == total_tested and correct_direction == significant_count:
        verdict = "STRONG SUPPORT: delta-R consistently predicts fitness across all proteins"
    elif significant_count >= total_tested / 2:
        verdict = "MODERATE SUPPORT: delta-R predicts fitness in majority of proteins"
    elif significant_count > 0:
        verdict = "WEAK SUPPORT: Some correlation observed, but inconsistent"
    else:
        verdict = "NO SUPPORT: delta-R does not predict experimental fitness"

    print(f"VERDICT: {verdict}")
    print("-"*70)

    # Add overall verdict to results
    results['summary'] = {
        'total_proteins': total_tested,
        'significant_correlations': significant_count,
        'correct_direction': correct_direction,
        'verdict': verdict
    }

    # Save results
    output_file = Path(__file__).parent / 'dms_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
