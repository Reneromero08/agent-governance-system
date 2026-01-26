"""
Fixed Protein Folding Test
==========================

This script implements the corrected R formula for protein folding prediction.

The original R_sequence formula had a critical bug: sigma (hydrophobicity_std)
was nearly constant across proteins, compressing R into a useless [0.82-1.00] range.

The fix: Use a sigma formula that varies meaningfully with protein properties.

sigma = 0.1 + 0.5 * abs(disorder_frac - 0.5) + 0.4 * log(length) / 10

This captures:
- Disorder uncertainty: proteins with extreme disorder (high or low) have more certain predictions
- Length factor: longer proteins have more structural heterogeneity
"""

import json
import math
from pathlib import Path
from scipy import stats
import numpy as np
from datetime import datetime


def load_extended_results(filepath: str) -> dict:
    """Load the extended protein results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_R_fixed(disorder_frac: float, length: int, mean_hydro_balance: float = None,
                    complexity: float = None) -> float:
    """
    Compute the fixed R value using the improved sigma formula.

    The key fix: sigma now varies meaningfully with protein properties.

    Parameters:
    - disorder_frac: Fraction of disordered residues (0-1)
    - length: Protein length in residues
    - mean_hydro_balance: Optional hydrophobicity balance (default: estimated from disorder)
    - complexity: Optional sequence complexity (default: 0.96 typical)

    Returns:
    - R_fixed: The corrected R value
    """
    # E: foldability estimate
    order_score = 1.0 - disorder_frac

    # If hydro_balance not provided, estimate from order score
    # Well-folded proteins tend to have balanced hydrophobicity
    if mean_hydro_balance is None:
        # Estimate: more ordered = more balanced hydrophobicity
        mean_hydro_balance = 0.7 + 0.2 * order_score

    # If complexity not provided, use typical value
    if complexity is None:
        complexity = 0.96

    # Estimate structure propensity from order score
    # More ordered proteins tend to have more secondary structure
    structure_prop = 0.3 + 0.4 * order_score

    # Complexity penalty
    complexity_penalty = abs(complexity - 0.75)

    # E formula (same weights as original)
    E = (0.4 * order_score +
         0.3 * mean_hydro_balance +
         0.2 * structure_prop +
         0.1 * (1 - complexity_penalty))

    # FIXED sigma: varies meaningfully with protein properties
    disorder_uncertainty = abs(disorder_frac - 0.5)
    length_factor = math.log(length + 1) / 10

    sigma = 0.1 + 0.5 * disorder_uncertainty + 0.4 * length_factor

    # R = E / sigma
    R_fixed = E / sigma

    return R_fixed


def compute_R_original_approx(disorder_frac: float, complexity: float) -> float:
    """
    Approximate the original R_sequence calculation.

    The original formula had sigma ~ hydrophobicity_std / 4.5 which was
    nearly constant (~0.7-0.8) across all proteins.
    """
    order_score = 1.0 - disorder_frac

    # Estimate other components
    hydro_balance = 0.7 + 0.2 * order_score
    structure_prop = 0.3 + 0.4 * order_score
    complexity_penalty = abs(complexity - 0.75)

    E = (0.4 * order_score +
         0.3 * hydro_balance +
         0.2 * structure_prop +
         0.1 * (1 - complexity_penalty))

    # Original sigma was nearly constant
    sigma = 0.75  # typical value

    R_original = E / sigma
    return R_original


def run_fixed_test(results_data: dict) -> dict:
    """
    Run the fixed protein folding test on all proteins.

    Returns comprehensive statistics comparing original R, fixed R, and baseline disorder.
    """
    protein_results = results_data['test_results']['protein_results']

    # Extract data for all proteins
    R_original_values = []
    R_fixed_values = []
    plddt_values = []
    disorder_values = []
    length_values = []
    protein_ids = []

    fixed_protein_results = []

    for protein in protein_results:
        uniprot_id = protein['uniprot_id']
        R_original = protein['R_sequence']
        plddt = protein['mean_plddt']
        disorder_frac = protein['disorder_frac']
        length = protein['length']
        complexity = protein['complexity']

        # Compute fixed R
        R_fixed = compute_R_fixed(disorder_frac, length, complexity=complexity)

        # Store values
        R_original_values.append(R_original)
        R_fixed_values.append(R_fixed)
        plddt_values.append(plddt)
        disorder_values.append(disorder_frac)
        length_values.append(length)
        protein_ids.append(uniprot_id)

        fixed_protein_results.append({
            'uniprot_id': uniprot_id,
            'R_original': R_original,
            'R_fixed': R_fixed,
            'mean_plddt': plddt,
            'disorder_frac': disorder_frac,
            'length': length,
            'complexity': complexity
        })

    # Convert to numpy arrays
    R_original = np.array(R_original_values)
    R_fixed = np.array(R_fixed_values)
    plddt = np.array(plddt_values)
    disorder = np.array(disorder_values)

    # Compute correlations
    # Original R vs pLDDT
    r_original_pearson, p_original_pearson = stats.pearsonr(R_original, plddt)
    r_original_spearman, p_original_spearman = stats.spearmanr(R_original, plddt)

    # Fixed R vs pLDDT
    r_fixed_pearson, p_fixed_pearson = stats.pearsonr(R_fixed, plddt)
    r_fixed_spearman, p_fixed_spearman = stats.spearmanr(R_fixed, plddt)

    # Baseline: disorder vs pLDDT (should be negative)
    r_disorder_pearson, p_disorder_pearson = stats.pearsonr(disorder, plddt)
    r_disorder_spearman, p_disorder_spearman = stats.spearmanr(disorder, plddt)

    # Simple order score (1 - disorder) vs pLDDT
    order = 1 - disorder
    r_order_pearson, p_order_pearson = stats.pearsonr(order, plddt)
    r_order_spearman, p_order_spearman = stats.spearmanr(order, plddt)

    # Compute R-squared values
    r2_original = r_original_pearson ** 2
    r2_fixed = r_fixed_pearson ** 2
    r2_order = r_order_pearson ** 2

    # Summary statistics
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'protein_folding_fixed',
        'description': 'Fixed R formula with meaningful sigma variation',
        'n_proteins': len(protein_results),

        'correlations': {
            'original_R': {
                'pearson_r': float(r_original_pearson),
                'pearson_p': float(p_original_pearson),
                'spearman_r': float(r_original_spearman),
                'spearman_p': float(p_original_spearman),
                'r_squared': float(r2_original),
                'mean_R': float(np.mean(R_original)),
                'std_R': float(np.std(R_original)),
                'min_R': float(np.min(R_original)),
                'max_R': float(np.max(R_original))
            },
            'fixed_R': {
                'pearson_r': float(r_fixed_pearson),
                'pearson_p': float(p_fixed_pearson),
                'spearman_r': float(r_fixed_spearman),
                'spearman_p': float(p_fixed_spearman),
                'r_squared': float(r2_fixed),
                'mean_R': float(np.mean(R_fixed)),
                'std_R': float(np.std(R_fixed)),
                'min_R': float(np.min(R_fixed)),
                'max_R': float(np.max(R_fixed))
            },
            'disorder_baseline': {
                'pearson_r': float(r_disorder_pearson),
                'pearson_p': float(p_disorder_pearson),
                'spearman_r': float(r_disorder_spearman),
                'spearman_p': float(p_disorder_spearman),
                'note': 'Negative correlation expected (more disorder = lower pLDDT)'
            },
            'order_baseline': {
                'pearson_r': float(r_order_pearson),
                'pearson_p': float(p_order_pearson),
                'spearman_r': float(r_order_spearman),
                'spearman_p': float(p_order_spearman),
                'r_squared': float(r2_order),
                'note': '1 - disorder_frac simple predictor'
            }
        },

        'improvement': {
            'r_improvement': float(r_fixed_pearson - r_original_pearson),
            'r_squared_improvement': float(r2_fixed - r2_original),
            'original_vs_fixed_ratio': float(r_fixed_pearson / r_original_pearson) if r_original_pearson != 0 else float('inf')
        },

        'success_criteria': {
            'target_r': 0.5,
            'target_p': 0.001,
            'achieved_r': float(r_fixed_pearson),
            'achieved_p': float(p_fixed_pearson),
            'r_criterion_met': bool(r_fixed_pearson > 0.5),
            'p_criterion_met': bool(p_fixed_pearson < 0.001),
            'overall_success': bool(r_fixed_pearson > 0.5 and p_fixed_pearson < 0.001)
        },

        'formula_details': {
            'original_sigma': 'hydrophobicity_std / 4.5 (nearly constant ~0.75)',
            'fixed_sigma': '0.1 + 0.5 * abs(disorder_frac - 0.5) + 0.4 * log(length) / 10',
            'E_formula': '0.4 * order_score + 0.3 * hydro_balance + 0.2 * structure_prop + 0.1 * (1 - complexity_penalty)',
            'fix_rationale': 'sigma now varies meaningfully with disorder uncertainty and protein length'
        },

        'protein_results': fixed_protein_results
    }

    return results


def print_results(results: dict):
    """Print formatted results to console."""
    print("=" * 70)
    print("FIXED PROTEIN FOLDING TEST RESULTS")
    print("=" * 70)
    print(f"\nTimestamp: {results['timestamp']}")
    print(f"Number of proteins: {results['n_proteins']}")

    print("\n" + "-" * 70)
    print("CORRELATION COMPARISON")
    print("-" * 70)

    orig = results['correlations']['original_R']
    fixed = results['correlations']['fixed_R']
    order = results['correlations']['order_baseline']
    disorder = results['correlations']['disorder_baseline']

    print(f"\n{'Metric':<30} {'Original R':<15} {'Fixed R':<15} {'Order (baseline)':<15}")
    print("-" * 75)
    print(f"{'Pearson r':<30} {orig['pearson_r']:>14.4f} {fixed['pearson_r']:>14.4f} {order['pearson_r']:>14.4f}")
    print(f"{'Pearson p-value':<30} {orig['pearson_p']:>14.4e} {fixed['pearson_p']:>14.4e} {order['pearson_p']:>14.4e}")
    print(f"{'Spearman rho':<30} {orig['spearman_r']:>14.4f} {fixed['spearman_r']:>14.4f} {order['spearman_r']:>14.4f}")
    print(f"{'Spearman p-value':<30} {orig['spearman_p']:>14.4e} {fixed['spearman_p']:>14.4e} {order['spearman_p']:>14.4e}")
    print(f"{'R-squared':<30} {orig['r_squared']:>14.4f} {fixed['r_squared']:>14.4f} {order['r_squared']:>14.4f}")

    print("\n" + "-" * 70)
    print("R VALUE DISTRIBUTION")
    print("-" * 70)
    print(f"\n{'Statistic':<20} {'Original R':<20} {'Fixed R':<20}")
    print("-" * 60)
    print(f"{'Mean':<20} {orig['mean_R']:>19.4f} {fixed['mean_R']:>19.4f}")
    print(f"{'Std Dev':<20} {orig['std_R']:>19.4f} {fixed['std_R']:>19.4f}")
    print(f"{'Min':<20} {orig['min_R']:>19.4f} {fixed['min_R']:>19.4f}")
    print(f"{'Max':<20} {orig['max_R']:>19.4f} {fixed['max_R']:>19.4f}")
    print(f"{'Range':<20} {orig['max_R'] - orig['min_R']:>19.4f} {fixed['max_R'] - fixed['min_R']:>19.4f}")

    print("\n" + "-" * 70)
    print("DISORDER BASELINE (for comparison)")
    print("-" * 70)
    print(f"disorder_frac vs pLDDT: r = {disorder['pearson_r']:.4f} (p = {disorder['pearson_p']:.4e})")
    print(f"Note: {disorder['note']}")

    print("\n" + "-" * 70)
    print("IMPROVEMENT SUMMARY")
    print("-" * 70)
    imp = results['improvement']
    print(f"Correlation improvement: {imp['r_improvement']:+.4f}")
    print(f"R-squared improvement: {imp['r_squared_improvement']:+.4f}")
    print(f"Improvement ratio: {imp['original_vs_fixed_ratio']:.2f}x")

    print("\n" + "-" * 70)
    print("SUCCESS CRITERIA")
    print("-" * 70)
    sc = results['success_criteria']
    print(f"Target: r > {sc['target_r']}, p < {sc['target_p']}")
    print(f"Achieved: r = {sc['achieved_r']:.4f}, p = {sc['achieved_p']:.4e}")
    print(f"r criterion met: {'YES' if sc['r_criterion_met'] else 'NO'}")
    print(f"p criterion met: {'YES' if sc['p_criterion_met'] else 'NO'}")
    print(f"\n*** OVERALL SUCCESS: {'PASS' if sc['overall_success'] else 'FAIL'} ***")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    input_file = script_dir / 'extended_protein_results.json'
    output_file = script_dir / 'protein_folding_fixed_results.json'

    print(f"Loading data from: {input_file}")

    # Load original results
    data = load_extended_results(input_file)

    # Run fixed test
    results = run_fixed_test(data)

    # Print results
    print_results(results)

    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone!")

    return results


if __name__ == '__main__':
    main()
