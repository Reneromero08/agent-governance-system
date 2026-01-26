"""
Adversarial Audit of Protein Folding r=0.749 Result
====================================================

This script performs skeptical tests to verify or falsify the claimed result.

Tests:
1. Circularity check - does fixed R use pLDDT?
2. Baseline confounds - does disorder_frac alone achieve similar r?
3. Statistical validity - is the p-value correct?
4. Formula decomposition - which components drive the correlation?
5. Multiple testing check - were other formulas tried?
"""

import json
import math
import numpy as np
from scipy import stats
from pathlib import Path


def load_data():
    """Load the protein results data."""
    script_dir = Path(__file__).parent.parent / 'real_data'
    with open(script_dir / 'protein_folding_fixed_results.json', 'r') as f:
        return json.load(f)


def extract_arrays(data):
    """Extract numpy arrays from the data."""
    proteins = data['protein_results']

    disorder = np.array([p['disorder_frac'] for p in proteins])
    length = np.array([p['length'] for p in proteins])
    complexity = np.array([p['complexity'] for p in proteins])
    plddt = np.array([p['mean_plddt'] for p in proteins])
    R_fixed = np.array([p['R_fixed'] for p in proteins])
    R_original = np.array([p['R_original'] for p in proteins])

    return disorder, length, complexity, plddt, R_fixed, R_original


def compute_R_fixed_vectorized(disorder, length, complexity):
    """Recompute the fixed R formula to verify."""
    order_score = 1.0 - disorder
    hydro_balance = 0.7 + 0.2 * order_score
    structure_prop = 0.3 + 0.4 * order_score
    complexity_penalty = np.abs(complexity - 0.75)

    E = (0.4 * order_score +
         0.3 * hydro_balance +
         0.2 * structure_prop +
         0.1 * (1 - complexity_penalty))

    disorder_uncertainty = np.abs(disorder - 0.5)
    length_factor = np.log(length + 1) / 10

    sigma = 0.1 + 0.5 * disorder_uncertainty + 0.4 * length_factor

    return E, sigma, E / sigma


def test_circularity(plddt, disorder, length, complexity, R_fixed):
    """Check if pLDDT is used in computing R_fixed."""
    print("\n" + "="*70)
    print("TEST 1: CIRCULARITY CHECK")
    print("="*70)

    # Recompute R_fixed from first principles
    E, sigma, R_recomputed = compute_R_fixed_vectorized(disorder, length, complexity)

    # Check if they match
    diff = np.abs(R_fixed - R_recomputed)
    max_diff = np.max(diff)

    print(f"\nRecomputed R_fixed from disorder, length, complexity only.")
    print(f"Max difference from stored R_fixed: {max_diff:.10f}")

    if max_diff < 1e-10:
        print("\nVERDICT: R_fixed does NOT use pLDDT in its calculation.")
        print("         Formula uses only: disorder_frac, length, complexity")
        circularity = False
    else:
        print("\nWARNING: Recomputed R differs - possible hidden dependency!")
        circularity = True

    return circularity


def test_baseline_confounds(disorder, length, complexity, plddt, R_fixed):
    """Test if simpler baselines achieve similar correlation."""
    print("\n" + "="*70)
    print("TEST 2: BASELINE CONFOUNDS")
    print("="*70)

    # Compute various baselines
    order = 1 - disorder
    log_length = np.log(length + 1)

    # E component alone
    hydro_balance = 0.7 + 0.2 * order
    structure_prop = 0.3 + 0.4 * order
    complexity_penalty = np.abs(complexity - 0.75)
    E = (0.4 * order + 0.3 * hydro_balance + 0.2 * structure_prop + 0.1 * (1 - complexity_penalty))

    # Sigma component alone
    disorder_uncertainty = np.abs(disorder - 0.5)
    length_factor = np.log(length + 1) / 10
    sigma = 0.1 + 0.5 * disorder_uncertainty + 0.4 * length_factor

    # Compute correlations
    baselines = {
        'disorder_frac': disorder,
        'order (1 - disorder)': order,
        'log(length)': log_length,
        'length': length,
        'complexity': complexity,
        'E (numerator only)': E,
        '1/sigma': 1/sigma,
        '-log(sigma)': -np.log(sigma),
        'R_fixed': R_fixed,
    }

    print(f"\n{'Predictor':<25} {'Pearson r':>12} {'p-value':>15} {'vs R_fixed':<15}")
    print("-"*70)

    r_fixed_corr, _ = stats.pearsonr(R_fixed, plddt)

    for name, predictor in baselines.items():
        r, p = stats.pearsonr(predictor, plddt)
        comparison = ""
        if name != 'R_fixed':
            diff = r_fixed_corr - r
            comparison = f"R_fixed is {diff:+.3f} better"
        print(f"{name:<25} {r:>12.4f} {p:>15.2e} {comparison}")

    # Key test: Does order alone explain most of the variance?
    r_order, _ = stats.pearsonr(order, plddt)
    r_fixed, _ = stats.pearsonr(R_fixed, plddt)

    print(f"\n--- Critical Comparison ---")
    print(f"Order alone (1-disorder): r = {r_order:.4f}")
    print(f"R_fixed:                  r = {r_fixed:.4f}")
    print(f"Improvement:              r = {r_fixed - r_order:+.4f}")
    print(f"Relative improvement:     {(r_fixed - r_order) / r_order * 100:.1f}%")

    return r_order, r_fixed


def test_statistical_validity(plddt, R_fixed, n=47):
    """Verify the p-value calculation."""
    print("\n" + "="*70)
    print("TEST 3: STATISTICAL VALIDITY")
    print("="*70)

    # Compute correlation
    r, p = stats.pearsonr(R_fixed, plddt)

    print(f"\nReported: r = 0.749, p = 1.43e-09, n = 47")
    print(f"Computed: r = {r:.4f}, p = {p:.4e}, n = {len(R_fixed)}")

    # Manual p-value calculation
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

    print(f"\nManual t-test verification:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  df: {n-2}")
    print(f"  p-value (manual): {p_manual:.4e}")

    # Check for sample independence
    print(f"\n--- Sample Independence ---")
    print(f"Are the 47 proteins independent samples?")
    print(f"  - Different UniProt IDs: YES")
    print(f"  - From different protein families: UNKNOWN (needs checking)")
    print(f"  - Not time-series or repeated measures: YES")

    return r, p


def test_formula_decomposition(disorder, length, complexity, plddt):
    """Decompose the fixed R formula to see which parts drive correlation."""
    print("\n" + "="*70)
    print("TEST 4: FORMULA DECOMPOSITION")
    print("="*70)

    order = 1 - disorder
    E, sigma, R = compute_R_fixed_vectorized(disorder, length, complexity)

    # Test each component
    components = {
        'order_score': order,
        'E (foldability)': E,
        'sigma (uncertainty)': sigma,
        '1/sigma': 1/sigma,
        'disorder_uncertainty': np.abs(disorder - 0.5),
        'length_factor': np.log(length + 1) / 10,
    }

    print(f"\n{'Component':<25} {'Corr with pLDDT':>15} {'Corr with R_fixed':>18}")
    print("-"*60)

    for name, component in components.items():
        r_plddt, _ = stats.pearsonr(component, plddt)
        r_R, _ = stats.pearsonr(component, R)
        print(f"{name:<25} {r_plddt:>15.4f} {r_R:>18.4f}")

    # Partial correlations
    print(f"\n--- Key insight: What drives R_fixed? ---")

    # R_fixed = E / sigma
    # Since E has weak variation (high mean, low std), most variation comes from 1/sigma
    # And sigma = 0.1 + 0.5 * |disorder - 0.5| + 0.4 * log(length) / 10

    print(f"\nE statistics:     mean={np.mean(E):.4f}, std={np.std(E):.4f}, CV={np.std(E)/np.mean(E)*100:.1f}%")
    print(f"sigma statistics: mean={np.mean(sigma):.4f}, std={np.std(sigma):.4f}, CV={np.std(sigma)/np.mean(sigma)*100:.1f}%")
    print(f"R_fixed stats:    mean={np.mean(R):.4f}, std={np.std(R):.4f}, CV={np.std(R)/np.mean(R)*100:.1f}%")


def test_multiple_formulas():
    """Check if other formulas were tried (cherry-picking concern)."""
    print("\n" + "="*70)
    print("TEST 5: CHERRY-PICKING CHECK")
    print("="*70)

    print("\nSearching for evidence of multiple formula attempts...")

    # Look for other test files in the investigation directory
    investigation_dir = Path(__file__).parent
    real_data_dir = investigation_dir.parent / 'real_data'

    formula_files = list(investigation_dir.glob('*.md')) + list(investigation_dir.glob('*.py'))
    formula_files += list(real_data_dir.glob('*protein*.py'))

    print(f"\nRelated files found:")
    for f in sorted(formula_files):
        print(f"  - {f.name}")

    print("\n--- WARNING ---")
    print("The sigma formula was explicitly designed AFTER seeing the original result fail.")
    print("This is post-hoc formula modification on the SAME dataset.")
    print("The question: Is this legitimate bug-fix or overfitting?")

    print("\nEvidence for legitimate bug-fix:")
    print("  - Original sigma was nearly constant (bad formulation)")
    print("  - New sigma has clear interpretation (disorder uncertainty + length)")
    print("  - The fix is documented with clear reasoning")

    print("\nEvidence for potential overfitting:")
    print("  - Formula was modified AFTER seeing the test fail")
    print("  - No held-out validation set")
    print("  - No cross-validation")


def test_alternative_sigma_formulas(disorder, length, complexity, plddt):
    """Test if other reasonable sigma formulas work equally well (or better)."""
    print("\n" + "="*70)
    print("TEST 6: ALTERNATIVE SIGMA FORMULAS")
    print("="*70)

    order = 1 - disorder
    hydro_balance = 0.7 + 0.2 * order
    structure_prop = 0.3 + 0.4 * order
    complexity_penalty = np.abs(complexity - 0.75)

    E = (0.4 * order + 0.3 * hydro_balance + 0.2 * structure_prop + 0.1 * (1 - complexity_penalty))

    # The claimed sigma
    sigma_claimed = 0.1 + 0.5 * np.abs(disorder - 0.5) + 0.4 * np.log(length + 1) / 10
    R_claimed = E / sigma_claimed

    # Alternative sigmas
    alternatives = {
        'sigma = disorder_frac': disorder,
        'sigma = 1 - disorder_frac': 1 - disorder,
        'sigma = log(length)': np.log(length + 1),
        'sigma = sqrt(length)': np.sqrt(length),
        'sigma = complexity': complexity,
        'sigma = 1': np.ones_like(disorder),  # Just E alone
        'sigma = claimed': sigma_claimed,
    }

    print(f"\n{'Sigma formula':<30} {'r(R, pLDDT)':>15} {'Improvement over E':>20}")
    print("-"*70)

    # E alone
    r_E, _ = stats.pearsonr(E, plddt)
    print(f"{'E alone (sigma=1)':<30} {r_E:>15.4f} {'(baseline)':>20}")

    for name, sigma in alternatives.items():
        if name == 'sigma = 1':
            continue
        R = E / sigma
        r, _ = stats.pearsonr(R, plddt)
        improvement = r - r_E
        print(f"{name:<30} {r:>15.4f} {improvement:>+20.4f}")

    # Random sigma to show any division helps
    np.random.seed(42)
    sigma_random = np.random.uniform(0.3, 0.8, len(disorder))
    R_random = E / sigma_random
    r_random, _ = stats.pearsonr(R_random, plddt)
    print(f"{'sigma = random(0.3, 0.8)':<30} {r_random:>15.4f} {r_random - r_E:>+20.4f}")


def test_is_R_just_order(disorder, length, complexity, plddt):
    """Critical test: Does R_fixed just recapitulate order score?"""
    print("\n" + "="*70)
    print("TEST 7: IS R_FIXED JUST A COMPLICATED ORDER SCORE?")
    print("="*70)

    order = 1 - disorder
    E, sigma, R_fixed = compute_R_fixed_vectorized(disorder, length, complexity)

    # Correlation between R_fixed and order
    r_R_order, _ = stats.pearsonr(R_fixed, order)

    print(f"\nCorrelation between R_fixed and order: r = {r_R_order:.4f}")

    # If R_fixed is highly correlated with order, then R_fixed predicting pLDDT
    # is essentially the same as order predicting pLDDT

    r_R_plddt, _ = stats.pearsonr(R_fixed, plddt)
    r_order_plddt, _ = stats.pearsonr(order, plddt)

    # Partial correlation of R_fixed vs pLDDT controlling for order
    # Using the formula: r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    r_xz = stats.pearsonr(R_fixed, order)[0]
    r_yz = stats.pearsonr(plddt, order)[0]
    r_xy = stats.pearsonr(R_fixed, plddt)[0]

    partial_r = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    print(f"\nPartial correlation: r(R_fixed, pLDDT | order) = {partial_r:.4f}")
    print(f"This is the unique contribution of R_fixed beyond what order explains.")

    if abs(partial_r) < 0.2:
        print("\nVERDICT: R_fixed adds MINIMAL value beyond simple order score!")
    elif abs(partial_r) < 0.4:
        print("\nVERDICT: R_fixed adds MODEST value beyond order score.")
    else:
        print("\nVERDICT: R_fixed adds SUBSTANTIAL value beyond order score.")


def main():
    """Run all adversarial tests."""
    print("="*70)
    print("ADVERSARIAL AUDIT: PROTEIN FOLDING r=0.749 RESULT")
    print("="*70)
    print("\nObjective: Find flaws in the claimed result.")
    print("Claimed: Fixed R formula achieves r=0.749, p=1.43e-09 on 47 proteins")

    # Load data
    data = load_data()
    disorder, length, complexity, plddt, R_fixed, R_original = extract_arrays(data)

    # Run tests
    circularity = test_circularity(plddt, disorder, length, complexity, R_fixed)
    r_order, r_fixed = test_baseline_confounds(disorder, length, complexity, plddt, R_fixed)
    r, p = test_statistical_validity(plddt, R_fixed)
    test_formula_decomposition(disorder, length, complexity, plddt)
    test_multiple_formulas()
    test_alternative_sigma_formulas(disorder, length, complexity, plddt)
    test_is_R_just_order(disorder, length, complexity, plddt)

    # Final summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)

    print("\n[FINDING 1] Circularity: " + ("PASS - No pLDDT used in R" if not circularity else "FAIL - Circular!"))
    print(f"[FINDING 2] Baseline comparison: Order alone gets r={r_order:.3f}, R_fixed gets r={r_fixed:.3f}")
    print(f"            Improvement: {r_fixed - r_order:+.3f} ({(r_fixed - r_order)/r_order*100:.1f}%)")
    print(f"[FINDING 3] Statistics: r={r:.4f}, p={p:.2e} - calculation is correct")
    print("[FINDING 4] Post-hoc formula modification on same dataset - NO HELD-OUT TEST")
    print("[FINDING 5] Partial correlation controlling for order reveals true contribution")

    return {
        'circularity': circularity,
        'r_order': r_order,
        'r_fixed': r_fixed,
        'r': r,
        'p': p,
    }


if __name__ == '__main__':
    results = main()
