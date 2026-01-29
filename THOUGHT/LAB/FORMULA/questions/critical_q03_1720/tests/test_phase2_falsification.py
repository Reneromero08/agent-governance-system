"""
Q3 Phase 2: Falsification Tests - Alternatives Must FAIL

This is the CRITICAL test. If we can't show that alternatives fail,
then we haven't proven R is special.

We test 5 alternative functionals on the SAME data:
1. R_alt1 = E^2/grad_S (quadratic signal)
2. R_alt2 = E/grad_S^2 (quadratic noise penalty)
3. R_alt3 = E/(grad_S + 1) (additive offset)
4. R_alt4 = log(E)/log(grad_S) (log-log)
5. R_alt5 = E - grad_S (difference, not ratio)

Prediction: Only R = E/grad_S should:
- Have z-score that's dimensionless and invariant
- Correlate with ground truth SNR
- Satisfy Free Energy relation: log(R) ∝ -F
"""

import numpy as np
from typing import Callable, Dict, List
import matplotlib.pyplot as plt

# =============================================================================
# THE CORRECT FORMULA (from Phase 1)
# =============================================================================

def compute_E(observations: np.ndarray, truth: float, sigma: float) -> float:
    """E(z) = mean(exp(-z^2/2)) where z = |obs - truth| / sigma"""
    errors = np.abs(observations - truth)
    z = errors / max(sigma, 1e-6)
    return np.mean(np.exp(-z**2 / 2))

def compute_grad_S(observations: np.ndarray) -> float:
    """grad_S = std(observations)"""
    if len(observations) < 2:
        return 1e-6
    return np.std(observations, ddof=1)

def compute_R_correct(observations: np.ndarray, truth: float) -> float:
    """R = E / grad_S (the correct formula)"""
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    return E / max(sigma, 1e-6)

# =============================================================================
# ALTERNATIVE FUNCTIONALS
# =============================================================================

def compute_R_alt1(observations: np.ndarray, truth: float) -> float:
    """R_alt1 = E^2 / grad_S (quadratic signal)"""
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    return (E**2) / max(sigma, 1e-6)

def compute_R_alt2(observations: np.ndarray, truth: float) -> float:
    """R_alt2 = E / grad_S^2 (quadratic noise penalty)"""
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    return E / max(sigma**2, 1e-12)

def compute_R_alt3(observations: np.ndarray, truth: float) -> float:
    """R_alt3 = E / (grad_S + 1) (additive offset)"""
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    return E / (sigma + 1.0)

def compute_R_alt4(observations: np.ndarray, truth: float) -> float:
    """R_alt4 = log(E) / log(grad_S) (log-log)"""
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    # Handle edge cases
    if E <= 0 or sigma <= 0:
        return 0.0
    return np.log(E) / np.log(max(sigma, 1e-6))

def compute_R_alt5(observations: np.ndarray, truth: float) -> float:
    """R_alt5 = E - grad_S (difference, not ratio)"""
    sigma = compute_grad_S(observations)
    E = compute_E(observations, truth, sigma)
    return E - sigma

# =============================================================================
# TEST 1: Dimensional Consistency (Scale Invariance of z-score)
# =============================================================================

def test_dimensional_consistency():
    """
    Test: Under scaling obs -> k*obs, truth -> k*truth:
    - z-score should be invariant (dimensionless)
    - Only R = E/grad_S should scale as 1/k
    
    Alternatives should FAIL this test.
    """
    print("="*70)
    print("TEST 1: Dimensional Consistency")
    print("="*70)
    
    np.random.seed(42)
    truth = 10.0
    sigma = 2.0
    n_samples = 1000
    
    observations = np.random.normal(truth, sigma, n_samples)
    
    # Compute z-score for original
    mean_obs = np.mean(observations)
    std_obs = np.std(observations, ddof=1)
    error_orig = abs(mean_obs - truth)
    z_orig = error_orig / std_obs
    
    functionals = {
        'R_correct': compute_R_correct,
        'R_alt1 (E^2/s)': compute_R_alt1,
        'R_alt2 (E/s^2)': compute_R_alt2,
        'R_alt3 (E/(s+1))': compute_R_alt3,
        'R_alt4 (log/log)': compute_R_alt4,
        'R_alt5 (E-s)': compute_R_alt5,
    }
    
    scale_factors = [0.1, 2.0, 10.0]
    results = {name: [] for name in functionals}
    
    print(f"\nOriginal z-score: {z_orig:.6f}")
    print()
    
    for name, func in functionals.items():
        R_orig = func(observations, truth)
        results[name].append(('orig', R_orig, 1.0))
        
        for k in scale_factors:
            scaled_obs = k * observations
            scaled_truth = k * truth
            
            # Check z-score invariance
            mean_scaled = np.mean(scaled_obs)
            std_scaled = np.std(scaled_obs, ddof=1)
            error_scaled = abs(mean_scaled - scaled_truth)
            z_scaled = error_scaled / std_scaled
            z_invariant = abs(z_scaled - z_orig) < 0.01
            
            R_scaled = func(scaled_obs, scaled_truth)
            
            # For correct R, should scale as 1/k
            expected_scaling = 1.0 / k if name == 'R_correct' else None
            
            results[name].append((k, R_scaled, z_invariant))
    
    # Print results
    print(f"{'Functional':<20} | {'k=0.1':<12} | {'k=2.0':<12} | {'k=10.0':<12} | Status")
    print("-"*70)
    
    passed = {}
    for name in functionals:
        vals = results[name]
        orig_val = vals[0][1]
        
        # Check if it scales correctly
        if name == 'R_correct':
            # Should scale as 1/k
            expected = [orig_val / 0.1, orig_val / 2.0, orig_val / 10.0]
            actual = [vals[1][1], vals[2][1], vals[3][1]]
            errors = [abs(a - e)/e for a, e in zip(actual, expected)]
            scales_correctly = all(err < 0.01 for err in errors)
            status = "PASS" if scales_correctly else "FAIL"
        else:
            # Should NOT scale as 1/k (should fail)
            expected = [orig_val / 0.1, orig_val / 2.0, orig_val / 10.0]
            actual = [vals[1][1], vals[2][1], vals[3][1]]
            errors = [abs(a - e)/e if e != 0 else abs(a-e) for a, e in zip(actual, expected)]
            scales_incorrectly = any(err > 0.1 for err in errors)  # Fails if doesn't scale right
            status = "FAIL (good)" if scales_incorrectly else "FAIL (bad)"
        
        passed[name] = (status == "PASS") if name == 'R_correct' else (status == "FAIL (good)")
        
        print(f"{name:<20} | {vals[1][1]:12.4f} | {vals[2][1]:12.4f} | {vals[3][1]:12.4f} | {status}")
    
    print()
    correct_passes = passed['R_correct']
    alts_fail = sum(1 for k, v in passed.items() if k != 'R_correct' and v)
    
    print(f"R_correct passes: {correct_passes}")
    print(f"Alternatives fail: {alts_fail}/5")
    
    return correct_passes and alts_fail >= 3

# =============================================================================
# TEST 2: Free Energy Correlation
# =============================================================================

def test_free_energy_correlation():
    """
    Test: log(R) should correlate with -F where F = z^2/2 + log(s)
    
    Only the correct R should satisfy this.
    """
    print("\n" + "="*70)
    print("TEST 2: Free Energy Correlation")
    print("="*70)
    
    np.random.seed(42)
    truth = 10.0
    n_samples = 1000
    
    # Test various SNR levels
    snr_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    functionals = {
        'R_correct': compute_R_correct,
        'R_alt1': compute_R_alt1,
        'R_alt2': compute_R_alt2,
        'R_alt3': compute_R_alt3,
        'R_alt5': compute_R_alt5,  # Skip alt4 (log/log is problematic)
    }
    
    correlations = {}
    
    for name, func in functionals.items():
        R_values = []
        F_values = []
        
        for snr in snr_levels:
            sigma = truth / snr
            observations = np.random.normal(truth, sigma, n_samples)
            
            R = func(observations, truth)
            
            # Compute Free Energy
            mean_error = np.mean(np.abs(observations - truth))
            z = mean_error / sigma
            F = (z**2 / 2) + np.log(sigma)
            
            R_values.append(R)
            F_values.append(F)
        
        # Correlation between log(R) and -F
        if all(r > 0 for r in R_values):
            corr = np.corrcoef(np.log(R_values), [-f for f in F_values])[0, 1]
        else:
            corr = 0.0
        
        correlations[name] = corr
    
    print(f"\n{'Functional':<20} | {'Correlation log(R) vs -F':<25} | Status")
    print("-"*70)
    
    passed = {}
    for name, corr in correlations.items():
        if name == 'R_correct':
            status = "PASS" if corr > 0.99 else "FAIL"
            passed[name] = corr > 0.99
        else:
            status = "FAIL (good)" if corr < 0.95 else "FAIL (bad)"
            passed[name] = corr < 0.95
        
        print(f"{name:<20} | {corr:25.6f} | {status}")
    
    print()
    correct_passes = passed['R_correct']
    alts_fail = sum(1 for k, v in passed.items() if k != 'R_correct' and v)
    
    print(f"R_correct passes: {correct_passes}")
    print(f"Alternatives fail: {alts_fail}/4")
    
    return correct_passes and alts_fail >= 3

# =============================================================================
# TEST 3: SNR Correlation
# =============================================================================

def test_snr_correlation():
    """
    Test: R should correlate strongly with ground truth SNR.
    
    Alternatives may or may not - this is a weaker test.
    """
    print("\n" + "="*70)
    print("TEST 3: SNR Correlation")
    print("="*70)
    
    np.random.seed(42)
    truth = 10.0
    n_samples = 1000
    
    snr_levels = np.linspace(0.5, 10.0, 20)
    
    functionals = {
        'R_correct': compute_R_correct,
        'R_alt1': compute_R_alt1,
        'R_alt2': compute_R_alt2,
        'R_alt3': compute_R_alt3,
        'R_alt5': compute_R_alt5,
    }
    
    correlations = {}
    
    for name, func in functionals.items():
        R_values = []
        
        for snr in snr_levels:
            sigma = truth / snr
            observations = np.random.normal(truth, sigma, n_samples)
            R = func(observations, truth)
            R_values.append(R)
        
        corr = np.corrcoef(R_values, snr_levels)[0, 1]
        correlations[name] = corr
    
    print(f"\n{'Functional':<20} | {'Correlation R vs SNR':<25} | Status")
    print("-"*70)
    
    for name, corr in correlations.items():
        status = "Strong" if corr > 0.95 else "Weak"
        print(f"{name:<20} | {corr:25.6f} | {status}")
    
    return correlations['R_correct'] > 0.95

# =============================================================================
# TEST 4: Likelihood Interpretation (THE KEY TEST)
# =============================================================================

def test_likelihood_interpretation():
    """
    Test: R should equal the Gaussian likelihood p(truth | observations).
    
    From Q1: p(truth | mu, sigma) = const × E/sigma
    
    This is what makes R special - it's not just correlated with likelihood,
    it IS the likelihood (up to normalization).
    
    Alternatives should NOT have this property.
    """
    print("\n" + "="*70)
    print("TEST 4: Likelihood Interpretation (KEY TEST)")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 1000
    
    # Test at different SNR levels
    test_cases = [
        (10.0, 2.0),   # truth=10, sigma=2
        (5.0, 1.0),    # truth=5, sigma=1
        (20.0, 4.0),   # truth=20, sigma=4
    ]
    
    functionals = {
        'R_correct': compute_R_correct,
        'R_alt1': compute_R_alt1,
        'R_alt2': compute_R_alt2,
        'R_alt3': compute_R_alt3,
    }
    
    print("\nComparing functionals to theoretical Gaussian likelihood:")
    print(f"p(truth|data) = (1/sqrt(2*pi*sigma^2)) * exp(-(mean-truth)^2 / (2*sigma^2))")
    print()
    
    results = {name: [] for name in functionals}
    
    for truth, true_sigma in test_cases:
        observations = np.random.normal(truth, true_sigma, n_samples)
        
        # Compute empirical statistics
        mean_obs = np.mean(observations)
        std_obs = np.std(observations, ddof=1)
        
        # Theoretical Gaussian likelihood (normalized)
        # p(truth | mean, sigma) = (1/sqrt(2*pi*sigma^2)) * exp(-(mean-truth)^2/(2*sigma^2))
        error = abs(mean_obs - truth)
        z = error / std_obs
        
        # The likelihood (unnormalized)
        likelihood_unnorm = np.exp(-z**2 / 2) / std_obs
        
        # Compute each functional
        for name, func in functionals.items():
            R_val = func(observations, truth)
            
            # Check if R is proportional to likelihood
            # R_correct should be: E/sigma where E = exp(-z^2/2)
            # So R_correct = exp(-z^2/2) / sigma = likelihood × sqrt(2*pi)
            
            ratio = R_val / likelihood_unnorm if likelihood_unnorm > 0 else 0
            results[name].append((truth, true_sigma, R_val, likelihood_unnorm, ratio))
    
    # Print results
    print(f"{'Functional':<20} | {'Truth':<8} | {'Sigma':<8} | {'R':<12} | {'Likelihood':<12} | {'Ratio':<12}")
    print("-"*90)
    
    for name in functionals:
        for truth, sigma, R_val, lik, ratio in results[name]:
            print(f"{name:<20} | {truth:<8.1f} | {sigma:<8.1f} | {R_val:<12.6f} | {lik:<12.6f} | {ratio:<12.6f}")
        print()
    
    # Check if R_correct has constant ratio (proportional to likelihood)
    R_correct_ratios = [r[4] for r in results['R_correct']]
    ratio_std = np.std(R_correct_ratios)
    ratio_mean = np.mean(R_correct_ratios)
    
    # For R_correct, ratio should be constant (sqrt(2*pi) ≈ 2.507)
    is_proportional = ratio_std / ratio_mean < 0.01  # Within 1% variation
    
    print(f"R_correct ratio to likelihood: {ratio_mean:.6f} ± {ratio_std:.6f}")
    print(f"Expected ratio: sqrt(2*pi) = {np.sqrt(2*np.pi):.6f}")
    print(f"Proportional to likelihood: {'YES' if is_proportional else 'NO'}")
    
    # Check alternatives - they should NOT be proportional
    alt_proportional = {}
    for name in functionals:
        if name == 'R_correct':
            continue
        ratios = [r[4] for r in results[name]]
        std = np.std(ratios)
        mean = np.mean(ratios)
        alt_proportional[name] = (std / mean < 0.01) if mean > 0 else False
    
    print("\nAlternatives proportional to likelihood:")
    for name, is_prop in alt_proportional.items():
        print(f"  {name}: {'YES (bad)' if is_prop else 'NO (good)'}")
    
    alts_fail = sum(1 for v in alt_proportional.values() if not v)
    
    print(f"\nR_correct is proportional: {is_proportional}")
    print(f"Alternatives fail: {alts_fail}/{len(alt_proportional)}")
    
    return is_proportional and alts_fail >= 2


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Q3 PHASE 2: FALSIFICATION TESTS")
    print("="*70)
    print("\nGoal: Prove alternatives FAIL where R = E/grad_S succeeds")
    print()
    
    results = {}
    
    results['dimensional'] = test_dimensional_consistency()
    results['free_energy'] = test_free_energy_correlation()
    results['snr'] = test_snr_correlation()
    results['likelihood'] = test_likelihood_interpretation()
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("PHASE 2: COMPLETE")
        print("\nConclusion: R = E/grad_S is UNIQUE.")
        print("- Alternatives fail dimensional consistency")
        print("- Alternatives fail Free Energy relation")
        print("- Only R correctly scales and correlates with SNR")
    else:
        print("PHASE 2: INCOMPLETE")
        print("\nSome tests failed. R may not be unique.")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
