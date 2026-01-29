#!/usr/bin/env python3
"""
ADVERSARIAL AUDIT: Verify or Falsify 8e Embedding Result

CLAIMED RESULT:
- R-modulated sinusoidal embedding produces Df x alpha = 21.15 (2.9% from 8e)
- This is claimed as evidence that 8e emerges from structured representations

THIS SCRIPT ATTEMPTS TO BREAK THE RESULT BY:
1. Testing for mathematical tautology
2. Testing parameter sensitivity
3. Testing data dependence
4. Verifying computation correctness
5. Assessing statistical significance

Author: Adversarial Auditor
Date: 2026-01-26
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
EIGHT_E = 8 * np.e  # ~21.746

# ============================================================================
# AUDIT 1: MATHEMATICAL TAUTOLOGY CHECK
# ============================================================================

def audit_tautology():
    """
    Check if the sinusoidal formula MATHEMATICALLY GUARANTEES 8e.

    The claim is that:
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    produces Df x alpha = 8e.

    QUESTION: Is this a tautology? Does the formula construction force 8e?
    """
    print("\n" + "="*80)
    print("AUDIT 1: MATHEMATICAL TAUTOLOGY CHECK")
    print("="*80)

    results = {}

    # Test A: What happens with CONSTANT R values?
    # If 8e is built into the formula, it should still produce 8e
    n_samples = 2500
    n_dims = 50
    seed = 42

    test_cases = [
        ("constant_r_1", np.ones(n_samples) * 1.0),
        ("constant_r_5", np.ones(n_samples) * 5.0),
        ("constant_r_10", np.ones(n_samples) * 10.0),
        ("constant_r_20", np.ones(n_samples) * 20.0),
        ("constant_r_50", np.ones(n_samples) * 50.0),
    ]

    print("\nTest A: Constant R values (if 8e is formula-intrinsic, should still produce 8e)")
    print("-"*70)

    for name, R_values in test_cases:
        np.random.seed(seed)
        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_values):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[name] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        print(f"  {name}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation)")

    # Test B: Does the formula work with ANY sinusoidal frequency?
    # Change the scaling constant (currently 10.0)
    print("\nTest B: Different sinusoidal frequency scaling (r/X instead of r/10)")
    print("-"*70)

    # Use gene expression R distribution
    R_values = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    R_values = np.clip(R_values, 0.3, 55)  # Similar range to real data

    for freq_scale in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        np.random.seed(seed)
        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_values):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / freq_scale)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"freq_scale_{freq_scale}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        print(f"  r/{freq_scale}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation)")

    return results


# ============================================================================
# AUDIT 2: PARAMETER SENSITIVITY
# ============================================================================

def audit_parameter_sensitivity():
    """
    Test if scale=10, noise=0.5, dim=50 were CHOSEN to produce 8e.
    What happens with other parameter values?
    """
    print("\n" + "="*80)
    print("AUDIT 2: PARAMETER SENSITIVITY")
    print("="*80)

    results = {}
    n_samples = 2500
    seed = 42

    # Generate R values similar to gene expression
    np.random.seed(seed)
    R_values = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    R_values = np.clip(R_values, 0.3, 55)

    # Test A: Varying dimensions
    print("\nTest A: Varying embedding dimensions")
    print("-"*70)

    for n_dims in [10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]:
        np.random.seed(seed)
        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_values):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"dims_{n_dims}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  {n_dims}D: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    # Test B: Varying noise scale
    print("\nTest B: Varying noise scale (scale = X / (r + 0.1))")
    print("-"*70)
    n_dims = 50

    for noise_mult in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        np.random.seed(seed)
        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_values):
            np.random.seed(i + seed)
            scale = noise_mult / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"noise_mult_{noise_mult}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  X={noise_mult}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    # Test C: Varying number of samples
    print("\nTest C: Varying number of samples")
    print("-"*70)
    n_dims = 50

    for n_samples_test in [100, 250, 500, 1000, 2500, 5000, 10000]:
        np.random.seed(seed)
        R_test = np.random.lognormal(mean=2, sigma=1, size=n_samples_test)
        R_test = np.clip(R_test, 0.3, 55)

        embeddings = np.zeros((n_samples_test, n_dims))
        for i, r in enumerate(R_test):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"samples_{n_samples_test}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  n={n_samples_test}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    return results


# ============================================================================
# AUDIT 3: DATA DEPENDENCE
# ============================================================================

def audit_data_dependence():
    """
    Does this only work with gene expression R values?
    What if we use completely random data?
    """
    print("\n" + "="*80)
    print("AUDIT 3: DATA DEPENDENCE")
    print("="*80)

    results = {}
    n_samples = 2500
    n_dims = 50
    seed = 42

    # Test A: Uniform random values in same range
    print("\nTest A: Uniform random values (range [0.3, 55])")
    print("-"*70)

    for trial in range(5):
        np.random.seed(seed + trial * 100)
        R_uniform = np.random.uniform(0.3, 55, n_samples)

        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_uniform):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"uniform_trial_{trial}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  Trial {trial}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    # Test B: Gaussian random values
    print("\nTest B: Gaussian random values (mean=11.7, std=13.2 - matching gene data)")
    print("-"*70)

    for trial in range(5):
        np.random.seed(seed + trial * 200)
        R_gaussian = np.random.randn(n_samples) * 13.2 + 11.7
        R_gaussian = np.clip(R_gaussian, 0.3, 55)

        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_gaussian):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"gaussian_trial_{trial}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  Trial {trial}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    # Test C: Log-normal (matching gene data shape)
    print("\nTest C: Log-normal random values (similar distribution shape)")
    print("-"*70)

    for trial in range(5):
        np.random.seed(seed + trial * 300)
        R_lognorm = np.random.lognormal(mean=2, sigma=1, size=n_samples)
        R_lognorm = np.clip(R_lognorm, 0.3, 55)

        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_lognorm):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"lognorm_trial_{trial}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  Trial {trial}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    # Test D: Completely random (no relation to biology)
    print("\nTest D: Random values in completely different ranges")
    print("-"*70)

    test_ranges = [
        (0.01, 1.0, "tiny [0.01, 1]"),
        (0.1, 10.0, "small [0.1, 10]"),
        (1.0, 100.0, "medium [1, 100]"),
        (10.0, 1000.0, "large [10, 1000]"),
        (100.0, 10000.0, "huge [100, 10000]"),
    ]

    for r_min, r_max, label in test_ranges:
        np.random.seed(seed)
        R_test = np.random.uniform(r_min, r_max, n_samples)

        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_test):
            np.random.seed(i + seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        product = Df * alpha
        deviation = abs(product - EIGHT_E) / EIGHT_E * 100

        results[f"range_{label}"] = {"Df": Df, "alpha": alpha, "product": product, "deviation_pct": deviation}
        status = "PASS" if deviation < 15 else "FAIL"
        print(f"  {label}: Df x alpha = {product:.2f} ({deviation:.1f}% deviation) [{status}]")

    return results


# ============================================================================
# AUDIT 4: COMPUTATION VERIFICATION
# ============================================================================

def compute_from_embeddings(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Compute Df and alpha from embeddings (copy from original code)."""
    centered = embeddings - embeddings.mean(axis=0)
    n_samples = embeddings.shape[0]
    cov = np.dot(centered.T, centered) / max(n_samples - 1, 1)

    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) < 2:
        return 1.0, 1.0, eigenvalues

    # Df: Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    Df = (sum_lambda ** 2) / (sum_lambda_sq + 1e-10)

    # alpha: Power-law fit
    k = np.arange(1, len(eigenvalues) + 1)
    log_k = np.log(k)
    log_lambda = np.log(eigenvalues + 1e-10)

    n_pts = len(log_k)
    slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
    slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2 + 1e-10)
    alpha = -slope

    return Df, alpha, eigenvalues


def audit_computation():
    """
    Verify the Df and alpha computations are correct.
    Compare against alternative implementations.
    """
    print("\n" + "="*80)
    print("AUDIT 4: COMPUTATION VERIFICATION")
    print("="*80)

    results = {}

    # Test with known eigenvalue distribution
    print("\nTest A: Known eigenvalue distributions")
    print("-"*70)

    # Flat spectrum (Df should be N, alpha should be 0)
    n = 50
    eigenvalues_flat = np.ones(n)
    sum_lambda = np.sum(eigenvalues_flat)
    sum_lambda_sq = np.sum(eigenvalues_flat ** 2)
    Df_flat = sum_lambda ** 2 / sum_lambda_sq

    print(f"  Flat spectrum (50 equal eigenvalues):")
    print(f"    Df = {Df_flat:.2f} (expected: 50)")

    # Single dominant eigenvalue (Df should be ~1)
    eigenvalues_single = np.zeros(n)
    eigenvalues_single[0] = 1.0
    eigenvalues_single[1:] = 1e-6
    eigenvalues_single = eigenvalues_single[eigenvalues_single > 1e-10]
    sum_lambda = np.sum(eigenvalues_single)
    sum_lambda_sq = np.sum(eigenvalues_single ** 2)
    Df_single = sum_lambda ** 2 / sum_lambda_sq

    print(f"  Single dominant eigenvalue:")
    print(f"    Df = {Df_single:.4f} (expected: ~1)")

    # Power-law spectrum lambda_k = k^(-alpha) with known alpha
    for true_alpha in [0.5, 1.0, 1.5, 2.0]:
        eigenvalues_power = np.arange(1, n+1, dtype=float) ** (-true_alpha)

        k = np.arange(1, n + 1)
        log_k = np.log(k)
        log_lambda = np.log(eigenvalues_power)

        n_pts = len(log_k)
        slope = (n_pts * np.sum(log_k * log_lambda) - np.sum(log_k) * np.sum(log_lambda))
        slope /= (n_pts * np.sum(log_k ** 2) - np.sum(log_k) ** 2)
        estimated_alpha = -slope

        error = abs(estimated_alpha - true_alpha) / true_alpha * 100
        print(f"  Power-law alpha={true_alpha}: estimated = {estimated_alpha:.4f} (error: {error:.2f}%)")

    # Test B: Alternative Df computation
    print("\nTest B: Alternative Df formulas")
    print("-"*70)

    # Generate test embedding
    np.random.seed(42)
    n_samples, n_dims = 2500, 50
    R_values = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    R_values = np.clip(R_values, 0.3, 55)

    embeddings = np.zeros((n_samples, n_dims))
    for i, r in enumerate(R_values):
        np.random.seed(i + 42)
        scale = 1.0 / (r + 0.1)
        direction = np.random.randn(n_dims)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        base_pos = np.sin(np.arange(n_dims) * r / 10.0)
        embeddings[i] = base_pos + scale * direction

    # Method 1: Participation ratio (used in code)
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.dot(centered.T, centered) / (n_samples - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    Df_pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)

    # Method 2: Using normalized eigenvalues
    eigenvalues_norm = eigenvalues / np.sum(eigenvalues)
    Df_entropy = np.exp(-np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10)))

    # Method 3: Effective rank (nuclear norm / spectral norm)
    Df_rank = np.sum(eigenvalues) / (eigenvalues[0] + 1e-10)

    print(f"  Method 1 (Participation ratio): Df = {Df_pr:.2f}")
    print(f"  Method 2 (Exp entropy): Df = {Df_entropy:.2f}")
    print(f"  Method 3 (Nuclear/Spectral): Df = {Df_rank:.2f}")

    results["Df_methods"] = {
        "participation_ratio": Df_pr,
        "exp_entropy": Df_entropy,
        "nuclear_spectral": Df_rank
    }

    return results


# ============================================================================
# AUDIT 5: STATISTICAL SIGNIFICANCE
# ============================================================================

def audit_statistical_significance():
    """
    Is 2.9% deviation statistically meaningful?
    What's the variance across bootstrap samples?
    """
    print("\n" + "="*80)
    print("AUDIT 5: STATISTICAL SIGNIFICANCE")
    print("="*80)

    results = {}
    n_samples = 2500
    n_dims = 50
    seed = 42

    # Test A: Bootstrap resampling
    print("\nTest A: Bootstrap resampling (100 iterations)")
    print("-"*70)

    np.random.seed(seed)
    R_values = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    R_values = np.clip(R_values, 0.3, 55)

    bootstrap_products = []
    n_bootstrap = 100

    for b in range(n_bootstrap):
        # Resample R values with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        R_boot = R_values[indices]

        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_boot):
            np.random.seed(i + seed + b * 10000)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        bootstrap_products.append(Df * alpha)

    bootstrap_products = np.array(bootstrap_products)
    mean_product = np.mean(bootstrap_products)
    std_product = np.std(bootstrap_products)
    ci_low = np.percentile(bootstrap_products, 2.5)
    ci_high = np.percentile(bootstrap_products, 97.5)

    print(f"  Bootstrap mean: {mean_product:.2f}")
    print(f"  Bootstrap std: {std_product:.2f}")
    print(f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  8e = {EIGHT_E:.2f}")
    print(f"  Is 8e in CI? {ci_low <= EIGHT_E <= ci_high}")

    results["bootstrap"] = {
        "mean": mean_product,
        "std": std_product,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "eight_e_in_ci": bool(ci_low <= EIGHT_E <= ci_high)
    }

    # Test B: Different random seeds
    print("\nTest B: Different random seeds for embedding construction")
    print("-"*70)

    seed_products = []
    for test_seed in range(1, 51):
        np.random.seed(test_seed)
        R_values = np.random.lognormal(mean=2, sigma=1, size=n_samples)
        R_values = np.clip(R_values, 0.3, 55)

        embeddings = np.zeros((n_samples, n_dims))
        for i, r in enumerate(R_values):
            np.random.seed(i + test_seed)
            scale = 1.0 / (r + 0.1)
            direction = np.random.randn(n_dims)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            base_pos = np.sin(np.arange(n_dims) * r / 10.0)
            embeddings[i] = base_pos + scale * direction

        Df, alpha, _ = compute_from_embeddings(embeddings)
        seed_products.append(Df * alpha)

    seed_products = np.array(seed_products)
    print(f"  Mean across seeds: {np.mean(seed_products):.2f}")
    print(f"  Std across seeds: {np.std(seed_products):.2f}")
    print(f"  Range: [{np.min(seed_products):.2f}, {np.max(seed_products):.2f}]")
    print(f"  % within 15% of 8e: {np.mean(np.abs(seed_products - EIGHT_E) / EIGHT_E < 0.15) * 100:.1f}%")

    results["seed_variance"] = {
        "mean": np.mean(seed_products),
        "std": np.std(seed_products),
        "min": np.min(seed_products),
        "max": np.max(seed_products),
        "pct_within_15": float(np.mean(np.abs(seed_products - EIGHT_E) / EIGHT_E < 0.15) * 100)
    }

    return results


# ============================================================================
# MAIN AUDIT RUNNER
# ============================================================================

def run_full_audit():
    """Run all adversarial audits."""
    print("=" * 80)
    print("ADVERSARIAL AUDIT: 8e EMBEDDING RESULT")
    print("Attempting to verify or falsify the claimed 2.9% deviation from 8e")
    print("=" * 80)
    print(f"\nTheoretical 8e = 8 * e = {EIGHT_E:.4f}")
    print(f"Claimed result: Df x alpha = 21.15 (2.9% deviation)")

    all_results = {}

    # Run all audits
    all_results["tautology"] = audit_tautology()
    all_results["parameter_sensitivity"] = audit_parameter_sensitivity()
    all_results["data_dependence"] = audit_data_dependence()
    all_results["computation"] = audit_computation()
    all_results["statistical"] = audit_statistical_significance()

    # Summary
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = run_full_audit()

    # Save results
    output_path = Path(__file__).parent / "adversarial_audit_results.json"

    def to_builtin(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_builtin(x) for x in obj]
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(to_builtin(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
