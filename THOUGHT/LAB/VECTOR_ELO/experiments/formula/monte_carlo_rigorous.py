"""
Monte Carlo - RIGOROUS VERSION (per GPT critique)

Changes:
1. NO outlier trimming
2. Lognormal noise (more realistic)
3. Sweep noise levels and sigma values
4. Pass criteria: CV < 0.5 for noise <= 10% WITHOUT trimming
"""

import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def compute_R_calibrated(H: float, nabla_H: float, d: int = 1, sigma: float = np.e) -> float:
    """Calibrated formula: everything derives from H."""
    if nabla_H < 1e-10:
        nabla_H = 1e-10
    if H < 1e-10:
        H = 1e-10

    alpha = 3 ** (d / 2 - 1)
    Df = max(5 - H, 0.1)
    E = H ** alpha

    R = (E / nabla_H) * (sigma ** Df)
    return R


def monte_carlo_rigorous(
    n_samples: int = 10000,
    noise_level: float = 0.1,
    sigma: float = np.e,
    use_lognormal: bool = True,
    trim_outliers: bool = False
) -> Dict:
    """
    Rigorous Monte Carlo with:
    - Lognormal noise (multiplicative, realistic)
    - No outlier trimming (unless explicitly enabled)
    - Configurable sigma
    """
    np.random.seed(42)

    H_base = 2.5
    nabla_H_base = 0.3

    if use_lognormal:
        # Lognormal: multiplicative noise with specified CV
        # For lognormal, if we want CV = noise_level:
        # sigma_ln = sqrt(ln(1 + CV^2))
        sigma_ln = np.sqrt(np.log(1 + noise_level**2))
        mu_ln = -sigma_ln**2 / 2  # Ensures mean = 1

        H_multiplier = np.random.lognormal(mu_ln, sigma_ln, n_samples)
        nabla_H_multiplier = np.random.lognormal(mu_ln, sigma_ln, n_samples)
    else:
        # Gaussian noise (original)
        H_multiplier = 1 + np.random.randn(n_samples) * noise_level
        nabla_H_multiplier = 1 + np.random.randn(n_samples) * noise_level

    H = H_base * H_multiplier
    H = np.clip(H, 0.1, 4.9)

    nabla_H = nabla_H_base * nabla_H_multiplier
    nabla_H = np.clip(nabla_H, 0.01, 10)

    # Compute R
    R = np.array([compute_R_calibrated(h, nh, sigma=sigma) for h, nh in zip(H, nabla_H)])

    if trim_outliers:
        R_final = R[(R > np.percentile(R, 1)) & (R < np.percentile(R, 99))]
    else:
        R_final = R

    # Handle infinities/NaNs
    R_final = R_final[np.isfinite(R_final)]

    return {
        'R_mean': np.mean(R_final),
        'R_std': np.std(R_final),
        'R_cv': np.std(R_final) / np.mean(R_final),
        'R_median': np.median(R_final),
        'R_iqr': np.percentile(R_final, 75) - np.percentile(R_final, 25),
        'n_valid': len(R_final),
        'noise_level': noise_level,
        'sigma': sigma,
        'trimmed': trim_outliers
    }


def sweep_noise_levels(sigma: float = np.e) -> List[Dict]:
    """Sweep noise from 1% to 20%."""
    noise_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    results = []

    for noise in noise_levels:
        result = monte_carlo_rigorous(
            noise_level=noise,
            sigma=sigma,
            use_lognormal=True,
            trim_outliers=False
        )
        results.append(result)

    return results


def sweep_sigma_values(noise_level: float = 0.1) -> List[Dict]:
    """Sweep sigma from 2 to e to 3."""
    sigma_values = [2.0, np.e, 3.0]
    results = []

    for sigma in sigma_values:
        result = monte_carlo_rigorous(
            noise_level=noise_level,
            sigma=sigma,
            use_lognormal=True,
            trim_outliers=False
        )
        results.append(result)

    return results


def compare_trimmed_vs_untrimmed(noise_level: float = 0.1) -> Dict:
    """Direct comparison of trimmed vs untrimmed."""
    trimmed = monte_carlo_rigorous(
        noise_level=noise_level,
        use_lognormal=True,
        trim_outliers=True
    )

    untrimmed = monte_carlo_rigorous(
        noise_level=noise_level,
        use_lognormal=True,
        trim_outliers=False
    )

    return {
        'trimmed_cv': trimmed['R_cv'],
        'untrimmed_cv': untrimmed['R_cv'],
        'difference': trimmed['R_cv'] - untrimmed['R_cv']
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO - RIGOROUS (GPT Critique Version)")
    print("=" * 60)
    print()
    print("Pass criteria: CV < 0.5 for noise <= 10% WITHOUT trimming")
    print()

    # Test 1: Sweep noise levels
    print("-" * 60)
    print("Test 1: Noise level sweep (lognormal, no trimming)")
    print("-" * 60)

    noise_results = sweep_noise_levels()

    print(f"\n{'Noise':>8} | {'CV':>10} | {'Mean':>12} | {'Std':>12} | {'Status':>10}")
    print("-" * 60)

    all_pass = True
    for r in noise_results:
        status = "PASS" if r['R_cv'] < 0.5 else "FAIL"
        if r['noise_level'] <= 0.10 and r['R_cv'] >= 0.5:
            all_pass = False
            status = "**FAIL**"
        print(f"{r['noise_level']:>8.0%} | {r['R_cv']:>10.4f} | {r['R_mean']:>12.2f} | {r['R_std']:>12.2f} | {status:>10}")

    # Test 2: Sweep sigma values
    print("\n" + "-" * 60)
    print("Test 2: Sigma sweep (lognormal, no trimming, 10% noise)")
    print("-" * 60)

    sigma_results = sweep_sigma_values()

    print(f"\n{'Sigma':>8} | {'CV':>10} | {'Mean':>12} | {'Status':>10}")
    print("-" * 50)

    for r in sigma_results:
        status = "PASS" if r['R_cv'] < 0.5 else "FAIL"
        print(f"{r['sigma']:>8.4f} | {r['R_cv']:>10.4f} | {r['R_mean']:>12.2f} | {status:>10}")

    # Test 3: Trimmed vs untrimmed comparison
    print("\n" + "-" * 60)
    print("Test 3: Trimmed vs Untrimmed (10% noise)")
    print("-" * 60)

    comparison = compare_trimmed_vs_untrimmed()
    print(f"\n  Trimmed CV:   {comparison['trimmed_cv']:.4f}")
    print(f"  Untrimmed CV: {comparison['untrimmed_cv']:.4f}")
    print(f"  Difference:   {comparison['difference']:.4f}")

    if abs(comparison['difference']) < 0.05:
        print("  -> Trimming has minimal effect")
    else:
        print("  -> Trimming significantly affects results")

    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Check pass criteria: CV < 0.5 for noise <= 10% without trimming
    passes_at_10pct = noise_results[3]['R_cv'] < 0.5  # 10% is index 3
    passes_at_5pct = noise_results[2]['R_cv'] < 0.5   # 5% is index 2
    passes_at_2pct = noise_results[1]['R_cv'] < 0.5   # 2% is index 1

    print(f"\nCV at  2% noise: {noise_results[1]['R_cv']:.4f} {'PASS' if passes_at_2pct else 'FAIL'}")
    print(f"CV at  5% noise: {noise_results[2]['R_cv']:.4f} {'PASS' if passes_at_5pct else 'FAIL'}")
    print(f"CV at 10% noise: {noise_results[3]['R_cv']:.4f} {'PASS' if passes_at_10pct else 'FAIL'}")

    if passes_at_10pct:
        print("\n** RIGOROUS MONTE CARLO: VALIDATED")
        print("   CV < 0.5 for noise <= 10% WITHOUT trimming")
    elif passes_at_5pct:
        print("\n*  PARTIAL PASS: CV < 0.5 for noise <= 5%")
    else:
        print("\nX  RIGOROUS MONTE CARLO: FAILED")
        print("   Formula is too sensitive to noise")
