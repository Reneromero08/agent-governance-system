"""
Q44: Statistical Validation
===========================

Bootstrap confidence intervals and permutation tests for
validating the correlation between R and Born rule probability.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    r: float                    # Pearson correlation
    r_normalized: float         # Correlation on normalized values
    mae: float                  # Mean absolute error (normalized)
    ci_low: float               # 95% CI lower bound
    ci_high: float              # 95% CI upper bound
    p_value: float              # Permutation test p-value
    verdict: str                # QUANTUM / NEEDS_ADJUSTMENT / NOT_QUANTUM


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2:
        return 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def normalize_to_01(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] range."""
    min_val, max_val = values.min(), values.max()
    if max_val - min_val < 1e-10:
        return np.full_like(values, 0.5)
    return (values - min_val) / (max_val - min_val)


def bootstrap_confidence_interval(
    R_values: List[float],
    P_values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, List[float]]:
    """
    Compute bootstrap confidence interval for correlation.

    Args:
        R_values: List of R values
        P_values: List of Born probabilities
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)
        seed: Random seed for reproducibility

    Returns:
        (ci_low, ci_high, all_correlations)
    """
    np.random.seed(seed)

    R_arr = np.array(R_values)
    P_arr = np.array(P_values)
    n = len(R_values)

    correlations = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        R_boot = R_arr[indices]
        P_boot = P_arr[indices]

        # Compute correlation
        corr = pearson_correlation(R_boot, P_boot)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Compute percentiles
    alpha = 1 - confidence
    ci_low = float(np.percentile(correlations, 100 * alpha / 2))
    ci_high = float(np.percentile(correlations, 100 * (1 - alpha / 2)))

    return ci_low, ci_high, list(correlations)


def permutation_test(
    R_values: List[float],
    P_values: List[float],
    n_permutations: int = 10000,
    seed: int = 42
) -> Tuple[float, List[float]]:
    """
    Permutation test for correlation significance.

    H0: R and P_born are independent (rho = 0)
    H1: R and P_born are correlated (rho > 0)

    Args:
        R_values: List of R values
        P_values: List of Born probabilities
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        (p_value, null_distribution)
    """
    np.random.seed(seed)

    R_arr = np.array(R_values)
    P_arr = np.array(P_values)

    # Observed correlation
    observed_corr = pearson_correlation(R_arr, P_arr)

    # Generate null distribution
    null_correlations = []
    for _ in range(n_permutations):
        P_shuffled = np.random.permutation(P_arr)
        null_corr = pearson_correlation(R_arr, P_shuffled)
        null_correlations.append(null_corr)

    null_correlations = np.array(null_correlations)

    # One-tailed p-value (testing rho > 0)
    p_value = float(np.mean(null_correlations >= observed_corr))

    return p_value, list(null_correlations)


def compute_mae(R_values: List[float], P_values: List[float]) -> float:
    """
    Compute Mean Absolute Error on normalized values.

    Both R and P_born are normalized to [0, 1] before computing MAE.
    """
    R_norm = normalize_to_01(np.array(R_values))
    P_norm = normalize_to_01(np.array(P_values))
    return float(np.mean(np.abs(R_norm - P_norm)))


def get_verdict(r: float) -> str:
    """
    Determine verdict based on correlation.

    r > 0.9 -> QUANTUM
    0.7 < r < 0.9 -> NEEDS_ADJUSTMENT
    r < 0.7 -> NOT_QUANTUM
    """
    if r > 0.9:
        return "QUANTUM"
    elif r > 0.7:
        return "NEEDS_ADJUSTMENT"
    else:
        return "NOT_QUANTUM"


def full_correlation_analysis(
    R_values: List[float],
    P_values: List[float],
    n_bootstrap: int = 1000,
    n_permutations: int = 10000,
    seed: int = 42
) -> CorrelationResult:
    """
    Complete correlation analysis with bootstrap CI and permutation test.

    Args:
        R_values: List of R values from formula
        P_values: List of Born rule probabilities
        n_bootstrap: Number of bootstrap samples
        n_permutations: Number of permutation samples
        seed: Random seed

    Returns:
        CorrelationResult with all statistics
    """
    R_arr = np.array(R_values)
    P_arr = np.array(P_values)

    # Raw correlation
    r = pearson_correlation(R_arr, P_arr)

    # Normalized correlation (remove scale effects)
    R_norm = normalize_to_01(R_arr)
    P_norm = normalize_to_01(P_arr)
    r_normalized = pearson_correlation(R_norm, P_norm)

    # Mean absolute error
    mae = compute_mae(R_values, P_values)

    # Bootstrap CI
    ci_low, ci_high, _ = bootstrap_confidence_interval(
        R_values, P_values, n_bootstrap, seed=seed
    )

    # Permutation test
    p_value, _ = permutation_test(
        R_values, P_values, n_permutations, seed=seed
    )

    # Verdict based on normalized correlation
    verdict = get_verdict(r_normalized)

    return CorrelationResult(
        r=r,
        r_normalized=r_normalized,
        mae=mae,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        verdict=verdict
    )


def analyze_by_category(
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlation by test case category.

    Args:
        results: List of result dicts with 'category', 'R', 'P_born'

    Returns:
        Dict of category -> {r, r_normalized, n}
    """
    categories = {}

    for result in results:
        cat = result.get('category', 'UNKNOWN')
        if cat not in categories:
            categories[cat] = {'R': [], 'P_born': []}
        categories[cat]['R'].append(result['R'])
        categories[cat]['P_born'].append(result['P_born'])

    analysis = {}
    for cat, data in categories.items():
        R_arr = np.array(data['R'])
        P_arr = np.array(data['P_born'])

        r = pearson_correlation(R_arr, P_arr)
        R_norm = normalize_to_01(R_arr)
        P_norm = normalize_to_01(P_arr)
        r_normalized = pearson_correlation(R_norm, P_norm)

        analysis[cat] = {
            'r': r,
            'r_normalized': r_normalized,
            'n': len(R_arr),
            'R_mean': float(np.mean(R_arr)),
            'P_mean': float(np.mean(P_arr)),
        }

    return analysis


def check_monotonicity(R_values: List[float], P_values: List[float]) -> Dict[str, float]:
    """
    Check if R and P_born have monotonic relationship.

    High R should correspond to high P_born, low R to low P_born.
    """
    R_arr = np.array(R_values)
    P_arr = np.array(P_values)

    # Sort by R
    sort_idx = np.argsort(R_arr)
    R_sorted = R_arr[sort_idx]
    P_sorted = P_arr[sort_idx]

    # Check if P is generally increasing with R
    # Use Spearman rank correlation
    n = len(R_arr)
    R_ranks = np.argsort(np.argsort(R_arr))
    P_ranks = np.argsort(np.argsort(P_arr))

    rank_diff = R_ranks - P_ranks
    spearman_rho = 1 - (6 * np.sum(rank_diff**2)) / (n * (n**2 - 1))

    # Check quartile alignment
    q1_idx = n // 4
    q3_idx = 3 * n // 4

    low_R_mean_P = float(np.mean(P_sorted[:q1_idx]))
    high_R_mean_P = float(np.mean(P_sorted[q3_idx:]))

    return {
        'spearman_rho': float(spearman_rho),
        'low_R_mean_P': low_R_mean_P,
        'high_R_mean_P': high_R_mean_P,
        'monotonic': high_R_mean_P > low_R_mean_P
    }


if __name__ == "__main__":
    # Quick test
    print("Q44 Statistics - Test")
    print("=" * 50)

    # Generate synthetic correlated data
    np.random.seed(42)
    n = 100
    P_born = np.random.uniform(0, 1, n)
    noise = np.random.normal(0, 0.1, n)
    R = P_born * 10 + noise  # R proportional to P_born with noise

    result = full_correlation_analysis(list(R), list(P_born))

    print(f"Pearson r: {result.r:.4f}")
    print(f"Normalized r: {result.r_normalized:.4f}")
    print(f"MAE: {result.mae:.4f}")
    print(f"95% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
    print(f"p-value: {result.p_value:.6f}")
    print(f"Verdict: {result.verdict}")

    mono = check_monotonicity(list(R), list(P_born))
    print(f"\nMonotonicity:")
    print(f"  Spearman rho: {mono['spearman_rho']:.4f}")
    print(f"  Low R mean P: {mono['low_R_mean_P']:.4f}")
    print(f"  High R mean P: {mono['high_R_mean_P']:.4f}")
    print(f"  Monotonic: {mono['monotonic']}")
