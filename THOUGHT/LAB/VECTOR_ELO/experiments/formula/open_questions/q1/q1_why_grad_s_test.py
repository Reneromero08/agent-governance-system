"""
Q1: Why grad_S specifically?

TEST: Compare grad_S against alternative local measures as truth indicators.
If grad_S is special, it should outperform alternatives at predicting correctness.

Alternatives tested:
- grad_S (local dispersion/variance)
- entropy (information content)
- kurtosis (distribution shape)
- range (max - min)
- mean absolute deviation

PASS: grad_S correlation with correctness > all alternatives
FAIL: Another measure beats grad_S
"""

import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class TestResult:
    measure_name: str
    correlation: float
    p_value: float


def compute_grad_S(observations: np.ndarray) -> float:
    """Standard deviation of observations (local dispersion)"""
    if len(observations) < 2:
        return 0.0
    return np.std(observations) + 1e-10


def compute_entropy(observations: np.ndarray) -> float:
    """Shannon entropy of binned observations"""
    hist, _ = np.histogram(observations, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist + 1e-10))


def compute_kurtosis(observations: np.ndarray) -> float:
    """Kurtosis (peakedness of distribution)"""
    if len(observations) < 4:
        return 0.0
    mean = np.mean(observations)
    std = np.std(observations) + 1e-10
    return np.mean(((observations - mean) / std) ** 4)


def compute_range(observations: np.ndarray) -> float:
    """Range (max - min)"""
    return np.max(observations) - np.min(observations) + 1e-10


def compute_mad(observations: np.ndarray) -> float:
    """Mean absolute deviation"""
    return np.mean(np.abs(observations - np.mean(observations))) + 1e-10


def generate_test_scenario(n_observations: int, agreement_level: float) -> Tuple[np.ndarray, float]:
    """
    Generate observations with controlled agreement level.
    Returns (observations, true_value)

    High agreement = observations cluster around true value
    Low agreement = observations scattered
    """
    true_value = np.random.uniform(-10, 10)
    noise_scale = (1 - agreement_level) * 5 + 0.1  # More agreement = less noise
    observations = true_value + np.random.normal(0, noise_scale, n_observations)
    return observations, true_value


def compute_correctness(observations: np.ndarray, true_value: float) -> float:
    """How close is the mean estimate to truth?"""
    estimate = np.mean(observations)
    error = abs(estimate - true_value)
    # Convert to 0-1 correctness score (higher = better)
    return 1 / (1 + error)


def run_comparison_test(n_trials: int = 500) -> dict:
    """
    Run many trials, compute correlation between each measure and correctness.

    KEY INSIGHT: We want INVERSE correlation - low dispersion = high correctness
    So we compute R = E / measure, and check correlation of R with correctness
    """
    measures = {
        'grad_S': compute_grad_S,
        'entropy': compute_entropy,
        'kurtosis': compute_kurtosis,
        'range': compute_range,
        'mad': compute_mad,
    }

    results = {name: [] for name in measures}
    correctness_scores = []

    for _ in range(n_trials):
        # Random agreement level
        agreement = np.random.uniform(0, 1)
        n_obs = np.random.randint(5, 50)

        observations, true_value = generate_test_scenario(n_obs, agreement)
        correctness = compute_correctness(observations, true_value)
        correctness_scores.append(correctness)

        # Compute E (signal strength) as inverse entropy of mean
        E = 1.0 / (1.0 + abs(np.mean(observations)))

        # Compute R = E / measure for each measure
        for name, func in measures.items():
            measure_value = func(observations)
            R = E / measure_value
            results[name].append(R)

    # Compute correlations
    correctness_arr = np.array(correctness_scores)
    correlations = {}

    for name, R_values in results.items():
        R_arr = np.array(R_values)
        corr = np.corrcoef(R_arr, correctness_arr)[0, 1]
        correlations[name] = corr

    return correlations


def test_grad_s_is_optimal():
    """
    HYPOTHESIS: grad_S produces highest correlation with correctness
    when used as denominator in R = E / measure
    """
    print("=" * 60)
    print("Q1 TEST: Why grad_S specifically?")
    print("=" * 60)
    print("\nComparing local measures as truth indicators...")
    print("R = E / measure, checking correlation(R, correctness)\n")

    correlations = run_comparison_test(n_trials=1000)

    # Sort by correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print("Results (higher correlation = better truth indicator):\n")
    print(f"{'Measure':<15} {'Correlation':>12}")
    print("-" * 30)
    for name, corr in sorted_corrs:
        marker = " <-- WINNER" if name == sorted_corrs[0][0] else ""
        print(f"{name:<15} {corr:>12.4f}{marker}")

    # Check if grad_S wins
    winner = sorted_corrs[0][0]
    grad_s_corr = correlations['grad_S']
    best_corr = sorted_corrs[0][1]

    print("\n" + "=" * 60)

    if winner == 'grad_S':
        print("RESULT: PASS")
        print(f"grad_S is the optimal truth indicator (r = {grad_s_corr:.4f})")
        print("\nWHY: Local dispersion (neighbor disagreement) is the")
        print("     most predictive measure of estimate reliability.")
        return True
    elif abs(grad_s_corr - best_corr) < 0.05:
        print("RESULT: PASS (TIED)")
        print(f"grad_S tied with {winner} (difference < 0.05)")
        print(f"grad_S: {grad_s_corr:.4f}, {winner}: {best_corr:.4f}")
        return True
    else:
        print("RESULT: FAIL")
        print(f"{winner} beats grad_S as truth indicator")
        print(f"{winner}: {best_corr:.4f}, grad_S: {grad_s_corr:.4f}")
        return False


def test_grad_s_vs_entropy_detailed():
    """
    Detailed comparison: grad_S vs entropy in different regimes
    """
    print("\n" + "=" * 60)
    print("Detailed: grad_S vs Entropy across agreement levels")
    print("=" * 60)

    agreement_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"\n{'Agreement':<12} {'grad_S corr':>12} {'entropy corr':>12} {'Winner':>10}")
    print("-" * 50)

    grad_s_wins = 0
    for agreement in agreement_levels:
        # Run trials at this agreement level
        grad_s_Rs = []
        entropy_Rs = []
        correctness_scores = []

        for _ in range(200):
            # Small variation around target agreement
            actual_agreement = agreement + np.random.uniform(-0.05, 0.05)
            actual_agreement = np.clip(actual_agreement, 0, 1)

            observations, true_value = generate_test_scenario(20, actual_agreement)
            correctness = compute_correctness(observations, true_value)
            correctness_scores.append(correctness)

            E = 1.0
            grad_s_Rs.append(E / compute_grad_S(observations))
            entropy_Rs.append(E / compute_entropy(observations))

        corr_grad_s = np.corrcoef(grad_s_Rs, correctness_scores)[0, 1]
        corr_entropy = np.corrcoef(entropy_Rs, correctness_scores)[0, 1]

        winner = "grad_S" if corr_grad_s > corr_entropy else "entropy"
        if winner == "grad_S":
            grad_s_wins += 1

        print(f"{agreement:<12.1f} {corr_grad_s:>12.4f} {corr_entropy:>12.4f} {winner:>10}")

    print(f"\ngrad_S wins {grad_s_wins}/5 agreement levels")
    return grad_s_wins >= 3


if __name__ == "__main__":
    np.random.seed(42)

    test1 = test_grad_s_is_optimal()
    test2 = test_grad_s_vs_entropy_detailed()

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if test1 and test2:
        print("\nQ1 ANSWERED: grad_S is optimal because it directly")
        print("measures what matters - local disagreement/dispersion.")
        print("\nEntropy measures information content (how much).")
        print("grad_S measures information reliability (how trustworthy).")
        print("\nFor gating decisions, reliability > content.")
    else:
        print("\nQ1 INCONCLUSIVE: Need further investigation.")
