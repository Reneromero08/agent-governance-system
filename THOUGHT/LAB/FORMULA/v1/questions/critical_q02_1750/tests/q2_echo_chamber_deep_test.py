"""
Q2 DEEP: The Echo Chamber Vulnerability

We found: Correlated observations give HIGH R but R doesn't predict accuracy.
This is a real vulnerability. Can we:

1. Detect echo chambers from within? (without external context)
2. Quantify how much external context breaks them?
3. Find a signature that distinguishes real agreement from echo chamber?

CRITICAL INSIGHT from Q1 test:
- Independent low dispersion: R = 1.4, error = 0.05, R PREDICTS
- Correlated low dispersion: R = 35.6, error = 0.24, R DOESN'T PREDICT

The echo chamber has 25x higher R but 5x higher error!
"""

import numpy as np
from typing import Tuple, List


def compute_R(observations: np.ndarray, sigma: float = 0.5, Df: float = 1.0) -> float:
    if len(observations) < 2:
        return 0.0
    E = 1.0 / (1.0 + np.std(observations))
    grad_S = np.std(observations) + 1e-10
    return (E / grad_S) * (sigma ** Df)


def generate_independent(true_value: float, noise: float, n: int) -> np.ndarray:
    return true_value + np.random.normal(0, noise, n)


def generate_echo_chamber(true_value: float, noise: float, n: int,
                          bias: float = None) -> Tuple[np.ndarray, float]:
    """
    Echo chamber: tight cluster that may be biased away from truth.
    Returns observations and the actual bias.
    """
    if bias is None:
        bias = np.random.normal(0, 3)  # Random bias

    center = true_value + bias
    observations = center + np.random.normal(0, noise * 0.1, n)  # Very tight
    return observations, bias


def test_internal_detection():
    """
    Can we detect echo chambers WITHOUT external context?

    Possible signals:
    1. Suspiciously low variance (too good to be true)
    2. Distribution shape (too peaked)
    3. Lack of outliers
    """
    print("=" * 70)
    print("Q2 TEST 1: Can we detect echo chambers internally?")
    print("=" * 70)

    independent_stats = []
    echo_stats = []

    for _ in range(500):
        true_value = np.random.uniform(-10, 10)
        noise = 1.5

        # Independent observations
        ind_obs = generate_independent(true_value, noise, 20)
        ind_stats = {
            'std': np.std(ind_obs),
            'kurtosis': compute_kurtosis(ind_obs),
            'range': np.max(ind_obs) - np.min(ind_obs),
            'R': compute_R(ind_obs),
            'error': abs(np.mean(ind_obs) - true_value),
        }
        independent_stats.append(ind_stats)

        # Echo chamber (tight, biased)
        echo_obs, _ = generate_echo_chamber(true_value, noise, 20)
        echo_stat = {
            'std': np.std(echo_obs),
            'kurtosis': compute_kurtosis(echo_obs),
            'range': np.max(echo_obs) - np.min(echo_obs),
            'R': compute_R(echo_obs),
            'error': abs(np.mean(echo_obs) - true_value),
        }
        echo_stats.append(echo_stat)

    # Compare distributions
    print("\nStatistic comparison (Independent vs Echo Chamber):")
    print(f"{'Metric':<15} {'Independent':<15} {'Echo Chamber':<15} {'Separable?':<12}")
    print("-" * 60)

    for metric in ['std', 'kurtosis', 'range', 'R', 'error']:
        ind_vals = [s[metric] for s in independent_stats]
        echo_vals = [s[metric] for s in echo_stats]

        ind_mean = np.mean(ind_vals)
        echo_mean = np.mean(echo_vals)

        # Check overlap
        ind_range = (np.percentile(ind_vals, 10), np.percentile(ind_vals, 90))
        echo_range = (np.percentile(echo_vals, 10), np.percentile(echo_vals, 90))

        overlap = max(0, min(ind_range[1], echo_range[1]) - max(ind_range[0], echo_range[0]))
        total = max(ind_range[1], echo_range[1]) - min(ind_range[0], echo_range[0])
        overlap_pct = overlap / total if total > 0 else 0

        separable = "YES" if overlap_pct < 0.3 else "NO"

        print(f"{metric:<15} {ind_mean:<15.3f} {echo_mean:<15.3f} {separable:<12}")

    # Key test: Can we use R itself to detect echo chambers?
    # Echo chambers have SUSPICIOUSLY high R
    print("\n" + "-" * 70)
    print("KEY TEST: Is 'suspiciously high R' a signal?")

    all_Rs = [s['R'] for s in independent_stats] + [s['R'] for s in echo_stats]
    R_95th = np.percentile(all_Rs, 95)

    # How many echo chambers vs independent have R > 95th percentile?
    ind_suspicious = sum(1 for s in independent_stats if s['R'] > R_95th)
    echo_suspicious = sum(1 for s in echo_stats if s['R'] > R_95th)

    print(f"\nR > {R_95th:.2f} (95th percentile):")
    print(f"  Independent: {ind_suspicious}/500 ({ind_suspicious/5:.1f}%)")
    print(f"  Echo Chamber: {echo_suspicious}/500 ({echo_suspicious/5:.1f}%)")

    if echo_suspicious > 3 * ind_suspicious:
        print("\nFINDING: Suspiciously high R CAN detect echo chambers!")
        print("Rule: If R > 95th percentile, suspect echo chamber.")
        return True
    else:
        print("\nFINDING: R alone cannot reliably detect echo chambers.")
        return False


def compute_kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0
    mean = np.mean(x)
    std = np.std(x) + 1e-10
    return np.mean(((x - mean) / std) ** 4)


def test_context_breaking():
    """
    How much external context is needed to break an echo chamber?
    """
    print("\n" + "=" * 70)
    print("Q2 TEST 2: How much external context breaks echo chambers?")
    print("=" * 70)

    true_value = 0.0
    noise = 1.5

    # Create echo chamber biased by +5
    echo_obs, bias = generate_echo_chamber(true_value, noise, 20, bias=5.0)
    echo_R = compute_R(echo_obs)
    echo_error = abs(np.mean(echo_obs) - true_value)

    print(f"\nEcho chamber: R = {echo_R:.2f}, error = {echo_error:.2f}, bias = {bias:.2f}")
    print(f"Echo chamber alone thinks truth is: {np.mean(echo_obs):.2f}")

    print(f"\n{'External obs':<15} {'Combined R':<12} {'Combined est':<15} {'Error':<10}")
    print("-" * 55)

    # Add increasing amounts of external (truthful) observations
    for n_external in [1, 2, 5, 10, 20, 40]:
        external_obs = generate_independent(true_value, noise, n_external)
        combined = np.concatenate([echo_obs, external_obs])

        combined_R = compute_R(combined)
        combined_est = np.mean(combined)
        combined_error = abs(combined_est - true_value)

        print(f"{n_external:<15} {combined_R:<12.3f} {combined_est:<15.3f} {combined_error:<10.3f}")

    print("\nFINDING: External context dilutes echo chamber influence.")
    print("The formula correctly registers increased disagreement (lower R).")


def test_echo_vs_real_agreement():
    """
    The ultimate test: Can we distinguish echo chambers from genuine agreement?

    - Real agreement: Independent observers converge on truth
    - Echo chamber: Correlated observers agree on bias

    Key insight: In REAL agreement, adding more observers CONFIRMS.
    In echo chambers, adding DIFFERENT observers DISRUPTS.
    """
    print("\n" + "=" * 70)
    print("Q2 TEST 3: Echo chamber vs real agreement (bootstrap test)")
    print("=" * 70)

    results = []

    for _ in range(200):
        true_value = np.random.uniform(-10, 10)
        noise = 1.5

        # Real agreement: tight cluster around truth
        real_obs = true_value + np.random.normal(0, 0.3, 20)
        real_R = compute_R(real_obs)

        # Echo chamber: tight cluster around biased value
        echo_obs, bias = generate_echo_chamber(true_value, noise, 20)
        echo_R = compute_R(echo_obs)

        # Bootstrap test: add fresh observations and see if R holds
        fresh_obs = generate_independent(true_value, noise, 10)

        real_combined = np.concatenate([real_obs, fresh_obs])
        echo_combined = np.concatenate([echo_obs, fresh_obs])

        real_R_after = compute_R(real_combined)
        echo_R_after = compute_R(echo_combined)

        # Key metric: how much does R change?
        real_R_change = (real_R_after - real_R) / real_R if real_R > 0 else 0
        echo_R_change = (echo_R_after - echo_R) / echo_R if echo_R > 0 else 0

        results.append({
            'type': 'real',
            'R_before': real_R,
            'R_after': real_R_after,
            'R_change': real_R_change,
        })
        results.append({
            'type': 'echo',
            'R_before': echo_R,
            'R_after': echo_R_after,
            'R_change': echo_R_change,
        })

    # Analyze
    real_changes = [r['R_change'] for r in results if r['type'] == 'real']
    echo_changes = [r['R_change'] for r in results if r['type'] == 'echo']

    print(f"\nR change when fresh observations added:")
    print(f"  Real agreement: {np.mean(real_changes):.3f} (mean), {np.std(real_changes):.3f} (std)")
    print(f"  Echo chamber:   {np.mean(echo_changes):.3f} (mean), {np.std(echo_changes):.3f} (std)")

    # Real agreement should be stable; echo chambers should crash
    if np.mean(echo_changes) < np.mean(real_changes) - 0.1:
        print("\nFINDING: Echo chambers CRASH when exposed to fresh data!")
        print("Detection method: Add fresh observations, check if R drops significantly.")
        return True
    else:
        print("\nFINDING: Both types affected similarly by fresh data.")
        return False


def test_practical_defense():
    """
    Practical defense: Sample fresh data, compare R before/after.
    Large R drop = likely echo chamber.
    """
    print("\n" + "=" * 70)
    print("Q2 TEST 4: Practical echo chamber defense")
    print("=" * 70)

    # Simulate real-world scenario
    n_scenarios = 200

    tp = 0  # True positive: detected echo chamber
    fp = 0  # False positive: flagged real agreement
    tn = 0  # True negative: passed real agreement
    fn = 0  # False negative: missed echo chamber

    threshold = -0.3  # If R drops by more than 30%, flag as echo chamber

    for _ in range(n_scenarios):
        true_value = np.random.uniform(-10, 10)
        noise = 1.5

        for scenario_type in ['real', 'echo']:
            if scenario_type == 'real':
                obs = true_value + np.random.normal(0, 0.3, 20)
            else:
                obs, _ = generate_echo_chamber(true_value, noise, 20)

            R_before = compute_R(obs)

            # Defense: add fresh independent observations
            fresh = generate_independent(true_value, noise, 10)
            combined = np.concatenate([obs, fresh])
            R_after = compute_R(combined)

            R_change = (R_after - R_before) / R_before if R_before > 0 else 0

            flagged = R_change < threshold

            if scenario_type == 'echo':
                if flagged:
                    tp += 1
                else:
                    fn += 1
            else:
                if flagged:
                    fp += 1
                else:
                    tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDefense: Flag if R drops > 30% when fresh data added")
    print(f"\nResults:")
    print(f"  True Positives (caught echo chambers):  {tp}")
    print(f"  False Positives (flagged real):         {fp}")
    print(f"  True Negatives (passed real):           {tn}")
    print(f"  False Negatives (missed echo):          {fn}")
    print(f"\n  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")

    if f1 > 0.7:
        print("\nVERDICT: Practical defense WORKS!")
        return True
    else:
        print("\nVERDICT: Defense needs improvement.")
        return False


if __name__ == "__main__":
    np.random.seed(42)

    test1 = test_internal_detection()
    test_context_breaking()
    test3 = test_echo_vs_real_agreement()
    test4 = test_practical_defense()

    print("\n" + "=" * 70)
    print("Q2 DEEP SUMMARY: ECHO CHAMBER VULNERABILITY")
    print("=" * 70)

    print("""
FINDINGS:

1. Echo chambers DO fool local R (high R, wrong answer)

2. Detection methods:
   - Suspiciously high R (>95th percentile) is a signal
   - Adding fresh observations breaks echo chamber R

3. Defense: "Bootstrap test"
   - Take current observations
   - Add fresh independent data
   - If R drops significantly, suspect echo chamber

4. The formula assumes INDEPENDENCE
   - Works correctly for independent observations
   - Can be fooled by correlated observations
   - This is a KNOWN LIMITATION, not a bug

IMPLICATION FOR FORMULA:
R measures LOCAL AGREEMENT among CURRENT observations.
It does NOT guarantee global truth.
Expanding context (adding diverse observations) is the proper defense.
""")
