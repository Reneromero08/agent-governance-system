"""
Q1 DEEP: Why does grad_S track truth?

The shallow answer: "low dispersion = neighbors agree = reliable"
The deep question: WHY should neighbor agreement indicate truth?

HYPOTHESIS TO TEST:
Agreement tracks truth ONLY when observations are INDEPENDENT.
Correlated observations (echo chambers) have low grad_S but are unreliable.

If this is true, then grad_S isn't measuring "agreement" - it's measuring
"independent verification". The formula should fail when observations
are correlated.

TEST DESIGN:
1. Independent observations → grad_S should predict reliability
2. Correlated observations → grad_S should FAIL to predict reliability
3. If grad_S works in both cases, then it's deeper than independence
"""

import numpy as np
from typing import Tuple, List


def compute_R(observations: np.ndarray, truth: float,
              sigma: float = 0.5, Df: float = 1.0) -> float:
    """
    Use the Gaussian free-energy-derived form:
      z = error/std
      E(z) = exp(-z^2/2)
      R = (E/std) * sigma^Df
    """
    if len(observations) < 2:
        return 0.0

    mean_obs = float(np.mean(observations))
    std = max(float(np.std(observations)), 0.001)
    error = abs(mean_obs - truth)

    z = error / std
    E = float(np.exp(-0.5 * (z ** 2)))

    return (E / std) * (sigma ** Df)


def generate_independent_observations(true_value: float, noise: float, n: int) -> np.ndarray:
    """Each observation is independent sample around truth"""
    return true_value + np.random.normal(0, noise, n)


def generate_correlated_observations(true_value: float, noise: float, n: int,
                                     correlation: float = 0.9) -> np.ndarray:
    """
    Observations are correlated - they share a common error term.
    This simulates echo chamber / groupthink / shared bias.

    High correlation = observations move together (echo chamber)
    Low correlation = independent observations
    """
    # Shared error component
    shared_error = np.random.normal(0, noise)

    # Individual error components
    individual_errors = np.random.normal(0, noise * (1 - correlation), n)

    # Combine: high correlation means shared error dominates
    return true_value + correlation * shared_error + individual_errors


def test_independence_hypothesis():
    """
    CRITICAL TEST: Does grad_S distinguish independent from correlated agreement?
    """
    print("=" * 70)
    print("Q1 DEEP: Does grad_S require INDEPENDENT observations?")
    print("=" * 70)

    results_independent = []
    results_correlated = []

    n_trials = 500

    for _ in range(n_trials):
        true_value = np.random.uniform(-10, 10)
        noise = np.random.uniform(0.5, 2.0)
        n_obs = 20

        # Independent observations
        ind_obs = generate_independent_observations(true_value, noise, n_obs)
        ind_R = compute_R(ind_obs, true_value)
        ind_error = abs(np.mean(ind_obs) - true_value)
        results_independent.append((ind_R, ind_error, np.std(ind_obs)))

        # Highly correlated observations (echo chamber)
        corr_obs = generate_correlated_observations(true_value, noise, n_obs, correlation=0.95)
        corr_R = compute_R(corr_obs, true_value)
        corr_error = abs(np.mean(corr_obs) - true_value)
        results_correlated.append((corr_R, corr_error, np.std(corr_obs)))

    # Analyze
    ind_Rs = np.array([r[0] for r in results_independent])
    ind_errors = np.array([r[1] for r in results_independent])
    ind_stds = np.array([r[2] for r in results_independent])

    corr_Rs = np.array([r[0] for r in results_correlated])
    corr_errors = np.array([r[1] for r in results_correlated])
    corr_stds = np.array([r[2] for r in results_correlated])

    # Key metrics
    ind_R_error_corr = np.corrcoef(ind_Rs, ind_errors)[0, 1]
    corr_R_error_corr = np.corrcoef(corr_Rs, corr_errors)[0, 1]

    print("\n1. OBSERVATION STATISTICS:")
    print(f"   Independent: mean std = {np.mean(ind_stds):.3f}, mean R = {np.mean(ind_Rs):.3f}")
    print(f"   Correlated:  mean std = {np.mean(corr_stds):.3f}, mean R = {np.mean(corr_Rs):.3f}")

    print("\n2. R-ERROR CORRELATION (negative = R predicts low error):")
    print(f"   Independent: r = {ind_R_error_corr:.4f}")
    print(f"   Correlated:  r = {corr_R_error_corr:.4f}")

    print("\n3. ACTUAL ERRORS:")
    print(f"   Independent: mean error = {np.mean(ind_errors):.3f}")
    print(f"   Correlated:  mean error = {np.mean(corr_errors):.3f}")

    # The critical question: does high R in correlated case predict low error?
    corr_high_R = corr_Rs > np.median(corr_Rs)
    corr_high_R_error = np.mean(corr_errors[corr_high_R])
    corr_low_R_error = np.mean(corr_errors[~corr_high_R])

    print("\n4. HIGH R vs LOW R ERROR (correlated case):")
    print(f"   High R mean error: {corr_high_R_error:.3f}")
    print(f"   Low R mean error:  {corr_low_R_error:.3f}")

    print("\n" + "=" * 70)

    # VERDICT
    if abs(corr_R_error_corr) < 0.1:
        print("FINDING: grad_S FAILS for correlated observations!")
        print("\nThis means grad_S isn't just measuring 'agreement' -")
        print("it's measuring 'independent agreement'. Correlated observers")
        print("can have low dispersion but high error.")
        print("\nIMPLICATION: The formula assumes independence.")
        return False, "independence_required"
    elif corr_R_error_corr < -0.2:
        print("FINDING: grad_S WORKS even for correlated observations!")
        print("\nThis is surprising. Even when observations are correlated,")
        print("low dispersion still predicts low error.")
        print("\nIMPLICATION: The formula is deeper than independence.")
        return True, "deeper_than_independence"
    else:
        print("FINDING: UNCLEAR - weak correlation in both cases")
        return None, "unclear"


def test_correlation_gradient():
    """
    Test how R's predictive power degrades as correlation increases.
    """
    print("\n" + "=" * 70)
    print("CORRELATION GRADIENT: How does R degrade with observer correlation?")
    print("=" * 70)

    correlations = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    print(f"\n{'Correlation':<12} {'R-Error Corr':<15} {'Mean Error':<12} {'Mean R':<10}")
    print("-" * 55)

    degradation = []

    for corr in correlations:
        results = []
        for _ in range(300):
            true_value = np.random.uniform(-10, 10)
            noise = 1.5

            obs = generate_correlated_observations(true_value, noise, 20, correlation=corr)
            R = compute_R(obs, true_value)
            error = abs(np.mean(obs) - true_value)
            results.append((R, error))

        Rs = np.array([r[0] for r in results])
        errors = np.array([r[1] for r in results])

        r_error_corr = np.corrcoef(Rs, errors)[0, 1]
        degradation.append((corr, r_error_corr))

        print(f"{corr:<12.2f} {r_error_corr:<15.4f} {np.mean(errors):<12.3f} {np.mean(Rs):<10.3f}")

    # Check if there's a clear degradation pattern
    corrs = [d[0] for d in degradation]
    r_err_corrs = [d[1] for d in degradation]

    # Does R-error correlation get weaker as observation correlation increases?
    trend = np.corrcoef(corrs, np.abs(r_err_corrs))[0, 1]

    print(f"\nTrend: correlation between observer_corr and |R-error_corr|: {trend:.4f}")

    if trend < -0.5:
        print("\nFINDING: R's predictive power DEGRADES with observer correlation")
        print("This confirms: grad_S assumes independent observations.")
        return True
    else:
        print("\nFINDING: R's predictive power is ROBUST to observer correlation")
        print("This suggests: grad_S captures something deeper than independence.")
        return False


def test_what_grad_s_actually_measures():
    """
    Direct test: what IS grad_S measuring if not independence?

    Compare scenarios where:
    - A: Low dispersion, independent, correct (should work)
    - B: Low dispersion, correlated, wrong (echo chamber)
    - C: High dispersion, independent, uncertain (should report uncertainty)
    - D: High dispersion, correlated, chaotic (should fail)
    """
    print("\n" + "=" * 70)
    print("DIRECT COMPARISON: What does grad_S actually measure?")
    print("=" * 70)

    true_value = 0.0
    n_trials = 200

    scenarios = {
        'A: Low disp, independent': lambda: generate_independent_observations(true_value, 0.3, 20),
        'B: Low disp, correlated': lambda: generate_correlated_observations(true_value, 0.3, 20, 0.95),
        'C: High disp, independent': lambda: generate_independent_observations(true_value, 3.0, 20),
        'D: High disp, correlated': lambda: generate_correlated_observations(true_value, 3.0, 20, 0.95),
    }

    print(f"\n{'Scenario':<25} {'Mean R':<10} {'Mean Error':<12} {'R predicts?':<12}")
    print("-" * 60)

    for name, generator in scenarios.items():
        results = []
        for _ in range(n_trials):
            obs = generator()
            R = compute_R(obs, true_value)
            error = abs(np.mean(obs) - true_value)
            results.append((R, error))

        Rs = np.array([r[0] for r in results])
        errors = np.array([r[1] for r in results])

        # Does R predict error within this scenario?
        high_R_error = np.mean(errors[Rs > np.median(Rs)])
        low_R_error = np.mean(errors[Rs <= np.median(Rs)])
        predicts = "YES" if high_R_error < low_R_error else "NO"

        print(f"{name:<25} {np.mean(Rs):<10.3f} {np.mean(errors):<12.3f} {predicts:<12}")


if __name__ == "__main__":
    np.random.seed(42)

    result1, finding1 = test_independence_hypothesis()
    result2 = test_correlation_gradient()
    test_what_grad_s_actually_measures()

    print("\n" + "=" * 70)
    print("Q1 DEEP VERDICT")
    print("=" * 70)

    if finding1 == "independence_required":
        print("""
WHY grad_S WORKS: It measures REDUNDANT INDEPENDENT VERIFICATION.

Low dispersion among INDEPENDENT observers = multiple paths to same answer
Low dispersion among CORRELATED observers = single path copied many times

The formula implicitly assumes observations are independent.
When they're not (echo chambers), it can be fooled.

This answers Q1: grad_S works because independent agreement IS truth.
Not agreement in general - INDEPENDENT agreement.

This also explains Q2 (echo chambers): they violate the independence assumption.
""")
    else:
        print("""
SURPRISING: grad_S works even without independence.

This suggests the formula captures something deeper than
"independent verification" - possibly the geometric structure
of the information space itself.

More investigation needed.
""")
