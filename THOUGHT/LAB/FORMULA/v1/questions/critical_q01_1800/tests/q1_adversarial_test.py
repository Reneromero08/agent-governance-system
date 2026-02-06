"""
Q1 ADVERSARIAL: Try to BREAK grad_S

The question: Why grad_S specifically?

WRONG APPROACH: Show grad_S correlates with correctness (trivial, confirmatory)
RIGHT APPROACH: Try to construct cases where grad_S is FOOLED

Attack vectors:
1. Construct low grad_S (looks like agreement) that is WRONG
2. Construct high grad_S (looks like disagreement) that is RIGHT
3. Find the boundary conditions where grad_S fails

If we can break it systematically, grad_S has a flaw.
If we can only break it in degenerate cases, those are the known limits.
If we can't break it at all, grad_S is fundamental.
"""

import numpy as np
from typing import Tuple, List, Callable


def compute_grad_S(observations: np.ndarray) -> float:
    """Local dispersion - standard deviation of observations"""
    return max(float(np.std(observations)), 0.001)


def compute_essence_v1(observations: np.ndarray) -> float:
    """Original E: inverse of dispersion (CIRCULAR - don't use)"""
    return 1.0 / (1.0 + np.std(observations))


def compute_essence_v2(observations: np.ndarray, target: float = None) -> float:
    """
    Better E: measures how much the mean RESOLVES uncertainty.
    If we know target, E = closeness to target.
    If we don't know target, E = concentration (kurtosis? negative entropy?)
    """
    if target is not None:
        # We have ground truth - use a likelihood-shaped compatibility term.
        # z = error/std is dimensionless and kills "confident but wrong" clusters.
        mean = float(np.mean(observations))
        error = abs(mean - target)
        std = compute_grad_S(observations)
        z = error / std
        return float(np.exp(-0.5 * (z ** 2)))
    else:
        # No ground truth - E is concentration of distribution
        # High kurtosis = peaked = high essence
        mean = np.mean(observations)
        std = compute_grad_S(observations)
        kurtosis = np.mean(((observations - mean) / std) ** 4)
        return kurtosis / 3.0  # Normalize (normal dist has kurtosis=3)


def compute_R(observations: np.ndarray, target: float = None,
              sigma: float = 0.5, Df: float = 1.0,
              use_v2_essence: bool = True) -> float:
    """
    R = E / grad_S * sigma^Df

    With v2 essence, R measures: resolvability / dispersion * scale
    """
    if use_v2_essence:
        E = compute_essence_v2(observations, target)
    else:
        E = compute_essence_v1(observations)

    grad_S = compute_grad_S(observations)
    return (E / grad_S) * (sigma ** Df)


# =============================================================================
# ATTACK 1: Low grad_S, Wrong Answer (The Echo Chamber)
# =============================================================================
def attack_low_grad_s_wrong():
    """
    Construct observations with LOW grad_S but WRONG answer.
    This should fool the formula if it only looks at dispersion.

    Key insight: echo chamber has low dispersion but wrong mean.
    The question is: does the formula catch this?
    """
    print("=" * 70)
    print("ATTACK 1: Low grad_S, Wrong Answer")
    print("=" * 70)

    true_value = 0.0
    results = []

    print("\nConstructing echo chambers with varying tightness and bias...")
    print(f"{'Bias':>8} {'Tightness':>10} {'grad_S':>10} {'Error':>10} {'R (v1)':>10} {'R (v2)':>10}")
    print("-" * 65)

    for bias in [1, 3, 5, 10]:  # How wrong the echo chamber is
        for tightness in [0.01, 0.1, 0.5]:  # How tight the agreement is
            # Echo chamber: tight cluster around wrong value
            wrong_center = true_value + bias
            observations = wrong_center + np.random.normal(0, tightness, 20)

            grad_S = compute_grad_S(observations)
            error = abs(np.mean(observations) - true_value)

            # R with circular E (should be fooled)
            R_v1 = compute_R(observations, use_v2_essence=False)

            # R with proper E that knows ground truth
            R_v2 = compute_R(observations, target=true_value, use_v2_essence=True)

            results.append({
                'bias': bias,
                'tightness': tightness,
                'grad_S': grad_S,
                'error': error,
                'R_v1': R_v1,
                'R_v2': R_v2,
            })

            print(f"{bias:>8} {tightness:>10.2f} {grad_S:>10.4f} {error:>10.2f} {R_v1:>10.2f} {R_v2:>10.2f}")

    # Analysis
    print("\nAnalysis:")
    high_error_results = [r for r in results if r['error'] > 3]

    v1_fooled = [r for r in high_error_results if r['R_v1'] > 1.0]
    v2_fooled = [r for r in high_error_results if r['R_v2'] > 1.0]

    print(f"  High error cases (error > 3): {len(high_error_results)}")
    print(f"  R_v1 fooled (high R despite high error): {len(v1_fooled)}")
    print(f"  R_v2 fooled (high R despite high error): {len(v2_fooled)}")

    if len(v1_fooled) > 0:
        print("\n  FINDING: grad_S alone (v1) IS FOOLED by echo chambers!")
        print("  The circular E definition makes R = 1/grad_S^2, which is useless.")

    if len(v2_fooled) == 0:
        print("\n  FINDING: With proper E (v2), formula is NOT fooled.")
        print("  E must measure RESOLUTION, not just concentration.")

    return len(v1_fooled), len(v2_fooled)


# =============================================================================
# ATTACK 2: High grad_S, Right Answer (The Noisy Truth)
# =============================================================================
def attack_high_grad_s_right():
    """
    Construct observations with HIGH grad_S but RIGHT answer (on average).

    If the formula says "don't trust" but the answer is actually correct,
    is that a failure or a feature?
    """
    print("\n" + "=" * 70)
    print("ATTACK 2: High grad_S, Right Answer (Noisy Truth)")
    print("=" * 70)

    true_value = 0.0
    results = []

    print("\nConstructing noisy-but-correct observations...")
    print(f"{'Noise':>8} {'N':>6} {'grad_S':>10} {'Error':>10} {'R (v2)':>10} {'Correct?':>10}")
    print("-" * 60)

    for noise in [1, 3, 5, 10]:
        for n in [10, 50, 200]:
            # Noisy observations around truth
            observations = true_value + np.random.normal(0, noise, n)

            grad_S = compute_grad_S(observations)
            estimate = np.mean(observations)
            error = abs(estimate - true_value)

            R = compute_R(observations, target=true_value, use_v2_essence=True)
            correct = error < 0.5

            results.append({
                'noise': noise,
                'n': n,
                'grad_S': grad_S,
                'error': error,
                'R': R,
                'correct': correct,
            })

            print(f"{noise:>8} {n:>6} {grad_S:>10.4f} {error:>10.4f} {R:>10.4f} {'YES' if correct else 'NO':>10}")

    # Analysis
    print("\nAnalysis:")
    low_R_correct = [r for r in results if r['R'] < 0.5 and r['correct']]
    high_R_wrong = [r for r in results if r['R'] > 0.5 and not r['correct']]

    print(f"  Low R but correct: {len(low_R_correct)} (conservative, not a bug)")
    print(f"  High R but wrong: {len(high_R_wrong)} (false confidence, IS a bug)")

    if len(low_R_correct) > 0:
        print("\n  FINDING: Formula is CONSERVATIVE - says 'don't trust' when uncertain,")
        print("  even if the answer happens to be right. This is CORRECT BEHAVIOR.")

    if len(high_R_wrong) > 0:
        print("\n  FINDING: Formula has FALSE POSITIVES - high R but wrong answer!")
        print("  This is a real vulnerability.")

    return len(low_R_correct), len(high_R_wrong)


# =============================================================================
# ATTACK 3: Adversarial Construction
# =============================================================================
def attack_adversarial():
    """
    Try to construct observations that MAXIMIZE R while being WRONG.
    This is the worst-case attack on the formula.
    """
    print("\n" + "=" * 70)
    print("ATTACK 3: Adversarial - Maximize R while being wrong")
    print("=" * 70)

    true_value = 0.0

    # Strategy 1: Extremely tight cluster at wrong value
    print("\nStrategy 1: Extremely tight wrong cluster")
    wrong_value = 100
    tight_wrong = wrong_value + np.random.normal(0, 0.001, 100)
    R_tight = compute_R(tight_wrong, target=true_value, use_v2_essence=True)
    error_tight = abs(np.mean(tight_wrong) - true_value)
    print(f"  grad_S = {compute_grad_S(tight_wrong):.6f}")
    print(f"  Error = {error_tight:.2f}")
    print(f"  R (v2) = {R_tight:.4f}")

    # Strategy 2: Bimodal - half right, half very wrong
    print("\nStrategy 2: Bimodal (half right, half very wrong)")
    right_half = true_value + np.random.normal(0, 0.1, 50)
    wrong_half = 100 + np.random.normal(0, 0.1, 50)
    bimodal = np.concatenate([right_half, wrong_half])
    R_bimodal = compute_R(bimodal, target=true_value, use_v2_essence=True)
    error_bimodal = abs(np.mean(bimodal) - true_value)
    print(f"  grad_S = {compute_grad_S(bimodal):.4f}")
    print(f"  Error = {error_bimodal:.2f}")
    print(f"  R (v2) = {R_bimodal:.4f}")

    # Strategy 3: Uniform spread centered on truth
    print("\nStrategy 3: Uniform spread centered on truth (high grad_S, low error)")
    uniform_right = true_value + np.random.uniform(-10, 10, 100)
    R_uniform = compute_R(uniform_right, target=true_value, use_v2_essence=True)
    error_uniform = abs(np.mean(uniform_right) - true_value)
    print(f"  grad_S = {compute_grad_S(uniform_right):.4f}")
    print(f"  Error = {error_uniform:.4f}")
    print(f"  R (v2) = {R_uniform:.4f}")

    print("\n" + "-" * 70)
    print("ADVERSARIAL FINDINGS:")

    if R_tight > 1.0:
        print("  - Formula FOOLED by extremely tight wrong cluster")
    else:
        print("  - Formula RESISTS extremely tight wrong cluster (R = {:.4f})".format(R_tight))

    if R_bimodal > 0.5:
        print("  - Formula might be confused by bimodal (R = {:.4f})".format(R_bimodal))
    else:
        print("  - Formula correctly identifies bimodal as uncertain (R = {:.4f})".format(R_bimodal))


# =============================================================================
# CRITICAL TEST: What is grad_S actually measuring?
# =============================================================================
def test_grad_s_meaning():
    """
    The fundamental question: What property does grad_S capture that matters?

    Hypothesis 1: grad_S measures noise level
    Hypothesis 2: grad_S measures independence
    Hypothesis 3: grad_S measures something about the generating process

    Design: Create different generating processes, measure grad_S,
    see which hypothesis explains the results.
    """
    print("\n" + "=" * 70)
    print("CRITICAL: What is grad_S actually measuring?")
    print("=" * 70)

    true_value = 0.0
    n = 50

    processes = {
        'Independent Gaussian': lambda: true_value + np.random.normal(0, 1, n),
        'Independent Uniform': lambda: true_value + np.random.uniform(-2, 2, n),
        'Correlated (shared bias)': lambda: true_value + np.random.normal(0, 1) + np.random.normal(0, 0.1, n),
        'Heavy-tailed (Cauchy)': lambda: true_value + np.random.standard_cauchy(n) * 0.5,
        'Mixture (two sources)': lambda: np.concatenate([
            true_value + np.random.normal(0, 0.5, n//2),
            true_value + np.random.normal(0, 2, n//2)
        ]),
    }

    print(f"\n{'Process':<25} {'Mean grad_S':>12} {'Mean Error':>12} {'R predicts?':>12}")
    print("-" * 65)

    for name, generator in processes.items():
        grad_Ss = []
        errors = []
        Rs = []

        for _ in range(100):
            obs = generator()
            grad_S = compute_grad_S(obs)
            error = abs(np.mean(obs) - true_value)
            R = compute_R(obs, target=true_value, use_v2_essence=True)

            grad_Ss.append(grad_S)
            errors.append(error)
            Rs.append(R)

        # Does R predict error in this process?
        corr = np.corrcoef(Rs, errors)[0, 1]
        predicts = "YES" if corr < -0.2 else "NO" if corr > 0.2 else "WEAK"

        print(f"{name:<25} {np.mean(grad_Ss):>12.4f} {np.mean(errors):>12.4f} {predicts:>12}")

    print("\nIf R predicts error for independent processes but NOT for correlated,")
    print("then grad_S is measuring EFFECTIVE INDEPENDENCE, not just dispersion.")


if __name__ == "__main__":
    np.random.seed(42)

    v1_fooled, v2_fooled = attack_low_grad_s_wrong()
    conservative, false_pos = attack_high_grad_s_right()
    attack_adversarial()
    test_grad_s_meaning()

    print("\n" + "=" * 70)
    print("Q1 ADVERSARIAL SUMMARY")
    print("=" * 70)

    print(f"""
FINDINGS:

1. Circular E (v1) is USELESS
   - R = E/grad_S where E = 1/(1+std) makes R = 1/std^2
   - This is fooled by ANY tight cluster, right or wrong
   - {v1_fooled} echo chambers fooled it

2. Proper E (v2) that measures resolution is HARDER to fool
   - {v2_fooled} echo chambers fooled it (should be 0 if E is correct)

3. The formula is CONSERVATIVE
   - {conservative} cases where R was low but answer was right
   - This is correct behavior: "I don't know" is better than false confidence

4. False positives are the real vulnerability
   - {false_pos} cases where R was high but answer was wrong
   - These need investigation

CONCLUSION for Q1:
grad_S works ONLY when combined with proper E (essence).
E must measure whether observations RESOLVE to truth, not just cluster.
The formula is: Resolvability / Uncertainty, not Agreement / Dispersion.
""")
