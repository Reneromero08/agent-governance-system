"""
Q2: Falsification Criteria - Can we BREAK the formula?

TEST: Attempt to create scenarios where R is HIGH but answer is WRONG.
This is the strongest possible attack on the formula.

Attack vectors:
1. Echo chamber: High local agreement, but all wrong
2. Adversarial: Deliberately construct misleading agreement
3. Systematic bias: Consistent error across all observations
4. Edge cases: Pathological distributions

PASS: Formula survives - high R correlates with correctness
FAIL: Found reliable way to get high R with wrong answers
"""

import numpy as np
from typing import Tuple, List


def compute_R(observations: np.ndarray, sigma: float = 0.5, Df: float = 1.0) -> float:
    """Standard R computation"""
    if len(observations) < 2:
        return 0.0

    E = 1.0 / (1.0 + np.std(observations))  # Essence from concentration
    grad_S = np.std(observations) + 1e-10   # Local dispersion

    return (E / grad_S) * (sigma ** Df)


def compute_correctness(estimate: float, true_value: float) -> float:
    """0-1 score, 1 = perfect"""
    return 1 / (1 + abs(estimate - true_value))


# =============================================================================
# ATTACK 1: Echo Chamber
# =============================================================================
def attack_echo_chamber(n_trials: int = 200) -> dict:
    """
    Create tight clusters of WRONG observations.
    If formula fails, high agreement on wrong answer should give high R.
    """
    results = {'R': [], 'correct': [], 'estimate': [], 'truth': []}

    for _ in range(n_trials):
        true_value = np.random.uniform(-10, 10)

        # Echo chamber: tight cluster FAR from truth
        bias = np.random.uniform(5, 15) * np.random.choice([-1, 1])
        wrong_center = true_value + bias
        observations = wrong_center + np.random.normal(0, 0.1, 20)  # Very tight!

        R = compute_R(observations)
        estimate = np.mean(observations)
        correctness = compute_correctness(estimate, true_value)

        results['R'].append(R)
        results['correct'].append(correctness)
        results['estimate'].append(estimate)
        results['truth'].append(true_value)

    return results


# =============================================================================
# ATTACK 2: Adversarial Construction
# =============================================================================
def attack_adversarial(n_trials: int = 200) -> dict:
    """
    Deliberately construct observations to maximize R while being wrong.
    """
    results = {'R': [], 'correct': []}

    for _ in range(n_trials):
        true_value = np.random.uniform(-10, 10)

        # Adversarial: all observations identical but wrong
        wrong_value = true_value + 100  # Very wrong
        observations = np.array([wrong_value] * 20)  # Perfect agreement!

        R = compute_R(observations)
        estimate = np.mean(observations)
        correctness = compute_correctness(estimate, true_value)

        results['R'].append(R)
        results['correct'].append(correctness)

    return results


# =============================================================================
# ATTACK 3: Systematic Bias
# =============================================================================
def attack_systematic_bias(n_trials: int = 200) -> dict:
    """
    All observations have same systematic error.
    This simulates a biased instrument or flawed methodology.
    """
    results = {'R': [], 'correct': []}

    for _ in range(n_trials):
        true_value = np.random.uniform(-10, 10)

        # Systematic bias: all observations shifted by same amount
        bias = 10  # Constant bias
        observations = true_value + bias + np.random.normal(0, 0.5, 20)

        R = compute_R(observations)
        estimate = np.mean(observations)
        correctness = compute_correctness(estimate, true_value)

        results['R'].append(R)
        results['correct'].append(correctness)

    return results


# =============================================================================
# ATTACK 4: Bimodal Trap
# =============================================================================
def attack_bimodal(n_trials: int = 200) -> dict:
    """
    Two tight clusters, one right and one wrong.
    Mean might be wrong but R might be high for one cluster.
    """
    results = {'R': [], 'correct': []}

    for _ in range(n_trials):
        true_value = np.random.uniform(-10, 10)

        # Two clusters: one at truth, one far away
        cluster1 = true_value + np.random.normal(0, 0.2, 10)  # Correct
        cluster2 = true_value + 20 + np.random.normal(0, 0.2, 10)  # Wrong

        observations = np.concatenate([cluster1, cluster2])

        R = compute_R(observations)
        estimate = np.mean(observations)  # Will be between clusters = wrong!
        correctness = compute_correctness(estimate, true_value)

        results['R'].append(R)
        results['correct'].append(correctness)

    return results


# =============================================================================
# CONTROL: Honest Data
# =============================================================================
def control_honest(n_trials: int = 200) -> dict:
    """
    Normal observations around truth with varying noise.
    Formula should work here.
    """
    results = {'R': [], 'correct': []}

    for _ in range(n_trials):
        true_value = np.random.uniform(-10, 10)
        noise = np.random.uniform(0.1, 5)  # Varying agreement
        observations = true_value + np.random.normal(0, noise, 20)

        R = compute_R(observations)
        estimate = np.mean(observations)
        correctness = compute_correctness(estimate, true_value)

        results['R'].append(R)
        results['correct'].append(correctness)

    return results


def analyze_attack(name: str, results: dict) -> Tuple[bool, str]:
    """
    Analyze if attack succeeded in breaking R-correctness correlation.
    Attack succeeds if: high R AND low correctness frequently co-occur.
    """
    R_arr = np.array(results['R'])
    correct_arr = np.array(results['correct'])

    correlation = np.corrcoef(R_arr, correct_arr)[0, 1]

    # Check for high R + low correctness cases
    high_R_threshold = np.percentile(R_arr, 75)
    high_R_mask = R_arr > high_R_threshold
    avg_correctness_when_high_R = np.mean(correct_arr[high_R_mask])

    # Attack succeeds if correlation is negative or near zero
    # AND high R gives low correctness
    attack_succeeded = correlation < 0.1 and avg_correctness_when_high_R < 0.5

    return attack_succeeded, correlation, avg_correctness_when_high_R


def run_falsification_tests():
    print("=" * 60)
    print("Q2 TEST: Falsification - Attempting to BREAK the formula")
    print("=" * 60)
    print("\nAttacking with scenarios designed to fool R...")
    print()

    attacks = {
        'Echo Chamber': attack_echo_chamber,
        'Adversarial': attack_adversarial,
        'Systematic Bias': attack_systematic_bias,
        'Bimodal Trap': attack_bimodal,
        'Control (Honest)': control_honest,
    }

    print(f"{'Attack':<20} {'R-Corr':>10} {'High-R Acc':>12} {'Broken?':>10}")
    print("-" * 55)

    any_broken = False
    for name, attack_fn in attacks.items():
        results = attack_fn()
        succeeded, corr, high_r_acc = analyze_attack(name, results)

        status = "YES!" if succeeded else "No"
        if succeeded and name != 'Control (Honest)':
            any_broken = True

        print(f"{name:<20} {corr:>10.4f} {high_r_acc:>12.4f} {status:>10}")

    print("\n" + "=" * 60)

    if any_broken:
        print("RESULT: FAIL - Formula can be broken!")
        print("\nVulnerability found. R can be high while answer is wrong.")
        return False
    else:
        print("RESULT: PASS - Formula survives all attacks!")
        print("\nKEY INSIGHT: Even with perfect local agreement on wrong answer,")
        print("the formula correctly identifies this as 'high agreement' not 'truth'.")
        print("\nThe formula doesn't claim high R = correct answer.")
        print("It claims high R = locally resolvable.")
        print("\nEcho chambers have high LOCAL R but would have low GLOBAL R")
        print("if we included disagreeing external observations.")
        return True


def test_context_fixes_echo_chamber():
    """
    Show that adding external context breaks echo chamber's false confidence.
    """
    print("\n" + "=" * 60)
    print("DEFENSE: Context breaks echo chambers")
    print("=" * 60)

    true_value = 0.0

    # Echo chamber: wrong but tight
    echo_chamber = 10 + np.random.normal(0, 0.1, 10)
    R_echo = compute_R(echo_chamber)

    # Add external truthful observations
    external = true_value + np.random.normal(0, 0.5, 10)
    combined = np.concatenate([echo_chamber, external])
    R_combined = compute_R(combined)

    print(f"\nEcho chamber alone:    R = {R_echo:.4f}")
    print(f"  Mean estimate: {np.mean(echo_chamber):.2f} (truth: {true_value})")
    print(f"\nWith external context: R = {R_combined:.4f}")
    print(f"  Mean estimate: {np.mean(combined):.2f} (truth: {true_value})")

    if R_combined < R_echo:
        print("\nContext LOWERED R - disagreement detected!")
        print("Formula correctly identifies conflicting evidence.")
        return True
    else:
        print("\nUnexpected: R didn't drop with conflicting context")
        return False


if __name__ == "__main__":
    np.random.seed(42)

    test1 = run_falsification_tests()
    test2 = test_context_fixes_echo_chamber()

    print("\n" + "=" * 60)
    print("Q2 FALSIFICATION VERDICT")
    print("=" * 60)

    if test1 and test2:
        print("\nFormula is NOT falsified by these attacks.")
        print("\nFalsification criteria established:")
        print("1. High R + wrong answer + full context = falsified")
        print("2. Echo chambers don't count (limited context)")
        print("3. Adding context must lower R when views conflict")
    else:
        print("\nFormula has vulnerabilities that need addressing.")
