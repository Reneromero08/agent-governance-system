"""
Q1 REVISITED: Essence is the amount of truth.

The formula: R = E / grad_S × σ^Df

Where:
- E = amount of truth (must be measured against reality)
- grad_S = local dispersion
- σ^Df = scale factor

KEY INSIGHT: You cannot compute E without testing against reality.
The formula doesn't claim to detect truth from agreement alone.
It combines EXTERNAL truth signal (E) with LOCAL structure (grad_S).

TEST DESIGN:
1. Simulate scenarios where we KNOW the truth
2. Compute E as actual closeness to truth
3. See how R behaves in different regimes
4. Verify: high E + low grad_S = high R (resolvable truth)
          low E + low grad_S = low R (echo chamber)
          low E + high grad_S = low R (noise)
          high E + high grad_S = medium R (noisy truth)
"""

import numpy as np
from typing import Tuple


def compute_essence(observations: np.ndarray, truth: float) -> float:
    """
    E = amount of truth in the observations.
    Measured as: how close is the signal to actual truth?

    This REQUIRES knowing truth - you can't compute E from observations alone.
    """
    estimate = np.mean(observations)
    error = abs(estimate - truth)
    # E is inverse of error, normalized
    # E = 1 when perfect, E → 0 as error → ∞
    return 1.0 / (1.0 + error)


def compute_grad_S(observations: np.ndarray) -> float:
    """Local dispersion - how much do observations disagree?"""
    return np.std(observations) + 1e-10


def compute_R(observations: np.ndarray, truth: float,
              sigma: float = 0.5, Df: float = 1.0) -> float:
    """
    R = E / grad_S × σ^Df

    Resolvability = (amount of truth) / (local uncertainty) × (scale)
    """
    E = compute_essence(observations, truth)
    grad_S = compute_grad_S(observations)
    return (E / grad_S) * (sigma ** Df)


def test_four_regimes():
    """
    Test the four fundamental regimes:

    1. High E, Low grad_S: RESOLVABLE TRUTH (should have HIGH R)
    2. Low E, Low grad_S: ECHO CHAMBER (should have LOW R)
    3. Low E, High grad_S: NOISE (should have LOW R)
    4. High E, High grad_S: NOISY TRUTH (should have MEDIUM R)
    """
    print("=" * 70)
    print("Q1 TEST: E = Amount of Truth")
    print("=" * 70)

    truth = 0.0

    regimes = {
        "High E, Low grad_S (Truth)": {
            "generator": lambda: truth + np.random.normal(0, 0.1, 20),
            "expected_R": "HIGH",
        },
        "Low E, Low grad_S (Echo)": {
            "generator": lambda: (truth + 10) + np.random.normal(0, 0.1, 20),  # Tight but wrong
            "expected_R": "LOW",
        },
        "Low E, High grad_S (Noise)": {
            "generator": lambda: (truth + 10) + np.random.normal(0, 5, 20),  # Scattered and wrong
            "expected_R": "LOW",
        },
        "High E, High grad_S (Noisy Truth)": {
            "generator": lambda: truth + np.random.normal(0, 3, 20),  # Scattered but right on average
            "expected_R": "MEDIUM",
        },
    }

    print(f"\n{'Regime':<35} {'E':>8} {'grad_S':>10} {'R':>10} {'Expected':>10} {'Match?':>8}")
    print("-" * 90)

    results = []
    for name, config in regimes.items():
        Es = []
        grad_Ss = []
        Rs = []

        for _ in range(100):
            obs = config["generator"]()
            E = compute_essence(obs, truth)
            grad_S = compute_grad_S(obs)
            R = compute_R(obs, truth)

            Es.append(E)
            grad_Ss.append(grad_S)
            Rs.append(R)

        mean_E = np.mean(Es)
        mean_grad_S = np.mean(grad_Ss)
        mean_R = np.mean(Rs)

        # Classify actual R
        if mean_R > 1.0:
            actual_R = "HIGH"
        elif mean_R > 0.1:
            actual_R = "MEDIUM"
        else:
            actual_R = "LOW"

        match = "YES" if actual_R == config["expected_R"] else "NO"

        results.append({
            "name": name,
            "E": mean_E,
            "grad_S": mean_grad_S,
            "R": mean_R,
            "expected": config["expected_R"],
            "actual": actual_R,
            "match": match,
        })

        print(f"{name:<35} {mean_E:>8.4f} {mean_grad_S:>10.4f} {mean_R:>10.4f} {config['expected_R']:>10} {match:>8}")

    # Summary
    matches = sum(1 for r in results if r["match"] == "YES")
    print(f"\n{matches}/4 regimes behave as expected")

    return matches == 4


def test_echo_chamber_detection():
    """
    With proper E (measuring actual truth), can we distinguish echo chambers?

    Echo chamber: tight agreement on WRONG answer
    - Low grad_S (tight)
    - Low E (wrong)
    - Therefore: LOW R

    This is the KEY TEST: does low E correctly penalize wrong agreement?
    """
    print("\n" + "=" * 70)
    print("ECHO CHAMBER TEST: Does low E penalize wrong agreement?")
    print("=" * 70)

    truth = 0.0

    print(f"\n{'Bias':>8} {'Tightness':>10} {'E':>10} {'grad_S':>10} {'R':>10}")
    print("-" * 55)

    for bias in [0, 1, 5, 10, 50]:  # 0 = truth, others = echo chambers
        for tightness in [0.1, 1.0]:
            center = truth + bias
            observations = center + np.random.normal(0, tightness, 20)

            E = compute_essence(observations, truth)
            grad_S = compute_grad_S(observations)
            R = compute_R(observations, truth)

            print(f"{bias:>8} {tightness:>10.1f} {E:>10.4f} {grad_S:>10.4f} {R:>10.4f}")

    print("\nIf E correctly measures truth:")
    print("  - Bias=0 should have HIGH E, HIGH R")
    print("  - Bias>0 should have LOW E, LOW R (despite tight agreement)")


def test_entropy_from_action():
    """
    The user said: echo chambers cause entropy when applied to reality.

    Test: Take observations, ACT on them, measure resulting entropy.
    Echo chambers should produce HIGH entropy (disorder) when acted upon.
    """
    print("\n" + "=" * 70)
    print("ENTROPY FROM ACTION: Do echo chambers cause disorder?")
    print("=" * 70)

    truth = 0.0
    n_trials = 100

    scenarios = {
        "Truth (bias=0)": 0,
        "Small bias (1)": 1,
        "Medium bias (5)": 5,
        "Large bias (10)": 10,
        "Echo chamber (50)": 50,
    }

    print(f"\n{'Scenario':<25} {'Action Error':>15} {'Outcome Entropy':>18}")
    print("-" * 60)

    for name, bias in scenarios.items():
        action_errors = []
        outcome_entropies = []

        for _ in range(n_trials):
            # Generate observations
            observations = (truth + bias) + np.random.normal(0, 0.5, 20)

            # ACT based on observations (use mean as decision)
            action = np.mean(observations)

            # Measure error of action
            action_error = abs(action - truth)
            action_errors.append(action_error)

            # Outcome entropy: if we repeatedly act, how much variance in outcomes?
            # Simulate: action + noise from reality
            outcomes = action + np.random.normal(0, action_error, 10)
            outcome_entropy = np.std(outcomes)
            outcome_entropies.append(outcome_entropy)

        mean_error = np.mean(action_errors)
        mean_entropy = np.mean(outcome_entropies)

        print(f"{name:<25} {mean_error:>15.4f} {mean_entropy:>18.4f}")

    print("\nEcho chambers (high bias) should produce:")
    print("  - HIGH action error (acting on wrong belief)")
    print("  - HIGH outcome entropy (disorder from wrong actions)")


def test_formula_prevents_entropy():
    """
    The formula should PREVENT acting on echo chambers.

    Test: Use R to gate actions. Compare entropy with/without R-gating.
    """
    print("\n" + "=" * 70)
    print("R-GATING: Does the formula prevent entropy?")
    print("=" * 70)

    truth = 0.0
    n_trials = 500

    # Mix of true and echo chamber observations
    ungated_errors = []
    gated_errors = []
    gated_abstentions = 0

    R_threshold = 0.5  # Only act if R > threshold

    for _ in range(n_trials):
        # Randomly choose: truth or echo chamber
        if np.random.random() < 0.5:
            # Truth
            observations = truth + np.random.normal(0, 0.5, 20)
        else:
            # Echo chamber (random large bias)
            bias = np.random.uniform(5, 20)
            observations = (truth + bias) + np.random.normal(0, 0.3, 20)

        R = compute_R(observations, truth)
        action = np.mean(observations)
        error = abs(action - truth)

        # Ungated: always act
        ungated_errors.append(error)

        # Gated: only act if R > threshold
        if R > R_threshold:
            gated_errors.append(error)
        else:
            gated_abstentions += 1

    ungated_entropy = np.std(ungated_errors)
    gated_entropy = np.std(gated_errors) if gated_errors else 0

    print(f"\nUngated actions:")
    print(f"  Mean error: {np.mean(ungated_errors):.4f}")
    print(f"  Error std (entropy): {ungated_entropy:.4f}")
    print(f"  N actions: {len(ungated_errors)}")

    print(f"\nR-gated actions (R > {R_threshold}):")
    print(f"  Mean error: {np.mean(gated_errors) if gated_errors else 'N/A':.4f}")
    print(f"  Error std (entropy): {gated_entropy:.4f}")
    print(f"  N actions: {len(gated_errors)}")
    print(f"  Abstentions: {gated_abstentions}")

    if gated_errors:
        improvement = (ungated_entropy - gated_entropy) / ungated_entropy * 100
        print(f"\nEntropy reduction: {improvement:.1f}%")

        if improvement > 30:
            print("\nFINDING: R-gating SIGNIFICANTLY reduces entropy!")
            print("The formula correctly identifies when NOT to act.")
            return True

    return False


if __name__ == "__main__":
    np.random.seed(42)

    test1 = test_four_regimes()
    test_echo_chamber_detection()
    test_entropy_from_action()
    test2 = test_formula_prevents_entropy()

    print("\n" + "=" * 70)
    print("Q1 FINAL: E = Amount of Truth")
    print("=" * 70)

    if test1 and test2:
        print("""
ANSWER TO Q1: Why grad_S?

grad_S alone is NOT sufficient. The formula requires BOTH:
  - E (essence) = amount of truth, measured against reality
  - grad_S = local dispersion

The formula R = E / grad_S means:
  RESOLVABILITY = TRUTH / UNCERTAINTY

For echo chambers:
  - Low grad_S (tight) doesn't help
  - Low E (wrong) kills R
  - Result: LOW R, don't act

For truth:
  - Low grad_S (tight) helps
  - High E (right) helps
  - Result: HIGH R, act with confidence

WHY grad_S works: It measures the RELIABILITY of the truth signal.
  - High E + Low grad_S = reliable truth → HIGH R
  - High E + High grad_S = unreliable truth → MEDIUM R
  - Low E + anything = no truth → LOW R

The formula prevents entropy by refusing to act on low R.
Echo chambers have low R because E is low (wrong).
""")
    else:
        print("\nTests did not fully confirm. Need more investigation.")
