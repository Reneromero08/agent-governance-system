"""
Q14: Complete Analysis Suite (Final Version)

Run all tests to characterize gate's category-theoretic structure.
"""

import numpy as np


def compute_R(observations: np.ndarray, truth: float) -> float:
    """R = E / grad_S"""
    E = 1.0 / (1.0 + abs(np.mean(observations) - truth))
    grad_S = np.std(observations) + 1e-10
    return E / grad_S


def gate_state(R: float, threshold: float) -> bool:
    """OPEN if R > threshold"""
    return R > threshold


# ============================================================================
# TEST 1: R-COVER GROTHENDIECK TOPOLOGY
# ============================================================================

def generate_r_cover(base_obs: np.ndarray, n_sub: int, truth: float) -> list:
    """
    Generate an R-cover: all sub-contexts have R >= parent R.
    """
    R_full = compute_R(base_obs, truth)
    covers = []

    n = len(base_obs)
    window_size = max(3, n // (n_sub - 1))
    overlap = window_size // 2

    for i in range(n_sub - 1):
        start = i * (window_size - overlap)
        end = min(start + window_size, n)
        if end > start:
            obs_i = base_obs[start:end].copy()
            R_i = compute_R(obs_i, truth)

            if R_i >= R_full * 0.95:
                covers.append(obs_i)

    start = (n_sub - 2) * (window_size - overlap)
    if start < n:
        obs_i = base_obs[start:].copy()
        R_i = compute_R(obs_i, truth)
        if R_i >= R_full * 0.95:
            covers.append(obs_i)

    return covers


def test_r_cover_sheaf():
    """Test R-cover Grothendieck topology"""
    print("=" * 70)
    print("TEST 1: R-COVER Grothendieck Topology")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 500

    locality_pass = 0
    locality_total = 0
    gluing_pass = 0
    gluing_total = 0

    for _ in range(n_tests):
        base_obs = np.random.normal(0, 1, 15)
        n_sub = np.random.randint(2, 4)
        covers = generate_r_cover(base_obs, n_sub, truth)

        if len(covers) == 0:
            continue

        all_cover_obs = np.concatenate(covers)
        if len(np.unique(all_cover_obs)) < len(np.unique(base_obs)):
            continue

        locality_total += 1

        R_full = compute_R(base_obs, truth)
        gate_full = gate_state(R_full, threshold)

        gate_subs = []
        for cover in covers:
            if len(cover) > 0:
                R_sub = compute_R(cover, truth)
                gate_sub = gate_state(R_sub, threshold)
                gate_subs.append(gate_sub)

        if len(set(gate_subs)) == 1:
            expected = gate_subs[0]
            if gate_full == expected:
                locality_pass += 1

        gluing_total += 1

        compatible = True
        for i in range(len(covers)):
            for j in range(i + 1, len(covers)):
                obs_i = covers[i]
                obs_j = covers[j]

                intersection = np.intersect1d(obs_i, obs_j)
                if len(intersection) > 0:
                    R_intersect = compute_R(intersection, truth)
                    gate_intersect = gate_state(R_intersect, threshold)

                    if gate_subs[i] != gate_intersect or gate_subs[j] != gate_intersect:
                        compatible = False
                        break

        if compatible:
            if len(set(gate_subs)) == 1:
                expected_glue = gate_subs[0]
                if gate_full == expected_glue:
                    gluing_pass += 1

    locality_rate = locality_pass / locality_total if locality_total > 0 else 0
    gluing_rate = gluing_pass / gluing_total if gluing_total > 0 else 0

    print(f"\nLocality: {locality_pass}/{locality_total} ({locality_rate*100:.1f}%)")
    print(f"Gluing: {gluing_pass}/{gluing_total} ({gluing_rate*100:.1f}%)")

    return locality_rate, gluing_rate


# ============================================================================
# TEST 2: MONOTONICITY ANALYSIS
# ============================================================================

def test_monotonicity():
    """Characterize when U subseteq V implies R(U) >= R(V)"""
    print("\n" + "=" * 70)
    print("TEST 2: Monotonicity Analysis")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 5000

    monotonicity_holds = 0
    monotonicity_fails = 0

    for _ in range(n_tests):
        base_size = np.random.randint(10, 30)
        base_obs = np.random.normal(0, 1, base_size)
        sub_size = np.random.randint(5, base_size)
        sub_obs = base_obs[:sub_size].copy()

        if not np.allclose(np.sort(sub_obs), np.sort(base_obs[:sub_size])):
            continue

        R_full = compute_R(base_obs, truth)
        R_sub = compute_R(sub_obs, truth)

        if R_sub >= R_full:
            monotonicity_holds += 1
        else:
            monotonicity_fails += 1

    total = monotonicity_holds + monotonicity_fails
    monotonicity_rate = monotonicity_holds / total if total > 0 else 0

    print(f"\nTests: {total}")
    print(f"Monotonicity holds: {monotonicity_holds} ({monotonicity_rate*100:.1f}%)")
    print(f"Monotonicity fails: {monotonicity_fails} ({(1-monotonicity_rate)*100:.1f}%)")

    return monotonicity_rate


# ============================================================================
# TEST 3: MONOTONICITY BY VARIANCE REGIONS
# ============================================================================

def test_monotonicity_by_variance():
    """Find parameter regions where monotonicity holds"""
    print("\n" + "=" * 70)
    print("TEST 3: Monotonicity by Variance Regions")
    print("=" * 70)

    truth = 0.0
    n_tests = 2000

    variance_ranges = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
    monotonicity_by_variance = []

    for base_std in variance_ranges:
        monotonicity_holds = 0
        total = 0

        for _ in range(n_tests):
            base_size = 20
            base_obs = np.random.normal(0, base_std, base_size)
            sub_size = np.random.randint(5, 15)
            sub_obs = base_obs[:sub_size].copy()

            R_full = compute_R(base_obs, truth)
            R_sub = compute_R(sub_obs, truth)

            if R_sub >= R_full:
                monotonicity_holds += 1

            total += 1

        rate = monotonicity_holds / total if total > 0 else 0
        monotonicity_by_variance.append(rate)

    print(f"\nMonotonicity rate by base std:")
    for i, std_val in enumerate(variance_ranges):
        print(f"  std={std_val:.1f}: {monotonicity_by_variance[i]*100:.1f}%")

    monotonicity_by_variance = np.array(monotonicity_by_variance)

    print(f"\nMean rate: {np.mean(monotonicity_by_variance)*100:.1f}%")
    print(f"Std correlation: {np.corrcoef(variance_ranges, monotonicity_by_variance)[0,1]:.4f}")

    low_variance = np.mean(monotonicity_by_variance[np.array(variance_ranges) < 0.7])
    high_variance = np.mean(monotonicity_by_variance[np.array(variance_ranges) > 1.0])

    print(f"\nLow variance (std < 0.7): {low_variance*100:.1f}%")
    print(f"High variance (std > 1.0): {high_variance*100:.1f}%")


# ============================================================================
# TEST 4: VARIANCE EFFECT ON R
# ============================================================================

def test_variance_effect():
    """How does variance change when extending context affect R?"""
    print("\n" + "=" * 70)
    print("TEST 4: Variance Effect on R")
    print("=" * 70)

    truth = 0.0
    n_tests = 1000

    R_decrease_count = 0
    R_increase_count = 0
    std_diff_when_decrease = []
    std_diff_when_increase = []

    for _ in range(n_tests):
        base_size = 20
        base_obs = np.random.normal(0, 1, base_size)

        extra_size = np.random.randint(1, 20)
        extra_obs = np.random.normal(0, 1, extra_size)

        extended_obs = np.concatenate([base_obs, extra_obs])

        R_base = compute_R(base_obs, truth)
        R_ext = compute_R(extended_obs, truth)

        diff = R_ext - R_base

        base_std = np.std(base_obs)
        ext_std = np.std(extended_obs)

        std_diff = ext_std - base_std

        if diff < 0:
            R_decrease_count += 1
            std_diff_when_decrease.append(std_diff)
        elif diff > 0:
            R_increase_count += 1
            std_diff_when_increase.append(std_diff)

    print(f"\nR decreases: {R_decrease_count} ({R_decrease_count/n_tests*100:.1f}%)")
    print(f"R increases: {R_increase_count} ({R_increase_count/n_tests*100:.1f}%)")

    if len(std_diff_when_decrease) > 0:
        print(f"\nWhen R decreases:")
        print(f"  Mean std diff: {np.mean(std_diff_when_decrease):.4f}")
        print(f"  Std std diff: {np.std(std_diff_when_decrease):.4f}")

    if len(std_diff_when_increase) > 0:
        print(f"\nWhen R increases:")
        print(f"  Mean std diff: {np.mean(std_diff_when_increase):.4f}")
        print(f"  Std std diff: {np.std(std_diff_when_increase):.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("Q14: Complete Category Theory Analysis")
    print("=" * 70)

    locality_rate, gluing_rate = test_r_cover_sheaf()
    monotonicity_rate = test_monotonicity()
    test_monotonicity_by_variance()
    test_variance_effect()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n1. R-COVER Grothendieck Topology:")
    print(f"   Locality: {locality_rate*100:.1f}%")
    print(f"   Gluing: {gluing_rate*100:.1f}%")

    print("\n2. Monotonicity:")
    print(f"   Overall rate: {monotonicity_rate*100:.1f}%")

    if locality_rate > 0.9 and gluing_rate > 0.9:
        print("\nCONCLUSION: Gate is EXCELLENT SHEAF with R-COVER topology")
        print("   Local agreement leads to global consistency.")
        print("   R-cover definition: all sub-contexts have R >= parent R.")
    elif monotonicity_rate > 0.5:
        print("\nCONCLUSION: Gate is PARTIAL SHEAF (monotonicity holds but sheaf weak)")
        print("   R-cover requires sub-contexts to have high R, which is restrictive.")
        print("   Most contexts will not satisfy this constraint.")
    else:
        print("\nCONCLUSION: Gate is WEAK SHEAF (poor monotonicity)")
        print("   Non-monotone: adding observations often decreases R.")
        print("   Requires careful threshold tuning and variance management.")
