"""
Q14: Grothendieck Topology Definition

Define proper covering families for the gate sheaf.
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
# GROTHENDIECK TOPOLOGY ON OBSERVATION CATEGORY C
# ============================================================================

"""
DEFINITION: A Grothendieck topology J on a category C assigns to each
object U a collection J(U) of "covering families" satisfying:

AXIOMS:
1. COVERAGE: If {V_i} in J(U), then ∪ V_i = U
2. STABILITY: If {V_i} in J(U) and W ⊆ U,
   then {V_i ∩ W} in J(W)
3. TRANSITIVITY: If {V_i} in J(U) and for each V_i,
   {W_ij} in J(V_i) covers V_i, then {W_ij} in J(U)

For the observation category C (poset of observation contexts):
- Objects: Observation contexts (sets of observations)
- Morphisms: Inclusions U → V when U ⊆ V
- Structure: Poset category ordered by inclusion
"""

# ============================================================================
# COVERING FAMILIES FOR THE GATE SHEAF
# ============================================================================

"""
We define covering families based on the gate's R-values.

INTUITION: A cover of U is a family of sub-contexts that
"collectively justify" the gate state of U.

Two possible definitions:

1. R-COVER (Consensus Cover):
   {V_i} is an R-cover of U if:
   - Each V_i ⊆ U
   - ∪ V_i = U (full coverage)
   - R(V_i) ≥ R(U) for all i (no sub-context has lower R)

   MEANING: All sub-contexts agree with or exceed parent's R-value.
   This ensures local agreement → global consistency.

2. OVERLAP-COVER (Topological Cover):
   {V_i} is an overlap-cover of U if:
   - Each V_i ⊆ U
   - ∪ V_i = U (full coverage)
   - |V_i| > 0 (non-empty)
   - V_i ∩ V_j ≠ ∅ for at least one pair i≠j (overlaps)

   MEANING: Sub-contexts overlap enough to preserve information.
   This is the standard sheaf-theoretic cover.

WHICH TO USE?

For the gate sheaf, we need to determine which definition
makes the sheaf axioms hold.

HYPOTHESIS: The gate sheaf with R-cover definition satisfies
sheaf axioms better than with overlap-cover.
"""


def generate_r_cover(base_obs: np.ndarray, n_sub: int, threshold: float, truth: float) -> list:
    """
    Generate an R-cover: all sub-contexts have R ≥ parent R.

    Returns list of sub-contexts (may be empty if impossible)
    """
    R_full = compute_R(base_obs, truth)

    # Try to find sub-contexts with R ≥ R_full
    covers = []

    # Sliding window approach
    n = len(base_obs)
    window_size = max(3, n // (n_sub - 1))
    overlap = window_size // 2

    for i in range(n_sub - 1):
        start = i * (window_size - overlap)
        end = min(start + window_size, n)
        if end > start:
            obs_i = base_obs[start:end].copy()
            R_i = compute_R(obs_i, truth)

            # Check if R_i ≥ R_full
            if R_i >= R_full * 0.95:  # Allow small margin
                covers.append(obs_i)

    # Add final segment
    start = (n_sub - 2) * (window_size - overlap)
    if start < n:
        obs_i = base_obs[start:].copy()
        R_i = compute_R(obs_i, truth)
        if R_i >= R_full * 0.95:
            covers.append(obs_i)

    return covers


def test_r_cover_sheaf_property():
    """
    Test: Does R-cover definition make gate sheaf axioms hold?
    """
    print("=" * 70)
    print("R-COVER Grothendieck Topology Test")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 500

    locality_pass = 0
    locality_total = 0
    gluing_pass = 0
    gluing_total = 0

    for _ in range(n_tests):
        # Generate base context U
        base_obs = np.random.normal(0, 1, 15)

        # Generate R-cover
        n_sub = np.random.randint(2, 4)
        covers = generate_r_cover(base_obs, n_sub, threshold, truth)

        # Skip if cover is empty or doesn't reconstruct
        if len(covers) == 0:
            continue

        # Verify coverage
        all_cover_obs = np.concatenate(covers)
        if len(np.unique(all_cover_obs)) < len(np.unique(base_obs)):
            continue

        # Compute gate states
        R_full = compute_R(base_obs, truth)
        gate_full = gate_state(R_full, threshold)

        gate_subs = []
        for cover in covers:
            if len(cover) > 0:
                R_sub = compute_R(cover, truth)
                gate_sub = gate_state(R_sub, threshold)
                gate_subs.append(gate_sub)

        # LOCALITY: If all sub-contexts agree, parent should agree
        locality_total += 1
        if len(set(gate_subs)) == 1:
            expected = gate_subs[0]
            if gate_full == expected:
                locality_pass += 1

        # GLUING: If compatible on overlaps, should glue
        gluing_total += 1

        # Check compatibility on overlaps
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
            else:
                # Compatible but different values
                if gate_full in gate_subs:
                    gluing_pass += 1

    locality_rate = locality_pass / locality_total if locality_total > 0 else 0
    gluing_rate = gluing_pass / gluing_total if gluing_total > 0 else 0

    print(f"\nTests: {locality_total}")
    print(f"Locality: {locality_pass}/{locality_total} ({locality_rate*100:.1f}%)")
    print(f"Gluing: {gluing_pass}/{gluing_total} ({gluing_rate*100:.1f}%)")

    return locality_rate, gluing_rate


def analyze_violation_cases():
    """
    Analyze specific cases where sheaf axioms fail.
    Try to identify patterns.
    """
    print("\n" + "=" * 70)
    print("Violation Analysis")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 500

    violations = []
    stats = {
        'R_full_low': 0,
        'R_full_high': 0,
        'low_variance_full': 0,
        'high_variance_full': 0,
        'parent_OPEN_subs_CLOSED': 0,
        'parent_CLOSED_subs_OPEN': 0,
    }

    for test_idx in range(n_tests):
        # Generate base context
        base_obs = np.random.normal(0, 1, 15)

        # Generate overlapping cover (2 parts)
        n = len(base_obs)
        split_point = n // 3

        sub1 = base_obs[:n - split_point]
        sub2 = base_obs[split_point:]

        # Verify overlap
        overlap = np.intersect1d(sub1, sub2)
        if len(overlap) == 0:
            continue

        # Compute gate states
        R_full = compute_R(base_obs, truth)
        R1 = compute_R(sub1, truth)
        R2 = compute_R(sub2, truth)

        gate_full = gate_state(R_full, threshold)
        gate1 = gate_state(R1, threshold)
        gate2 = gate_state(R2, threshold)

        # Collect stats
        stats['R_full_low'] += 1 if R_full < 0.5 else 0
        stats['R_full_high'] += 1 if R_full > 0.7 else 0
        stats['low_variance_full'] += 1 if np.std(base_obs) < 0.8 else 0
        stats['high_variance_full'] += 1 if np.std(base_obs) > 1.2 else 0

        # Check for violations
        # Type 1: Parent OPEN, both subs CLOSED
        if gate_full and not gate1 and not gate2:
            violations.append({
                'type': 'parent_OPEN_subs_CLOSED',
                'test': test_idx,
                'R_full': R_full,
                'R1': R1,
                'R2': R2,
                'std_full': np.std(base_obs),
                'std1': np.std(sub1),
                'std2': np.std(sub2),
            })
            stats['parent_OPEN_subs_CLOSED'] += 1

        # Type 2: Parent CLOSED, both subs OPEN
        if not gate_full and gate1 and gate2:
            violations.append({
                'type': 'parent_CLOSED_subs_OPEN',
                'test': test_idx,
                'R_full': R_full,
                'R1': R1,
                'R2': R2,
                'std_full': np.std(base_obs),
                'std1': np.std(sub1),
                'std2': np.std(sub2),
            })
            stats['parent_CLOSED_subs_OPEN'] += 1

    print(f"\nAnalysis of {len(violations)} violations:")

    # Analyze violation types
    type1_count = sum(1 for v in violations if v['type'] == 'parent_OPEN_subs_CLOSED')
    type2_count = sum(1 for v in violations if v['type'] == 'parent_CLOSED_subs_OPEN')

    print(f"\nViolation Types:")
    print(f"  Parent OPEN, Subs CLOSED: {type1_count}")
    print(f"  Parent CLOSED, Subs OPEN: {type2_count}")

    # Analyze R-value distribution
    print(f"\nParent R Distribution:")
    print(f"  Low R (< 0.5): {stats['R_full_low']}")
    print(f"  High R (> 0.7): {stats['R_full_high']}")

    print(f"\nVariance Distribution:")
    print(f"  Low variance (< 0.8): {stats['low_variance_full']}")
    print(f"  High variance (> 1.2): {stats['high_variance_full']}")

    # Analyze violation cases specifically
    if type2_count > 0:
        print(f"\nAnalysis of 'Parent CLOSED, Subs OPEN' cases:")
        type2_violations = [v for v in violations if v['type'] == 'parent_CLOSED_subs_OPEN']

        R_full_vals = [v['R_full'] for v in type2_violations]
        R_sub_vals = [(v['R1'] + v['R2']) / 2 for v in type2_violations]

        print(f"  Mean R_full: {np.mean(R_full_vals):.4f}")
        print(f"  Mean R_sub: {np.mean(R_sub_vals):.4f}")
        print(f"  Mean std_full: {np.mean([v['std_full'] for v in type2_violations]):.4f}")
        print(f"  Mean std_sub: {np.mean([v['std1'] + v['std2'] for v in type2_violations]) / 2:.4f}")

    return violations


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("GROTHENDIECK TOPOLOGY DEFINITION")
    print("=" * 70)

    # Test R-cover definition
    locality_rate, gluing_rate = test_r_cover_sheaf_property()

    # Analyze violations
    violations = analyze_violation_cases()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nR-COVER Topology:")
    print(f"  Locality: {locality_rate*100:.1f}%")
    print(f"  Gluing: {gluing_rate*100:.1f}%")

    if locality_rate > 0.9 and gluing_rate > 0.9:
        print("\nFINDING: R-cover Grothendieck topology makes gate a GOOD SHEAF!")
    else:
        print("\nFINDING: R-cover topology improves but doesn't fully resolve violations.")
