"""
Q14: Fixed Sheaf Test (Simple Overlapping Covers)

Simplified approach: 2 overlapping sub-contexts that cover full context.
"""

import numpy as np
from typing import Tuple


def compute_R(observations: np.ndarray, truth: float) -> float:
    """R = E / grad_S"""
    E = 1.0 / (1.0 + abs(np.mean(observations) - truth))
    grad_S = np.std(observations) + 1e-10
    return E / grad_S


def gate_state(R: float, threshold: float) -> bool:
    """OPEN if R > threshold"""
    return R > threshold


def generate_simple_overlap_cover(base_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2 overlapping sub-contexts that cover the full context.

    Strategy:
    - Sub-context 1: first 2/3 of observations
    - Sub-context 2: last 2/3 of observations
    - Together they cover full context with 1/3 overlap
    """
    n = len(base_obs)

    # Split into two overlapping parts
    split_point = n // 3  # 1/3 from start

    # Sub1: from start to 2/3 point
    sub1_end = min(n, n * 2 // 3 + 1)
    sub1 = base_obs[:sub1_end].copy()

    # Sub2: from 1/3 point to end
    sub2_start = split_point
    sub2 = base_obs[sub2_start:].copy()

    return sub1, sub2


def test_locality():
    """
    LOCALITY: If two sections agree on all restrictions, they are equal.

    For gate: if both sub-contexts have same gate state,
    does parent have that gate state?
    """
    print("=" * 70)
    print("Sheaf Locality Test (2 overlapping sub-contexts)")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 1000

    locality_pass = 0
    locality_total = 0
    violations = []

    for test_idx in range(n_tests):
        # Generate base context
        base_obs = np.random.normal(0, 1, 15)

        # Generate overlapping cover
        sub1, sub2 = generate_simple_overlap_cover(base_obs)

        # Verify coverage (they should overlap and cover)
        # Find overlap
        overlap_len = len(np.intersect1d(sub1, sub2))

        if overlap_len == 0:
            # No overlap, skip
            continue

        # Check if they cover full context
        combined = np.concatenate([sub1, sub2])
        unique_combined = np.unique(combined)
        unique_base = np.unique(base_obs)

        if len(unique_combined) < len(unique_base):
            # Don't cover full context, skip
            continue

        locality_total += 1

        # Compute gate states
        R_full = compute_R(base_obs, truth)
        gate_full = gate_state(R_full, threshold)

        R1 = compute_R(sub1, truth)
        gate1 = gate_state(R1, threshold)

        R2 = compute_R(sub2, truth)
        gate2 = gate_state(R2, threshold)

        # Locality: if both sub-contexts agree, parent should agree
        if gate1 == gate2:
            # Both same - locality should hold
            if gate_full == gate1:
                locality_pass += 1
            else:
                violations.append({
                    'test': test_idx,
                    'parent_gate': gate_full,
                    'sub_gates': [gate1, gate2],
                    'R_parent': R_full,
                    'R_subs': [R1, R2],
                })

    locality_rate = locality_pass / locality_total if locality_total > 0 else 0

    print(f"\nValid tests: {locality_total}")
    print(f"Locality passes: {locality_pass} ({locality_rate*100:.1f}%)")
    print(f"Violations: {len(violations)}")

    if len(violations) > 0:
        print(f"\nSample violations:")
        for i, v in enumerate(violations[:3]):
            print(f"  {i+1}. Parent: {v['parent_gate']}, Subs: {v['sub_gates']}")
            print(f"     R_parent: {v['R_parent']:.4f}, R_subs: {[f'{r:.4f}' for r in v['R_subs']]}")

    if locality_rate > 0.9:
        print("\nFINDING: Gate SATISFIES locality axiom!")
        print("If both sub-contexts agree, parent agrees.")
        return True
    else:
        print("\nFINDING: Gate does NOT satisfy locality axiom.")
        print(f"High violation rate: {(1-locality_rate)*100:.1f}%")
        return False


def test_gluing():
    """
    GLUING: If sections agree on overlap, they glue uniquely.

    For gate: if both sub-contexts have same gate state on overlap,
    can they be glued to parent?
    """
    print("\n" + "=" * 70)
    print("Sheaf Gluing Test (2 overlapping sub-contexts)")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 1000

    gluing_pass = 0
    gluing_total = 0

    for _ in range(n_tests):
        # Generate base context
        base_obs = np.random.normal(0, 1, 15)

        # Generate overlapping cover
        sub1, sub2 = generate_simple_overlap_cover(base_obs)

        # Verify coverage
        overlap_len = len(np.intersect1d(sub1, sub2))
        combined = np.concatenate([sub1, sub2])
        unique_combined = np.unique(combined)
        unique_base = np.unique(base_obs)

        if overlap_len == 0 or len(unique_combined) < len(unique_base):
            continue

        gluing_total += 1

        # Compute gate states
        R_full = compute_R(base_obs, truth)
        gate_full = gate_state(R_full, threshold)

        R1 = compute_R(sub1, truth)
        gate1 = gate_state(R1, threshold)

        R2 = compute_R(sub2, truth)
        gate2 = gate_state(R2, threshold)

        # Check compatibility on overlap
        overlap = np.intersect1d(sub1, sub2)
        R_overlap = compute_R(overlap, truth)
        gate_overlap = gate_state(R_overlap, threshold)

        # Compatibility: both should agree with overlap
        compatible = (gate1 == gate_overlap) and (gate2 == gate_overlap)

        if compatible:
            # Try to glue: should give parent gate state
            if len(set([gate1, gate2])) == 1:
                # Both same - glue should be that value
                expected_glue = gate1
                if gate_full == expected_glue:
                    gluing_pass += 1
            else:
                # Different but compatible with overlap
                # For gate, this means they're both compatible
                # Glue should give parent value
                if gate_full in [gate1, gate2]:
                    gluing_pass += 1

    gluing_rate = gluing_pass / gluing_total if gluing_total > 0 else 0

    print(f"\nValid tests: {gluing_total}")
    print(f"Gluing passes: {gluing_pass} ({gluing_rate*100:.1f}%)")

    if gluing_rate > 0.9:
        print("\nFINDING: Gate SATISFIES gluing axiom!")
        print("Compatible sections glue to parent gate state.")
        return True
    else:
        print("\nFINDING: Gate does NOT satisfy gluing axiom.")
        print(f"Gluing failure rate: {(1-gluing_rate)*100:.1f}%")
        return False


def test_sheaf_comprehensive():
    """
    Comprehensive test: both locality and gluing.
    """
    print("\n" + "=" * 70)
    print("Comprehensive Sheaf Test")
    print("=" * 70)

    locality_result = test_locality()
    gluing_result = test_gluing()

    print("\n" + "=" * 70)
    print("Sheaf Axioms Summary")
    print("=" * 70)

    print(f"\nLocality: {'PASS' if locality_result else 'FAIL'}")
    print(f"Gluing: {'PASS' if gluing_result else 'FAIL'}")

    if locality_result and gluing_result:
        print("\n" + "=" * 70)
        print("FINDING: Gate IS a sheaf!")
        print("=" * 70)
        print("\nINTERPRETATION:")
        print("Local agreement leads to global consistency.")
        print("Gate state on parent context is determined by sub-contexts.")
        print("Gate presheaf satisfies both sheaf axioms.")
        return True
    else:
        print("\n" + "=" * 70)
        print("FINDING: Gate is NOT a sheaf")
        print("=" * 70)
        print("\nINTERPRETATION:")
        print("Local agreement does NOT lead to global consistency.")
        print("Gate state on parent context is NOT determined by sub-contexts.")
        print("Each context must be evaluated independently.")
        return False


if __name__ == "__main__":
    np.random.seed(42)

    result = test_sheaf_comprehensive()

    if result:
        print("\nFINAL ANSWER: Gate IS a sheaf")
    else:
        print("\nFINAL ANSWER: Gate is NOT a sheaf")
