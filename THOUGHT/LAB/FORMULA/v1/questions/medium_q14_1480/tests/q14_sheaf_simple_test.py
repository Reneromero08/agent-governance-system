"""
Q14: Simple Sheaf Test (Direct Verification)

TEST: If all sub-contexts have same gate state, does parent have that state?

This is the core sheaf property: local consistency ⇒ global consistency.
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


def test_simple_sheaf_property():
    """
    Simple test: split context into sub-contexts, check if gate states align.

    If gate is a sheaf:
    - When all sub-contexts have same gate state → parent should have that state
    """
    print("=" * 70)
    print("Simple Sheaf Test: Local Consistency => Global Consistency")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 1000

    matches = 0
    total = 0
    violations = []

    for _ in range(n_tests):
        # Generate base context
        base_obs = np.random.normal(0, 1, 20)

        # Compute gate state on full context
        R_full = compute_R(base_obs, truth)
        gate_full = gate_state(R_full, threshold)

        # Split into 2-4 sub-contexts (non-overlapping for simplicity)
        n_sub = np.random.randint(2, 5)
        sub_gates = []

        # Simple split: divide base_obs into equal parts
        indices = np.array_split(np.arange(len(base_obs)), n_sub)
        sub_contexts = []

        for idx in indices:
            if len(idx) > 0:
                sub_obs = base_obs[idx]
                R_sub = compute_R(sub_obs, truth)
                gate_sub = gate_state(R_sub, threshold)
                sub_gates.append(gate_sub)
                sub_contexts.append(sub_obs)

        # Check: if all sub-contexts agree, does parent agree?
        if len(sub_gates) > 0:
            total += 1
            all_same = len(set(sub_gates)) == 1

            if all_same:
                # All sub-contexts have same gate state
                # Parent should have that state
                expected = sub_gates[0]

                if gate_full == expected:
                    matches += 1
                else:
                    violations.append({
                        'parent_gate': gate_full,
                        'sub_gates': sub_gates,
                        'R_parent': R_full,
                        'R_subs': [compute_R(obs, truth) for obs in sub_contexts]
                    })

    match_rate = matches / total if total > 0 else 0
    violation_rate = (total - matches) / total if total > 0 else 0

    print(f"\nTests: {total}")
    print(f"Matches (parent agrees with sub-contexts): {matches} ({match_rate*100:.1f}%)")
    print(f"Violations: {total - matches} ({violation_rate*100:.1f}%)")

    if len(violations) > 0:
        print(f"\nSample violations:")
        for i, v in enumerate(violations[:3]):
            print(f"  {i+1}. Parent: {v['parent_gate']}, Subs: {v['sub_gates']}")
            print(f"     R_parent: {v['R_parent']:.4f}, R_subs: {[f'{r:.4f}' for r in v['R_subs']]}")

    if violation_rate < 0.1:  # Less than 10% violations
        print("\nFINDING: Gate SATISFIES simple sheaf property!")
        print("Local consistency => global consistency holds most of the time.")
        return True
    else:
        print("\nFINDING: Gate does NOT fully satisfy sheaf property.")
        print("High violation rate suggests local != global consistency.")
        return False


def test_overlap_sheaf_property():
    """
    Test with overlapping sub-contexts (more realistic sheaf test).

    If gate is a sheaf:
    - When overlapping sub-contexts agree on overlaps → can glue to parent
    """
    print("\n" + "=" * 70)
    print("Overlap Sheaf Test: Compatibility on Intersections")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 500

    compatible_cases = 0
    glued_correctly = 0
    total_compatible = 0

    for _ in range(n_tests):
        # Generate base context
        base_obs = np.random.normal(0, 1, 15)

        # Create 2 overlapping sub-contexts
        start1 = np.random.randint(0, 8)
        end1 = np.random.randint(start1 + 3, 15)
        start2 = np.random.randint(0, 8)
        end2 = np.random.randint(start2 + 3, 15)

        obs1 = base_obs[start1:end1]
        obs2 = base_obs[start2:end2]

        R1 = compute_R(obs1, truth)
        R2 = compute_R(obs2, truth)
        gate1 = gate_state(R1, threshold)
        gate2 = gate_state(R2, threshold)

        # Find intersection
        indices1 = set(range(start1, end1))
        indices2 = set(range(start2, end2))
        intersection_indices = indices1 & indices2

        if len(intersection_indices) > 0:
            # Have overlap - check compatibility
            obs_intersect = base_obs[list(intersection_indices)]
            R_intersect = compute_R(obs_intersect, truth)
            gate_intersect = gate_state(R_intersect, threshold)

            # Compatibility: both sub-contexts should agree with intersection
            if gate1 == gate_intersect and gate2 == gate_intersect:
                total_compatible += 1

                # Try to glue: parent gate state
                R_full = compute_R(base_obs, truth)
                gate_full = gate_state(R_full, threshold)

                # Glued should match parent
                expected_glue = gate1  # Both same

                if gate_full == expected_glue:
                    glued_correctly += 1
            else:
                # Incompatible on intersection - cannot glue
                pass

    glue_rate = glued_correctly / total_compatible if total_compatible > 0 else 0

    print(f"\nCompatible cases (agreement on overlap): {total_compatible}")
    print(f"Glued correctly: {glued_correctly} ({glue_rate*100:.1f}%)")

    if glue_rate > 0.9:
        print("\nFINDING: Overlapping sub-contexts glue correctly!")
        print("Sheaf property holds for overlapping covers.")
        return True
    else:
        print("\nFINDING: Sheaf property does NOT fully hold for overlaps.")
        return False


if __name__ == "__main__":
    np.random.seed(42)

    test1 = test_simple_sheaf_property()
    test2 = test_overlap_sheaf_property()

    print("\n" + "=" * 70)
    print("Simplified Sheaf Test Summary")
    print("=" * 70)

    if test1 and test2:
        print("\nFINDING: Gate satisfies SHEAF PROPERTIES!")
        print("Local consistency leads to global consistency.")
        print("Gate is a sheaf on observation category.")
    else:
        print("\nFINDING: Sheaf properties not fully satisfied.")
        print("Gate may not be a standard sheaf.")
