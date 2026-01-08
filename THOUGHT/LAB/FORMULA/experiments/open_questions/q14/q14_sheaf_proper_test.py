"""
Q14: Proper Sheaf Test (Valid Covering Families)

TEST: Does gate presheaf satisfy sheaf axioms with valid covers?

KEY: Covering families must be overlapping and reconstructable.
Non-overlapping splits change std too much to test sheaf property.
"""

import numpy as np
from typing import List, Tuple


def compute_R(observations: np.ndarray, truth: float) -> float:
    """R = E / grad_S"""
    E = 1.0 / (1.0 + abs(np.mean(observations) - truth))
    grad_S = np.std(observations) + 1e-10
    return E / grad_S


def gate_state(R: float, threshold: float) -> bool:
    """OPEN if R > threshold"""
    return R > threshold


def generate_overlap_covering(base_obs: np.ndarray, n_cover: int) -> List[np.ndarray]:
    """
    Generate a valid covering family with overlaps.

    A covering {U_1, ..., U_k} of U is valid if:
    - Each U_i ⊆ U
    - Union of all U_i = U (reconstructable)
    - Overlaps ensure information consistency

    Strategy: Sliding windows with overlap
    """
    n = len(base_obs)
    covers = []

    if n <= n_cover:
        # If base is small, just use full context
        covers = [base_obs]
        return covers

    # Sliding window approach
    window_size = max(3, n // (n_cover - 1))
    overlap = window_size // 2

    for i in range(n_cover - 1):
        start = i * (window_size - overlap)
        end = min(start + window_size, n)
        covers.append(base_obs[start:end].copy())

    # Add final segment (ensure coverage)
    covers.append(base_obs[max(0, (n_cover - 2) * (window_size - overlap)):].copy())

    return covers


def test_sheaf_locality():
    """
    LOCALITY AXIOM: If two sections agree on all restrictions, they are equal.

    For gate presheaf G: C^op → Set:
    - Sections are gate states (OPEN/CLOSED)
    - If s, t ∈ G(U) and s|V_i = t|V_i for all V_i in a cover,
    - Then s = t

    TEST: If all sub-contexts have same gate state, does parent have that state?
    """
    print("=" * 70)
    print("Sheaf Locality Test")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 500

    locality_pass = 0
    locality_total = 0
    violations = []

    for test_idx in range(n_tests):
        # Generate base context U
        base_obs = np.random.normal(0, 1, 20)

        # Generate overlapping covering
        n_cover = np.random.randint(2, 5)
        covers = generate_overlap_covering(base_obs, n_cover)

        # Verify coverage: union should reconstruct base
        # (allow duplicates - sliding windows overlap)
        all_cover_obs = np.concatenate(covers)
        # Remove duplicates for verification
        unique_cover = np.unique(all_cover_obs)
        unique_base = np.unique(base_obs)

        if len(unique_cover) != len(unique_base) or not np.allclose(np.sort(unique_cover), np.sort(unique_base)):
            # Not a valid cover - skip
            continue

        locality_total += 1

        # Compute gate states
        R_full = compute_R(base_obs, truth)
        gate_full = gate_state(R_full, threshold)

        gate_subs = []
        for cover in covers:
            if len(cover) > 0:
                R_sub = compute_R(cover, truth)
                gate_sub = gate_state(R_sub, threshold)
                gate_subs.append(gate_sub)

        # Locality: if all sub-contexts agree, parent should agree
        if len(set(gate_subs)) == 1:
            # All sub-contexts have same gate state
            expected = gate_subs[0]

            if gate_full == expected:
                locality_pass += 1
            else:
                violations.append({
                    'test': test_idx,
                    'parent_gate': gate_full,
                    'sub_gates': gate_subs,
                    'R_parent': R_full,
                    'R_subs': [compute_R(c, truth) for c in covers if len(c) > 0],
                    'base_obs': base_obs,
                    'covers': covers
                })

    locality_rate = locality_pass / locality_total if locality_total > 0 else 0

    print(f"\nTests: {locality_total}")
    print(f"Locality passes: {locality_pass} ({locality_rate*100:.1f}%)")
    print(f"Violations: {len(violations)}")

    if len(violations) > 0:
        print(f"\nSample violations:")
        for i, v in enumerate(violations[:3]):
            print(f"  {i+1}. Parent: {v['parent_gate']}, Subs: {v['sub_gates']}")
            print(f"     R_parent: {v['R_parent']:.4f}, R_subs: {[f'{r:.4f}' for r in v['R_subs']]}")

    if locality_rate > 0.9:
        print("\nFINDING: Gate SATISFIES locality axiom!")
        print("If all sub-contexts agree, parent agrees.")
        return True
    else:
        print("\nFINDING: Gate does NOT satisfy locality axiom.")
        print(f"High violation rate: {(1-locality_rate)*100:.1f}%")
        return False


def test_sheaf_gluing():
    """
    GLUING AXIOM: If sections agree on overlaps, they glue uniquely.

    For gate presheaf G: C^op → Set:
    - If {U_i} covers U and s_i ∈ G(U_i)
    - And s_i|U_i∩U_j = s_j|U_i∩U_j for all i,j
    - Then there exists unique s ∈ G(U) with s|U_i = s_i

    TEST: For overlapping cover with compatible sections, do they glue to parent?
    """
    print("\n" + "=" * 70)
    print("Sheaf Gluing Test")
    print("=" * 70)

    truth = 0.0
    threshold = 0.5
    n_tests = 500

    gluing_pass = 0
    gluing_total = 0

    for _ in range(n_tests):
        # Generate base context U
        base_obs = np.random.normal(0, 1, 15)

        # Generate overlapping covering
        n_cover = np.random.randint(2, 4)
        covers = generate_overlap_covering(base_obs, n_cover)

        # Verify coverage
        all_cover_obs = np.concatenate(covers)
        if len(np.sort(all_cover_obs)) != len(np.sort(base_obs)):
            continue

        # Compute gate states for each cover
        gate_subs = []
        for cover in covers:
            if len(cover) > 0:
                R_sub = compute_R(cover, truth)
                gate_sub = gate_state(R_sub, threshold)
                gate_subs.append(gate_sub)

        # Check compatibility on overlaps
        compatible = True
        for i in range(len(covers)):
            for j in range(i + 1, len(covers)):
                # Find intersection
                obs_i = covers[i]
                obs_j = covers[j]

                # Intersection (values that appear in both)
                intersection = np.intersect1d(obs_i, obs_j)

                if len(intersection) > 0:
                    # On intersection, gate states must be compatible
                    # For gate presheaf, they must be equal
                    if gate_subs[i] != gate_subs[j]:
                        compatible = False
                        break

        if compatible:
            gluing_total += 1

            # Try to glue: all compatible, should glue to parent
            R_full = compute_R(base_obs, truth)
            gate_full = gate_state(R_full, threshold)

            # If all compatible sections are same, glue should be that value
            if len(set(gate_subs)) == 1:
                expected_glue = gate_subs[0]
                if gate_full == expected_glue:
                    gluing_pass += 1
            else:
                # Different gate states in compatible cover
                # For gate presheaf, this means they're all compatible
                # (no overlap disagreement) but have different values
                # Gluing should give parent value
                if gate_full in gate_subs:
                    gluing_pass += 1

    gluing_rate = gluing_pass / gluing_total if gluing_total > 0 else 0

    print(f"\nTests: {gluing_total}")
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
    COMPREHENSIVE SHEAF TEST: Both locality and gluing together.

    For gate to be a sheaf, BOTH axioms must hold.
    """
    print("\n" + "=" * 70)
    print("Comprehensive Sheaf Test (Locality + Gluing)")
    print("=" * 70)

    locality_result = test_sheaf_locality()
    gluing_result = test_sheaf_gluing()

    print("\n" + "=" * 70)
    print("Sheaf Axioms Summary")
    print("=" * 70)

    print(f"\nLocality: {'PASS' if locality_result else 'FAIL'}")
    print(f"Gluing: {'PASS' if gluing_result else 'FAIL'}")

    if locality_result and gluing_result:
        print("\nFINDING: Gate presheaf IS a SHEAF!")
        print("Both locality and gluing axioms hold.")
        print("\nINTERPRETATION:")
        print("Local agreement DOES lead to global consistency.")
        print("Gate state on parent context is determined by sub-contexts.")
        return True
    else:
        print("\nFINDING: Gate presheaf is NOT a sheaf.")
        print("One or both sheaf axioms fail.")
        print("\nINTERPRETATION:")
        print("Gate state on parent context is NOT determined by sub-contexts.")
        print("Each context must be evaluated independently.")
        return False


if __name__ == "__main__":
    np.random.seed(42)

    result = test_sheaf_comprehensive()

    if result:
        print("\n" + "=" * 70)
        print("FINAL ANSWER: Gate IS a sheaf")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FINAL ANSWER: Gate is NOT a sheaf")
        print("=" * 70)
