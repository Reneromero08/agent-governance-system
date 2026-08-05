#!/usr/bin/env python3
"""Independent particle-label oracle for the pair-signature transfer no-go."""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math


GRID = 17
ROTOR_COUNTS = (2, 3, 4, 5, 6)
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


def histogram(particles: tuple[int, ...]) -> Histogram:
    values = [0] * GRID
    for mode in particles:
        values[mode] += 1
    return tuple(values)


def particles(item: Histogram) -> tuple[int, ...]:
    return tuple(mode for mode, count in enumerate(item) for _ in range(count))


def rotate(item: Histogram, amount: int) -> Histogram:
    return tuple(item[(mode - amount) % GRID] for mode in range(GRID))


def cyclic_representative(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflection_representative(item: Histogram) -> Histogram:
    mirrored = tuple(item[(-mode) % GRID] for mode in range(GRID))
    return cyclic_representative(mirrored)


def bracelet_orbit(item: Histogram) -> frozenset[Histogram]:
    return frozenset((cyclic_representative(item), reflection_representative(item)))


def particle_pair_signature(item: Histogram) -> Signature:
    modes = particles(item)
    values = [0] * 9
    for left in range(len(modes)):
        for right in range(left + 1, len(modes)):
            delta = (modes[right] - modes[left]) % GRID
            values[min(delta, GRID - delta)] += 1
    if sum(values) != math.comb(len(modes), 2):
        raise RuntimeError("oracle unordered-particle signature failed")
    return tuple(values)


def ordered_particle_profile(
    item: Histogram, shift: int
) -> dict[Signature, int]:
    modes = particles(item)
    result: collections.Counter[Signature] = collections.Counter()
    for first, second in itertools.permutations(range(len(modes)), 2):
        moved = list(modes)
        moved[first] = (moved[first] - shift) % GRID
        moved[second] = (moved[second] + shift) % GRID
        result[particle_pair_signature(histogram(tuple(moved)))] += 1
    if sum(result.values()) != len(modes) * (len(modes) - 1):
        raise RuntimeError("oracle ordered-particle total failed")
    return dict(result)


def burnside(rotors: int) -> tuple[int, int, int, int]:
    occupation_count = math.comb(rotors + GRID - 1, rotors)
    necklaces = occupation_count // GRID
    reflection_fixed = sum(
        math.comb(pair_particles + 7, 7)
        for pair_particles in range(rotors // 2 + 1)
    )
    bracelets = (necklaces + reflection_fixed) // 2
    return occupation_count, necklaces, reflection_fixed, bracelets


def difference_summary(
    left: dict[Signature, int], right: dict[Signature, int]
) -> tuple[int, dict[str, object]]:
    differing_count = 0
    first: Signature | None = None
    for value in sorted(set(left) | set(right)):
        if left.get(value, 0) == right.get(value, 0):
            continue
        differing_count += 1
        if first is None:
            first = value
    if first is None:
        raise RuntimeError("difference summary requires unequal profiles")
    return differing_count, {
        "destination_signature": list(first),
        "left_coefficient": left.get(first, 0),
        "right_coefficient": right.get(first, 0),
    }


def profile_commitment(profile: dict[Signature, int]) -> str:
    encoded = ";".join(
        f"{','.join(map(str, value))}:{coefficient}"
        for value, coefficient in sorted(profile.items())
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_case(rotors: int) -> dict[str, object]:
    expected_occupations, expected_necklaces, reflection_fixed, expected_bracelets = burnside(rotors)
    occupations = tuple(
        histogram(item)
        for item in itertools.combinations_with_replacement(range(GRID), rotors)
    )
    if len(occupations) != expected_occupations:
        raise RuntimeError("oracle occupation count failed")
    necklaces = tuple(sorted({cyclic_representative(item) for item in occupations}))
    if len(necklaces) != expected_necklaces:
        raise RuntimeError("oracle necklace count failed")

    fibers: dict[Signature, list[Histogram]] = collections.defaultdict(list)
    for item in necklaces:
        fibers[particle_pair_signature(item)].append(item)
    bracelet_orbits = {bracelet_orbit(item) for item in necklaces}
    if len(bracelet_orbits) != expected_bracelets:
        raise RuntimeError("oracle bracelet count failed")

    fiber_sizes = collections.Counter(len(items) for items in fibers.values())
    non_dihedral: list[tuple[Signature, list[Histogram]]] = []
    for value, items in fibers.items():
        if set(items) != set(bracelet_orbit(items[0])):
            non_dihedral.append((value, items))

    reflection_checks = 0
    nonequitable = 0
    first_witness: dict[str, object] | None = None
    for value, items in fibers.items():
        base = items[0]
        mirrored = reflection_representative(base)
        for shift in range(1, GRID):
            if ordered_particle_profile(base, shift) != ordered_particle_profile(
                mirrored, shift
            ):
                raise RuntimeError("oracle reflection profile failed")
            reflection_checks += 1
        if set(items) == set(bracelet_orbit(base)):
            continue
        rejected = False
        for other in items[1:]:
            for shift in range(1, GRID):
                left = ordered_particle_profile(base, shift)
                right = ordered_particle_profile(other, shift)
                if left == right:
                    continue
                rejected = True
                if first_witness is None:
                    difference_count, first_difference = difference_summary(
                        left, right
                    )
                    first_witness = {
                        "target_signature": list(value),
                        "target_left": list(base),
                        "target_right": list(other),
                        "targets_are_dihedrally_related": other in bracelet_orbit(base),
                        "signed_shift": shift,
                        "left_row_sum": sum(left.values()),
                        "right_row_sum": sum(right.values()),
                        "differing_destination_count": difference_count,
                        "first_difference": first_difference,
                        "left_row_commitment": profile_commitment(left),
                        "right_row_commitment": profile_commitment(right),
                    }
                break
            if rejected:
                break
        if rejected:
            nonequitable += 1

    if rotors <= 5 and (non_dihedral or nonequitable):
        raise RuntimeError("oracle predecessor closure changed")
    if rotors == 6 and (len(non_dihedral) != 24 or not nonequitable):
        raise RuntimeError("oracle transfer obstruction changed")
    return {
        "rotors": rotors,
        "occupation_histograms": expected_occupations,
        "necklace_cells": expected_necklaces,
        "reflection_fixed_necklaces": reflection_fixed,
        "burnside_bracelet_cells": expected_bracelets,
        "pair_signature_cells": len(fibers),
        "pair_signature_minus_bracelet_cells": len(fibers) - expected_bracelets,
        "fiber_size_histogram": {
            str(size): count for size, count in sorted(fiber_sizes.items())
        },
        "non_dihedral_pair_signature_fibers": len(non_dihedral),
        "nonequitable_pair_signature_fibers": nonequitable,
        "pair_signatures_equal_dihedral_orbits": not non_dihedral,
        "all_sixteen_shift_bases_equitable": nonequitable == 0,
        "reflection_profile_checks": reflection_checks,
        "first_nonequitable_witness": first_witness,
    }


def main() -> None:
    cases = [run_case(rotors) for rotors in ROTOR_COUNTS]
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_PAIR_SIGNATURE_FIBERS_EQUAL_DIHEDRAL_BRACELET_ORBITS_FOR_GRID17_ROTORS2_TO5_BUT_AT_ROTOR6_24_HOMOMETRIC_NONDIHEDRAL_FIBERS_MERGE_2277_BRACELETS_INTO2253_SIGNATURES_AND_AT_LEAST_ONE_SHARED_SIGNATURE_HAS39_DIFFERENT_SHIFT1_DESTINATION_COEFFICIENTS_SO_PAIR_SIGNATURE_ONLY_SCATTERING_CLOSURE_FAILS_AT_THE_TRANSFER_CEILING",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_NECKLACES_ROTORS2_TO6_ALL16_SIGNED_PAIR_SHIFTS_EXACT_INTEGER_TOPOLOGY_DIAGNOSTIC_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "NO_RESTORATION_CLAIM",
                "result": "PASS_TRANSFER_NO_GO",
                "production_source_imported": False,
                "production_transition_called": False,
                "unordered_particle_signature_used": True,
                "ordered_particle_shift_used": True,
                "cases": cases,
                "burnside_law": "BRACELETS_EQUALS_HALF_OF_NECKLACES_PLUS_REFLECTION_FIXED_NECKLACES_FOR_ODD_GRID17_AND_ROTORS_BELOW17",
                "matched_classical_diagnostic": "IDENTICAL_HOMOMETRY_AND_SIGNED_SHIFT_PROFILE_COMPARISON",
                "pair_signature_only_analytic_kernel_transferable_through_rotor6": False,
                "higher_order_signature_repair_required": True,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
