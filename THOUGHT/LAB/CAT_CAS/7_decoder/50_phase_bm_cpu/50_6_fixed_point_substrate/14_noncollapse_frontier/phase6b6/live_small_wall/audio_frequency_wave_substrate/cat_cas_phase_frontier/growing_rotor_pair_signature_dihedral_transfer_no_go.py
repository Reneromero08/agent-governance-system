#!/usr/bin/env python3
"""Exact dihedral identity and R6 transfer no-go for pair signatures.

M193/M194 close pair-signature fibers through five rotors.  This diagnostic
identifies those fibers with dihedral bracelet orbits and tests the first
transfer rotor.  At six rotors, homometric non-dihedral bracelets merge under
the nine pair counts and have different signed-shift transition profiles.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
from typing import Iterator


GRID = 17
PAIR_CHANNELS = 9
ROTOR_COUNTS = (2, 3, 4, 5, 6)
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


def rotate(item: Histogram, amount: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(item):
        result[(mode + amount) % GRID] = count
    return tuple(result)


def rotation_canonical(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflect(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def dihedral_orbit(item: Histogram) -> frozenset[Histogram]:
    return frozenset(
        (rotation_canonical(item), rotation_canonical(reflect(item)))
    )


def iter_histograms(rotors: int) -> Iterator[Histogram]:
    working = [0] * GRID

    def visit(position: int, remaining: int) -> Iterator[Histogram]:
        if position == GRID - 1:
            working[position] = remaining
            yield tuple(working)
            return
        for count in range(remaining + 1):
            working[position] = count
            yield from visit(position + 1, remaining - count)

    yield from visit(0, rotors)


def pair_signature(item: Histogram, rotors: int) -> Signature:
    values = [sum(count * (count - 1) // 2 for count in item)]
    values.extend(
        sum(item[mode] * item[(mode + distance) % GRID] for mode in range(GRID))
        for distance in range(1, PAIR_CHANNELS)
    )
    if sum(values) != math.comb(rotors, 2):
        raise RuntimeError("pair-signature total failed")
    return tuple(values)


def reflection_fixed_histograms(rotors: int) -> int:
    return sum(
        math.comb(pairs + 7, 7) for pairs in range(rotors // 2 + 1)
    )


def burnside_counts(rotors: int) -> tuple[int, int, int]:
    occupations = math.comb(rotors + GRID - 1, rotors)
    if not 0 < rotors < GRID or occupations % GRID:
        raise RuntimeError("free rotation Burnside scope failed")
    necklaces = occupations // GRID
    reflection_fixed = reflection_fixed_histograms(rotors)
    if (necklaces + reflection_fixed) % 2:
        raise RuntimeError("bracelet Burnside parity failed")
    return occupations, necklaces, (necklaces + reflection_fixed) // 2


def shift_profile(
    item: Histogram, rotors: int, shift: int
) -> dict[Signature, int]:
    result: collections.Counter[Signature] = collections.Counter()
    for first in range(GRID):
        if item[first] == 0:
            continue
        for second in range(GRID):
            multiplicity = item[first] * (
                item[second] - (1 if first == second else 0)
            )
            if multiplicity == 0:
                continue
            moved = list(item)
            moved[first] -= 1
            moved[second] -= 1
            moved[(first - shift) % GRID] += 1
            moved[(second + shift) % GRID] += 1
            result[pair_signature(tuple(moved), rotors)] += multiplicity
    if sum(result.values()) != rotors * (rotors - 1):
        raise RuntimeError("shift-profile ordered-pair total failed")
    return dict(result)


def profile_difference_summary(
    left: dict[Signature, int], right: dict[Signature, int]
) -> tuple[int, dict[str, object]]:
    differing_count = 0
    first: Signature | None = None
    for signature in sorted(set(left) | set(right)):
        if left.get(signature, 0) == right.get(signature, 0):
            continue
        differing_count += 1
        if first is None:
            first = signature
    if first is None:
        raise RuntimeError("difference summary requires unequal profiles")
    return differing_count, {
        "destination_signature": list(first),
        "left_coefficient": left.get(first, 0),
        "right_coefficient": right.get(first, 0),
    }


def profile_commitment(profile: dict[Signature, int]) -> str:
    encoded = ";".join(
        f"{','.join(map(str, signature))}:{coefficient}"
        for signature, coefficient in sorted(profile.items())
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_case(rotors: int) -> dict[str, object]:
    expected_occupations, expected_necklaces, expected_bracelets = burnside_counts(
        rotors
    )
    necklaces: list[Histogram] = []
    occupations_visited = 0
    for item in iter_histograms(rotors):
        occupations_visited += 1
        if rotation_canonical(item) == item:
            necklaces.append(item)
    if occupations_visited != expected_occupations or len(necklaces) != expected_necklaces:
        raise RuntimeError("enumerated topology differs from Burnside count")

    fibers: dict[Signature, list[Histogram]] = collections.defaultdict(list)
    for item in necklaces:
        fibers[pair_signature(item, rotors)].append(item)
    fiber_size_histogram = collections.Counter(len(items) for items in fibers.values())
    non_dihedral: list[tuple[Signature, list[Histogram]]] = []
    dihedral_orbits: set[frozenset[Histogram]] = set()
    for signature, items in fibers.items():
        item_set = set(items)
        dihedral_orbits.update(dihedral_orbit(item) for item in items)
        if item_set != set(dihedral_orbit(items[0])):
            non_dihedral.append((signature, items))
    if len(dihedral_orbits) != expected_bracelets:
        raise RuntimeError("enumerated bracelet count differs from Burnside")

    nonequitable_fibers = 0
    first_witness: dict[str, object] | None = None
    reflection_profile_checks = 0
    for signature, items in fibers.items():
        base = items[0]
        reflected = rotation_canonical(reflect(base))
        for shift in range(1, GRID):
            if shift_profile(base, rotors, shift) != shift_profile(
                reflected, rotors, shift
            ):
                raise RuntimeError("signed shift failed reflection equivariance")
            reflection_profile_checks += 1
        if len(items) == len(dihedral_orbit(base)) and set(items) == set(
            dihedral_orbit(base)
        ):
            continue
        failed = False
        for other in items[1:]:
            for shift in range(1, GRID):
                left = shift_profile(base, rotors, shift)
                right = shift_profile(other, rotors, shift)
                if left == right:
                    continue
                failed = True
                if first_witness is None:
                    difference_count, first_difference = profile_difference_summary(
                        left, right
                    )
                    first_witness = {
                        "target_signature": list(signature),
                        "target_left": list(base),
                        "target_right": list(other),
                        "targets_are_dihedrally_related": other
                        in dihedral_orbit(base),
                        "signed_shift": shift,
                        "left_row_sum": sum(left.values()),
                        "right_row_sum": sum(right.values()),
                        "differing_destination_count": difference_count,
                        "first_difference": first_difference,
                        "left_row_commitment": profile_commitment(left),
                        "right_row_commitment": profile_commitment(right),
                    }
                break
            if failed:
                break
        if failed:
            nonequitable_fibers += 1

    if rotors <= 5 and (non_dihedral or nonequitable_fibers):
        raise RuntimeError("predecessor pair-signature closure changed")
    if rotors == 6 and (
        len(non_dihedral) != 24
        or nonequitable_fibers == 0
        or first_witness is None
    ):
        raise RuntimeError("six-rotor transfer obstruction changed")

    return {
        "rotors": rotors,
        "occupation_histograms": expected_occupations,
        "necklace_cells": expected_necklaces,
        "reflection_fixed_necklaces": reflection_fixed_histograms(rotors),
        "burnside_bracelet_cells": expected_bracelets,
        "pair_signature_cells": len(fibers),
        "pair_signature_minus_bracelet_cells": len(fibers) - expected_bracelets,
        "fiber_size_histogram": {
            str(size): count for size, count in sorted(fiber_size_histogram.items())
        },
        "non_dihedral_pair_signature_fibers": len(non_dihedral),
        "nonequitable_pair_signature_fibers": nonequitable_fibers,
        "pair_signatures_equal_dihedral_orbits": not non_dihedral,
        "all_sixteen_shift_bases_equitable": nonequitable_fibers == 0,
        "reflection_profile_checks": reflection_profile_checks,
        "first_nonequitable_witness": first_witness,
    }


def main() -> None:
    cases = [run_case(rotors) for rotors in ROTOR_COUNTS]
    if [case["necklace_cells"] for case in cases] != [9, 57, 285, 1197, 4389]:
        raise RuntimeError("necklace law changed")
    if [case["burnside_bracelet_cells"] for case in cases] != [9, 33, 165, 621, 2277]:
        raise RuntimeError("bracelet law changed")
    if [case["pair_signature_cells"] for case in cases] != [9, 33, 165, 621, 2253]:
        raise RuntimeError("pair-signature law changed")
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_PAIR_SIGNATURE_FIBERS_EQUAL_DIHEDRAL_BRACELET_ORBITS_FOR_GRID17_ROTORS2_TO5_BUT_AT_ROTOR6_24_HOMOMETRIC_NONDIHEDRAL_FIBERS_MERGE_2277_BRACELETS_INTO2253_SIGNATURES_AND_AT_LEAST_ONE_SHARED_SIGNATURE_HAS39_DIFFERENT_SHIFT1_DESTINATION_COEFFICIENTS_SO_PAIR_SIGNATURE_ONLY_SCATTERING_CLOSURE_FAILS_AT_THE_TRANSFER_CEILING",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_NECKLACES_ROTORS2_TO6_ALL16_SIGNED_PAIR_SHIFTS_EXACT_INTEGER_TOPOLOGY_DIAGNOSTIC_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "NO_RESTORATION_CLAIM",
                "result": "PASS_TRANSFER_NO_GO",
                "cases": cases,
                "burnside_law": "BRACELETS_EQUALS_HALF_OF_NECKLACES_PLUS_REFLECTION_FIXED_NECKLACES_FOR_ODD_GRID17_AND_ROTORS_BELOW17",
                "matched_classical_diagnostic": "IDENTICAL_HOMOMETRY_AND_SIGNED_SHIFT_PROFILE_COMPARISON",
                "pair_signature_only_analytic_kernel_transferable_through_rotor6": False,
                "higher_order_signature_repair_required": True,
                "catvm_custody": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "physical_bit_replacement": False,
                "unbounded_computation_established": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
