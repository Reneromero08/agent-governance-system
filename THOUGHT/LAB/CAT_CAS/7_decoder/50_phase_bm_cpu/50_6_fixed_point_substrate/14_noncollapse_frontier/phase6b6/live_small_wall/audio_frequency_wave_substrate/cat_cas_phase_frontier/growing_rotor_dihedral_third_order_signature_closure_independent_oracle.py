#!/usr/bin/env python3
"""Independent particle-enumeration oracle for the M196 refined quotient."""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Literal


GRID = 17
ROTOR_COUNTS = (2, 3, 4, 5, 6)
PRIMES = (103, 239)
PRIMARY_DEPTH = 3
REUSE_DEPTH = 2
SELECTED_TRIANGLES = ((1, 2, 3), (1, 4, 5))
Histogram = tuple[int, ...]
Signature = tuple[int, ...]
Triangle = tuple[int, int, int]
Control = Literal["correct", "missing", "wrong", "reordered"]


def histogram(modes: tuple[int, ...]) -> Histogram:
    result = [0] * GRID
    for mode in modes:
        result[mode] += 1
    return tuple(result)


def particle_modes(item: Histogram) -> tuple[int, ...]:
    return tuple(mode for mode, count in enumerate(item) for _ in range(count))


def rotate(item: Histogram, amount: int) -> Histogram:
    return tuple(item[(mode - amount) % GRID] for mode in range(GRID))


def canonical(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflected_representative(item: Histogram) -> Histogram:
    return canonical(tuple(item[(-mode) % GRID] for mode in range(GRID)))


def bracelet(item: Histogram) -> frozenset[Histogram]:
    return frozenset((canonical(item), reflected_representative(item)))


def triangle_shape(left: int, middle: int, right: int) -> Triangle:
    values = sorted(
        min((first - second) % GRID, (second - first) % GRID)
        for first, second in (
            (left, middle),
            (left, right),
            (middle, right),
        )
    )
    return values[0], values[1], values[2]


def triangle_counts(item: Histogram) -> dict[Triangle, int]:
    result: collections.Counter[Triangle] = collections.Counter()
    for left, middle, right in itertools.combinations(particle_modes(item), 3):
        result[triangle_shape(left, middle, right)] += 1
    return dict(result)


def particle_pair_signature(item: Histogram) -> tuple[int, ...]:
    modes = particle_modes(item)
    values = [0] * 9
    for left, right in itertools.combinations(range(len(modes)), 2):
        delta = (modes[right] - modes[left]) % GRID
        values[min(delta, GRID - delta)] += 1
    if sum(values) != math.comb(len(modes), 2):
        raise RuntimeError("oracle pair total failed")
    return tuple(values)


def refined_signature(item: Histogram) -> Signature:
    triangles = triangle_counts(item)
    return particle_pair_signature(item) + tuple(
        triangles.get(shape, 0) for shape in SELECTED_TRIANGLES
    )


def particle_shift_profile(
    item: Histogram, signatures: dict[Signature, int]
) -> dict[tuple[int, int], int]:
    modes = particle_modes(item)
    result: collections.Counter[tuple[int, int]] = collections.Counter()
    for first, second in itertools.permutations(range(len(modes)), 2):
        for shift in range(1, GRID):
            moved = list(modes)
            moved[first] = (moved[first] - shift) % GRID
            moved[second] = (moved[second] + shift) % GRID
            destination = refined_signature(histogram(tuple(moved)))
            if destination not in signatures:
                raise RuntimeError("oracle transition escaped refined chart")
            result[(shift, signatures[destination])] += 1
    return dict(result)


@dataclass(frozen=True)
class OracleTopology:
    rotors: int
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]
    shift_rows: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    occupation_histograms: int
    necklace_cells: int
    bracelet_cells: int
    mode_pair_shift_terms: int
    weighted_particle_shift_terms: int
    plan_nonzeros: int
    equitability_comparisons: int
    verification_peak_entries: int


def occupations(rotors: int) -> tuple[Histogram, ...]:
    return tuple(
        histogram(values)
        for values in itertools.combinations_with_replacement(range(GRID), rotors)
    )


def compile_topology(rotors: int, prime: int, root: int) -> OracleTopology:
    all_occupations = occupations(rotors)
    expected = math.comb(rotors + GRID - 1, rotors)
    if len(all_occupations) != expected:
        raise RuntimeError("oracle occupation count failed")
    necklaces = tuple(sorted({canonical(item) for item in all_occupations}))
    if len(necklaces) != expected // GRID:
        raise RuntimeError("oracle necklace count failed")

    fibers: dict[Signature, list[Histogram]] = collections.defaultdict(list)
    boundary: dict[Signature, int] = collections.defaultdict(int)
    for index, item in enumerate(necklaces):
        signature = refined_signature(item)
        fibers[signature].append(item)
        boundary[signature] = (
            boundary[signature]
            + pow(root, (11 * index + 5 * signature[0] + 1) % GRID, prime)
        ) % prime
    signatures = tuple(sorted(fibers))
    signature_index = {value: index for index, value in enumerate(signatures)}
    representatives = tuple(fibers[value][0] for value in signatures)

    reference_profiles = [
        particle_shift_profile(item, signature_index) for item in representatives
    ]
    comparisons = 0
    peak_entries = 0
    for signature in signatures:
        reference = reference_profiles[signature_index[signature]]
        for item in fibers[signature]:
            candidate = particle_shift_profile(item, signature_index)
            peak_entries = max(peak_entries, len(candidate) + len(reference))
            if candidate != reference:
                raise RuntimeError("oracle refined fiber is not equitable")
            if item != fibers[signature][0]:
                comparisons += GRID - 1

    shift_builders: list[list[tuple[tuple[int, int], ...]]] = [
        [] for _ in range(GRID - 1)
    ]
    for profile in reference_profiles:
        for shift in range(1, GRID):
            shift_builders[shift - 1].append(
                tuple(
                    sorted(
                        (source, coefficient)
                        for (row_shift, source), coefficient in profile.items()
                        if row_shift == shift
                    )
                )
            )
    shift_rows = tuple(tuple(rows) for rows in shift_builders)
    plan_nonzeros = sum(len(row) for basis in shift_rows for row in basis)
    mode_terms = 0
    for item in representatives:
        occupied_pairs = sum(
            item[first]
            * (item[second] - (1 if first == second else 0))
            > 0
            for first in range(GRID)
            for second in range(GRID)
        )
        mode_terms += occupied_pairs * (GRID - 1)
    bracelet_count = len({bracelet(item) for item in necklaces})
    if len(signatures) != bracelet_count:
        raise RuntimeError("oracle refined signature differs from bracelet quotient")
    return OracleTopology(
        rotors=rotors,
        signatures=signatures,
        representatives=representatives,
        boundary_weights=tuple(boundary[value] for value in signatures),
        shift_rows=shift_rows,
        occupation_histograms=len(all_occupations),
        necklace_cells=len(necklaces),
        bracelet_cells=bracelet_count,
        mode_pair_shift_terms=mode_terms,
        weighted_particle_shift_terms=len(signatures) * rotors * (rotors - 1) * (GRID - 1),
        plan_nonzeros=plan_nonzeros,
        equitability_comparisons=comparisons,
        verification_peak_entries=peak_entries,
    )


def verify_minimality() -> dict[str, object]:
    necklaces = tuple(sorted({canonical(item) for item in occupations(6)}))
    data = {item: triangle_counts(item) for item in necklaces}
    coordinates = tuple(sorted({shape for values in data.values() for shape in values}))
    pair_fibers: dict[Signature, list[Histogram]] = collections.defaultdict(list)
    for item in necklaces:
        pair_fibers[particle_pair_signature(item)].append(item)
    collisions: list[tuple[Histogram, Histogram]] = []
    for items in pair_fibers.values():
        orbits: list[frozenset[Histogram]] = []
        for item in items:
            value = bracelet(item)
            if value not in orbits:
                orbits.append(value)
        collisions.extend(
            (min(left), min(right))
            for left, right in itertools.combinations(orbits, 2)
        )
    successful_singles = sum(
        all(data[left].get(shape, 0) != data[right].get(shape, 0) for left, right in collisions)
        for shape in coordinates
    )
    successful_pairs = 0
    first_pair: tuple[Triangle, Triangle] | None = None
    for pair in itertools.combinations(coordinates, 2):
        if all(
            any(data[left].get(shape, 0) != data[right].get(shape, 0) for shape in pair)
            for left, right in collisions
        ):
            successful_pairs += 1
            if first_pair is None:
                first_pair = pair
    if len(collisions) != 24 or successful_singles or first_pair != SELECTED_TRIANGLES:
        raise RuntimeError("oracle minimality certificate changed")
    return {
        "observed_triangle_coordinate_candidates": len(coordinates),
        "homometric_bracelet_pairs_requiring_separation": len(collisions),
        "successful_single_coordinates": successful_singles,
        "successful_coordinate_pairs": successful_pairs,
        "lexicographically_first_successful_pair": [list(value) for value in SELECTED_TRIANGLES],
        "minimal_selected_coordinate_count_within_candidate_family": 2,
        "verification_only_necklace_records": len(necklaces),
        "verification_only_triangle_coordinate_cells": sum(len(values) for values in data.values()),
    }


def primitive_generator(prime: int) -> int:
    factors = []
    remainder = prime - 1
    divisor = 2
    while divisor * divisor <= remainder:
        if remainder % divisor == 0:
            factors.append(divisor)
            while remainder % divisor == 0:
                remainder //= divisor
        divisor += 1
    if remainder > 1:
        factors.append(remainder)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    raise RuntimeError("oracle primitive generator failed")


def scattering_integer(shift: int, step: int, tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + ((distance + 2) * (step + 1) + (3 * distance + 1) * (tag + 2)) % GRID % 5
    return -magnitude if (distance + step + tag) % GRID % 3 == 0 else magnitude


def pair_weight(distance: int, step: int, tag: int) -> int:
    return 1 + ((distance + 1) * (distance + 3) + (2 * distance + 1) * (step + 1) + (3 * distance + 2) * tag) % GRID % (GRID - 1)


def program(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple((step, (family + 3 * step + step * step) % 7) for step in range(depth))


def diagonal(state: list[int], topology: OracleTopology, prime: int, root: int, step: int, tag: int) -> list[int]:
    return [
        value * pow(root, sum(count * pair_weight(distance, step, tag) for distance, count in enumerate(signature[:9])) % GRID, prime) % prime
        for value, signature in zip(state, topology.signatures, strict=True)
    ]


def scatter(state: list[int], topology: OracleTopology, prime: int, step: int, tag: int) -> list[int]:
    output = [0] * len(state)
    for target in range(len(state)):
        total = 0
        for shift, basis in enumerate(topology.shift_rows, 1):
            weight = scattering_integer(shift, step, tag)
            total += sum(coefficient * weight * state[source] for source, coefficient in basis[target])
        output[target] = total % prime
    return output


def execute(source: list[int], topology: OracleTopology, prime: int, root: int, word: tuple[tuple[int, int], ...], reordered: bool = False) -> list[int]:
    state = source.copy()
    for step, tag in word:
        if reordered:
            state = diagonal(scatter(state, topology, prime, step, tag), topology, prime, root, step, tag)
        else:
            state = scatter(diagonal(state, topology, prime, root, step, tag), topology, prime, step, tag)
    return state


def source_state(signatures: tuple[Signature, ...], prime: int, family: int) -> list[int]:
    return [
        (1 + (family + 3) * (index + 1) + sum((coordinate + 2 + family) * (count + 1) ** 2 for coordinate, count in enumerate(signature))) % prime
        for index, signature in enumerate(signatures)
    ]


def boundary(state: list[int], topology: OracleTopology, prime: int) -> int:
    return sum(value * weight for value, weight in zip(state, topology.boundary_weights, strict=True)) % prime


def commitment(state: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, state)).encode()).hexdigest()


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def transaction(carrier: Carrier, expected: list[int], topology: OracleTopology, prime: int, root: int, word: tuple[tuple[int, int], ...], inverse_word: tuple[tuple[int, int], ...], control: Control) -> dict[str, object]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("oracle null carrier")
    source_id, target_id = id(carrier.source), id(carrier.target)
    forward = execute(carrier.source, topology, prime, root, word)
    carrier.target[:] = [(left + right) % prime for left, right in zip(carrier.target, forward, strict=True)]
    projected = boundary(carrier.target, topology, prime)
    if control != "missing":
        inverse = execute(carrier.source, topology, prime, root, inverse_word, reordered=control == "reordered")
        carrier.target[:] = [(left - right) % prime for left, right in zip(carrier.target, inverse, strict=True)]
    error = sum(left != right for left, right in zip(carrier.source, expected, strict=True)) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    return {
        "boundary": projected,
        "forward_commitment": commitment(forward),
        "restoration_error_field_cells": error,
        "same_backing": id(carrier.source) == source_id and id(carrier.target) == target_id,
        "generation": carrier.generation,
    }


def run_transaction_case(topology: OracleTopology, prime: int) -> dict[str, object]:
    generator = primitive_generator(prime)
    root = pow(generator, (prime - 1) // GRID, prime)
    primary_word, wrong_word, reuse_word = program(PRIMARY_DEPTH, 0), program(PRIMARY_DEPTH, 1), program(REUSE_DEPTH, 4)
    source = source_state(topology.signatures, prime, 0)
    carrier = Carrier(source.copy(), [0] * len(source))
    primary = transaction(carrier, source, topology, prime, root, primary_word, primary_word, "correct")
    reuse = transaction(carrier, source, topology, prime, root, reuse_word, reuse_word, "correct")
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse = transaction(fresh, source, topology, prime, root, reuse_word, reuse_word, "correct")
    if primary["restoration_error_field_cells"] or reuse["restoration_error_field_cells"] or reuse["boundary"] != fresh_reuse["boundary"]:
        raise RuntimeError("oracle restoration failed")
    controls: dict[str, int | bool] = {}
    for control, inverse in (("missing", primary_word), ("wrong", wrong_word), ("reordered", primary_word)):
        result = transaction(Carrier(source.copy(), [0] * len(source)), source, topology, prime, root, primary_word, inverse, control)
        controls[f"{control}_inverse_error_field_cells"] = int(result["restoration_error_field_cells"])
    if min(int(value) for value in controls.values()) == 0:
        raise RuntimeError("oracle controls failed")
    try:
        transaction(Carrier([], []), [], topology, prime, root, primary_word, primary_word, "correct")
    except ValueError:
        controls["null_carrier_rejected"] = True
    cells, nonzeros = len(topology.signatures), topology.plan_nonzeros
    return {
        "prime": prime,
        "multiplicative_generator": generator,
        "seventeenth_root": root,
        "refined_signature_cells": cells,
        "primary_depth": PRIMARY_DEPTH,
        "reuse_depth": REUSE_DEPTH,
        "primary_boundary": primary["boundary"],
        "reuse_boundary": reuse["boundary"],
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "primary_output_commitment": primary["forward_commitment"],
        "primary_restoration_error_field_cells": primary["restoration_error_field_cells"],
        "reuse_restoration_error_field_cells": reuse["restoration_error_field_cells"],
        "same_backing_primary": primary["same_backing"],
        "same_backing_reuse": reuse["same_backing"],
        "restoration_generation_after_reuse": carrier.generation,
        "controls": controls,
        "retained_shift_basis_plans": GRID - 1,
        "retained_shift_plan_nonzero_entries": nonzeros,
        "public_signature_descriptor_integer_cells": 11 * cells,
        "public_representative_descriptor_integer_cells": GRID * cells,
        "public_boundary_weight_field_cells": cells,
        "streamed_mode_pair_shift_terms_per_scattering": topology.mode_pair_shift_terms,
        "streamed_weighted_particle_shift_terms_per_scattering": topology.weighted_particle_shift_terms,
        "public_plan_compiler_third_order_triangle_shape_tests": topology.mode_pair_shift_terms * math.comb(6, 3),
        "primary_forward_plan_nonzero_applications": PRIMARY_DEPTH * nonzeros,
        "primary_forward_inverse_plan_nonzero_applications": 2 * PRIMARY_DEPTH * nonzeros,
        "reuse_forward_inverse_plan_nonzero_applications": 2 * REUSE_DEPTH * nonzeros,
    }


def main() -> None:
    generator = primitive_generator(PRIMES[0])
    root = pow(generator, (PRIMES[0] - 1) // GRID, PRIMES[0])
    topologies = [compile_topology(rotors, PRIMES[0], root) for rotors in ROTOR_COUNTS]
    topology_cases = [
        {
            "rotors": value.rotors,
            "occupation_histograms": value.occupation_histograms,
            "necklace_cells": value.necklace_cells,
            "bracelet_cells": value.bracelet_cells,
            "refined_signature_cells": len(value.signatures),
            "refined_signatures_equal_dihedral_orbits": True,
            "all_sixteen_shift_bases_equitable": True,
            "equitability_comparisons": value.equitability_comparisons,
            "verification_only_occupation_histograms_visited": value.occupation_histograms,
            "verification_only_peak_sparse_row_entries": value.verification_peak_entries,
            "prior_materialized_plan_nonzeros": value.plan_nonzeros,
        }
        for value in topologies
    ]
    minimality = verify_minimality()
    transaction_cases = [run_transaction_case(topologies[-1], prime) for prime in PRIMES]
    print(json.dumps({
        "claim_candidate": "BOUNDED_EXACT_MINIMAL_WITHIN33_TRIANGLE_COORDINATES_TWO_COORDINATE_DIHEDRAL_THIRD_ORDER_PHASE_SIGNATURE_REFINEMENT_SEPARATES_ALL24_ROTOR6_HOMOMETRIC_PAIR_FIBERS_AND_CLOSES_ALL16_SIGNED_SCATTERING_BASES_ON2277_CELLS_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS28S_PUBLIC_INTEGER_DESCRIPTORS_16_SHIFT_PLANS_AND_AN_IDENTICAL_CLASSICAL_RECURRENCE",
        "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_REFINED_SIGNATURE_INPUTS_ROTORS2_TO6_TOPOLOGY_F103_F239_ROTOR6_PRIMARY_DEPTH3_REUSE_DEPTH2_DIRECT_PROCESS_SOFTWARE_ONLY",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "result": "PASS_REFINED_CLOSURE",
        "production_source_imported": False,
        "production_transition_called": False,
        "multiset_occupation_oracle": True,
        "ordered_particle_transition_oracle": True,
        "selected_triangle_coordinates": [list(value) for value in SELECTED_TRIANGLES],
        "minimality": minimality,
        "topology_cases": topology_cases,
        "transaction_cases": transaction_cases,
        "matched_classical_recurrence": "IDENTICAL_PLAN_COMPILED_REFINED_SIGNATURE_QUOTIENT_RECURRENCE",
        "full_bracelet_fallback_required": False,
        "refinement_smaller_than_full_bracelet_identity": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
