#!/usr/bin/env python3
"""Exact two-coordinate third-order repair of the R6 homometry collision.

M195 showed that unordered pair distances identify dihedral bracelets only
through five rotors.  This successor augments the nine pair counts with two
public dihedral triangle-count coordinates.  It verifies minimality within the
33 observed single-coordinate triangle family, exact signed-shift closure,
and an in-place two-register transaction on the repaired R6 quotient.  The
first repaired path retains sixteen topology-compiled shift plans and counts
that regression from M194 explicitly.
"""

from __future__ import annotations

import bisect
import collections
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Iterator, Literal

import growing_rotor_pair_signature_streamed_quotient as predecessor


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


def rotate(item: Histogram, amount: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(item):
        result[(mode + amount) % GRID] = count
    return tuple(result)


def canonical(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflect(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def bracelet_orbit(item: Histogram) -> frozenset[Histogram]:
    return frozenset((canonical(item), canonical(reflect(item))))


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


def particle_modes(item: Histogram) -> tuple[int, ...]:
    return tuple(mode for mode, count in enumerate(item) for _ in range(count))


def triangle_shape(left: int, middle: int, right: int) -> Triangle:
    return tuple(
        sorted(
            min((first - second) % GRID, (second - first) % GRID)
            for first, second in (
                (left, middle),
                (left, right),
                (middle, right),
            )
        )
    )  # type: ignore[return-value]


def all_triangle_counts(item: Histogram) -> dict[Triangle, int]:
    result: collections.Counter[Triangle] = collections.Counter()
    modes = particle_modes(item)
    for left, middle, right in itertools.combinations(modes, 3):
        result[triangle_shape(left, middle, right)] += 1
    if sum(result.values()) != math.comb(len(modes), 3):
        raise RuntimeError("third-order triangle total failed")
    return dict(result)


def selected_triangle_counts(item: Histogram) -> tuple[int, int]:
    counts = [0, 0]
    modes = particle_modes(item)
    for left, middle, right in itertools.combinations(modes, 3):
        shape = triangle_shape(left, middle, right)
        for index, selected in enumerate(SELECTED_TRIANGLES):
            if shape == selected:
                counts[index] += 1
    return counts[0], counts[1]


def refined_signature(item: Histogram, rotors: int) -> Signature:
    return predecessor.pair_signature(item, rotors) + selected_triangle_counts(item)


def moved_signature(
    item: Histogram,
    signature: Signature,
    first: int,
    second: int,
    shift: int,
) -> Signature:
    pair = predecessor.moved_signature(
        item, signature[: predecessor.PAIR_CHANNELS], first, second, shift
    )
    moved = list(item)
    moved[first] -= 1
    moved[second] -= 1
    moved[(first - shift) % GRID] += 1
    moved[(second + shift) % GRID] += 1
    return pair + selected_triangle_counts(tuple(moved))


def sparse_profile(
    item: Histogram, signature: Signature, signatures: tuple[Signature, ...]
) -> dict[tuple[int, int], int]:
    result: collections.Counter[tuple[int, int]] = collections.Counter()
    for first in range(GRID):
        if item[first] == 0:
            continue
        for second in range(GRID):
            multiplicity = item[first] * (
                item[second] - (1 if first == second else 0)
            )
            if multiplicity == 0:
                continue
            for shift in range(1, GRID):
                destination = moved_signature(
                    item, signature, first, second, shift
                )
                index = bisect.bisect_left(signatures, destination)
                if index == len(signatures) or signatures[index] != destination:
                    raise RuntimeError("third-order transition escaped signature chart")
                result[(shift, index)] += multiplicity
    return dict(result)


@dataclass(frozen=True)
class RefinedTopology:
    rotors: int
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]
    occupation_histograms: int
    necklace_cells: int
    bracelet_cells: int
    shift_rows: tuple[
        tuple[tuple[tuple[int, int], ...], ...], ...
    ]
    streamed_mode_pair_shift_terms_per_scattering: int
    streamed_weighted_particle_shift_terms_per_scattering: int
    prior_materialized_plan_nonzeros: int


def compile_topology(rotors: int, prime: int, root: int) -> RefinedTopology:
    representatives: dict[Signature, Histogram] = {}
    boundary_by_signature: dict[Signature, int] = {}
    occupations = 0
    necklaces = 0
    bracelets: set[frozenset[Histogram]] = set()
    for item in iter_histograms(rotors):
        occupations += 1
        if canonical(item) != item:
            continue
        signature = refined_signature(item, rotors)
        representatives.setdefault(signature, item)
        boundary_by_signature[signature] = (
            boundary_by_signature.get(signature, 0)
            + pow(root, (11 * necklaces + 5 * signature[0] + 1) % GRID, prime)
        ) % prime
        bracelets.add(bracelet_orbit(item))
        necklaces += 1
    expected = math.comb(rotors + GRID - 1, rotors)
    if occupations != expected or expected % GRID or necklaces != expected // GRID:
        raise RuntimeError("third-order topology count failed")

    signatures = tuple(sorted(representatives))
    representative_tuple = tuple(representatives[value] for value in signatures)
    boundary_weights = tuple(boundary_by_signature[value] for value in signatures)
    distinct_terms = 0
    weighted_terms = 0
    shift_row_builders: list[list[tuple[tuple[int, int], ...]]] = [
        [] for _ in range(GRID - 1)
    ]
    for item, signature in zip(representative_tuple, signatures, strict=True):
        row = sparse_profile(item, signature, signatures)
        for shift in range(1, GRID):
            shift_row_builders[shift - 1].append(
                tuple(
                    sorted(
                        (source, coefficient)
                        for (row_shift, source), coefficient in row.items()
                        if row_shift == shift
                    )
                )
            )
        for first in range(GRID):
            if item[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = item[first] * (
                    item[second] - (1 if first == second else 0)
                )
                if multiplicity:
                    distinct_terms += GRID - 1
                    weighted_terms += multiplicity * (GRID - 1)
    shift_rows = tuple(tuple(rows) for rows in shift_row_builders)
    plan_nonzeros = sum(
        len(row) for basis in shift_rows for row in basis
    )
    return RefinedTopology(
        rotors=rotors,
        signatures=signatures,
        representatives=representative_tuple,
        boundary_weights=boundary_weights,
        occupation_histograms=occupations,
        necklace_cells=necklaces,
        bracelet_cells=len(bracelets),
        shift_rows=shift_rows,
        streamed_mode_pair_shift_terms_per_scattering=distinct_terms,
        streamed_weighted_particle_shift_terms_per_scattering=weighted_terms,
        prior_materialized_plan_nonzeros=plan_nonzeros,
    )


def verify_fibers(topology: RefinedTopology) -> tuple[int, int, int]:
    comparisons = 0
    visits = 0
    peak_entries = 0
    for item in iter_histograms(topology.rotors):
        visits += 1
        if canonical(item) != item:
            continue
        signature = refined_signature(item, topology.rotors)
        target = bisect.bisect_left(topology.signatures, signature)
        if target == len(topology.signatures) or topology.signatures[target] != signature:
            raise RuntimeError("unknown refined signature")
        candidate = sparse_profile(item, signature, topology.signatures)
        reference = sparse_profile(
            topology.representatives[target], signature, topology.signatures
        )
        peak_entries = max(peak_entries, len(candidate) + len(reference))
        if candidate != reference:
            raise RuntimeError("third-order signature fiber is not equitable")
        if item != topology.representatives[target]:
            comparisons += GRID - 1
    return comparisons, visits, peak_entries


def verify_minimality() -> dict[str, object]:
    rotors = 6
    necklaces = tuple(
        item for item in iter_histograms(rotors) if canonical(item) == item
    )
    triangle_data = {item: all_triangle_counts(item) for item in necklaces}
    coordinates = tuple(
        sorted({shape for values in triangle_data.values() for shape in values})
    )
    pair_fibers: dict[Signature, list[Histogram]] = collections.defaultdict(list)
    for item in necklaces:
        pair_fibers[predecessor.pair_signature(item, rotors)].append(item)

    collisions: list[tuple[Histogram, Histogram]] = []
    for items in pair_fibers.values():
        orbits: list[frozenset[Histogram]] = []
        for item in items:
            orbit = bracelet_orbit(item)
            if orbit not in orbits:
                orbits.append(orbit)
        for left, right in itertools.combinations(orbits, 2):
            collisions.append((min(left), min(right)))
    if len(collisions) != 24:
        raise RuntimeError("homometric collision count changed")

    successful_singles = 0
    for coordinate in coordinates:
        if all(
            triangle_data[left].get(coordinate, 0)
            != triangle_data[right].get(coordinate, 0)
            for left, right in collisions
        ):
            successful_singles += 1

    successful_pairs = 0
    first_pair: tuple[Triangle, Triangle] | None = None
    for pair in itertools.combinations(coordinates, 2):
        if all(
            any(
                triangle_data[left].get(coordinate, 0)
                != triangle_data[right].get(coordinate, 0)
                for coordinate in pair
            )
            for left, right in collisions
        ):
            successful_pairs += 1
            if first_pair is None:
                first_pair = pair
    if successful_singles or first_pair != SELECTED_TRIANGLES:
        raise RuntimeError("minimal triangle-coordinate certificate changed")
    return {
        "observed_triangle_coordinate_candidates": len(coordinates),
        "homometric_bracelet_pairs_requiring_separation": len(collisions),
        "successful_single_coordinates": successful_singles,
        "successful_coordinate_pairs": successful_pairs,
        "lexicographically_first_successful_pair": [
            list(value) for value in SELECTED_TRIANGLES
        ],
        "minimal_selected_coordinate_count_within_candidate_family": 2,
        "verification_only_necklace_records": len(necklaces),
        "verification_only_triangle_coordinate_cells": sum(
            len(values) for values in triangle_data.values()
        ),
    }


def apply_diagonal(
    state: list[int], topology: RefinedTopology, prime: int, root: int,
    step: int, tag: int
) -> list[int]:
    return [
        value
        * pow(
            root,
            predecessor.phase_exponent(signature[: predecessor.PAIR_CHANNELS], step, tag),
            prime,
        )
        % prime
        for value, signature in zip(state, topology.signatures, strict=True)
    ]


def apply_scattering(
    state: list[int], topology: RefinedTopology, prime: int, step: int, tag: int
) -> list[int]:
    output = [0] * len(topology.signatures)
    for target in range(len(topology.signatures)):
        accumulator = 0
        for shift, basis in enumerate(topology.shift_rows, 1):
            weight = predecessor.public_scattering_integer(shift, step, tag)
            for source, coefficient in basis[target]:
                accumulator += coefficient * weight * state[source]
        output[target] = accumulator % prime
    return output


def execute_word(
    source: list[int], topology: RefinedTopology, prime: int, root: int,
    operations: tuple[tuple[int, int], ...], reordered: bool = False
) -> list[int]:
    current = source.copy()
    for step, tag in operations:
        if reordered:
            current = apply_diagonal(
                apply_scattering(current, topology, prime, step, tag),
                topology, prime, root, step, tag,
            )
        else:
            current = apply_scattering(
                apply_diagonal(current, topology, prime, root, step, tag),
                topology, prime, step, tag,
            )
    return current


def source_state(signatures: tuple[Signature, ...], prime: int, family: int) -> list[int]:
    return [
        (
            1
            + (family + 3) * (index + 1)
            + sum(
                (coordinate + 2 + family) * (count + 1) ** 2
                for coordinate, count in enumerate(signature)
            )
        )
        % prime
        for index, signature in enumerate(signatures)
    ]


def boundary(state: list[int], topology: RefinedTopology, prime: int) -> int:
    return sum(
        value * weight
        for value, weight in zip(state, topology.boundary_weights, strict=True)
    ) % prime


def commitment(state: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, state)).encode()).hexdigest()


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    topology: RefinedTopology,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
    inverse_operations: tuple[tuple[int, int], ...],
    control: Control,
) -> dict[str, object]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed third-order carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward = execute_word(carrier.source, topology, prime, root, operations)
    carrier.target[:] = [
        (left + right) % prime
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    projected = boundary(carrier.target, topology, prime)
    if control != "missing":
        inverse = execute_word(
            carrier.source,
            topology,
            prime,
            root,
            inverse_operations,
            reordered=control == "reordered",
        )
        carrier.target[:] = [
            (left - right) % prime
            for left, right in zip(carrier.target, inverse, strict=True)
        ]
    error = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    return {
        "boundary": projected,
        "forward_commitment": commitment(forward),
        "restoration_error_field_cells": error,
        "same_backing": id(carrier.source) == source_backing
        and id(carrier.target) == target_backing,
        "generation": carrier.generation,
    }


def run_transaction_case(topology: RefinedTopology, prime: int) -> dict[str, object]:
    generator = predecessor.primitive_generator(prime)
    root = pow(generator, (prime - 1) // GRID, prime)
    if root == 1 or pow(root, GRID, prime) != 1:
        raise RuntimeError("third-order seventeenth root failed")
    primary_word = predecessor.public_program(PRIMARY_DEPTH, 0)
    wrong_word = predecessor.public_program(PRIMARY_DEPTH, 1)
    reuse_word = predecessor.public_program(REUSE_DEPTH, 4)
    source = source_state(topology.signatures, prime, 0)
    carrier = Carrier(source.copy(), [0] * len(source))
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    primary = transaction(
        carrier, source, topology, prime, root, primary_word, primary_word, "correct"
    )
    reuse = transaction(
        carrier, source, topology, prime, root, reuse_word, reuse_word, "correct"
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse = transaction(
        fresh, source, topology, prime, root, reuse_word, reuse_word, "correct"
    )
    if (
        primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or reuse["boundary"] != fresh_reuse["boundary"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or id(carrier.source) != source_backing
        or id(carrier.target) != target_backing
    ):
        raise RuntimeError("third-order carrier restoration or reuse failed")

    controls: dict[str, int | bool] = {}
    for control, inverse_word in (
        ("missing", primary_word),
        ("wrong", wrong_word),
        ("reordered", primary_word),
    ):
        controlled = Carrier(source.copy(), [0] * len(source))
        result = transaction(
            controlled,
            source,
            topology,
            prime,
            root,
            primary_word,
            inverse_word,
            control,
        )
        controls[f"{control}_inverse_error_field_cells"] = int(
            result["restoration_error_field_cells"]
        )
    if min(int(value) for value in controls.values()) == 0:
        raise RuntimeError("third-order inverse controls did not discriminate")

    null_rejected = False
    try:
        transaction(
            Carrier([], []), [], topology, prime, root,
            primary_word, primary_word, "correct"
        )
    except ValueError:
        null_rejected = True
    if not null_rejected:
        raise RuntimeError("null third-order carrier was accepted")
    controls["null_carrier_rejected"] = null_rejected

    cells = len(topology.signatures)
    terms = topology.streamed_mode_pair_shift_terms_per_scattering
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
        "retained_shift_plan_nonzero_entries": topology.prior_materialized_plan_nonzeros,
        "public_signature_descriptor_integer_cells": 11 * cells,
        "public_representative_descriptor_integer_cells": GRID * cells,
        "public_boundary_weight_field_cells": cells,
        "streamed_mode_pair_shift_terms_per_scattering": terms,
        "streamed_weighted_particle_shift_terms_per_scattering": topology.streamed_weighted_particle_shift_terms_per_scattering,
        "public_plan_compiler_third_order_triangle_shape_tests": terms * math.comb(6, 3),
        "primary_forward_plan_nonzero_applications": PRIMARY_DEPTH * topology.prior_materialized_plan_nonzeros,
        "primary_forward_inverse_plan_nonzero_applications": 2 * PRIMARY_DEPTH * topology.prior_materialized_plan_nonzeros,
        "reuse_forward_inverse_plan_nonzero_applications": 2 * REUSE_DEPTH * topology.prior_materialized_plan_nonzeros,
    }


def main() -> None:
    generator = predecessor.primitive_generator(PRIMES[0])
    root = pow(generator, (PRIMES[0] - 1) // GRID, PRIMES[0])
    topologies = [compile_topology(rotors, PRIMES[0], root) for rotors in ROTOR_COUNTS]
    expected_necklaces = [9, 57, 285, 1197, 4389]
    expected_cells = [9, 33, 165, 621, 2277]
    if [value.necklace_cells for value in topologies] != expected_necklaces:
        raise RuntimeError("third-order necklace law changed")
    if [len(value.signatures) for value in topologies] != expected_cells:
        raise RuntimeError("third-order signature law changed")
    if any(len(value.signatures) != value.bracelet_cells for value in topologies):
        raise RuntimeError("refined signatures do not equal bracelet orbits")

    topology_cases = []
    for topology in topologies:
        comparisons, visits, peak = verify_fibers(topology)
        topology_cases.append(
            {
                "rotors": topology.rotors,
                "occupation_histograms": topology.occupation_histograms,
                "necklace_cells": topology.necklace_cells,
                "bracelet_cells": topology.bracelet_cells,
                "refined_signature_cells": len(topology.signatures),
                "refined_signatures_equal_dihedral_orbits": True,
                "all_sixteen_shift_bases_equitable": True,
                "equitability_comparisons": comparisons,
                "verification_only_occupation_histograms_visited": visits,
                "verification_only_peak_sparse_row_entries": peak,
                "prior_materialized_plan_nonzeros": topology.prior_materialized_plan_nonzeros,
            }
        )
    minimality = verify_minimality()
    rotor6 = topologies[-1]
    transaction_cases = [run_transaction_case(rotor6, prime) for prime in PRIMES]

    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_MINIMAL_WITHIN33_TRIANGLE_COORDINATES_TWO_COORDINATE_DIHEDRAL_THIRD_ORDER_PHASE_SIGNATURE_REFINEMENT_SEPARATES_ALL24_ROTOR6_HOMOMETRIC_PAIR_FIBERS_AND_CLOSES_ALL16_SIGNED_SCATTERING_BASES_ON2277_CELLS_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS28S_PUBLIC_INTEGER_DESCRIPTORS_16_SHIFT_PLANS_AND_AN_IDENTICAL_CLASSICAL_RECURRENCE",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_REFINED_SIGNATURE_INPUTS_ROTORS2_TO6_TOPOLOGY_F103_F239_ROTOR6_PRIMARY_DEPTH3_REUSE_DEPTH2_DIRECT_PROCESS_SOFTWARE_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS_REFINED_CLOSURE",
                "selected_triangle_coordinates": [list(value) for value in SELECTED_TRIANGLES],
                "minimality": minimality,
                "topology_cases": topology_cases,
                "transaction_cases": transaction_cases,
                "resource_law": {
                    "accepted_carrier_resident_field_cells": "TWO_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_word_scratch_field_cells": "TWO_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_public_boundary_weight_field_cells": "REFINED_SIGNATURE_COUNT",
                    "accepted_public_signature_descriptor_integer_cells": "ELEVEN_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_public_representative_descriptor_integer_cells": "SEVENTEEN_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_retained_shift_basis_plans": 16,
                    "accepted_retained_shift_plan_nonzero_entries": "REPORTED_PER_CASE",
                    "accepted_retained_inverse_history_bytes": 0,
                    "accepted_relation_table_or_assignment_expansion_cells": 0,
                    "public_plan_compiler_mode_pair_shift_and_triangle_work": "REPORTED_PER_CASE",
                    "verification_only_full_fiber_rows_triangle_search_and_necklace_records": "REPORTED_AND_EXCLUDED_FROM_ACCEPTED_PATH",
                    "python_containers_allocator_bigints_expression_temporaries_and_whole_process_peak": "EXCLUDED_NOT_ZERO",
                },
                "matched_classical_recurrence": "IDENTICAL_PLAN_COMPILED_REFINED_SIGNATURE_QUOTIENT_RECURRENCE",
                "full_bracelet_fallback_required": False,
                "refinement_smaller_than_full_bracelet_identity": False,
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
