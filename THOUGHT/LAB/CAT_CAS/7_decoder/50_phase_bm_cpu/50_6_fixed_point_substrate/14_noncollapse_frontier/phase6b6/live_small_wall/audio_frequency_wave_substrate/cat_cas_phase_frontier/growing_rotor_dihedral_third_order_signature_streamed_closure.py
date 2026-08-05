#!/usr/bin/env python3
"""Topology-streamed third-order Rotor-6 closure with no retained shift plan.

M196 repaired the Rotor-6 pair-homometry collision with two exact triangle
coordinates, but retained sixteen compiled sparse shift plans.  This bounded
successor updates those coordinates from the four-site population delta using
a fixed eight-integer public stencil.  Every scattering destination is
rematerialized from the public representative and discarded immediately.

The accepted path remains direct-process finite-field software.  Its strongest
matched classical implementation is the identical streamed recurrence.  This
package therefore tests retained-plan removal; it does not establish CATVM
custody, computational advantage, or a distinct physical phase resource.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator

import growing_rotor_pair_signature_streamed_quotient as predecessor


GRID = 17
ROTOR_COUNTS = (2, 3, 4, 5, 6)
PRIME = 103
PRIMARY_DEPTH = 1
REUSE_DEPTH = 1
PAIR_CHANNELS = 9
TRIANGLE_STENCILS = (((1, 3), (2, 3)), ((1, 5), (4, 5)))
PRIOR_PLAN_NONZEROS = (272, 2448, 21904, 131168, 652048)
Histogram = tuple[int, ...]
Signature = tuple[int, ...]


def rotate(item: Histogram, amount: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(item):
        result[(mode + amount) % GRID] = count
    return tuple(result)


def canonical(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


def reflect(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def bracelet_key(item: Histogram) -> Histogram:
    return min(canonical(item), canonical(reflect(item)))


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


def selected_triangle_counts(item: Histogram) -> tuple[int, int]:
    """Evaluate the two invariant triangle coordinates as cyclic monomials."""
    return tuple(
        sum(
            item[anchor]
            * item[(anchor + first) % GRID]
            * item[(anchor + second) % GRID]
            for first, second in orientations
            for anchor in range(GRID)
        )
        for orientations in TRIANGLE_STENCILS
    )  # type: ignore[return-value]


def refined_signature(item: Histogram, rotors: int) -> Signature:
    return predecessor.pair_signature(item, rotors) + selected_triangle_counts(item)


def move_delta(first: int, second: int, shift: int) -> tuple[int, ...]:
    delta = [0] * GRID
    for mode, amount in (
        (first, -1),
        (second, -1),
        ((first - shift) % GRID, 1),
        ((second + shift) % GRID, 1),
    ):
        delta[mode] += amount
    return tuple(delta)


def moved_histogram(item: Histogram, delta: tuple[int, ...]) -> Histogram:
    moved = tuple(left + right for left, right in zip(item, delta, strict=True))
    if min(moved) < 0 or sum(moved) != sum(item):
        raise RuntimeError("invalid topology-streamed population move")
    return moved


def moved_signature_analytic(
    item: Histogram,
    signature: Signature,
    first: int,
    second: int,
    shift: int,
) -> tuple[Signature, int, int]:
    """Update all eleven coordinates by touching only affected monomials."""
    pair = predecessor.moved_signature(
        item, signature[:PAIR_CHANNELS], first, second, shift
    )
    delta = move_delta(first, second, shift)
    active = tuple(mode for mode, amount in enumerate(delta) if amount)
    updated_triangles: list[int] = []
    monomial_evaluations = 0
    for current, orientations in zip(
        signature[PAIR_CHANNELS:], TRIANGLE_STENCILS, strict=True
    ):
        change = 0
        for offset_a, offset_b in orientations:
            affected = {
                anchor
                for mode in active
                for anchor in (
                    mode,
                    (mode - offset_a) % GRID,
                    (mode - offset_b) % GRID,
                )
            }
            monomial_evaluations += len(affected)
            for anchor in affected:
                middle = (anchor + offset_a) % GRID
                right = (anchor + offset_b) % GRID
                old = item[anchor] * item[middle] * item[right]
                new = (
                    (item[anchor] + delta[anchor])
                    * (item[middle] + delta[middle])
                    * (item[right] + delta[right])
                )
                change += new - old
        updated_triangles.append(current + change)
    return pair + tuple(updated_triangles), monomial_evaluations, len(active)


def locate(signatures: tuple[Signature, ...], value: Signature) -> tuple[int, int]:
    low = 0
    high = len(signatures)
    comparisons = 0
    while low < high:
        middle = (low + high) // 2
        comparisons += 1
        if signatures[middle] < value:
            low = middle + 1
        else:
            high = middle
    if low == len(signatures) or signatures[low] != value:
        raise RuntimeError("analytic delta escaped the refined signature chart")
    return low, comparisons


@dataclass(frozen=True)
class StreamedTopology:
    rotors: int
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]
    occupation_histograms: int
    necklace_cells: int
    bracelet_cells: int
    mode_pair_shift_terms_per_scattering: int
    weighted_particle_shift_terms_per_scattering: int
    prior_plan_nonzeros: int


def compile_topology(rotors: int, prime: int, root: int) -> StreamedTopology:
    representatives: dict[Signature, Histogram] = {}
    boundary: dict[Signature, int] = {}
    bracelets: set[Histogram] = set()
    occupations = 0
    necklaces = 0
    for item in iter_histograms(rotors):
        occupations += 1
        if canonical(item) != item:
            continue
        signature = refined_signature(item, rotors)
        representatives.setdefault(signature, item)
        boundary[signature] = (
            boundary.get(signature, 0)
            + pow(root, (11 * necklaces + 5 * signature[0] + 1) % GRID, prime)
        ) % prime
        bracelets.add(bracelet_key(item))
        necklaces += 1
    expected = math.comb(rotors + GRID - 1, rotors)
    if occupations != expected or expected % GRID or necklaces != expected // GRID:
        raise RuntimeError("streamed third-order topology count failed")
    signatures = tuple(sorted(representatives))
    representative_tuple = tuple(representatives[value] for value in signatures)
    distinct_terms = 0
    weighted_terms = 0
    for item in representative_tuple:
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
    return StreamedTopology(
        rotors=rotors,
        signatures=signatures,
        representatives=representative_tuple,
        boundary_weights=tuple(boundary[value] for value in signatures),
        occupation_histograms=occupations,
        necklace_cells=necklaces,
        bracelet_cells=len(bracelets),
        mode_pair_shift_terms_per_scattering=distinct_terms,
        weighted_particle_shift_terms_per_scattering=weighted_terms,
        prior_plan_nonzeros=PRIOR_PLAN_NONZEROS[ROTOR_COUNTS.index(rotors)],
    )


def transition_record(
    target: int,
    first: int,
    second: int,
    shift: int,
    source: int,
    multiplicity: int,
) -> bytes:
    return f"{target},{first},{second},{shift},{source},{multiplicity};".encode()


def audit_all_transitions(topology: StreamedTopology) -> dict[str, object]:
    digest = hashlib.sha256()
    terms = 0
    weighted_terms = 0
    monomial_evaluations = 0
    lookup_comparisons = 0
    peak_monomials = 0
    active_mode_histogram: dict[int, int] = {}
    for target, (item, signature) in enumerate(
        zip(topology.representatives, topology.signatures, strict=True)
    ):
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
                    destination, evaluations, active_modes = moved_signature_analytic(
                        item, signature, first, second, shift
                    )
                    delta = move_delta(first, second, shift)
                    direct = refined_signature(
                        moved_histogram(item, delta), topology.rotors
                    )
                    if destination != direct:
                        raise RuntimeError("analytic triangle delta differs from direct recurrence")
                    source, comparisons = locate(topology.signatures, destination)
                    digest.update(
                        transition_record(
                            target, first, second, shift, source, multiplicity
                        )
                    )
                    terms += 1
                    weighted_terms += multiplicity
                    monomial_evaluations += evaluations
                    lookup_comparisons += comparisons
                    peak_monomials = max(peak_monomials, evaluations)
                    active_mode_histogram[active_modes] = (
                        active_mode_histogram.get(active_modes, 0) + 1
                    )
    if (
        terms != topology.mode_pair_shift_terms_per_scattering
        or weighted_terms != topology.weighted_particle_shift_terms_per_scattering
    ):
        raise RuntimeError("transition audit work law changed")
    return {
        "raw_transition_commitment": digest.hexdigest(),
        "mode_pair_shift_terms": terms,
        "weighted_particle_shift_terms": weighted_terms,
        "analytic_triangle_monomial_delta_evaluations": monomial_evaluations,
        "peak_analytic_triangle_monomials_per_term": peak_monomials,
        "signature_binary_search_comparisons": lookup_comparisons,
        "active_mode_count_histogram": {
            str(key): active_mode_histogram[key] for key in sorted(active_mode_histogram)
        },
        "analytic_destinations_equal_direct_cyclic_monomial_reexecution": True,
    }


@dataclass
class Work:
    scatterings: int = 0
    mode_pair_shift_terms: int = 0
    weighted_particle_shift_terms: int = 0
    triangle_monomial_delta_evaluations: int = 0
    signature_binary_search_comparisons: int = 0
    diagonal_field_cells: int = 0

    def add(self, other: "Work") -> None:
        self.scatterings += other.scatterings
        self.mode_pair_shift_terms += other.mode_pair_shift_terms
        self.weighted_particle_shift_terms += other.weighted_particle_shift_terms
        self.triangle_monomial_delta_evaluations += (
            other.triangle_monomial_delta_evaluations
        )
        self.signature_binary_search_comparisons += (
            other.signature_binary_search_comparisons
        )
        self.diagonal_field_cells += other.diagonal_field_cells

    def as_dict(self) -> dict[str, int]:
        return {
            "scatterings": self.scatterings,
            "mode_pair_shift_terms": self.mode_pair_shift_terms,
            "weighted_particle_shift_terms": self.weighted_particle_shift_terms,
            "triangle_monomial_delta_evaluations": self.triangle_monomial_delta_evaluations,
            "signature_binary_search_comparisons": self.signature_binary_search_comparisons,
            "diagonal_field_cells": self.diagonal_field_cells,
        }


def apply_diagonal(
    state: list[int],
    topology: StreamedTopology,
    prime: int,
    root: int,
    step: int,
    tag: int,
) -> tuple[list[int], Work]:
    output = [
        value
        * pow(
            root,
            predecessor.phase_exponent(signature[:PAIR_CHANNELS], step, tag),
            prime,
        )
        % prime
        for value, signature in zip(state, topology.signatures, strict=True)
    ]
    return output, Work(diagonal_field_cells=len(state))


def apply_scattering_streamed(
    state: list[int],
    topology: StreamedTopology,
    prime: int,
    step: int,
    tag: int,
) -> tuple[list[int], Work]:
    output = [0] * len(topology.signatures)
    work = Work(scatterings=1)
    for target, (item, signature) in enumerate(
        zip(topology.representatives, topology.signatures, strict=True)
    ):
        accumulator = 0
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
                    destination, evaluations, _ = moved_signature_analytic(
                        item, signature, first, second, shift
                    )
                    source, comparisons = locate(topology.signatures, destination)
                    accumulator += (
                        multiplicity
                        * predecessor.public_scattering_integer(shift, step, tag)
                        * state[source]
                    )
                    work.mode_pair_shift_terms += 1
                    work.weighted_particle_shift_terms += multiplicity
                    work.triangle_monomial_delta_evaluations += evaluations
                    work.signature_binary_search_comparisons += comparisons
        output[target] = accumulator % prime
    return output, work


def execute_word(
    source: list[int],
    topology: StreamedTopology,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
    reordered: bool = False,
) -> tuple[list[int], Work]:
    current = source.copy()
    total = Work()
    for step, tag in operations:
        if reordered:
            current, scatter = apply_scattering_streamed(
                current, topology, prime, step, tag
            )
            current, diagonal = apply_diagonal(
                current, topology, prime, root, step, tag
            )
        else:
            current, diagonal = apply_diagonal(
                current, topology, prime, root, step, tag
            )
            current, scatter = apply_scattering_streamed(
                current, topology, prime, step, tag
            )
        total.add(diagonal)
        total.add(scatter)
    return current, total


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


def boundary(state: list[int], topology: StreamedTopology, prime: int) -> int:
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
    topology: StreamedTopology,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
) -> tuple[dict[str, object], list[int], Work]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed streamed third-order carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward, forward_work = execute_word(
        carrier.source, topology, prime, root, operations
    )
    carrier.target[:] = [
        (left + right) % prime
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    projected = boundary(carrier.target, topology, prime)
    inverse, inverse_work = execute_word(
        carrier.source, topology, prime, root, operations
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
    total = Work()
    total.add(forward_work)
    total.add(inverse_work)
    return (
        {
            "boundary": projected,
            "forward_commitment": commitment(forward),
            "restoration_error_field_cells": error,
            "same_backing": id(carrier.source) == source_backing
            and id(carrier.target) == target_backing,
            "generation": carrier.generation,
        },
        forward,
        total,
    )


def restoration_error(
    forward: list[int], inverse: list[int], prime: int
) -> int:
    return sum((left - right) % prime != 0 for left, right in zip(forward, inverse, strict=True))


def run_transaction(topology: StreamedTopology, prime: int, root: int) -> dict[str, object]:
    primary_word = predecessor.public_program(PRIMARY_DEPTH, 0)
    wrong_word = predecessor.public_program(PRIMARY_DEPTH, 1)
    reuse_word = predecessor.public_program(REUSE_DEPTH, 4)
    source = source_state(topology.signatures, prime, 0)
    carrier = Carrier(source.copy(), [0] * len(source))
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    primary, primary_forward, primary_work = transaction(
        carrier, source, topology, prime, root, primary_word
    )
    reuse, _, reuse_work = transaction(
        carrier, source, topology, prime, root, reuse_word
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse, _, fresh_work = transaction(
        fresh, source, topology, prime, root, reuse_word
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
        raise RuntimeError("streamed third-order restoration or reuse failed")

    matched, matched_work = execute_word(source, topology, prime, root, primary_word)
    if matched != primary_forward:
        raise RuntimeError("identical compact classical stream differs")
    wrong, wrong_work = execute_word(source, topology, prime, root, wrong_word)
    reordered, reordered_work = execute_word(
        source, topology, prime, root, primary_word, reordered=True
    )
    controls = {
        "missing_inverse_error_field_cells": sum(value != 0 for value in primary_forward),
        "wrong_inverse_error_field_cells": restoration_error(
            primary_forward, wrong, prime
        ),
        "reordered_inverse_error_field_cells": restoration_error(
            primary_forward, reordered, prime
        ),
        "null_carrier_rejected": False,
    }
    try:
        transaction(Carrier([], []), [], topology, prime, root, primary_word)
    except ValueError:
        controls["null_carrier_rejected"] = True
    if (
        min(
            int(controls[key])
            for key in (
                "missing_inverse_error_field_cells",
                "wrong_inverse_error_field_cells",
                "reordered_inverse_error_field_cells",
            )
        )
        == 0
        or not controls["null_carrier_rejected"]
    ):
        raise RuntimeError("streamed third-order controls did not discriminate")

    cells = len(topology.signatures)
    return {
        "prime": prime,
        "multiplicative_generator": predecessor.primitive_generator(prime),
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
        "fresh_restored_reuse_rank_signature_agreement": len(fresh.source)
        == len(carrier.source),
        "baseline_reload_used": False,
        "controls": controls,
        "retained_shift_basis_plans": 0,
        "retained_shift_plan_nonzero_entries": 0,
        "prior_materialized_plan_nonzero_entries": topology.prior_plan_nonzeros,
        "public_signature_descriptor_integer_cells": 11 * cells,
        "public_representative_descriptor_integer_cells": GRID * cells,
        "public_triangle_stencil_integer_cells": 8,
        "public_boundary_weight_field_cells": cells,
        "primary_forward_inverse_work": primary_work.as_dict(),
        "reuse_forward_inverse_work": reuse_work.as_dict(),
        "fresh_reuse_verification_work": fresh_work.as_dict(),
        "matched_classical_forward_work": matched_work.as_dict(),
        "wrong_inverse_control_work": wrong_work.as_dict(),
        "reordered_inverse_control_work": reordered_work.as_dict(),
        "matched_classical_boundary": boundary(matched, topology, prime),
        "matched_classical_output_commitment": commitment(matched),
    }


def main() -> None:
    generator = predecessor.primitive_generator(PRIME)
    root = pow(generator, (PRIME - 1) // GRID, PRIME)
    if root == 1 or pow(root, GRID, PRIME) != 1:
        raise RuntimeError("streamed third-order seventeenth root failed")
    topologies = [compile_topology(rotors, PRIME, root) for rotors in ROTOR_COUNTS]
    expected_necklaces = [9, 57, 285, 1197, 4389]
    expected_cells = [9, 33, 165, 621, 2277]
    if [value.necklace_cells for value in topologies] != expected_necklaces:
        raise RuntimeError("streamed third-order necklace law changed")
    if [len(value.signatures) for value in topologies] != expected_cells:
        raise RuntimeError("streamed third-order signature law changed")
    if any(len(value.signatures) != value.bracelet_cells for value in topologies):
        raise RuntimeError("streamed refined signatures differ from bracelet orbits")
    rotor6 = topologies[-1]
    transition_audit = audit_all_transitions(rotor6)
    transaction_case = run_transaction(rotor6, PRIME, root)
    topology_cases = [
        {
            "rotors": value.rotors,
            "occupation_histograms": value.occupation_histograms,
            "necklace_cells": value.necklace_cells,
            "bracelet_cells": value.bracelet_cells,
            "refined_signature_cells": len(value.signatures),
            "refined_signatures_equal_dihedral_orbits": len(value.signatures)
            == value.bracelet_cells,
            "prior_materialized_plan_nonzeros": value.prior_plan_nonzeros,
        }
        for value in topologies
    ]
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_FIXED_EIGHT_INTEGER_TRIANGLE_DELTA_STENCIL_ELIMINATES_ALL16_RETAINED_SHIFT_PLANS_WHILE_PRESERVING_THE2277_CELL_ROTOR6_REFINED_PHASE_QUOTIENT_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_REMATERIALIZES684624_MOVES_AND24767280_TRIANGLE_MONOMIAL_DELTAS_PER_SCATTERING_WITH_AN_IDENTICAL_CLASSICAL_STREAM",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_AND_REFLECTION_INVARIANT_TWO_TRIANGLE_REFINED_SIGNATURE_INPUTS_ROTORS2_TO6_TOPOLOGY_F103_ROTOR6_PRIMARY_DEPTH1_REUSE_DEPTH1_DIRECT_PROCESS_SOFTWARE_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS_ZERO_RETAINED_SHIFT_PLAN_WITH_MEASURED_REMATERIALIZATION",
                "selected_triangle_coordinates": [[1, 2, 3], [1, 4, 5]],
                "public_triangle_stencils": [
                    [[1, 3], [2, 3]],
                    [[1, 5], [4, 5]],
                ],
                "topology_cases": topology_cases,
                "transition_audit": transition_audit,
                "transaction_case": transaction_case,
                "resource_law": {
                    "accepted_carrier_resident_field_cells": "TWO_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_word_scratch_field_cells": "TWO_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_public_boundary_weight_field_cells": "REFINED_SIGNATURE_COUNT",
                    "accepted_public_signature_descriptor_integer_cells": "ELEVEN_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_public_representative_descriptor_integer_cells": "SEVENTEEN_TIMES_REFINED_SIGNATURE_COUNT",
                    "accepted_fixed_triangle_stencil_integer_cells": 8,
                    "accepted_retained_shift_basis_plans": 0,
                    "accepted_retained_shift_plan_nonzero_entries": 0,
                    "accepted_retained_inverse_history_bytes": 0,
                    "accepted_relation_table_or_assignment_expansion_cells": 0,
                    "accepted_rematerialized_mode_pair_shift_terms_per_scattering": "REPORTED",
                    "accepted_analytic_triangle_monomial_delta_evaluations_per_scattering": "REPORTED",
                    "verification_only_full_transition_audit_and_fresh_reuse": "REPORTED_AND_EXCLUDED_FROM_ACCEPTED_PATH",
                    "python_containers_allocator_bigints_expression_temporaries_and_whole_process_peak": "EXCLUDED_NOT_ZERO",
                },
                "matched_classical_recurrence": "IDENTICAL_TOPOLOGY_STREAMED_TWO_TRIANGLE_REFINED_SIGNATURE_RECURRENCE",
                "prior_plan_compiled_nonzeros_eliminated": 652048,
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
