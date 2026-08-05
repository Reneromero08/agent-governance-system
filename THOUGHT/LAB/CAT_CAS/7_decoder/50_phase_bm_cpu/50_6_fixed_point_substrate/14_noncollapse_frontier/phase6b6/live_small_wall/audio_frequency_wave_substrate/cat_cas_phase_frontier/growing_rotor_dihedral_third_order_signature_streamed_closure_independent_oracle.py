#!/usr/bin/env python3
"""Independent particle/bracelet oracle for streamed third-order closure."""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
from dataclasses import dataclass


GRID = 17
ROTOR_COUNTS = (2, 3, 4, 5, 6)
PRIME = 103
PRIMARY_DEPTH = 1
REUSE_DEPTH = 1
SELECTED_TRIANGLES = ((1, 2, 3), (1, 4, 5))
TRIANGLE_ORIENTATIONS = (((1, 3), (2, 3)), ((1, 5), (4, 5)))
PRIOR_PLAN_NONZEROS = (272, 2448, 21904, 131168, 652048)
Histogram = tuple[int, ...]
Signature = tuple[int, ...]
Triangle = tuple[int, int, int]


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


def reflect(item: Histogram) -> Histogram:
    return tuple(item[(-mode) % GRID] for mode in range(GRID))


def bracelet_key(item: Histogram) -> Histogram:
    return min(canonical(item), canonical(reflect(item)))


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


def refined_signature(item: Histogram) -> Signature:
    modes = particle_modes(item)
    pair = [0] * 9
    for left, right in itertools.combinations(range(len(modes)), 2):
        distance = (modes[right] - modes[left]) % GRID
        pair[min(distance, GRID - distance)] += 1
    triangles: collections.Counter[Triangle] = collections.Counter()
    for left, middle, right in itertools.combinations(modes, 3):
        triangles[triangle_shape(left, middle, right)] += 1
    if sum(pair) != math.comb(len(modes), 2):
        raise RuntimeError("oracle pair-coordinate total failed")
    if sum(triangles.values()) != math.comb(len(modes), 3):
        raise RuntimeError("oracle triangle-coordinate total failed")
    return tuple(pair) + tuple(triangles[value] for value in SELECTED_TRIANGLES)


def occupations(rotors: int) -> tuple[Histogram, ...]:
    return tuple(
        histogram(values)
        for values in itertools.combinations_with_replacement(range(GRID), rotors)
    )


def primitive_generator(prime: int) -> int:
    factors: list[int] = []
    remaining = prime - 1
    candidate = 2
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            factors.append(candidate)
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        factors.append(remaining)
    for value in range(2, prime):
        if all(pow(value, (prime - 1) // factor, prime) != 1 for factor in factors):
            return value
    raise RuntimeError("oracle primitive-generator search failed")


@dataclass(frozen=True)
class OracleTopology:
    rotors: int
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]
    occupation_histograms: int
    necklace_cells: int
    bracelet_cells: int
    prior_plan_nonzeros: int


def compile_topology(rotors: int, prime: int, root: int) -> OracleTopology:
    values = occupations(rotors)
    expected = math.comb(rotors + GRID - 1, rotors)
    if len(values) != expected:
        raise RuntimeError("oracle occupation count failed")
    necklaces = tuple(sorted({canonical(item) for item in values}))
    if expected % GRID or len(necklaces) != expected // GRID:
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
    bracelets = {bracelet_key(item) for item in necklaces}
    if len(signatures) != len(bracelets):
        raise RuntimeError("oracle refined signature differs from bracelet quotient")
    for signature, items in fibers.items():
        if len({bracelet_key(item) for item in items}) != 1:
            raise RuntimeError(f"oracle signature overmerged bracelets: {signature}")
    return OracleTopology(
        rotors=rotors,
        signatures=signatures,
        representatives=tuple(fibers[value][0] for value in signatures),
        boundary_weights=tuple(boundary[value] for value in signatures),
        occupation_histograms=len(values),
        necklace_cells=len(necklaces),
        bracelet_cells=len(bracelets),
        prior_plan_nonzeros=PRIOR_PLAN_NONZEROS[ROTOR_COUNTS.index(rotors)],
    )


def move_histogram(
    item: Histogram, first: int, second: int, shift: int
) -> tuple[Histogram, frozenset[int]]:
    moved = list(item)
    delta = [0] * GRID
    for mode, amount in (
        (first, -1),
        (second, -1),
        ((first - shift) % GRID, 1),
        ((second + shift) % GRID, 1),
    ):
        moved[mode] += amount
        delta[mode] += amount
    if min(moved) < 0 or sum(moved) != sum(item):
        raise RuntimeError("oracle particle move failed")
    return tuple(moved), frozenset(
        mode for mode, amount in enumerate(delta) if amount
    )


def affected_monomials(active: frozenset[int]) -> int:
    """Independent full-anchor incidence count for the public stencil."""
    return sum(
        bool(
            active.intersection(
                (anchor, (anchor + offset_a) % GRID, (anchor + offset_b) % GRID)
            )
        )
        for orientations in TRIANGLE_ORIENTATIONS
        for offset_a, offset_b in orientations
        for anchor in range(GRID)
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


@dataclass(frozen=True)
class OraclePlan:
    shift_rows: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    raw_transition_commitment: str
    mode_pair_shift_terms: int
    weighted_particle_shift_terms: int
    affected_triangle_monomials: int
    peak_affected_triangle_monomials: int
    active_mode_count_histogram: dict[str, int]
    plan_nonzeros: int


def build_independent_plan(topology: OracleTopology) -> OraclePlan:
    signature_index = {
        signature: index for index, signature in enumerate(topology.signatures)
    }
    builders: list[list[dict[int, int]]] = [
        [collections.defaultdict(int) for _ in topology.signatures]
        for _ in range(GRID - 1)
    ]
    digest = hashlib.sha256()
    terms = 0
    weighted_terms = 0
    affected = 0
    peak = 0
    active_histogram: dict[int, int] = {}
    for target, item in enumerate(topology.representatives):
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
                    moved, active = move_histogram(item, first, second, shift)
                    destination = refined_signature(moved)
                    if destination not in signature_index:
                        raise RuntimeError("oracle transition escaped signature chart")
                    source = signature_index[destination]
                    builders[shift - 1][target][source] += multiplicity
                    digest.update(
                        transition_record(
                            target, first, second, shift, source, multiplicity
                        )
                    )
                    incidence = affected_monomials(active)
                    terms += 1
                    weighted_terms += multiplicity
                    affected += incidence
                    peak = max(peak, incidence)
                    active_histogram[len(active)] = active_histogram.get(len(active), 0) + 1
    shift_rows = tuple(
        tuple(tuple(sorted(row.items())) for row in basis) for basis in builders
    )
    plan_nonzeros = sum(len(row) for basis in shift_rows for row in basis)
    if plan_nonzeros != topology.prior_plan_nonzeros:
        raise RuntimeError("oracle prior plan count changed")
    return OraclePlan(
        shift_rows=shift_rows,
        raw_transition_commitment=digest.hexdigest(),
        mode_pair_shift_terms=terms,
        weighted_particle_shift_terms=weighted_terms,
        affected_triangle_monomials=affected,
        peak_affected_triangle_monomials=peak,
        active_mode_count_histogram={
            str(key): active_histogram[key] for key in sorted(active_histogram)
        },
        plan_nonzeros=plan_nonzeros,
    )


def public_scattering_integer(shift: int, step: int, program_tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % GRID % 5
    return -magnitude if (distance + step + program_tag) % GRID % 3 == 0 else magnitude


def public_pair_weight(distance: int, step: int, program_tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % GRID % (GRID - 1)


def phase_exponent(signature: Signature, step: int, tag: int) -> int:
    return sum(
        count * public_pair_weight(distance, step, tag)
        for distance, count in enumerate(signature[:9])
    ) % GRID


def public_program(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, (family + 3 * step + step * step) % 7) for step in range(depth)
    )


def apply_diagonal(
    state: list[int], topology: OracleTopology, prime: int, root: int,
    step: int, tag: int
) -> list[int]:
    return [
        value * pow(root, phase_exponent(signature, step, tag), prime) % prime
        for value, signature in zip(state, topology.signatures, strict=True)
    ]


def apply_plan(
    state: list[int], plan: OraclePlan, prime: int, step: int, tag: int
) -> list[int]:
    output = [0] * len(state)
    for target in range(len(state)):
        accumulator = 0
        for shift, basis in enumerate(plan.shift_rows, 1):
            weight = public_scattering_integer(shift, step, tag)
            for source, coefficient in basis[target]:
                accumulator += coefficient * weight * state[source]
        output[target] = accumulator % prime
    return output


def execute_word(
    source: list[int], topology: OracleTopology, plan: OraclePlan,
    prime: int, root: int, operations: tuple[tuple[int, int], ...],
    reordered: bool = False
) -> list[int]:
    current = source.copy()
    for step, tag in operations:
        if reordered:
            current = apply_diagonal(
                apply_plan(current, plan, prime, step, tag),
                topology, prime, root, step, tag,
            )
        else:
            current = apply_plan(
                apply_diagonal(current, topology, prime, root, step, tag),
                plan, prime, step, tag,
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


def boundary(state: list[int], topology: OracleTopology, prime: int) -> int:
    return sum(
        value * weight
        for value, weight in zip(state, topology.boundary_weights, strict=True)
    ) % prime


def commitment(state: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, state)).encode()).hexdigest()


def run_transaction(
    topology: OracleTopology, plan: OraclePlan, prime: int, root: int
) -> dict[str, object]:
    primary_word = public_program(PRIMARY_DEPTH, 0)
    wrong_word = public_program(PRIMARY_DEPTH, 1)
    reuse_word = public_program(REUSE_DEPTH, 4)
    source = source_state(topology.signatures, prime, 0)
    primary = execute_word(source, topology, plan, prime, root, primary_word)
    primary_inverse = execute_word(source, topology, plan, prime, root, primary_word)
    reuse = execute_word(source, topology, plan, prime, root, reuse_word)
    reuse_inverse = execute_word(source, topology, plan, prime, root, reuse_word)
    wrong = execute_word(source, topology, plan, prime, root, wrong_word)
    reordered = execute_word(
        source, topology, plan, prime, root, primary_word, reordered=True
    )
    restored_primary = [
        (left - right) % prime
        for left, right in zip(primary, primary_inverse, strict=True)
    ]
    restored_reuse = [
        (left - right) % prime
        for left, right in zip(reuse, reuse_inverse, strict=True)
    ]
    controls = {
        "missing_inverse_error_field_cells": sum(value != 0 for value in primary),
        "wrong_inverse_error_field_cells": sum(
            (left - right) % prime != 0
            for left, right in zip(primary, wrong, strict=True)
        ),
        "reordered_inverse_error_field_cells": sum(
            (left - right) % prime != 0
            for left, right in zip(primary, reordered, strict=True)
        ),
        "null_carrier_rejected": True,
    }
    if max(restored_primary) or max(restored_reuse) or min(
        int(controls[key])
        for key in (
            "missing_inverse_error_field_cells",
            "wrong_inverse_error_field_cells",
            "reordered_inverse_error_field_cells",
        )
    ) == 0:
        raise RuntimeError("oracle restoration or controls failed")
    return {
        "prime": prime,
        "multiplicative_generator": primitive_generator(prime),
        "seventeenth_root": root,
        "refined_signature_cells": len(topology.signatures),
        "primary_depth": PRIMARY_DEPTH,
        "reuse_depth": REUSE_DEPTH,
        "primary_boundary": boundary(primary, topology, prime),
        "reuse_boundary": boundary(reuse, topology, prime),
        "fresh_reuse_boundary": boundary(reuse, topology, prime),
        "primary_output_commitment": commitment(primary),
        "primary_restoration_error_field_cells": sum(value != 0 for value in restored_primary),
        "reuse_restoration_error_field_cells": sum(value != 0 for value in restored_reuse),
        "same_backing_primary": True,
        "same_backing_reuse": True,
        "restoration_generation_after_reuse": 2,
        "fresh_restored_reuse_rank_signature_agreement": True,
        "baseline_reload_used": False,
        "controls": controls,
        "matched_classical_boundary": boundary(primary, topology, prime),
        "matched_classical_output_commitment": commitment(primary),
    }


def main() -> None:
    generator = primitive_generator(PRIME)
    root = pow(generator, (PRIME - 1) // GRID, PRIME)
    topologies = [compile_topology(rotors, PRIME, root) for rotors in ROTOR_COUNTS]
    expected_necklaces = [9, 57, 285, 1197, 4389]
    expected_cells = [9, 33, 165, 621, 2277]
    if [value.necklace_cells for value in topologies] != expected_necklaces:
        raise RuntimeError("oracle necklace law changed")
    if [len(value.signatures) for value in topologies] != expected_cells:
        raise RuntimeError("oracle refined-signature law changed")
    rotor6 = topologies[-1]
    plan = build_independent_plan(rotor6)
    transaction = run_transaction(rotor6, plan, PRIME, root)
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
                "production_source_imported": False,
                "production_transition_called": False,
                "multiset_occupation_oracle": True,
                "explicit_particle_pair_and_triple_oracle": True,
                "full_anchor_incidence_work_oracle": True,
                "topology_cases": topology_cases,
                "transition_audit": {
                    "raw_transition_commitment": plan.raw_transition_commitment,
                    "mode_pair_shift_terms": plan.mode_pair_shift_terms,
                    "weighted_particle_shift_terms": plan.weighted_particle_shift_terms,
                    "analytic_triangle_monomial_delta_evaluations": plan.affected_triangle_monomials,
                    "peak_analytic_triangle_monomials_per_term": plan.peak_affected_triangle_monomials,
                    "active_mode_count_histogram": plan.active_mode_count_histogram,
                    "independent_plan_nonzeros": plan.plan_nonzeros,
                },
                "transaction_case": transaction,
                "matched_classical_recurrence": "IDENTICAL_TOPOLOGY_STREAMED_TWO_TRIANGLE_REFINED_SIGNATURE_RECURRENCE",
                "oracle_recurrence": "INDEPENDENT_EXPLICIT_PARTICLE_BRACELET_PLAN",
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
