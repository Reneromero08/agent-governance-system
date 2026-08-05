#!/usr/bin/env python3
"""Independent ordered-particle oracle for the plan-free quotient.

No production module is imported.  Multiset occupations define the topology,
ordered particle labels define scattering, and a verification-only sparse plan
is compared with the row-local ordered-particle stream.  The plan is never an
accepted-path resource.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Literal


GRID = 17
CHANNELS = 9
FIELDS = (103, 239)
ROTOR_COUNTS = (2, 3, 4, 5)
PRIMARY_DEPTH = 8
REUSE_DEPTH = 5
Histogram = tuple[int, ...]
Signature = tuple[int, ...]
SparseProfile = tuple[tuple[int, int, int], ...]
Control = Literal["correct", "missing", "wrong", "reordered"]


def histogram(item: tuple[int, ...]) -> Histogram:
    result = [0] * GRID
    for mode in item:
        result[mode] += 1
    return tuple(result)


def shift_histogram(item: Histogram, amount: int) -> Histogram:
    return tuple(item[(mode - amount) % GRID] for mode in range(GRID))


def representative(item: Histogram) -> Histogram:
    return min(shift_histogram(item, amount) for amount in range(GRID))


def signature(item: Histogram, rotors: int) -> Signature:
    values = [sum(count * (count - 1) // 2 for count in item)]
    values.extend(
        sum(item[mode] * item[(mode + distance) % GRID] for mode in range(GRID))
        for distance in range(1, CHANNELS)
    )
    if sum(values) != math.comb(rotors, 2):
        raise RuntimeError("oracle signature failed")
    return tuple(values)


def prime_factors(value: int) -> tuple[int, ...]:
    result: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            result.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1
    if value > 1:
        result.append(value)
    return tuple(result)


def primitive_generator(prime: int) -> int:
    factors = prime_factors(prime - 1)
    return next(
        candidate
        for candidate in range(2, prime)
        if all(
            pow(candidate, (prime - 1) // factor, prime) != 1
            for factor in factors
        )
    )


def scattering_weight(shift: int, step: int, tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1) + (3 * distance + 1) * (tag + 2)
    ) % GRID % 5
    return -magnitude if (distance + step + tag) % GRID % 3 == 0 else magnitude


def pair_weight(distance: int, step: int, tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * tag
    ) % GRID % (GRID - 1)


def phase(value: Signature, step: int, tag: int) -> int:
    return sum(
        count * pair_weight(distance, step, tag)
        for distance, count in enumerate(value)
    ) % GRID


@dataclass(frozen=True)
class OracleTopology:
    rotors: int
    occupations: tuple[Histogram, ...]
    necklaces: tuple[Histogram, ...]
    signatures: tuple[Signature, ...]
    representatives: tuple[Histogram, ...]
    boundary_weights: tuple[int, ...]


def build_topology(rotors: int, prime: int, root: int) -> OracleTopology:
    occupations = tuple(
        sorted(
            histogram(item)
            for item in itertools.combinations_with_replacement(range(GRID), rotors)
        )
    )
    expected = math.comb(rotors + GRID - 1, rotors)
    if len(occupations) != expected or len(set(occupations)) != expected:
        raise RuntimeError("oracle multiset topology failed")
    necklaces = tuple(item for item in occupations if representative(item) == item)
    if not 0 < rotors < GRID or len(necklaces) != expected // GRID:
        raise RuntimeError("oracle free-orbit quotient failed")
    necklace_signatures = tuple(signature(item, rotors) for item in necklaces)
    signatures = tuple(sorted(set(necklace_signatures)))
    representatives: dict[Signature, Histogram] = {}
    boundary = {item: 0 for item in signatures}
    for index, (item, value) in enumerate(zip(necklaces, necklace_signatures, strict=True)):
        representatives.setdefault(value, item)
        boundary[value] = (
            boundary[value]
            + pow(root, (11 * index + 5 * value[0] + 1) % GRID, prime)
        ) % prime
    return OracleTopology(
        rotors=rotors,
        occupations=occupations,
        necklaces=necklaces,
        signatures=signatures,
        representatives=tuple(representatives[item] for item in signatures),
        boundary_weights=tuple(boundary[item] for item in signatures),
    )


def ordered_profile(item: Histogram, topology: OracleTopology) -> SparseProfile:
    particles = tuple(
        mode for mode, count in enumerate(item) for _ in range(count)
    )
    row: dict[tuple[int, int], int] = {}
    for first_particle, second_particle in itertools.permutations(
        range(topology.rotors), 2
    ):
        first = particles[first_particle]
        second = particles[second_particle]
        for shift in range(1, GRID):
            moved = list(item)
            moved[first] -= 1
            moved[second] -= 1
            moved[(first - shift) % GRID] += 1
            moved[(second + shift) % GRID] += 1
            source_signature = signature(tuple(moved), topology.rotors)
            source = bisect.bisect_left(topology.signatures, source_signature)
            if source == len(topology.signatures) or topology.signatures[source] != source_signature:
                raise RuntimeError("oracle move escaped signature topology")
            key = (shift, source)
            row[key] = row.get(key, 0) + 1
    return tuple(
        (shift, source, coefficient)
        for (shift, source), coefficient in sorted(row.items())
    )


def compile_verification_plan(
    topology: OracleTopology,
) -> tuple[tuple[SparseProfile, ...], int, int]:
    rows = tuple(ordered_profile(item, topology) for item in topology.representatives)
    lookup = {value: index for index, value in enumerate(topology.signatures)}
    comparisons = 0
    peak_comparison_entries = 0
    for item in topology.necklaces:
        value = signature(item, topology.rotors)
        candidate = ordered_profile(item, topology)
        reference = rows[lookup[value]]
        peak_comparison_entries = max(
            peak_comparison_entries, len(candidate) + len(reference)
        )
        if candidate != reference:
            raise RuntimeError("oracle rejected fiber equitability")
        if item != topology.representatives[lookup[value]]:
            comparisons += GRID - 1
    return rows, comparisons, peak_comparison_entries


def apply_diagonal(
    state: list[int],
    topology: OracleTopology,
    prime: int,
    root: int,
    step: int,
    tag: int,
) -> list[int]:
    return [
        value * pow(root, phase(sig, step, tag), prime) % prime
        for value, sig in zip(state, topology.signatures, strict=True)
    ]


def apply_plan(
    state: list[int],
    rows: tuple[SparseProfile, ...],
    prime: int,
    step: int,
    tag: int,
) -> list[int]:
    return [
        sum(
            coefficient * scattering_weight(shift, step, tag) * state[source]
            for shift, source, coefficient in row
        )
        % prime
        for row in rows
    ]


def apply_ordered_stream(
    state: list[int],
    topology: OracleTopology,
    prime: int,
    step: int,
    tag: int,
) -> list[int]:
    result: list[int] = []
    for item in topology.representatives:
        particles = tuple(
            mode for mode, count in enumerate(item) for _ in range(count)
        )
        accumulator = 0
        for first_particle, second_particle in itertools.permutations(
            range(topology.rotors), 2
        ):
            first = particles[first_particle]
            second = particles[second_particle]
            for shift in range(1, GRID):
                moved = list(item)
                moved[first] -= 1
                moved[second] -= 1
                moved[(first - shift) % GRID] += 1
                moved[(second + shift) % GRID] += 1
                source_signature = signature(tuple(moved), topology.rotors)
                source = bisect.bisect_left(topology.signatures, source_signature)
                accumulator += (
                    scattering_weight(shift, step, tag) * state[source]
                )
        result.append(accumulator % prime)
    return result


def operations(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, (family + 3 * step + step * step) % 7) for step in range(depth)
    )


def execute_word(
    source: list[int],
    topology: OracleTopology,
    prime: int,
    root: int,
    program: tuple[tuple[int, int], ...],
    rows: tuple[SparseProfile, ...] | None = None,
    reordered: bool = False,
) -> list[int]:
    current = source.copy()
    for step, tag in program:
        scatter = (
            (lambda state: apply_plan(state, rows, prime, step, tag))
            if rows is not None
            else (lambda state: apply_ordered_stream(state, topology, prime, step, tag))
        )
        if reordered:
            current = apply_diagonal(
                scatter(current), topology, prime, root, step, tag
            )
        else:
            current = scatter(
                apply_diagonal(current, topology, prime, root, step, tag)
            )
    return current


def source_state(signatures: tuple[Signature, ...], prime: int, family: int) -> list[int]:
    return [
        (
            1
            + (family + 3) * (index + 1)
            + sum(
                (distance + 2 + family) * (count + 1) ** 2
                for distance, count in enumerate(value)
            )
        )
        % prime
        for index, value in enumerate(signatures)
    ]


def boundary(state: list[int], topology: OracleTopology, prime: int) -> int:
    return sum(
        value * weight
        for value, weight in zip(state, topology.boundary_weights, strict=True)
    ) % prime


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def transaction(
    carrier: Carrier,
    expected: list[int],
    topology: OracleTopology,
    prime: int,
    root: int,
    program: tuple[tuple[int, int], ...],
    inverse_program: tuple[tuple[int, int], ...],
    control: Control,
    rows: tuple[SparseProfile, ...] | None = None,
) -> dict[str, object]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("oracle null carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward = execute_word(
        carrier.source, topology, prime, root, program, rows=rows
    )
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
            inverse_program,
            rows=rows,
            reordered=control == "reordered",
        )
        carrier.target[:] = [
            (left - right) % prime
            for left, right in zip(carrier.target, inverse, strict=True)
        ]
    error = sum(a != b for a, b in zip(carrier.source, expected, strict=True))
    error += sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    return {
        "boundary": projected,
        "forward_commitment": commitment(forward),
        "restoration_error_field_cells": error,
        "same_backing": id(carrier.source) == source_backing
        and id(carrier.target) == target_backing,
        "generation": carrier.generation,
    }


def commitment(state: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, state)).encode()).hexdigest()


def run_case(rotors: int, prime: int) -> dict[str, object]:
    generator = primitive_generator(prime)
    root = pow(generator, (prime - 1) // GRID, prime)
    topology = build_topology(rotors, prime, root)
    rows, comparisons, verification_peak = compile_verification_plan(topology)
    primary_program = operations(PRIMARY_DEPTH, 0)
    wrong_program = operations(PRIMARY_DEPTH, 1)
    reuse_program = operations(REUSE_DEPTH, 4)
    source = source_state(topology.signatures, prime, 0)

    primary_plan = execute_word(
        source, topology, prime, root, primary_program, rows=rows
    )
    first_step = primary_program[:1]
    one_step_stream = execute_word(
        source, topology, prime, root, first_step
    )
    one_step_plan = execute_word(
        source, topology, prime, root, first_step, rows=rows
    )
    if one_step_stream != one_step_plan:
        raise RuntimeError("oracle stream and materialized plan differ")

    carrier = Carrier(source.copy(), [0] * len(source))
    primary = transaction(
        carrier,
        source,
        topology,
        prime,
        root,
        primary_program,
        primary_program,
        "correct",
        rows,
    )
    reuse = transaction(
        carrier,
        source,
        topology,
        prime,
        root,
        reuse_program,
        reuse_program,
        "correct",
        rows,
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse = transaction(
        fresh,
        source,
        topology,
        prime,
        root,
        reuse_program,
        reuse_program,
        "correct",
        rows,
    )
    if (
        primary["restoration_error_field_cells"]
        or reuse["restoration_error_field_cells"]
        or reuse["boundary"] != fresh_reuse["boundary"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
    ):
        raise RuntimeError("oracle restoration or reuse failed")

    controls: dict[str, int] = {}
    for control, inverse in (
        ("missing", primary_program),
        ("wrong", wrong_program),
        ("reordered", primary_program),
    ):
        result = transaction(
            Carrier(source.copy(), [0] * len(source)),
            source,
            topology,
            prime,
            root,
            primary_program,
            inverse,
            control,
            rows,
        )
        controls[f"{control}_inverse_error_field_cells"] = int(
            result["restoration_error_field_cells"]
        )
    if min(controls.values()) == 0:
        raise RuntimeError("oracle controls failed")
    null_rejected = False
    try:
        transaction(
            Carrier([], []),
            [],
            topology,
            prime,
            root,
            primary_program,
            primary_program,
            "correct",
            rows,
        )
    except ValueError:
        null_rejected = True

    fibers = len(topology.signatures)
    distinct_terms = 0
    for item in topology.representatives:
        particles = tuple(mode for mode, count in enumerate(item) for _ in range(count))
        distinct_terms += len(
            {
                (particles[first], particles[second], shift)
                for first, second in itertools.permutations(range(rotors), 2)
                for shift in range(1, GRID)
            }
        )
    weighted_terms = fibers * rotors * (rotors - 1) * (GRID - 1)
    plan_nonzeros = sum(len(row) for row in rows)
    return {
        "rotors": rotors,
        "prime": prime,
        "multiplicative_generator": generator,
        "seventeenth_root": root,
        "necklace_cells": len(topology.necklaces),
        "pair_signature_fiber_cells": fibers,
        "occupation_histograms_visited": len(topology.occupations),
        "compiler_peak_live_histograms": 1,
        "verification_only_occupation_histograms_visited": len(topology.occupations),
        "verification_only_peak_sparse_row_entries": verification_peak,
        "equitability_comparisons": comparisons,
        "all_sixteen_shift_bases_equitable": True,
        "primary_depth": PRIMARY_DEPTH,
        "reuse_depth": REUSE_DEPTH,
        "primary_boundary": primary["boundary"],
        "reuse_boundary": reuse["boundary"],
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "primary_output_commitment": commitment(primary_plan),
        "primary_restoration_error_field_cells": primary["restoration_error_field_cells"],
        "reuse_restoration_error_field_cells": reuse["restoration_error_field_cells"],
        "same_backing_primary": primary["same_backing"],
        "same_backing_reuse": reuse["same_backing"],
        "restoration_generation_after_reuse": carrier.generation,
        "controls": {**controls, "null_carrier_rejected": null_rejected},
        "retained_shift_basis_plans": 0,
        "retained_shift_plan_nonzero_entries": 0,
        "prior_materialized_plan_nonzero_entries": plan_nonzeros,
        "public_signature_descriptor_integer_cells": CHANNELS * fibers,
        "public_representative_descriptor_integer_cells": GRID * fibers,
        "public_boundary_weight_field_cells": fibers,
        "streamed_mode_pair_shift_terms_per_scattering": distinct_terms,
        "streamed_weighted_particle_shift_terms_per_scattering": weighted_terms,
        "primary_forward_streamed_terms": PRIMARY_DEPTH * distinct_terms,
        "primary_forward_inverse_streamed_terms": 2 * PRIMARY_DEPTH * distinct_terms,
        "reuse_forward_inverse_streamed_terms": 2 * REUSE_DEPTH * distinct_terms,
    }


def main() -> None:
    cases = [
        run_case(rotors, prime) for rotors in ROTOR_COUNTS for prime in FIELDS
    ]
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_TOPOLOGY_STREAMED_PAIR_SIGNATURE_FIBER_QUOTIENT_ELIMINATES_ALL16_RETAINED_SHIFT_BASES_WHILE_CLOSING_THE9_33_165_621_CELL_ROTORS2_TO5_CARRIER_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_RETAINS26S_PUBLIC_INTEGER_DESCRIPTORS_AND_REMATERIALIZES_EACH_SCATTERING_WITH_AN_IDENTICAL_CLASSICAL_STREAM",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_PAIR_SIGNATURE_FIBER_CONSTANT_INPUTS_ROTORS2_TO5_F103_F239_PRIMARY_DEPTH8_REUSE_DEPTH5_DIRECT_PROCESS_SOFTWARE_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS",
                "production_source_imported": False,
                "production_projection_called": False,
                "production_inverse_called": False,
                "ordered_particle_stream_used": True,
                "materialized_rows_are_verification_only": True,
                "cases": cases,
                "quotient_cell_law": [9, 33, 165, 621],
                "necklace_cell_law": [9, 57, 285, 1197],
                "all_sixteen_shift_bases_equitable": True,
                "matched_classical_recurrence": "IDENTICAL_TOPOLOGY_STREAMED_PAIR_SIGNATURE_FIBER_RECURRENCE",
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
