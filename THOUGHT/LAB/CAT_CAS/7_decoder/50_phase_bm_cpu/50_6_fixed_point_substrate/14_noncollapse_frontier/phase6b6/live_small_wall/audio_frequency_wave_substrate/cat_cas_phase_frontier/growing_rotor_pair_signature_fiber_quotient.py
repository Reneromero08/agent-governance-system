#!/usr/bin/env python3
"""Exact pair-signature fiber quotient after the M192 chart escape.

The M192 nine-parameter monomial chart was too small, but its public
pair-distance signatures induce a larger structural partition.  This package
compiles all sixteen signed pair-scattering shift bases from public topology
and checks exact equitability of every signature fiber.  The resulting
quotient has 9, 33, 165, and 621 field cells at rotor counts two through five,
instead of 9, 57, 285, and 1197 necklace cells.

The accepted transaction is an exact two-register quotient shear.  It keeps
only the final scalar boundary, subtracts the recomputed public word, verifies
exact restoration, and reuses the same backing under a different program.
Full-necklace states and generators are verification-only parity resources.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal


GRID = 17
PAIR_CHANNELS = 9
PRIMES = (103, 239)
ROTOR_COUNTS = (2, 3, 4, 5)
PRIMARY_DEPTH = 8
REUSE_DEPTH = 5
Control = Literal["correct", "missing", "wrong", "reordered"]
Histogram = tuple[int, ...]
SparseRow = tuple[tuple[int, int], ...]


def rotate(histogram: Histogram, shift: int) -> Histogram:
    result = [0] * GRID
    for mode, count in enumerate(histogram):
        result[(mode + shift) % GRID] = count
    return tuple(result)


def canonical(histogram: Histogram) -> Histogram:
    return min(rotate(histogram, shift) for shift in range(GRID))


def generate_histograms(rotors: int) -> list[Histogram]:
    result: list[Histogram] = []
    working = [0] * GRID

    def visit(position: int, remaining: int) -> None:
        if position == GRID - 1:
            working[position] = remaining
            result.append(tuple(working))
            return
        for count in range(remaining + 1):
            working[position] = count
            visit(position + 1, remaining - count)

    visit(0, rotors)
    expected = math.comb(rotors + GRID - 1, rotors)
    if len(result) != expected:
        raise RuntimeError("fiber topology histogram count failed")
    return result


def pair_signature(histogram: Histogram, rotors: int) -> tuple[int, ...]:
    result = [sum(count * (count - 1) // 2 for count in histogram)]
    for distance in range(1, PAIR_CHANNELS):
        result.append(
            sum(
                histogram[mode] * histogram[(mode + distance) % GRID]
                for mode in range(GRID)
            )
        )
    if sum(result) != math.comb(rotors, 2):
        raise RuntimeError("fiber pair-signature partition failed")
    return tuple(result)


@dataclass(frozen=True)
class Topology:
    rotors: int
    necklaces: tuple[Histogram, ...]
    signatures: tuple[tuple[int, ...], ...]
    unique_signatures: tuple[tuple[int, ...], ...]
    fiber_labels: tuple[int, ...]
    lookup: dict[Histogram, int]
    occupation_histograms_visited: int


@dataclass(frozen=True)
class QuotientProgram:
    rotors: int
    signatures: tuple[tuple[int, ...], ...]
    shift_rows: tuple[tuple[SparseRow, ...], ...]
    boundary_weights: tuple[int, ...]
    necklace_cells: int
    occupation_histograms_visited: int
    enumerated_mode_pair_shift_terms: int
    weighted_particle_pair_shift_terms: int
    quotient_shift_plan_nonzero_entries: int
    equitability_comparisons: int


def compile_topology(rotors: int) -> Topology:
    if not 0 < rotors < GRID:
        raise RuntimeError("fiber free-orbit law is out of scope")
    occupations = generate_histograms(rotors)
    necklaces = tuple(item for item in occupations if canonical(item) == item)
    if len(occupations) % GRID or len(necklaces) != len(occupations) // GRID:
        raise RuntimeError("fiber free global-rotation orbit law failed")
    signatures = tuple(pair_signature(item, rotors) for item in necklaces)
    unique_signatures = tuple(sorted(set(signatures)))
    signature_lookup = {
        signature: index for index, signature in enumerate(unique_signatures)
    }
    return Topology(
        rotors=rotors,
        necklaces=necklaces,
        signatures=signatures,
        unique_signatures=unique_signatures,
        fiber_labels=tuple(signature_lookup[item] for item in signatures),
        lookup={item: index for index, item in enumerate(necklaces)},
        occupation_histograms_visited=len(occupations),
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
    raise RuntimeError("fiber primitive generator search failed")


def public_scattering_integer(shift: int, step: int, program_tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % GRID % 5
    return (
        -magnitude
        if (distance + step + program_tag) % GRID % 3 == 0
        else magnitude
    )


def public_pair_weight(distance: int, step: int, program_tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % GRID % (GRID - 1)


def phase_exponent(
    signature: tuple[int, ...], step: int, program_tag: int
) -> int:
    return sum(
        count * public_pair_weight(distance, step, program_tag)
        for distance, count in enumerate(signature)
    ) % GRID


def compile_quotient(
    topology: Topology, prime: int, root: int
) -> QuotientProgram:
    fiber_count = len(topology.unique_signatures)
    references: list[list[tuple[int, ...] | None]] = [
        [None] * fiber_count for _ in range(1, GRID)
    ]
    enumerated_terms = 0
    weighted_terms = 0
    comparisons = 0
    for target, histogram in enumerate(topology.necklaces):
        rows = [[0] * fiber_count for _ in range(1, GRID)]
        for first in range(GRID):
            if histogram[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = histogram[first] * (
                    histogram[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    source = list(histogram)
                    source[first] -= 1
                    source[second] -= 1
                    source[(first - shift) % GRID] += 1
                    source[(second + shift) % GRID] += 1
                    source_index = topology.lookup[canonical(tuple(source))]
                    rows[shift - 1][topology.fiber_labels[source_index]] += multiplicity
                    enumerated_terms += 1
                    weighted_terms += multiplicity
        target_fiber = topology.fiber_labels[target]
        for shift_index, row in enumerate(rows):
            candidate = tuple(row)
            reference = references[shift_index][target_fiber]
            if reference is None:
                references[shift_index][target_fiber] = candidate
            else:
                comparisons += 1
                if reference != candidate:
                    raise RuntimeError(
                        "pair-signature fiber is not equitable under a shift basis"
                    )
    if any(row is None for shift in references for row in shift):
        raise RuntimeError("fiber quotient has an uncompiled row")
    shift_rows = tuple(
        tuple(
            tuple((source, value) for source, value in enumerate(row) if value)
            for row in shift
            if row is not None
        )
        for shift in references
    )
    boundary_weights = [0] * fiber_count
    for index, signature in enumerate(topology.signatures):
        boundary_weights[topology.fiber_labels[index]] = (
            boundary_weights[topology.fiber_labels[index]]
            + pow(root, (11 * index + 5 * signature[0] + 1) % GRID, prime)
        ) % prime
    return QuotientProgram(
        rotors=topology.rotors,
        signatures=topology.unique_signatures,
        shift_rows=shift_rows,
        boundary_weights=tuple(boundary_weights),
        necklace_cells=len(topology.necklaces),
        occupation_histograms_visited=topology.occupation_histograms_visited,
        enumerated_mode_pair_shift_terms=enumerated_terms,
        weighted_particle_pair_shift_terms=weighted_terms,
        quotient_shift_plan_nonzero_entries=sum(
            len(row) for shift in shift_rows for row in shift
        ),
        equitability_comparisons=comparisons,
    )


def apply_diagonal(
    state: list[int],
    program: QuotientProgram,
    prime: int,
    root: int,
    step: int,
    tag: int,
) -> list[int]:
    return [
        value * pow(root, phase_exponent(signature, step, tag), prime) % prime
        for value, signature in zip(state, program.signatures, strict=True)
    ]


def apply_scattering(
    state: list[int],
    program: QuotientProgram,
    prime: int,
    step: int,
    tag: int,
) -> list[int]:
    output = [0] * len(program.signatures)
    for shift, rows in enumerate(program.shift_rows, 1):
        weight = public_scattering_integer(shift, step, tag)
        for target, row in enumerate(rows):
            output[target] = (
                output[target]
                + weight
                * sum(coefficient * state[source] for source, coefficient in row)
            ) % prime
    return output


def public_program(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, (family + 3 * step + step * step) % 7) for step in range(depth)
    )


def execute_word(
    source: list[int],
    quotient: QuotientProgram,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
    reordered: bool = False,
) -> list[int]:
    current = source.copy()
    for step, tag in operations:
        if reordered:
            current = apply_diagonal(
                apply_scattering(current, quotient, prime, step, tag),
                quotient,
                prime,
                root,
                step,
                tag,
            )
        else:
            current = apply_scattering(
                apply_diagonal(current, quotient, prime, root, step, tag),
                quotient,
                prime,
                step,
                tag,
            )
    return current


def quotient_source(
    signatures: tuple[tuple[int, ...], ...], prime: int, family: int
) -> list[int]:
    return [
        (
            1
            + (family + 3) * (index + 1)
            + sum(
                (distance + 2 + family) * (count + 1) ** 2
                for distance, count in enumerate(signature)
            )
        )
        % prime
        for index, signature in enumerate(signatures)
    ]


def public_boundary(
    state: list[int], program: QuotientProgram, prime: int
) -> int:
    return sum(
        value * weight
        for value, weight in zip(state, program.boundary_weights, strict=True)
    ) % prime


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    quotient: QuotientProgram,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
    inverse_operations: tuple[tuple[int, int], ...],
    control: Control,
) -> dict[str, object]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed fiber quotient carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward = execute_word(
        carrier.source, quotient, prime, root, operations
    )
    carrier.target[:] = [
        (left + right) % prime
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    boundary = public_boundary(carrier.target, quotient, prime)
    if control != "missing":
        inverse = execute_word(
            carrier.source,
            quotient,
            prime,
            root,
            inverse_operations,
            reordered=control == "reordered",
        )
        carrier.target[:] = [
            (left - right) % prime
            for left, right in zip(carrier.target, inverse, strict=True)
        ]
    error_cells = sum(
        left != right
        for left, right in zip(carrier.source, expected_source, strict=True)
    ) + sum(value != 0 for value in carrier.target)
    carrier.generation += 1
    return {
        "boundary": boundary,
        "restoration_error_field_cells": error_cells,
        "same_backing": id(carrier.source) == source_backing
        and id(carrier.target) == target_backing,
        "generation": carrier.generation,
    }


def compile_full_generator(
    topology: Topology, step: int, tag: int
) -> tuple[SparseRow, ...]:
    output: list[SparseRow] = []
    for histogram in topology.necklaces:
        row: dict[int, int] = {}
        for first in range(GRID):
            if histogram[first] == 0:
                continue
            for second in range(GRID):
                multiplicity = histogram[first] * (
                    histogram[second] - (1 if first == second else 0)
                )
                if multiplicity == 0:
                    continue
                for shift in range(1, GRID):
                    source = list(histogram)
                    source[first] -= 1
                    source[second] -= 1
                    source[(first - shift) % GRID] += 1
                    source[(second + shift) % GRID] += 1
                    source_index = topology.lookup[canonical(tuple(source))]
                    row[source_index] = row.get(source_index, 0) + (
                        multiplicity * public_scattering_integer(shift, step, tag)
                    )
        output.append(tuple(sorted((index, value) for index, value in row.items() if value)))
    return tuple(output)


def execute_full_word(
    source: list[int],
    topology: Topology,
    prime: int,
    root: int,
    operations: tuple[tuple[int, int], ...],
) -> list[int]:
    current = source.copy()
    for step, tag in operations:
        current = [
            value * pow(root, phase_exponent(signature, step, tag), prime) % prime
            for value, signature in zip(current, topology.signatures, strict=True)
        ]
        rows = compile_full_generator(topology, step, tag)
        current = [
            sum(coefficient * current[source_index] for source_index, coefficient in row)
            % prime
            for row in rows
        ]
    return current


def state_commitment(state: list[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in state).encode()).hexdigest()


def run_case(rotors: int, prime: int) -> dict[str, object]:
    topology = compile_topology(rotors)
    generator = primitive_generator(prime)
    root = pow(generator, (prime - 1) // GRID, prime)
    if root == 1 or pow(root, GRID, prime) != 1:
        raise RuntimeError("fiber seventeenth root failed")
    quotient = compile_quotient(topology, prime, root)
    primary_operations = public_program(PRIMARY_DEPTH, 0)
    wrong_operations = public_program(PRIMARY_DEPTH, 1)
    reuse_operations = public_program(REUSE_DEPTH, 4)
    source = quotient_source(quotient.signatures, prime, 0)

    quotient_primary = execute_word(
        source, quotient, prime, root, primary_operations
    )
    full_source = [source[fiber] for fiber in topology.fiber_labels]
    full_primary = execute_full_word(
        full_source, topology, prime, root, primary_operations
    )
    lifted_quotient = [
        quotient_primary[fiber] for fiber in topology.fiber_labels
    ]
    if full_primary != lifted_quotient:
        raise RuntimeError("full-necklace and fiber quotient execution differ")
    full_boundary = sum(
        value
        * pow(root, (11 * index + 5 * topology.signatures[index][0] + 1) % GRID, prime)
        for index, value in enumerate(full_primary)
    ) % prime
    if full_boundary != public_boundary(quotient_primary, quotient, prime):
        raise RuntimeError("full-necklace and fiber quotient boundaries differ")

    carrier = Carrier(source.copy(), [0] * len(source))
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    primary = transaction(
        carrier,
        source,
        quotient,
        prime,
        root,
        primary_operations,
        primary_operations,
        "correct",
    )
    reuse = transaction(
        carrier,
        source,
        quotient,
        prime,
        root,
        reuse_operations,
        reuse_operations,
        "correct",
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse = transaction(
        fresh,
        source,
        quotient,
        prime,
        root,
        reuse_operations,
        reuse_operations,
        "correct",
    )
    if (
        primary["restoration_error_field_cells"] != 0
        or reuse["restoration_error_field_cells"] != 0
        or reuse["boundary"] != fresh_reuse["boundary"]
        or not primary["same_backing"]
        or not reuse["same_backing"]
        or id(carrier.source) != source_backing
        or id(carrier.target) != target_backing
    ):
        raise RuntimeError("fiber quotient restoration or reuse failed")

    controls: dict[str, int] = {}
    for control, inverse_operations in (
        ("missing", primary_operations),
        ("wrong", wrong_operations),
        ("reordered", primary_operations),
    ):
        controlled = Carrier(source.copy(), [0] * len(source))
        result = transaction(
            controlled,
            source,
            quotient,
            prime,
            root,
            primary_operations,
            inverse_operations,
            control,
        )
        controls[f"{control}_inverse_error_field_cells"] = int(
            result["restoration_error_field_cells"]
        )
    if min(controls.values()) == 0:
        raise RuntimeError("fiber quotient inverse controls did not discriminate")

    null_rejected = False
    try:
        transaction(
            Carrier([], []),
            [],
            quotient,
            prime,
            root,
            primary_operations,
            primary_operations,
            "correct",
        )
    except ValueError:
        null_rejected = True
    if not null_rejected:
        raise RuntimeError("null fiber quotient carrier was accepted")

    return {
        "rotors": rotors,
        "prime": prime,
        "multiplicative_generator": generator,
        "seventeenth_root": root,
        "necklace_cells": quotient.necklace_cells,
        "pair_signature_fiber_cells": len(quotient.signatures),
        "occupation_histograms_visited": quotient.occupation_histograms_visited,
        "all_sixteen_shift_bases_equitable": True,
        "primary_depth": PRIMARY_DEPTH,
        "reuse_depth": REUSE_DEPTH,
        "primary_boundary": primary["boundary"],
        "full_necklace_primary_boundary": full_boundary,
        "reuse_boundary": reuse["boundary"],
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "primary_output_commitment": state_commitment(quotient_primary),
        "full_necklace_output_commitment": state_commitment(full_primary),
        "primary_restoration_error_field_cells": primary[
            "restoration_error_field_cells"
        ],
        "reuse_restoration_error_field_cells": reuse[
            "restoration_error_field_cells"
        ],
        "same_backing_primary": primary["same_backing"],
        "same_backing_reuse": reuse["same_backing"],
        "restoration_generation_after_reuse": carrier.generation,
        "controls": {**controls, "null_carrier_rejected": null_rejected},
        "enumerated_mode_pair_shift_terms": quotient.enumerated_mode_pair_shift_terms,
        "weighted_particle_pair_shift_terms": quotient.weighted_particle_pair_shift_terms,
        "quotient_shift_plan_nonzero_entries": quotient.quotient_shift_plan_nonzero_entries,
        "equitability_comparisons": quotient.equitability_comparisons,
        "compression_ratio_necklace_over_fiber": quotient.necklace_cells
        / len(quotient.signatures),
    }


def main() -> None:
    cases = [
        run_case(rotors, prime) for rotors in ROTOR_COUNTS for prime in PRIMES
    ]
    expected_necklaces = [9, 9, 57, 57, 285, 285, 1197, 1197]
    expected_fibers = [9, 9, 33, 33, 165, 165, 621, 621]
    if [case["necklace_cells"] for case in cases] != expected_necklaces:
        raise RuntimeError("fiber quotient necklace dimensions changed")
    if [case["pair_signature_fiber_cells"] for case in cases] != expected_fibers:
        raise RuntimeError("fiber quotient dimensions changed")
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_PAIR_SIGNATURE_FIBER_EQUITABLE_QUOTIENT_CLOSES_ALL16_OFFDIAGONAL_SHIFT_BASES_AND_DIAGONAL_TWO_BODY_PHASE_ON9_33_165_621_CELLS_FOR_ROTORS2_TO5_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_AND_REUSE_BUT_IDENTICAL_CLASSICAL_QUOTIENT_RECURRENCE_REMAINS",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS2_TO5_F103_F239_ALL16_SIGNED_PAIR_SHIFT_BASES_PRIMARY_DEPTH8_REUSE_DEPTH5_DIRECT_PROCESS_SOFTWARE_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS",
                "cases": cases,
                "quotient_cell_law": [9, 33, 165, 621],
                "necklace_cell_law": [9, 57, 285, 1197],
                "all_sixteen_shift_bases_equitable": True,
                "pair_diagonal_closed_by_signature": True,
                "resource_law": {
                    "accepted_resident_field_cells": "TWO_TIMES_PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_word_scratch_field_cells": "TWO_TIMES_PAIR_SIGNATURE_FIBER_COUNT",
                    "accepted_public_shift_basis_plans": 16,
                    "accepted_public_plan_nonzero_entries": "REPORTED_PER_CASE",
                    "accepted_retained_inverse_history_bytes": 0,
                    "accepted_assignment_or_relation_table_cells": 0,
                    "public_topology_compiler_necklace_cells": "REPORTED_PER_CASE",
                    "public_topology_compiler_occupation_histograms_visited": "REPORTED_PER_CASE",
                    "full_necklace_vectors_and_generators": "VERIFICATION_ONLY",
                },
                "matched_classical_recurrence": "IDENTICAL_PAIR_SIGNATURE_FIBER_QUOTIENT_RECURRENCE",
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
