#!/usr/bin/env python3
"""Exact nonlinear pair-signature chart diagnostic after M191.

The candidate compact state is a multiplicative Jastrow chart with one global
scale and eight relative cyclic-pair coordinates.  The public diagonal
two-body phase is closed on this chart by construction.  The experiment asks
whether the off-diagonal pair-scattering generator preserves it.  Exact
membership is decided by discrete logarithms and modular rank over every
prime factor of ``p - 1`` at F103 and F239.

The reversible accepted transaction is a two-register shear.  It writes the
actual ``K*D`` output into a resident necklace register, projects only one
public scalar, rematerializes the inverse word from public topology, restores
both registers exactly, and reuses the same backing under a different public
program.  The chart-membership matrices are verification-only.
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
Control = Literal["correct", "missing", "wrong", "reordered"]
Histogram = tuple[int, ...]


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
    if len(result) != math.comb(rotors + GRID - 1, rotors):
        raise RuntimeError("Jastrow topology histogram count failed")
    return result


def collision_count(histogram: Histogram) -> int:
    return sum(count * (count - 1) // 2 for count in histogram)


def pair_signature(histogram: Histogram, rotors: int) -> tuple[int, ...]:
    result = [collision_count(histogram)]
    for distance in range(1, PAIR_CHANNELS):
        result.append(
            sum(
                histogram[mode] * histogram[(mode + distance) % GRID]
                for mode in range(GRID)
            )
        )
    if sum(result) != math.comb(rotors, 2):
        raise RuntimeError("Jastrow pair-signature partition failed")
    return tuple(result)


def public_pair_weight(distance: int, step: int, program_tag: int) -> int:
    return 1 + (
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % GRID % (GRID - 1)


def pair_phase_exponent(
    signature: tuple[int, ...], step: int, program_tag: int
) -> int:
    return sum(
        signature[distance]
        * public_pair_weight(distance, step, program_tag)
        for distance in range(PAIR_CHANNELS)
    ) % GRID


def public_scattering_integer(
    signed_shift: int, step: int, program_tag: int
) -> int:
    positive = signed_shift % GRID
    if positive == 0:
        raise RuntimeError("zero Jastrow scattering shift")
    distance = min(positive, GRID - positive)
    magnitude = 1 + (
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % GRID % 5
    return (
        -magnitude
        if (distance + step + program_tag) % GRID % 3 == 0
        else magnitude
    )


@dataclass(frozen=True)
class Topology:
    rotors: int
    necklaces: tuple[Histogram, ...]
    signatures: tuple[tuple[int, ...], ...]
    collisions: tuple[int, ...]
    lookup: dict[Histogram, int]
    occupation_histograms_visited: int


@dataclass(frozen=True)
class Generator:
    rows: tuple[tuple[tuple[int, int], ...], ...]
    enumerated_terms: int
    unique_terms: int


def compile_topology(rotors: int) -> Topology:
    if not 0 < rotors < GRID:
        raise RuntimeError("Jastrow free-orbit law is out of scope")
    occupations = generate_histograms(rotors)
    necklaces = tuple(item for item in occupations if canonical(item) == item)
    if len(occupations) % GRID or len(necklaces) != len(occupations) // GRID:
        raise RuntimeError("Jastrow free global-rotation orbit law failed")
    lookup = {item: index for index, item in enumerate(necklaces)}
    signatures = tuple(pair_signature(item, rotors) for item in necklaces)
    return Topology(
        rotors=rotors,
        necklaces=necklaces,
        signatures=signatures,
        collisions=tuple(collision_count(item) for item in necklaces),
        lookup=lookup,
        occupation_histograms_visited=len(occupations),
    )


def compile_generator(
    topology: Topology, step: int, program_tag: int
) -> Generator:
    output: list[tuple[tuple[int, int], ...]] = []
    enumerated_terms = 0
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
                        multiplicity
                        * public_scattering_integer(shift, step, program_tag)
                    )
                    enumerated_terms += 1
        output.append(
            tuple(sorted((source, value) for source, value in row.items() if value))
        )
    return Generator(
        rows=tuple(output),
        enumerated_terms=enumerated_terms,
        unique_terms=sum(len(row) for row in output),
    )


def prime_factors(value: int) -> tuple[int, ...]:
    result: list[int] = []
    candidate = 2
    while candidate * candidate <= value:
        if value % candidate == 0:
            result.append(candidate)
            while value % candidate == 0:
                value //= candidate
        candidate += 1
    if value > 1:
        result.append(value)
    return tuple(result)


def primitive_generator(prime: int) -> int:
    factors = prime_factors(prime - 1)
    for candidate in range(2, prime):
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors):
            return candidate
    raise RuntimeError("primitive generator search failed")


def primitive_seventeenth_root(prime: int, generator: int) -> int:
    if (prime - 1) % GRID:
        raise RuntimeError("Jastrow prime lacks seventeenth roots")
    root = pow(generator, (prime - 1) // GRID, prime)
    if root == 1 or pow(root, GRID, prime) != 1:
        raise RuntimeError("Jastrow seventeenth root failed")
    return root


def logarithm_table(prime: int, generator: int) -> list[int]:
    result = [-1] * prime
    value = 1
    for exponent in range(prime - 1):
        result[value] = exponent
        value = value * generator % prime
    if value != 1 or any(item < 0 for item in result[1:]):
        raise RuntimeError("Jastrow logarithm table failed")
    return result


def chart_exponent_rows(topology: Topology) -> list[list[int]]:
    return [[1, *signature[1:]] for signature in topology.signatures]


def matrix_rank_mod(rows: list[list[int]], prime: int) -> int:
    matrix = [[value % prime for value in row] for row in rows]
    if not matrix:
        return 0
    pivot_row = 0
    for column in range(len(matrix[0])):
        pivot = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column], prime - 2, prime)
        matrix[pivot_row] = [value * inverse % prime for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or matrix[row][column] == 0:
                continue
            scale = matrix[row][column]
            matrix[row] = [
                (left - scale * right) % prime
                for left, right in zip(matrix[row], matrix[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    return pivot_row


def chart_membership(
    state: list[int],
    topology: Topology,
    prime: int,
    logs: list[int],
) -> dict[str, object]:
    zeros = sum(value % prime == 0 for value in state)
    factors = prime_factors(prime - 1)
    exponent_rows = chart_exponent_rows(topology)
    checks: list[dict[str, int | bool]] = []
    if zeros:
        for factor in factors:
            checks.append(
                {
                    "factor": factor,
                    "chart_rank": matrix_rank_mod(exponent_rows, factor),
                    "augmented_rank": -1,
                    "consistent": False,
                }
            )
        return {"closed": False, "zero_cells": zeros, "factor_checks": checks}
    logarithms = [logs[value % prime] for value in state]
    closed = True
    for factor in factors:
        chart_rank = matrix_rank_mod(exponent_rows, factor)
        augmented = [
            [*row, logarithm]
            for row, logarithm in zip(exponent_rows, logarithms, strict=True)
        ]
        augmented_rank = matrix_rank_mod(augmented, factor)
        consistent = chart_rank == augmented_rank
        closed = closed and consistent
        checks.append(
            {
                "factor": factor,
                "chart_rank": chart_rank,
                "augmented_rank": augmented_rank,
                "consistent": consistent,
            }
        )
    return {"closed": closed, "zero_cells": 0, "factor_checks": checks}


def make_chart_state(
    topology: Topology,
    prime: int,
    generator: int,
    identity: int,
) -> list[int]:
    scale = pow(generator, 3 + 5 * identity, prime)
    ratios = [
        pow(generator, 7 + 11 * distance + 13 * identity, prime)
        for distance in range(1, PAIR_CHANNELS)
    ]
    result: list[int] = []
    for signature in topology.signatures:
        value = scale
        for ratio, exponent in zip(ratios, signature[1:], strict=True):
            value = value * pow(ratio, exponent, prime) % prime
        result.append(value)
    return result


def apply_diagonal(
    state: list[int],
    topology: Topology,
    prime: int,
    root: int,
    step: int,
    program_tag: int,
    inverse: bool = False,
) -> list[int]:
    sign = -1 if inverse else 1
    return [
        value
        * pow(
            root,
            (sign * pair_phase_exponent(signature, step, program_tag)) % GRID,
            prime,
        )
        % prime
        for value, signature in zip(state, topology.signatures, strict=True)
    ]


def apply_generator(state: list[int], generator: Generator, prime: int) -> list[int]:
    return [
        sum(coefficient * state[source] for source, coefficient in row) % prime
        for row in generator.rows
    ]


def public_boundary(
    state: list[int], topology: Topology, prime: int, root: int
) -> int:
    return sum(
        value
        * pow(
            root,
            (11 * index + 5 * topology.collisions[index] + 1) % GRID,
            prime,
        )
        for index, value in enumerate(state)
    ) % prime


@dataclass
class Carrier:
    source: list[int]
    target: list[int]
    generation: int = 0


def word(
    source: list[int],
    topology: Topology,
    generator: Generator,
    prime: int,
    root: int,
    step: int,
    program_tag: int,
    reordered: bool = False,
) -> list[int]:
    if reordered:
        return apply_diagonal(
            apply_generator(source, generator, prime),
            topology,
            prime,
            root,
            step,
            program_tag,
        )
    return apply_generator(
        apply_diagonal(
            source, topology, prime, root, step, program_tag
        ),
        generator,
        prime,
    )


def transaction(
    carrier: Carrier,
    expected_source: list[int],
    topology: Topology,
    generators: dict[int, Generator],
    prime: int,
    root: int,
    program_tag: int,
    control: Control,
) -> dict[str, object]:
    if not carrier.source or len(carrier.source) != len(carrier.target):
        raise ValueError("null or malformed Jastrow carrier")
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    forward = word(
        carrier.source,
        topology,
        generators[program_tag],
        prime,
        root,
        0,
        program_tag,
    )
    carrier.target[:] = [
        (left + right) % prime
        for left, right in zip(carrier.target, forward, strict=True)
    ]
    boundary = public_boundary(carrier.target, topology, prime, root)
    if control != "missing":
        inverse_tag = program_tag + 1 if control == "wrong" else program_tag
        inverse_word = word(
            carrier.source,
            topology,
            generators[inverse_tag],
            prime,
            root,
            0,
            inverse_tag,
            reordered=control == "reordered",
        )
        carrier.target[:] = [
            (left - right) % prime
            for left, right in zip(carrier.target, inverse_word, strict=True)
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


def state_commitment(state: list[int]) -> str:
    payload = ",".join(str(value) for value in state).encode()
    return hashlib.sha256(payload).hexdigest()


def run_case(topology: Topology, prime: int) -> dict[str, object]:
    generator_value = primitive_generator(prime)
    root = primitive_seventeenth_root(prime, generator_value)
    logs = logarithm_table(prime, generator_value)
    generators = {
        tag: compile_generator(topology, 0, tag) for tag in (0, 1, 3, 4)
    }
    source = make_chart_state(topology, prime, generator_value, 0)
    source_membership = chart_membership(source, topology, prime, logs)
    diagonal = apply_diagonal(source, topology, prime, root, 0, 0)
    diagonal_membership = chart_membership(diagonal, topology, prime, logs)
    forward = apply_generator(diagonal, generators[0], prime)
    forward_membership = chart_membership(forward, topology, prime, logs)
    alternate_source = make_chart_state(topology, prime, generator_value, 1)
    alternate_diagonal = apply_diagonal(
        alternate_source, topology, prime, root, 0, 3
    )
    alternate_forward = apply_generator(
        alternate_diagonal, generators[3], prime
    )
    alternate_membership = chart_membership(
        alternate_forward, topology, prime, logs
    )
    if not source_membership["closed"] or not diagonal_membership["closed"]:
        raise RuntimeError("declared Jastrow chart or diagonal closure failed")

    carrier = Carrier(source.copy(), [0] * len(source))
    source_backing = id(carrier.source)
    target_backing = id(carrier.target)
    primary = transaction(
        carrier, source, topology, generators, prime, root, 0, "correct"
    )
    restored_commitment = state_commitment(carrier.source + carrier.target)
    reuse = transaction(
        carrier, source, topology, generators, prime, root, 3, "correct"
    )
    fresh = Carrier(source.copy(), [0] * len(source))
    fresh_reuse = transaction(
        fresh, source, topology, generators, prime, root, 3, "correct"
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
        raise RuntimeError("exact Jastrow shear restoration or reuse failed")

    controls: dict[str, int] = {}
    for control in ("missing", "wrong", "reordered"):
        control_carrier = Carrier(source.copy(), [0] * len(source))
        result = transaction(
            control_carrier,
            source,
            topology,
            generators,
            prime,
            root,
            0,
            control,
        )
        controls[f"{control}_inverse_error_field_cells"] = int(
            result["restoration_error_field_cells"]
        )
    if min(controls.values()) == 0:
        raise RuntimeError("Jastrow inverse controls did not discriminate")

    null_rejected = False
    try:
        transaction(
            Carrier([], []), [], topology, generators, prime, root, 0, "correct"
        )
    except ValueError:
        null_rejected = True
    if not null_rejected:
        raise RuntimeError("null Jastrow carrier was accepted")

    return {
        "rotors": topology.rotors,
        "prime": prime,
        "multiplicative_generator": generator_value,
        "seventeenth_root": root,
        "necklace_cells": len(topology.necklaces),
        "occupation_histograms_visited": topology.occupation_histograms_visited,
        "chart_parameter_cells": 9,
        "source_chart_membership": source_membership,
        "diagonal_only_chart_membership": diagonal_membership,
        "primary_scattering_chart_membership": forward_membership,
        "alternate_scattering_chart_membership": alternate_membership,
        "primary_output_commitment": state_commitment(forward),
        "alternate_output_commitment": state_commitment(alternate_forward),
        "primary_boundary": primary["boundary"],
        "reuse_boundary": reuse["boundary"],
        "fresh_reuse_boundary": fresh_reuse["boundary"],
        "primary_restoration_error_field_cells": primary[
            "restoration_error_field_cells"
        ],
        "reuse_restoration_error_field_cells": reuse[
            "restoration_error_field_cells"
        ],
        "restored_carrier_commitment": restored_commitment,
        "same_backing_primary": primary["same_backing"],
        "same_backing_reuse": reuse["same_backing"],
        "restoration_generation_after_reuse": carrier.generation,
        "controls": {**controls, "null_carrier_rejected": null_rejected},
        "generator_enumerated_terms": generators[0].enumerated_terms,
        "generator_unique_terms": generators[0].unique_terms,
    }


def main() -> None:
    cases: list[dict[str, object]] = []
    for rotors in ROTOR_COUNTS:
        topology = compile_topology(rotors)
        for prime in PRIMES:
            cases.append(run_case(topology, prime))
    escaped = [
        case
        for case in cases
        if not case["primary_scattering_chart_membership"]["closed"]  # type: ignore[index]
    ]
    alternate_escaped = [
        case
        for case in cases
        if not case["alternate_scattering_chart_membership"]["closed"]  # type: ignore[index]
    ]
    if not escaped or not alternate_escaped:
        raise RuntimeError("Jastrow scattering escape was not observed")
    first_escape_rotors = min(int(case["rotors"]) for case in escaped)
    print(
        json.dumps(
            {
                "claim_candidate": "BOUNDED_EXACT_NINE_PARAMETER_PAIR_SIGNATURE_JASTROW_PHASE_CHART_CLOSES_THE_PUBLIC_DIAGONAL_TWO_BODY_PHASE_AND_IS_DIMENSION_SATURATED_AT_TWO_ROTORS_BUT_ESCAPES_UNDER_OFFDIAGONAL_PAIR_SCATTERING_FROM_THREE_THROUGH_FIVE_ROTORS_WHILE_THE_FULL_NECKLACE_SHEAR_RESTORES_AND_REUSES_EXACTLY_WITH_AN_IDENTICAL_CLASSICAL_RECURRENCE",
                "claim_ceiling": "GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS2_TO5_F103_F239_ONE_PRIMARY_AND_ONE_ALTERNATE_PUBLIC_JASTROW_FAMILY_EXACT_K_TIMES_D_GENERATOR_WORD_DIRECT_PROCESS_SOFTWARE_ONLY",
                "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
                "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
                "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
                "result": "PASS",
                "cases": cases,
                "first_primary_escape_rotor_count": first_escape_rotors,
                "two_rotor_chart_closure_is_dimension_saturation_not_transfer": True,
                "all_diagonal_only_cases_closed": all(
                    case["diagonal_only_chart_membership"]["closed"]  # type: ignore[index]
                    for case in cases
                ),
                "all_rotor_counts_have_primary_and_alternate_escape": all(
                    any(
                        item["rotors"] == rotors
                        and not item["primary_scattering_chart_membership"]["closed"]  # type: ignore[index]
                        and not item["alternate_scattering_chart_membership"]["closed"]  # type: ignore[index]
                        for item in cases
                    )
                    for rotors in ROTOR_COUNTS
                ),
                "resource_law": {
                    "accepted_resident_field_cells": "TWO_TIMES_NECKLACE_DIMENSION",
                    "accepted_word_scratch_field_cells": "TWO_TIMES_NECKLACE_DIMENSION",
                    "accepted_maximum_public_generator_entries": "REPORTED_PER_CASE",
                    "accepted_retained_inverse_history_bytes": 0,
                    "inverse_word_rematerialized_from_public_topology": True,
                    "candidate_chart_parameter_field_cells": 9,
                    "chart_rank_and_discrete_log_matrices": "VERIFICATION_ONLY",
                    "accepted_assignment_or_relation_table_cells": 0,
                },
                "matched_classical_recurrence": "IDENTICAL_FULL_NECKLACE_EXACT_K_TIMES_D_SHEAR_RECURRENCE",
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
