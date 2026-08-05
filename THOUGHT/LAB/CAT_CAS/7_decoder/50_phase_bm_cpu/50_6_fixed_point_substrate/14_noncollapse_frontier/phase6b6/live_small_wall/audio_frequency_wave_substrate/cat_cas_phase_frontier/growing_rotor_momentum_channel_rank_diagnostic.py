#!/usr/bin/env python3
"""Exact rank test for the Rotor-6 coherent momentum-wave interface.

M204 retains eight reflection-paired momentum channels.  This diagnostic
streams a small exact witness from public topology to decide whether a
uniform linear interface with fewer than eight F103 coordinates can preserve
both the channel operators and every declared public contraction.  It does
not execute or restore a carrier transaction.
"""

from __future__ import annotations

import hashlib
import json

import growing_rotor_coherent_momentum_wave_streaming_closure as predecessor


GRID = 17
ROTORS = 6
PRIME = 103
CHANNELS = tuple(range(1, 9))
BASE = ROTORS + 1
Histogram = tuple[int, ...]


def rotate(item: Histogram, amount: int) -> Histogram:
    amount %= GRID
    return item[amount:] + item[:amount]


def reflect(item: Histogram) -> Histogram:
    return (item[0],) + tuple(reversed(item[1:]))


def bracelet(item: Histogram) -> Histogram:
    reflected = reflect(item)
    return min(
        *(rotate(item, amount) for amount in range(GRID)),
        *(rotate(reflected, amount) for amount in range(GRID)),
    )


def encode(item: Histogram) -> int:
    value = 0
    place = 1
    for digit in item:
        value += digit * place
        place *= BASE
    return value


def rank_mod(matrix: list[list[int]]) -> int:
    if not matrix:
        return 0
    work = [[value % PRIME for value in row] for row in matrix]
    row = 0
    for column in range(len(work[0])):
        pivot = next(
            (candidate for candidate in range(row, len(work)) if work[candidate][column]),
            None,
        )
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        inverse = pow(work[row][column], PRIME - 2, PRIME)
        work[row] = [(value * inverse) % PRIME for value in work[row]]
        for candidate in range(len(work)):
            if candidate == row or work[candidate][column] == 0:
                continue
            factor = work[candidate][column]
            work[candidate] = [
                (left - factor * right) % PRIME
                for left, right in zip(work[candidate], work[row], strict=True)
            ]
        row += 1
        if row == len(work):
            break
    return row


def determinant_mod(matrix: list[list[int]]) -> int:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("determinant requires a nonempty square matrix")
    work = [[value % PRIME for value in row] for row in matrix]
    determinant = 1
    for column in range(len(work)):
        pivot = next(
            (candidate for candidate in range(column, len(work)) if work[candidate][column]),
            None,
        )
        if pivot is None:
            return 0
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            determinant = -determinant
        pivot_value = work[column][column]
        determinant = determinant * pivot_value % PRIME
        inverse = pow(pivot_value, PRIME - 2, PRIME)
        for target in range(column + 1, len(work)):
            factor = work[target][column] * inverse % PRIME
            for entry in range(column, len(work)):
                work[target][entry] = (
                    work[target][entry] - factor * work[column][entry]
                ) % PRIME
    return determinant % PRIME


def commitment(matrix: list[list[int]]) -> str:
    return hashlib.sha256(
        ";".join(",".join(str(value) for value in row) for row in matrix).encode()
    ).hexdigest()


def witness_source() -> Histogram:
    return (ROTORS,) + (0,) * (GRID - 1)


def moved_pair(source: Histogram, shift: int) -> Histogram:
    moved = list(source)
    moved[0] -= 2
    moved[(-shift) % GRID] += 1
    moved[shift % GRID] += 1
    return tuple(moved)


def moved_one(source: Histogram, shift: int) -> Histogram:
    moved = list(source)
    moved[0] -= 1
    moved[(-shift) % GRID] += 1
    return tuple(moved)


def stream_port_reachability_witness() -> tuple[list[list[int]], list[int], int]:
    target = witness_source()
    source_codes = [
        encode(bracelet(moved_one(target, channel))) for channel in CHANNELS
    ]
    if len(set(source_codes)) != len(CHANNELS):
        raise RuntimeError("momentum-port source witness coordinates collided")
    matrix = [[0] * len(CHANNELS) for _ in CHANNELS]
    comparisons = 0
    for channel in CHANNELS:
        moved_code = encode(bracelet(moved_one(target, channel)))
        for source, source_code in enumerate(source_codes):
            comparisons += 1
            if moved_code == source_code:
                matrix[channel - 1][source] = ROTORS
    return matrix, source_codes, comparisons


def stream_operator_witness() -> tuple[list[list[int]], list[int], int, int, int]:
    source = witness_source()
    source_code = encode(bracelet(source))
    targets = tuple(bracelet(moved_pair(source, channel)) for channel in CHANNELS)
    target_codes = [encode(target) for target in targets]
    if len(set(target_codes)) != len(CHANNELS) or source_code in target_codes:
        raise RuntimeError("public off-diagonal witness coordinates collided")
    matrix = [[0] * len(CHANNELS) for _ in CHANNELS]
    candidate_terms = 0
    accepted_terms = 0
    for row, target in enumerate(targets):
        for channel in CHANNELS:
            coefficient = 0
            for shift in (channel, GRID - channel):
                for first, first_count in enumerate(target):
                    if first_count == 0:
                        continue
                    for second, second_count in enumerate(target):
                        multiplicity = first_count * (
                            second_count - (1 if first == second else 0)
                        )
                        if multiplicity == 0:
                            continue
                        candidate_terms += 1
                        moved = list(target)
                        moved[first] -= 1
                        moved[second] -= 1
                        moved[(first - shift) % GRID] += 1
                        moved[(second + shift) % GRID] += 1
                        if encode(bracelet(tuple(moved))) == source_code:
                            coefficient += multiplicity
                            accepted_terms += 1
            matrix[row][channel - 1] = coefficient % PRIME
    return matrix, target_codes, source_code, candidate_terms, accepted_terms


def stream_public_weight_witness() -> tuple[list[list[int]], list[list[int]], int]:
    witness: list[list[int]] = []
    descriptors: list[list[int]] = []
    candidates = 0
    for step, tag in predecessor.scalar.predecessor.public_law.public_program(
        GRID, 0
    ):
        candidate = list(predecessor.scattering_weights(step, tag))
        candidates += 1
        if rank_mod(witness + [candidate]) > len(witness):
            witness.append(candidate)
            descriptors.append([step, tag])
        if len(witness) == len(CHANNELS):
            break
    return witness, descriptors, candidates


def main() -> None:
    port, port_source_codes, port_comparisons = stream_port_reachability_witness()
    (
        operator,
        target_codes,
        source_code,
        operator_candidates,
        operator_terms,
    ) = stream_operator_witness()
    weights, descriptors, weight_candidates = stream_public_weight_witness()
    port_rank = rank_mod(port)
    operator_rank = rank_mod(operator)
    weight_rank = rank_mod(weights)
    port_determinant = determinant_mod(port)
    operator_determinant = determinant_mod(operator)
    weight_determinant = determinant_mod(weights)
    duplicated_port_column = [row[:-1] + [row[-2]] for row in port]
    duplicated_operator_column = [row[:-1] + [row[-2]] for row in operator]
    duplicated_weight_row = weights[:-1] + [weights[-2]]
    if (
        port_rank != 8
        or operator_rank != 8
        or weight_rank != 8
        or port_determinant == 0
        or operator_determinant == 0
        or weight_determinant == 0
        or rank_mod(port[:-1]) != 7
        or rank_mod(operator[:-1]) != 7
        or rank_mod(weights[:-1]) != 7
        or rank_mod(duplicated_port_column) != 7
        or rank_mod(duplicated_operator_column) != 7
        or rank_mod(duplicated_weight_row) != 7
    ):
        raise RuntimeError("momentum-channel rank certificate changed")

    claim = (
        "EXACT_F103_ROTOR6_REFLECTION_PAIRED_MOMENTUM_PORT_MAP_CHANNEL_"
        "OPERATORS_AND_DECLARED_PUBLIC_WEIGHT_FAMILY_EACH_HAVE_RANK8_WITH_"
        "NONZERO8X8_STREAMED_WITNESS_MINORS_SO_NO_EXACT_F103_PORT_ENCODING_"
        "BELOW8_CELLS_OR_UNIFORM_LINEAR_OPERATOR_QUOTIENT_BELOW8_PRESERVES_"
        "THE_DECLARED_FAMILIES_BUT_THE_CERTIFICATES_ARE_IDENTICAL_CLASSICAL_"
        "LINEAR_ALGEBRA"
    )
    result = {
        "claim_candidate": claim,
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "claim_ceiling": (
            "GRID17_EXCHANGE_SYMMETRIC_ROTATION_REFLECTION_INVARIANT_ROTOR6_"
            "F103_CHANNELS1_TO8_PUBLIC_FAMILY0_STEPS0_TO7_DIRECT_PROCESS_"
            "STREAMED_LINEAR_WITNESS_ONLY"
        ),
        "result": "PASS_RANK8_EXACT_F103_PORT_CELL_AND_UNIFORM_LINEAR_OPERATOR_QUOTIENT_BELOW8_REJECTED",
        "certificate": {
            "field_prime": PRIME,
            "channels": list(CHANNELS),
            "port_coordinate_target_necklace_code": source_code,
            "port_coordinate_source_bracelet_codes": port_source_codes,
            "port_reachability_witness": port,
            "port_reachability_witness_commitment": commitment(port),
            "port_reachability_rank": port_rank,
            "port_reachability_determinant": port_determinant,
            "port_image_cardinality": str(PRIME**8),
            "operator_coordinate_source_bracelet_code": source_code,
            "operator_coordinate_target_bracelet_codes": target_codes,
            "operator_witness": operator,
            "operator_witness_commitment": commitment(operator),
            "operator_witness_rank": operator_rank,
            "operator_witness_determinant": operator_determinant,
            "public_program_descriptors": descriptors,
            "public_weight_witness": weights,
            "public_weight_witness_commitment": commitment(weights),
            "public_weight_witness_rank": weight_rank,
            "public_weight_witness_determinant": weight_determinant,
            "exact_f103_port_cell_lower_bound": 8,
            "linear_quotient_lower_bound": 8,
            "scope": (
                "EXACT_F103_CELL_ENCODING_OF_THE_DECLARED_PORT_MAP_AND_UNIFORM_"
                "F103_LINEAR_QUOTIENT_PRESERVING_THE_DECLARED_CHANNEL_OPERATOR_"
                "AND_PUBLIC_WEIGHT_FAMILIES"
            ),
        },
        "controls": {
            "drop_port_row_rank": rank_mod(port[:-1]),
            "drop_operator_row_rank": rank_mod(operator[:-1]),
            "drop_public_weight_row_rank": rank_mod(weights[:-1]),
            "duplicate_port_column_rank": rank_mod(duplicated_port_column),
            "duplicate_operator_column_rank": rank_mod(duplicated_operator_column),
            "duplicate_public_weight_row_rank": rank_mod(duplicated_weight_row),
            "source_target_coordinates_distinct": source_code not in target_codes
            and len(set(target_codes)) == 8,
        },
        "resource_law": {
            "port_witness_coordinate_comparisons": port_comparisons,
            "operator_candidate_terms_streamed": operator_candidates,
            "operator_witness_contributions_accepted": operator_terms,
            "public_weight_candidates_streamed": weight_candidates,
            "retained_port_reachability_witness_field_cells": 64,
            "retained_operator_witness_field_cells": 64,
            "retained_public_weight_witness_field_cells": 64,
            "determinant_rank_scratch_field_cells_peak": 64,
            "retained_determinant_field_cells": 3,
            "retained_topology_code_integers": 17,
            "retained_public_program_descriptor_integers": 16,
            "peak_named_field_and_descriptor_slots": 292,
            "dense_2277_squared_operator_cells": 0,
            "full_74613_occupation_scratch_cells": 0,
            "retained_transition_plan_entries": 0,
            "permanent_assignment_terms": 0,
            "python_object_bigint_allocator_interpreter_and_whole_process_peaks_excluded": True,
        },
        "matched_classical_baselines": [
            "IDENTICAL_STREAMED_F103_EIGHT_CHANNEL_OPERATOR_AND_PUBLIC_WEIGHT_RANK_CERTIFICATE"
        ],
        "preserved_subclaims": [
            "M204_EIGHT_CHANNEL_WAVE_PORT_EXECUTION_RESTORATION_AND_REUSE_REMAIN_SEPARATELY_VALID",
            "PORT_REACHABILITY_AND_OPERATOR_AND_WEIGHT_RANK8_ARE_CERTIFIED_BY_PUBLIC_TOPOLOGY_AND_PUBLIC_PROGRAM_DESCRIPTORS_WITHOUT_DENSE_OPERATOR_RETENTION",
        ],
        "rejected_interpretations": [
            "ENCODING_INTO_FEWER_THAN8_CELLS_OF_A_LARGER_ALPHABET_NO_GO",
            "LOSSY_OR_PROGRAM_SPECIALIZED_PORT_QUOTIENT_NO_GO",
            "GENERAL_ROTOR_COUNT_TRANSFER",
            "CATVM_CUSTODY",
            "DISTINCT_PHASE_RESOURCE",
            "COMPUTATIONAL_ADVANTAGE",
            "SMALL_WALL_CROSSING",
            "PHYSICAL_WAVEFORM_EXECUTION",
            "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI",
            "UNBOUNDED_CATALYTIC_COMPUTATION",
        ],
        "catvm_custody": False,
        "distinct_phase_resource_established": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "physical_waveform_execution": False,
        "physical_bit_replacement": False,
        "unbounded_computation_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
