#!/usr/bin/env python3
"""Independent oracle for the Rotor-6 eight-channel rank diagnostic.

This implementation imports no CAT_CAS module.  It enumerates the complete
occupation topology, derives the witness once from direct two-body action and
again from one-body factor action, enumerates the declared public programs,
and evaluates the two witness determinants by a permutation formula.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator


GRID = 17
ROTORS = 6
PRIME = 103
CHANNELS = tuple(range(1, 9))
BASE = ROTORS + 1
Histogram = tuple[int, ...]


def histograms(total: int, width: int = GRID) -> Iterator[Histogram]:
    def visit(remaining: int, slots: int, prefix: tuple[int, ...]) -> Iterator[Histogram]:
        if slots == 1:
            yield prefix + (remaining,)
            return
        for value in range(remaining + 1):
            yield from visit(remaining - value, slots - 1, prefix + (value,))

    yield from visit(total, width, ())


def rotate(item: Histogram, amount: int) -> Histogram:
    amount %= GRID
    return item[amount:] + item[:amount]


def reflect(item: Histogram) -> Histogram:
    return (item[0],) + tuple(reversed(item[1:]))


def cyclic(item: Histogram) -> Histogram:
    return min(rotate(item, amount) for amount in range(GRID))


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
    work = [[entry % PRIME for entry in row] for row in matrix]
    row = 0
    for column in range(len(work[0])):
        pivot = next(
            (index for index in range(row, len(work)) if work[index][column]),
            None,
        )
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        inverse = pow(work[row][column], PRIME - 2, PRIME)
        work[row] = [(entry * inverse) % PRIME for entry in work[row]]
        for index in range(row + 1, len(work)):
            factor = work[index][column]
            if factor:
                work[index] = [
                    (left - factor * right) % PRIME
                    for left, right in zip(work[index], work[row], strict=True)
                ]
        row += 1
        if row == len(work):
            break
    return row


def permutation_determinant_mod(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("permutation determinant requires a square matrix")
    total = 0
    for permutation in itertools.permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        term = 1
        for row, column in enumerate(permutation):
            term = term * matrix[row][column] % PRIME
        total += -term if inversions % 2 else term
    return total % PRIME


def commitment(matrix: list[list[int]]) -> str:
    return hashlib.sha256(
        ";".join(",".join(str(value) for value in row) for row in matrix).encode()
    ).hexdigest()


def witness_source() -> Histogram:
    return (ROTORS,) + (0,) * (GRID - 1)


def witness_target(channel: int) -> Histogram:
    target = list(witness_source())
    target[0] -= 2
    target[channel] += 1
    target[-channel] += 1
    return bracelet(tuple(target))


def port_source(channel: int) -> Histogram:
    source = list(witness_source())
    source[0] -= 1
    source[-channel] += 1
    return bracelet(tuple(source))


def port_reachability_witness() -> tuple[list[list[int]], list[int], int]:
    source_codes = [encode(port_source(channel)) for channel in CHANNELS]
    matrix = [[0] * len(CHANNELS) for _ in CHANNELS]
    comparisons = 0
    target = witness_source()
    for channel in CHANNELS:
        for source, source_code in enumerate(source_codes):
            value = 0
            for mode, count in enumerate(target):
                if count == 0:
                    continue
                moved = list(target)
                moved[mode] -= 1
                moved[(mode - channel) % GRID] += 1
                comparisons += 1
                if encode(bracelet(tuple(moved))) == source_code:
                    value += count
            matrix[channel - 1][source] = value % PRIME
    return matrix, source_codes, comparisons


def direct_entry(target: Histogram, source_code: int, channel: int) -> tuple[int, int, int]:
    coefficient = 0
    candidates = 0
    accepted = 0
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
                candidates += 1
                moved = list(target)
                moved[first] -= 1
                moved[second] -= 1
                moved[(first - shift) % GRID] += 1
                moved[(second + shift) % GRID] += 1
                if encode(bracelet(tuple(moved))) == source_code:
                    coefficient += multiplicity
                    accepted += 1
    return coefficient % PRIME, candidates, accepted


def first_factor_value(target: Histogram, source_code: int, momentum: int) -> int:
    value = 0
    for mode, count in enumerate(target):
        if count == 0:
            continue
        moved = list(target)
        moved[mode] -= 1
        moved[(mode - momentum) % GRID] += 1
        if encode(bracelet(tuple(moved))) == source_code:
            value += count
    return value


def factor_entry(target: Histogram, source_code: int, momentum: int) -> int:
    closed = 0
    for mode, count in enumerate(target):
        if count == 0:
            continue
        middle = list(target)
        middle[mode] -= 1
        middle[(mode + momentum) % GRID] += 1
        closed += count * first_factor_value(tuple(middle), source_code, momentum)
    return closed % PRIME


def operator_witness() -> tuple[list[list[int]], list[list[int]], list[int], int, int, int]:
    source_code = encode(bracelet(witness_source()))
    targets = tuple(witness_target(channel) for channel in CHANNELS)
    target_codes = [encode(target) for target in targets]
    direct = []
    factor = []
    candidates = 0
    accepted = 0
    for target in targets:
        direct_row = []
        factor_row = []
        for channel in CHANNELS:
            entry, entry_candidates, entry_accepted = direct_entry(
                target, source_code, channel
            )
            direct_row.append(entry)
            factor_row.append(
                (
                    factor_entry(target, source_code, channel)
                    + factor_entry(target, source_code, GRID - channel)
                )
                % PRIME
            )
            candidates += entry_candidates
            accepted += entry_accepted
        direct.append(direct_row)
        factor.append(factor_row)
    return direct, factor, target_codes, source_code, candidates, accepted


def public_program(depth: int, family: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, (family + 3 * step + step * step) % 7) for step in range(depth)
    )


def public_weight(channel: int, step: int, tag: int) -> int:
    distance = min(channel % GRID, GRID - channel % GRID)
    magnitude = 1 + (
        (distance + 2) * (step + 1) + (3 * distance + 1) * (tag + 2)
    ) % GRID % 5
    return -magnitude if (distance + step + tag) % GRID % 3 == 0 else magnitude


def public_weight_witness() -> tuple[list[list[int]], list[list[int]], int, int]:
    witness: list[list[int]] = []
    descriptors: list[list[int]] = []
    candidates = 0
    for step, tag in public_program(GRID, 0):
        row = [public_weight(channel, step, tag) for channel in CHANNELS]
        candidates += 1
        if rank_mod(witness + [row]) > len(witness):
            witness.append(row)
            descriptors.append([step, tag])
        if len(witness) == len(CHANNELS):
            break
    all_declared_rows = [
        [public_weight(channel, step, tag) for channel in CHANNELS]
        for family in range(7)
        for step, tag in public_program(GRID, family)
    ]
    return witness, descriptors, candidates, rank_mod(all_declared_rows)


def main() -> None:
    occupations = tuple(histograms(ROTORS))
    necklaces = tuple(item for item in occupations if cyclic(item) == item)
    bracelets = tuple(sorted({bracelet(item) for item in necklaces}))
    bracelet_codes = {encode(item) for item in bracelets}
    port, port_source_codes, port_comparisons = port_reachability_witness()
    (
        direct,
        factor,
        target_codes,
        source_code,
        operator_candidates,
        operator_accepted,
    ) = operator_witness()
    weights, descriptors, weight_candidates, full_public_rank = public_weight_witness()
    port_rank = rank_mod(port)
    operator_rank = rank_mod(direct)
    weight_rank = rank_mod(weights)
    port_determinant = permutation_determinant_mod(port)
    operator_determinant = permutation_determinant_mod(direct)
    weight_determinant = permutation_determinant_mod(weights)
    duplicated_port_column = [row[:-1] + [row[-2]] for row in port]
    duplicated_operator_column = [row[:-1] + [row[-2]] for row in direct]
    duplicated_weight_row = weights[:-1] + [weights[-2]]
    if (
        len(occupations) != 74613
        or len(necklaces) != 4389
        or len(bracelets) != 2277
        or source_code not in bracelet_codes
        or any(code not in bracelet_codes for code in port_source_codes)
        or any(code not in bracelet_codes for code in target_codes)
        or len(set(port_source_codes)) != 8
        or len(set(target_codes)) != 8
        or direct != factor
        or port_rank != 8
        or operator_rank != 8
        or weight_rank != 8
        or full_public_rank != 8
        or port_determinant != 98
        or operator_determinant != 50
        or weight_determinant != 80
        or rank_mod(port[:-1]) != 7
        or rank_mod(direct[:-1]) != 7
        or rank_mod(weights[:-1]) != 7
        or rank_mod(duplicated_port_column) != 7
        or rank_mod(duplicated_operator_column) != 7
        or rank_mod(duplicated_weight_row) != 7
    ):
        raise RuntimeError("independent momentum-channel rank oracle changed")

    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "result": "PASS_RANK8_EXACT_F103_PORT_CELL_AND_UNIFORM_LINEAR_OPERATOR_QUOTIENT_BELOW8_REJECTED",
        "independent_topology": {
            "occupations_enumerated": len(occupations),
            "necklaces_enumerated": len(necklaces),
            "bracelets_enumerated": len(bracelets),
            "source_bracelet_code": source_code,
            "port_source_bracelet_codes": port_source_codes,
            "target_bracelet_codes": target_codes,
            "all_witness_coordinates_present": True,
            "full_topology_is_verification_only": True,
        },
        "independent_certificate": {
            "port_reachability_witness": port,
            "port_reachability_witness_commitment": commitment(port),
            "port_reachability_rank": port_rank,
            "port_reachability_permutation_determinant": port_determinant,
            "port_image_cardinality": str(PRIME**8),
            "direct_operator_witness": direct,
            "factor_operator_witness": factor,
            "direct_factor_witness_agreement": True,
            "operator_witness_commitment": commitment(direct),
            "operator_rank": operator_rank,
            "operator_permutation_determinant": operator_determinant,
            "public_program_descriptors": descriptors,
            "public_weight_witness": weights,
            "public_weight_witness_commitment": commitment(weights),
            "public_weight_rank": weight_rank,
            "all_seven_families_seventeen_steps_rank": full_public_rank,
            "public_weight_permutation_determinant": weight_determinant,
            "exact_f103_port_cell_lower_bound": 8,
            "linear_quotient_lower_bound": 8,
        },
        "controls": {
            "drop_port_row_rank": rank_mod(port[:-1]),
            "drop_operator_row_rank": rank_mod(direct[:-1]),
            "drop_public_weight_row_rank": rank_mod(weights[:-1]),
            "duplicate_port_column_rank": rank_mod(duplicated_port_column),
            "duplicate_operator_column_rank": rank_mod(duplicated_operator_column),
            "duplicate_public_weight_row_rank": rank_mod(duplicated_weight_row),
        },
        "resource_derivation": {
            "accepted_port_witness_coordinate_comparisons": port_comparisons,
            "accepted_operator_candidate_terms_streamed": operator_candidates,
            "accepted_operator_witness_contributions": operator_accepted,
            "accepted_public_weight_candidates_streamed": weight_candidates,
            "accepted_peak_named_field_and_descriptor_slots": 292,
            "accepted_dense_operator_cells": 0,
            "accepted_occupation_scratch_cells": 0,
            "accepted_transition_plan_entries": 0,
            "verification_only_occupation_cells": len(occupations),
            "verification_only_necklace_cells": len(necklaces),
            "verification_only_bracelet_cells": len(bracelets),
            "python_objects_allocator_interpreter_and_whole_process_peaks_excluded": True,
        },
        "production_module_imported": False,
        "matched_classical_baseline": (
            "IDENTICAL_STREAMED_F103_EIGHT_CHANNEL_OPERATOR_AND_PUBLIC_WEIGHT_"
            "RANK_CERTIFICATE"
        ),
        "nonlinear_or_program_specialized_quotients_rejected": False,
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
