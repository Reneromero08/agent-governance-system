#!/usr/bin/env python3
"""Exact F5 cyclotomic tensor-train carrier with cubic phase coupling."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache


Q = Fraction
ZERO = (Q(0), Q(0), Q(0), Q(0))
ONE = (Q(1), Q(0), Q(0), Q(0))
Element = tuple[Q, Q, Q, Q]
Matrix = list[list[Element]]
Tensor = list[list[list[Element]]]


def element_logical_bytes(value: Element) -> int:
    total = 0
    for coefficient in value:
        total += max(
            1,
            (abs(coefficient.numerator).bit_length() + 8) // 8,
        )
        total += max(
            1, (coefficient.denominator.bit_length() + 7) // 8
        )
    return total


def matrix_logical_bytes(matrix: Matrix) -> int:
    return sum(
        element_logical_bytes(value)
        for row in matrix
        for value in row
    )


def add(left: Element, right: Element) -> Element:
    return tuple(left[i] + right[i] for i in range(4))  # type: ignore[return-value]


def neg(value: Element) -> Element:
    return tuple(-item for item in value)  # type: ignore[return-value]


def sub(left: Element, right: Element) -> Element:
    return add(left, neg(right))


def scale(value: Element, factor: Q) -> Element:
    return tuple(item * factor for item in value)  # type: ignore[return-value]


def mul(left: Element, right: Element) -> Element:
    raw = [Q(0) for _ in range(7)]
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            raw[i + j] += left_value * right_value
    for degree in range(6, 3, -1):
        value = raw[degree]
        if value:
            for offset in range(1, 5):
                raw[degree - offset] -= value
    return tuple(raw[:4])  # type: ignore[return-value]


def power(value: Element, exponent: int) -> Element:
    result = ONE
    factor = value
    while exponent:
        if exponent & 1:
            result = mul(result, factor)
        factor = mul(factor, factor)
        exponent >>= 1
    return result


ZETA = (Q(0), Q(1), Q(0), Q(0))
ROOTS = tuple(power(ZETA, exponent) for exponent in range(5))
SQRT5 = (Q(-1), Q(0), Q(-2), Q(-2))
INV_SQRT5 = scale(SQRT5, Q(1, 5))


@lru_cache(maxsize=None)
def inv(value: Element) -> Element:
    if value == ZERO:
        raise ZeroDivisionError("zero cyclotomic inverse")
    multiplication = [
        list(mul(value, basis))
        for basis in (
            ONE,
            ZETA,
            mul(ZETA, ZETA),
            mul(mul(ZETA, ZETA), ZETA),
        )
    ]
    augmented = [
        [multiplication[column][row] for column in range(4)]
        + [Q(int(row == 0))]
        for row in range(4)
    ]
    for column in range(4):
        pivot = next(
            row
            for row in range(column, 4)
            if augmented[row][column]
        )
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [
            item / divisor for item in augmented[column]
        ]
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][index]
                    - factor * augmented[column][index]
                    for index in range(5)
                ]
    result = tuple(augmented[row][4] for row in range(4))
    if mul(value, result) != ONE:
        raise RuntimeError("cyclotomic inverse verification failed")
    return result  # type: ignore[return-value]


def div(left: Element, right: Element) -> Element:
    return mul(left, inv(right))


def conjugate(value: Element) -> Element:
    result = ZERO
    for exponent, coefficient in enumerate(value):
        if coefficient:
            result = add(
                result,
                scale(ROOTS[(-exponent) % 5], coefficient),
            )
    return result


def matrix_inverse(
    matrix: Matrix, stats: Stats | None = None
) -> Matrix:
    size = len(matrix)
    augmented = [
        list(row)
        + [ONE if row_index == column else ZERO for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            row
            for row in range(column, size)
            if augmented[row][column] != ZERO
        )
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [
            div(item, divisor) for item in augmented[column]
        ]
        if stats is not None:
            stats.factorization_field_divisions += 2 * size
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor != ZERO:
                augmented[row] = [
                    sub(
                        augmented[row][index],
                        mul(factor, augmented[column][index]),
                    )
                    for index in range(2 * size)
                ]
                if stats is not None:
                    stats.factorization_field_multiplications += (
                        2 * size
                    )
                    stats.factorization_field_additions += 2 * size
    return [row[size:] for row in augmented]


def matrix_multiply(
    left: Matrix, right: Matrix, stats: Stats | None = None
) -> Matrix:
    result = [
        [
            sum_elements(
                mul(left[row][inner], right[inner][column])
                for inner in range(len(right))
            )
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]
    if stats is not None:
        terms = len(left) * len(right[0]) * len(right)
        stats.factorization_field_multiplications += terms
        stats.factorization_field_additions += terms
    return result


def sum_elements(values: object) -> Element:
    result = ZERO
    for value in values:  # type: ignore[union-attr]
        result = add(result, value)
    return result


def skeleton_factor(
    matrix: Matrix, stats: Stats | None = None
) -> tuple[Matrix, Matrix, int]:
    rows = len(matrix)
    columns = len(matrix[0])
    work = [list(row) for row in matrix]
    row_order = list(range(rows))
    column_order = list(range(columns))
    rank = 0
    while rank < rows and rank < columns:
        found: tuple[int, int] | None = None
        for row in range(rank, rows):
            for column in range(rank, columns):
                if work[row][column] != ZERO:
                    found = (row, column)
                    break
            if found is not None:
                break
        if found is None:
            break
        pivot_row, pivot_column = found
        work[rank], work[pivot_row] = (
            work[pivot_row],
            work[rank],
        )
        row_order[rank], row_order[pivot_row] = (
            row_order[pivot_row],
            row_order[rank],
        )
        for row in range(rows):
            work[row][rank], work[row][pivot_column] = (
                work[row][pivot_column],
                work[row][rank],
            )
        column_order[rank], column_order[pivot_column] = (
            column_order[pivot_column],
            column_order[rank],
        )
        divisor = work[rank][rank]
        for row in range(rank + 1, rows):
            if work[row][rank] == ZERO:
                continue
            factor = div(work[row][rank], divisor)
            if stats is not None:
                stats.factorization_field_divisions += 1
            for column in range(rank, columns):
                work[row][column] = sub(
                    work[row][column],
                    mul(factor, work[rank][column]),
                )
                if stats is not None:
                    stats.factorization_field_multiplications += 1
                    stats.factorization_field_additions += 1
        rank += 1

    selected_rows = row_order[:rank]
    selected_columns = column_order[:rank]
    cross = [
        [matrix[row][column] for column in selected_columns]
        for row in selected_rows
    ]
    cross_inverse = matrix_inverse(cross, stats)
    columns_matrix = [
        [matrix[row][column] for column in selected_columns]
        for row in range(rows)
    ]
    left = matrix_multiply(columns_matrix, cross_inverse, stats)
    right = [
        list(matrix[row]) for row in selected_rows
    ]
    if matrix_multiply(left, right, stats) != matrix:
        raise RuntimeError("exact skeleton factorization failed")
    if stats is not None:
        stats.maximum_factorization_scratch_logical_bytes = max(
            stats.maximum_factorization_scratch_logical_bytes,
            sum(
                matrix_logical_bytes(item)
                for item in (
                    matrix,
                    work,
                    cross,
                    cross_inverse,
                    columns_matrix,
                    left,
                    right,
                )
            ),
        )
    return left, right, rank


@dataclass
class Stats:
    field_additions: int = 0
    field_multiplications: int = 0
    fourier_terms: int = 0
    cubic_phase_multiplications: int = 0
    factorization_calls: int = 0
    maximum_bond_rank: int = 1
    maximum_live_tensor_cells: int = 0
    maximum_merged_cells: int = 0
    maximum_numerator_bits: int = 0
    maximum_denominator_bits: int = 0
    maximum_logical_coefficient_bytes: int = 0
    boundary_contraction_terms: int = 0
    maximum_factorization_scratch_logical_bytes: int = 0
    factorization_field_additions: int = 0
    factorization_field_multiplications: int = 0
    factorization_field_divisions: int = 0
    restoration_field_inversions: int = 0
    restoration_field_multiplications: int = 0
    restoration_verification_cells: int = 0


@dataclass
class Carrier:
    tensors: list[Tensor]


def product_zero_state(width: int) -> Carrier:
    tensors = []
    for _ in range(width):
        tensor = [[[ZERO] for _ in range(5)]]
        tensor[0][0][0] = ONE
        tensors.append(tensor)
    return Carrier(tensors)


def tensor_shape(tensor: Tensor) -> tuple[int, int, int]:
    return len(tensor), len(tensor[0]), len(tensor[0][0])


def fourier_matrix(inverse: bool) -> Matrix:
    return [
        [
            mul(
                INV_SQRT5,
                ROOTS[((-1 if inverse else 1) * output * source) % 5],
            )
            for source in range(5)
        ]
        for output in range(5)
    ]


def apply_fourier(
    carrier: Carrier, site: int, inverse: bool, stats: Stats
) -> None:
    tensor = carrier.tensors[site]
    left_rank, _, right_rank = tensor_shape(tensor)
    matrix = fourier_matrix(inverse)
    result = [
        [[ZERO for _ in range(right_rank)] for _ in range(5)]
        for _ in range(left_rank)
    ]
    for left in range(left_rank):
        for output in range(5):
            for right in range(right_rank):
                value = ZERO
                for source in range(5):
                    value = add(
                        value,
                        mul(matrix[output][source], tensor[left][source][right]),
                    )
                    stats.field_additions += 1
                    stats.field_multiplications += 1
                    stats.fourier_terms += 1
                result[left][output][right] = value
    carrier.tensors[site] = result


def cubic_exponent(x: int, y: int, gamma: int) -> int:
    return gamma * (x * x * y + x * y * y) % 5


def apply_cubic_gate(
    carrier: Carrier,
    site: int,
    gamma: int,
    stats: Stats,
    rank_cap: int | None = None,
) -> None:
    left_tensor = carrier.tensors[site]
    right_tensor = carrier.tensors[site + 1]
    left_rank, _, shared_rank = tensor_shape(left_tensor)
    shared_right, _, right_rank = tensor_shape(right_tensor)
    if shared_rank != shared_right:
        raise RuntimeError("tensor-train bond mismatch")
    rows = left_rank * 5
    columns = 5 * right_rank
    merged = [[ZERO for _ in range(columns)] for _ in range(rows)]
    for left in range(left_rank):
        for x in range(5):
            row = left * 5 + x
            for y in range(5):
                phase = ROOTS[cubic_exponent(x, y, gamma)]
                for right in range(right_rank):
                    value = ZERO
                    for shared in range(shared_rank):
                        value = add(
                            value,
                            mul(
                                left_tensor[left][x][shared],
                                right_tensor[shared][y][right],
                            ),
                        )
                        stats.field_additions += 1
                        stats.field_multiplications += 1
                    merged[row][y * right_rank + right] = mul(
                        value, phase
                    )
                    stats.field_multiplications += 1
                    stats.cubic_phase_multiplications += 1
    stats.maximum_merged_cells = max(
        stats.maximum_merged_cells, rows * columns
    )
    left_matrix, right_matrix, rank = skeleton_factor(merged, stats)
    if rank_cap is not None and rank > rank_cap:
        raise RuntimeError("exact nonzero pivot exceeds rank cap")
    stats.factorization_calls += 1
    stats.maximum_bond_rank = max(stats.maximum_bond_rank, rank)
    carrier.tensors[site] = [
        [
            [
                left_matrix[left * 5 + physical][bond]
                for bond in range(rank)
            ]
            for physical in range(5)
        ]
        for left in range(left_rank)
    ]
    carrier.tensors[site + 1] = [
        [
            [
                right_matrix[bond][physical * right_rank + right]
                for right in range(right_rank)
            ]
            for physical in range(5)
        ]
        for bond in range(rank)
    ]
    observe(carrier, stats)


def bond_schedule(
    width: int, rounds: int, program: int = 0
) -> list[tuple[str, int, int]]:
    schedule: list[tuple[str, int, int]] = []
    for round_index in range(rounds):
        for site in range(width):
            schedule.append(("F", site, 1))
        parity = round_index % 2
        for site in range(parity, width - 1, 2):
            schedule.append(
                (
                    "D",
                    site,
                    1 + (round_index + 2 * program) % 4,
                )
            )
    return schedule


def apply_operation(
    carrier: Carrier,
    operation: tuple[str, int, int],
    inverse: bool,
    stats: Stats,
) -> None:
    kind, site, parameter = operation
    if kind == "F":
        apply_fourier(carrier, site, inverse, stats)
    else:
        apply_cubic_gate(
            carrier,
            site,
            (-parameter if inverse else parameter) % 5,
            stats,
        )


def observe(carrier: Carrier, stats: Stats) -> None:
    live_cells = 0
    logical_bytes = 0
    for tensor in carrier.tensors:
        left_rank, _, right_rank = tensor_shape(tensor)
        live_cells += left_rank * 5 * right_rank
        for left in tensor:
            for physical in left:
                for value in physical:
                    for coefficient in value:
                        logical_bytes += max(
                            1,
                            (
                                abs(coefficient.numerator).bit_length()
                                + 8
                            )
                            // 8,
                        )
                        logical_bytes += max(
                            1,
                            (
                                coefficient.denominator.bit_length()
                                + 7
                            )
                            // 8,
                        )
                        stats.maximum_numerator_bits = max(
                            stats.maximum_numerator_bits,
                            abs(coefficient.numerator).bit_length(),
                        )
                        stats.maximum_denominator_bits = max(
                            stats.maximum_denominator_bits,
                            coefficient.denominator.bit_length(),
                        )
    stats.maximum_live_tensor_cells = max(
        stats.maximum_live_tensor_cells, live_cells
    )
    stats.maximum_logical_coefficient_bytes = max(
        stats.maximum_logical_coefficient_bytes, logical_bytes
    )


def logical_payload_bytes(carrier: Carrier) -> int:
    total = 0
    for tensor in carrier.tensors:
        for left in tensor:
            for physical in left:
                for value in physical:
                    for coefficient in value:
                        total += max(
                            1,
                            (
                                abs(coefficient.numerator).bit_length()
                                + 8
                            )
                            // 8,
                        )
                        total += max(
                            1,
                            (
                                coefficient.denominator.bit_length()
                                + 7
                            )
                            // 8,
                        )
    return total


def bond_ranks(carrier: Carrier) -> list[int]:
    return [tensor_shape(tensor)[2] for tensor in carrier.tensors[:-1]]


def boundary_amplitude(
    carrier: Carrier, stats: Stats | None = None
) -> Element:
    state = [ONE]
    for site, tensor in enumerate(carrier.tensors):
        bra = [
            mul(
                INV_SQRT5,
                ROOTS[
                    (
                        (site + 1) * physical * physical
                        + 2 * physical
                    )
                    % 5
                ],
            )
            for physical in range(5)
        ]
        left_rank, _, right_rank = tensor_shape(tensor)
        if len(state) != left_rank:
            raise RuntimeError("boundary contraction rank mismatch")
        following = [ZERO for _ in range(right_rank)]
        for left in range(left_rank):
            for physical in range(5):
                for right in range(right_rank):
                    following[right] = add(
                        following[right],
                        mul(
                            mul(state[left], tensor[left][physical][right]),
                            bra[physical],
                        ),
                    )
                    if stats is not None:
                        stats.field_multiplications += 2
                        stats.field_additions += 1
                        stats.boundary_contraction_terms += 1
        state = following
    if len(state) != 1:
        raise RuntimeError("boundary contraction did not close")
    return state[0]


def canonicalize_restored_product(
    carrier: Carrier, stats: Stats | None = None
) -> None:
    for site in range(len(carrier.tensors) - 1):
        tensor = carrier.tensors[site]
        if tensor_shape(tensor) != (1, 5, 1):
            raise RuntimeError("restored carrier rank is not one")
        pivot = next(
            tensor[0][physical][0]
            for physical in range(5)
            if tensor[0][physical][0] != ZERO
        )
        inverse_pivot = inv(pivot)
        if stats is not None:
            stats.restoration_field_inversions += 1
        for physical in range(5):
            tensor[0][physical][0] = mul(
                tensor[0][physical][0], inverse_pivot
            )
            if stats is not None:
                stats.restoration_field_multiplications += 1
        following = carrier.tensors[site + 1]
        for left in range(len(following)):
            for physical in range(5):
                for right in range(len(following[left][physical])):
                    following[left][physical][right] = mul(
                        pivot, following[left][physical][right]
                    )
                    if stats is not None:
                        stats.restoration_field_multiplications += 1


def restored_exact(
    carrier: Carrier, stats: Stats | None = None
) -> bool:
    canonicalize_restored_product(carrier, stats)
    expected = product_zero_state(len(carrier.tensors))
    if stats is not None:
        stats.restoration_verification_cells += sum(
            tensor_shape(tensor)[0]
            * tensor_shape(tensor)[1]
            * tensor_shape(tensor)[2]
            for tensor in carrier.tensors
        )
    return carrier == expected


def safe_restored_exact(carrier: Carrier) -> bool:
    try:
        return restored_exact(carrier)
    except RuntimeError:
        return False


def transaction(
    carrier: Carrier, width: int, rounds: int, program: int = 0
) -> dict[str, object]:
    stats = Stats()
    schedule = bond_schedule(width, rounds, program)
    for operation in schedule:
        apply_operation(carrier, operation, False, stats)
    observe(carrier, stats)
    boundary = boundary_amplitude(carrier, stats)
    forward_ranks = bond_ranks(carrier)
    for operation in reversed(schedule):
        apply_operation(carrier, operation, True, stats)
    restored = restored_exact(carrier, stats)
    if not restored:
        raise RuntimeError("exact TT carrier restoration failed")
    return {
        "width": width,
        "rounds": rounds,
        "public_operations": len(schedule),
        "boundary_numerators": [
            coefficient.numerator for coefficient in boundary
        ],
        "boundary_denominators": [
            coefficient.denominator for coefficient in boundary
        ],
        "forward_bond_ranks": forward_ranks,
        "central_bond_rank": forward_ranks[(width - 2) // 2],
        "maximum_bond_rank": stats.maximum_bond_rank,
        "maximum_live_tensor_cells": stats.maximum_live_tensor_cells,
        "maximum_merged_cells": stats.maximum_merged_cells,
        "maximum_numerator_bits": stats.maximum_numerator_bits,
        "maximum_denominator_bits": stats.maximum_denominator_bits,
        "maximum_logical_coefficient_bytes": (
            stats.maximum_logical_coefficient_bytes
        ),
        "fourier_terms": stats.fourier_terms,
        "cubic_phase_multiplications": (
            stats.cubic_phase_multiplications
        ),
        "factorization_calls": stats.factorization_calls,
        "maximum_factorization_scratch_logical_bytes": (
            stats.maximum_factorization_scratch_logical_bytes
        ),
        "factorization_field_additions": (
            stats.factorization_field_additions
        ),
        "factorization_field_multiplications": (
            stats.factorization_field_multiplications
        ),
        "factorization_field_divisions": (
            stats.factorization_field_divisions
        ),
        "restoration_field_inversions": (
            stats.restoration_field_inversions
        ),
        "restoration_field_multiplications": (
            stats.restoration_field_multiplications
        ),
        "restoration_verification_cells": (
            stats.restoration_verification_cells
        ),
        "boundary_contraction_terms": (
            stats.boundary_contraction_terms
        ),
        "compiled_topology_logical_bytes": len(schedule) * 24,
        "carrier_creation_tensor_cells": width * 5,
        "matched_exact_tt_maximum_bond_rank": (
            stats.maximum_bond_rank
        ),
        "matched_exact_tt_maximum_live_tensor_cells": (
            stats.maximum_live_tensor_cells
        ),
        "matched_exact_tt_maximum_logical_coefficient_bytes": (
            stats.maximum_logical_coefficient_bytes
        ),
        "matched_exact_tt_is_same_as_accepted_representation": True,
        "actual_python_allocation_measured": False,
        "actual_inverse_restoration": restored,
        "retained_inverse_matrices": 0,
        "dense_assignment_cells": 0,
    }


def execute(width: int, rounds: int) -> dict[str, object]:
    return transaction(product_zero_state(width), width, rounds)


def phase_gate_rank(kind: str) -> int:
    if kind == "CUBIC":
        matrix = [
            [ROOTS[cubic_exponent(x, y, 1)] for y in range(5)]
            for x in range(5)
        ]
    elif kind == "BILINEAR":
        matrix = [
            [ROOTS[x * y % 5] for y in range(5)]
            for x in range(5)
        ]
    elif kind == "SEPARABLE":
        matrix = [
            [ROOTS[(x * x + 2 * y) % 5] for y in range(5)]
            for x in range(5)
        ]
    elif kind == "IDENTITY":
        matrix = [[ONE for _ in range(5)] for _ in range(5)]
    else:
        raise RuntimeError("unknown phase gate rank control")
    _, _, rank = skeleton_factor(matrix)
    return rank


def inverse_control(mode: str) -> bool:
    width = 4
    rounds = 2
    schedule = bond_schedule(width, rounds)
    carrier = product_zero_state(width)
    stats = Stats()
    for operation in schedule:
        apply_operation(carrier, operation, False, stats)
    inverse_schedule = list(reversed(schedule))
    if mode == "MISSING":
        inverse_schedule = inverse_schedule[1:]
    elif mode == "REORDERED":
        gate_index = next(
            index
            for index, operation in enumerate(inverse_schedule)
            if operation[0] == "D"
        )
        fourier_index = next(
            index
            for index in range(gate_index + 1, len(inverse_schedule))
            if (
                inverse_schedule[index][0] == "F"
                and inverse_schedule[index][1]
                in (
                    inverse_schedule[gate_index][1],
                    inverse_schedule[gate_index][1] + 1,
                )
            )
        )
        inverse_schedule[gate_index], inverse_schedule[fourier_index] = (
            inverse_schedule[fourier_index],
            inverse_schedule[gate_index],
        )
    for index, operation in enumerate(inverse_schedule):
        if mode == "WRONG" and index == 0 and operation[0] == "D":
            operation = (
                operation[0],
                operation[1],
                operation[2] % 4 + 1,
            )
        apply_operation(carrier, operation, True, stats)
    return safe_restored_exact(carrier)


def fourier_disabled_boundary(width: int, rounds: int) -> Element:
    carrier = product_zero_state(width)
    stats = Stats()
    for operation in bond_schedule(width, rounds):
        if operation[0] == "D":
            apply_operation(carrier, operation, False, stats)
    return boundary_amplitude(carrier)


def forced_rank_cap_rejected() -> bool:
    carrier = product_zero_state(2)
    stats = Stats()
    apply_fourier(carrier, 0, False, stats)
    apply_fourier(carrier, 1, False, stats)
    try:
        apply_cubic_gate(carrier, 0, 1, stats, rank_cap=1)
    except RuntimeError as error:
        return str(error) == "exact nonzero pivot exceeds rank cap"
    return False


def main() -> None:
    if sys.argv[1:] == ["--project-intermediate"]:
        raise RuntimeError(
            "cyclotomic TT intermediate projection denied"
        )
    if sys.argv[1:] == ["--null-carrier"]:
        raise RuntimeError("invalid cyclotomic TT carrier")
    if sys.argv[1:]:
        raise RuntimeError("unsupported cyclotomic TT request")
    fixtures = [(2, 1), (4, 4), (6, 5)]
    results = [execute(width, rounds) for width, rounds in fixtures]
    reuse_carrier = product_zero_state(4)
    primary_reuse_sentinel = transaction(reuse_carrier, 4, 2, 0)
    unrelated_reuse = transaction(reuse_carrier, 4, 3, 1)
    snapshot_image = product_zero_state(4)
    snapshot_carrier = deepcopy(snapshot_image)
    snapshot_baseline_tensor_cells = 4 * 5
    snapshot_baseline_payload_bytes = logical_payload_bytes(
        snapshot_carrier
    )
    snapshot_stats = Stats()
    for operation in bond_schedule(4, 2):
        apply_operation(
            snapshot_carrier, operation, False, snapshot_stats
        )
    snapshot_forward_boundary = boundary_amplitude(
        snapshot_carrier, snapshot_stats
    )
    snapshot_forward_tensor_cells = sum(
        tensor_shape(tensor)[0]
        * tensor_shape(tensor)[1]
        * tensor_shape(tensor)[2]
        for tensor in snapshot_carrier.tensors
    )
    snapshot_carrier = deepcopy(snapshot_image)
    snapshot_reuse = transaction(snapshot_carrier, 4, 3, 1)
    snapshot_forward_boundary_matches = (
        snapshot_forward_boundary
        == tuple(
            Q(numerator, denominator)
            for numerator, denominator in zip(
                primary_reuse_sentinel["boundary_numerators"],
                primary_reuse_sentinel["boundary_denominators"],
            )
        )
    )
    snapshot_actual_inverse = False
    snapshot_restoration_generation = int(
        snapshot_actual_inverse
        and safe_restored_exact(snapshot_carrier)
    )
    controls = {
        "identity_gate_operator_schmidt_rank": phase_gate_rank(
            "IDENTITY"
        ),
        "separable_gate_operator_schmidt_rank": phase_gate_rank(
            "SEPARABLE"
        ),
        "bilinear_clifford_gate_operator_schmidt_rank": phase_gate_rank(
            "BILINEAR"
        ),
        "cubic_gate_operator_schmidt_rank": phase_gate_rank("CUBIC"),
        "missing_inverse_restored": inverse_control("MISSING"),
        "wrong_inverse_restored": inverse_control("WRONG"),
        "reordered_inverse_restored": inverse_control("REORDERED"),
        "forced_rank_cap_rejected": forced_rank_cap_rejected(),
        "snapshot_baseline_tensor_cells": (
            snapshot_baseline_tensor_cells
        ),
        "snapshot_forward_tensor_cells": (
            snapshot_forward_tensor_cells
        ),
        "snapshot_creation_traffic_bytes": (
            snapshot_baseline_payload_bytes
        ),
        "snapshot_reload_traffic_bytes": (
            snapshot_baseline_payload_bytes
        ),
        "snapshot_forward_boundary_matches": (
            snapshot_forward_boundary_matches
        ),
        "snapshot_actual_inverse": snapshot_actual_inverse,
        "snapshot_restoration_generation": (
            snapshot_restoration_generation
        ),
        "snapshot_reuse_restored": snapshot_reuse[
            "actual_inverse_restoration"
        ],
    }
    if not (
        controls["identity_gate_operator_schmidt_rank"] == 1
        and controls["separable_gate_operator_schmidt_rank"] == 1
        and controls["bilinear_clifford_gate_operator_schmidt_rank"]
        == 5
        and controls["cubic_gate_operator_schmidt_rank"] == 4
        and not controls["missing_inverse_restored"]
        and not controls["wrong_inverse_restored"]
        and not controls["reordered_inverse_restored"]
        and controls["forced_rank_cap_rejected"]
        and controls["snapshot_forward_boundary_matches"]
        and not controls["snapshot_actual_inverse"]
        and primary_reuse_sentinel["actual_inverse_restoration"]
        and unrelated_reuse["actual_inverse_restoration"]
        and snapshot_reuse["actual_inverse_restoration"]
        and results[0]["central_bond_rank"]
        < results[1]["central_bond_rank"]
        < results[2]["central_bond_rank"]
        and fourier_disabled_boundary(4, 4)
        != tuple(
            Q(numerator, denominator)
            for numerator, denominator in zip(
                results[1]["boundary_numerators"],
                results[1]["boundary_denominators"],
            )
        )
    ):
        raise RuntimeError("cyclotomic cubic TT control failed")
    print(
        json.dumps(
            {
                "result": "PASS",
                "claim_candidate": (
                    "BOUNDED_EXACT_CYCLOTOMIC_CUBIC_PHASE_FOURIER_"
                    "TENSOR_TRAIN_RANK_GROWTH_WITH_RESTORATION"
                ),
                "field": "Q(ZETA5)",
                "fixtures": results,
                "central_bond_ranks": [
                    result["central_bond_rank"]
                    for result in results
                ],
                "controls": controls,
                "same_carrier_transactions": 2,
                "unrelated_reuse_boundary_numerators": (
                    unrelated_reuse["boundary_numerators"]
                ),
                "unrelated_reuse_boundary_denominators": (
                    unrelated_reuse["boundary_denominators"]
                ),
                "actual_restored_carrier_reuse": True,
                "fourier_disabled_boundary_differs": True,
                "phase_is_primitive_wave_coupling": True,
                "roots_alone_not_claimed_as_resource": True,
                "fixed_rank_closure_established": False,
                "distinct_phase_resource_established": False,
                "computational_advantage": False,
                "small_wall_crossed": False,
                "physical_waveform_execution": False,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ZeroDivisionError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
