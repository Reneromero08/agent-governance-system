#!/usr/bin/env python3
"""Independent dual-field TT certificate for the cubic F5 phase circuit."""

from __future__ import annotations

import json

import cyclotomic_f5_cubic_tt_phase as exact


PRIMES = ((11, 3), (31, 2))
Matrix = list[list[int]]
Tensor = list[list[list[int]]]


def inverse(value: int, prime: int) -> int:
    return pow(value, prime - 2, prime)


def matrix_inverse(matrix: Matrix, prime: int) -> Matrix:
    size = len(matrix)
    augmented = [
        [value % prime for value in matrix_row]
        + [int(row_index == column) for column in range(size)]
        for row_index, matrix_row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            row
            for row in range(column, size)
            if augmented[row][column] % prime
        )
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        factor = inverse(augmented[column][column], prime)
        augmented[column] = [
            value * factor % prime
            for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    (
                        augmented[row][index]
                        - factor * augmented[column][index]
                    )
                    % prime
                    for index in range(2 * size)
                ]
    return [row[size:] for row in augmented]


def matrix_multiply(
    left: Matrix, right: Matrix, prime: int
) -> Matrix:
    return [
        [
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(len(right))
            )
            % prime
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def skeleton(
    matrix: Matrix, prime: int
) -> tuple[Matrix, Matrix, int]:
    rows = len(matrix)
    columns = len(matrix[0])
    work = [
        [value % prime for value in row] for row in matrix
    ]
    row_order = list(range(rows))
    column_order = list(range(columns))
    rank = 0
    while rank < rows and rank < columns:
        found = next(
            (
                (row, column)
                for row in range(rank, rows)
                for column in range(rank, columns)
                if work[row][column]
            ),
            None,
        )
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
        divisor_inverse = inverse(work[rank][rank], prime)
        for row in range(rank + 1, rows):
            factor = work[row][rank] * divisor_inverse % prime
            if factor:
                for column in range(rank, columns):
                    work[row][column] = (
                        work[row][column]
                        - factor * work[rank][column]
                    ) % prime
        rank += 1
    selected_rows = row_order[:rank]
    selected_columns = column_order[:rank]
    cross = [
        [matrix[row][column] % prime for column in selected_columns]
        for row in selected_rows
    ]
    left = matrix_multiply(
        [
            [
                matrix[row][column] % prime
                for column in selected_columns
            ]
            for row in range(rows)
        ],
        matrix_inverse(cross, prime),
        prime,
    )
    right = [
        [value % prime for value in matrix[row]]
        for row in selected_rows
    ]
    if matrix_multiply(left, right, prime) != [
        [value % prime for value in row] for row in matrix
    ]:
        raise RuntimeError("modular skeleton reconstruction failed")
    return left, right, rank


def zero_state(width: int) -> list[Tensor]:
    tensors = []
    for _ in range(width):
        tensor = [[[0] for _ in range(5)]]
        tensor[0][0][0] = 1
        tensors.append(tensor)
    return tensors


def fourier(
    tensors: list[Tensor],
    site: int,
    inverse_transform: bool,
    prime: int,
    root: int,
) -> None:
    tensor = tensors[site]
    left_rank = len(tensor)
    right_rank = len(tensor[0][0])
    sqrt5 = (
        -1 - 2 * root**2 - 2 * root**3
    ) % prime
    normalization = inverse(sqrt5, prime)
    sign = -1 if inverse_transform else 1
    result = [
        [[0 for _ in range(right_rank)] for _ in range(5)]
        for _ in range(left_rank)
    ]
    for left in range(left_rank):
        for output in range(5):
            for right in range(right_rank):
                result[left][output][right] = sum(
                    (
                        normalization
                        * pow(
                            root,
                            (sign * output * source) % 5,
                            prime,
                        )
                        * tensor[left][source][right]
                    )
                    for source in range(5)
                ) % prime
    tensors[site] = result


def cubic(
    tensors: list[Tensor],
    site: int,
    gamma: int,
    prime: int,
    root: int,
) -> int:
    left_tensor = tensors[site]
    right_tensor = tensors[site + 1]
    left_rank = len(left_tensor)
    shared_rank = len(left_tensor[0][0])
    right_rank = len(right_tensor[0][0])
    matrix = [
        [0 for _ in range(5 * right_rank)]
        for _ in range(5 * left_rank)
    ]
    for left in range(left_rank):
        for x in range(5):
            for y in range(5):
                phase = pow(
                    root,
                    exact.cubic_exponent(x, y, gamma),
                    prime,
                )
                for right in range(right_rank):
                    matrix[left * 5 + x][y * right_rank + right] = (
                        phase
                        * sum(
                            left_tensor[left][x][shared]
                            * right_tensor[shared][y][right]
                            for shared in range(shared_rank)
                        )
                    ) % prime
    left_matrix, right_matrix, rank = skeleton(matrix, prime)
    tensors[site] = [
        [
            [
                left_matrix[left * 5 + physical][bond]
                for bond in range(rank)
            ]
            for physical in range(5)
        ]
        for left in range(left_rank)
    ]
    tensors[site + 1] = [
        [
            [
                right_matrix[bond][physical * right_rank + right]
                for right in range(right_rank)
            ]
            for physical in range(5)
        ]
        for bond in range(rank)
    ]
    return rank


def boundary(
    tensors: list[Tensor], prime: int, root: int
) -> int:
    sqrt5 = (-1 - 2 * root**2 - 2 * root**3) % prime
    normalization = inverse(sqrt5, prime)
    state = [1]
    for site, tensor in enumerate(tensors):
        right_rank = len(tensor[0][0])
        following = [0 for _ in range(right_rank)]
        for left in range(len(tensor)):
            for physical in range(5):
                bra = (
                    normalization
                    * pow(
                        root,
                        (
                            (site + 1) * physical * physical
                            + 2 * physical
                        )
                        % 5,
                        prime,
                    )
                ) % prime
                for right in range(right_rank):
                    following[right] = (
                        following[right]
                        + state[left]
                        * tensor[left][physical][right]
                        * bra
                    ) % prime
        state = following
    return state[0]


def map_exact_boundary(
    result: dict[str, object], prime: int, root: int
) -> int:
    numerators = result["boundary_numerators"]
    denominators = result["boundary_denominators"]
    return sum(
        int(numerators[exponent])
        * inverse(int(denominators[exponent]) % prime, prime)
        * pow(root, exponent, prime)
        for exponent in range(4)
    ) % prime


def execute(
    width: int, rounds: int, prime: int, root: int
) -> tuple[list[int], int]:
    tensors = zero_state(width)
    for kind, site, parameter in exact.bond_schedule(width, rounds):
        if kind == "F":
            fourier(tensors, site, False, prime, root)
        else:
            cubic(tensors, site, parameter, prime, root)
    ranks = [len(tensor[0][0]) for tensor in tensors[:-1]]
    return ranks, boundary(tensors, prime, root)


def main() -> None:
    fixtures = [(2, 1), (4, 4), (6, 5)]
    exact_results = [
        exact.execute(width, rounds) for width, rounds in fixtures
    ]
    fields = []
    for prime, root in PRIMES:
        records = []
        for (width, rounds), exact_result in zip(
            fixtures, exact_results
        ):
            ranks, observed_boundary = execute(
                width, rounds, prime, root
            )
            expected_boundary = map_exact_boundary(
                exact_result, prime, root
            )
            if (
                ranks != exact_result["forward_bond_ranks"]
                or observed_boundary != expected_boundary
            ):
                raise RuntimeError(
                    "independent modular TT certificate mismatch"
                )
            records.append(
                {
                    "width": width,
                    "rounds": rounds,
                    "bond_ranks": ranks,
                    "boundary_residue": observed_boundary,
                }
            )
        fields.append(
            {
                "prime": prime,
                "primitive_fifth_root": root,
                "fixtures": records,
            }
        )
    print(
        json.dumps(
            {
                "result": "PASS",
                "oracle": (
                    "INDEPENDENT_DUAL_FINITE_FIELD_TENSOR_TRAIN_"
                    "RANK_AND_BOUNDARY_CERTIFICATE"
                ),
                "fields": fields,
                "central_bond_ranks": [4, 14, 64],
                "dense_assignment_expansion": False,
                "modular_rank_lower_bounds_exact_rank": True,
                "terminal": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
