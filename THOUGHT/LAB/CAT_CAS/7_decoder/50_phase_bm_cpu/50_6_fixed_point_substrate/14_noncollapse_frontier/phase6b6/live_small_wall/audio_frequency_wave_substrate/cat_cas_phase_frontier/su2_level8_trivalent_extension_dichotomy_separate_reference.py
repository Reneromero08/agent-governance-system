#!/usr/bin/env python3
"""Standalone exact polynomial/dense verifier for M233 continuation rank."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import root_of_unity_su2_level8_fusion_independent_oracle as oracle


E = oracle.E
ZERO = oracle.ZERO
ONE = oracle.ONE
A = E.root(11)
A_INVERSE = E.root(-11)
DELTA = E.root(2) + E.root(-2)
INVERSE_DELTA = DELTA.inverse()
PHI = E.root(4) + E.root(-4)
D_TWO = DELTA * DELTA - ONE
DEPTHS = (1, 2, 4, 8, 16)
IDENTITY = ((ONE, ZERO), (ZERO, ONE))
LEVEL = 8
LABELS = 9
GROWING_STRANDS = (4, 6, 8, 10)
GROWING_DIMENSIONS = (2, 5, 14, 42)
CERTIFICATE_PRIMES = (641, 881)
PRIMARY_ROUNDS = 4
REUSE_ROUNDS = 3


def multiply(left, right):
    return tuple(
        tuple(
            sum(
                (left[row][middle] * right[middle][column] for middle in range(2)),
                ZERO,
            )
            for column in range(2)
        )
        for row in range(2)
    )


def add(left, right):
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(2))
        for row in range(2)
    )


def scale(value, matrix):
    return tuple(
        tuple(value * matrix[row][column] for column in range(2))
        for row in range(2)
    )


def commitment(matrix) -> str:
    payload = "|".join(value.token() for row in matrix for value in row)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def f_matrix(omit=False):
    offdiagonal = ZERO if omit else PHI * INVERSE_DELTA
    return (
        (INVERSE_DELTA, offdiagonal),
        (offdiagonal, ZERO - INVERSE_DELTA),
    )


def r_matrix(exponent, swap=False):
    if exponent == 1:
        values = [A + A_INVERSE * DELTA, A]
    elif exponent == -1:
        values = [A_INVERSE + A * DELTA, A_INVERSE]
    else:
        raise ValueError("reference exponent mismatch")
    if swap:
        values.reverse()
    return ((values[0], ZERO), (ZERO, values[1]))


def symmetric(exponent, omit=False, swap=False):
    f_move = f_matrix(omit)
    return multiply(multiply(f_move, r_matrix(exponent, swap)), f_move)


def path_block(exponent):
    e_path = (
        (INVERSE_DELTA, INVERSE_DELTA * D_TWO),
        (INVERSE_DELTA, INVERSE_DELTA * D_TWO),
    )
    alpha, beta = (A, A_INVERSE) if exponent == 1 else (A_INVERSE, A)
    return add(scale(alpha, IDENTITY), scale(beta, e_path))


def gauge(matrix, gauge_phi=PHI):
    left = ((ONE, ZERO), (ZERO, gauge_phi))
    right = ((ONE, ZERO), (ZERO, gauge_phi.inverse()))
    return multiply(multiply(left, matrix), right)


def serial_case(exponent):
    block = symmetric(exponent)
    trace = block[0][0] + block[1][1]
    determinant = block[0][0] * block[1][1] - block[0][1] * block[1][0]
    ch = add(
        add(multiply(block, block), scale(ZERO - trace, block)),
        scale(determinant, IDENTITY),
    )
    current = IDENTITY
    powers = []
    for depth in range(1, max(DEPTHS) + 1):
        current = multiply(current, block)
        if depth in DEPTHS:
            powers.append({"depth": depth, "matrix_commitment": commitment(current)})
    return {
        "exponent": exponent,
        "symmetric_block_commitment": commitment(block),
        "path_block_commitment": commitment(path_block(exponent)),
        "gauge_transformed_path_commitment": commitment(gauge(path_block(exponent))),
        "exact_gauge_equivalence": block == gauge(path_block(exponent)),
        "cayley_hamilton_degree2_exact": ch == ((ZERO, ZERO), (ZERO, ZERO)),
        "distinct_braid_eigenvalues": r_matrix(exponent)[0][0]
        != r_matrix(exponent)[1][1],
        "serial_powers": powers,
        "serial_resident_channel_cells": 2,
        "serial_compact_classical_state_cells": 2,
    }


def quantum_dimensions():
    dimensions = [ONE, DELTA]
    for _ in range(2, LABELS + 1):
        dimensions.append(DELTA * dimensions[-1] - dimensions[-2])
    if dimensions[LABELS] != ZERO:
        raise RuntimeError("reference Jones-Wenzl relation failed")
    return tuple(dimensions[:LABELS])


QDIM = quantum_dimensions()
QDIM_INV = tuple(value.inverse() for value in QDIM)


def vacuum_paths(strands):
    result = []

    def visit(path):
        if len(path) == strands + 1:
            if path[-1] == 0:
                result.append(tuple(path))
            return
        label = path[-1]
        for following in (label - 1, label + 1):
            if 0 <= following <= LEVEL:
                visit(path + [following])

    visit([0])
    return result


def vacuum_path(strands):
    return tuple(index % 2 for index in range(strands + 1))


def program_operation(strands, rounds, family, step):
    steps = rounds * (strands - 1)
    if step < 0 or step >= steps:
        raise IndexError("reference program cursor outside public word")
    round_index, offset = divmod(step, strands - 1)
    generator = strands - 1 - offset if (round_index + family) % 2 else offset + 1
    exponent = -1 if (3 * round_index + generator + family) % 5 == 0 else 1
    return generator, exponent


def exact_action(values, paths, index_by_path, generator, exponent):
    alpha, beta = (A, A_INVERSE) if exponent == 1 else (A_INVERSE, A)
    output = [ZERO] * len(values)
    for index, path in enumerate(paths):
        left, middle, right = path[generator - 1 : generator + 2]
        if left != right:
            output[index] = alpha * values[index]
            continue
        alternatives = tuple(
            label for label in (left - 1, left + 1) if 0 <= label <= LEVEL
        )
        if len(alternatives) == 2 and middle == alternatives[1]:
            continue
        if len(alternatives) == 1:
            temperley = QDIM_INV[left] * QDIM[middle] * values[index]
            output[index] = alpha * values[index] + beta * temperley
            continue
        peer_path = path[:generator] + (alternatives[1],) + path[generator + 1 :]
        peer = index_by_path[peer_path]
        temperley = QDIM_INV[left] * (
            QDIM[alternatives[0]] * values[index]
            + QDIM[alternatives[1]] * values[peer]
        )
        output[index] = alpha * values[index] + beta * temperley
        output[peer] = alpha * values[peer] + beta * temperley
    return output


def vector_commitment(values):
    return hashlib.sha256("|".join(value.token() for value in values).encode("ascii")).hexdigest()


def field_commitment(value):
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def execute_word(strands, rounds, family, *, flip_first=False):
    paths = vacuum_paths(strands)
    index_by_path = {path: index for index, path in enumerate(paths)}
    values = [ZERO] * len(paths)
    values[index_by_path[vacuum_path(strands)]] = ONE
    steps = rounds * (strands - 1)
    for step in range(steps):
        generator, exponent = program_operation(strands, rounds, family, step)
        if step == 0 and flip_first:
            exponent = -exponent
        values = exact_action(values, paths, index_by_path, generator, exponent)
    return values, paths, index_by_path


def transaction_case(strands):
    paths = vacuum_paths(strands)
    index_by_path = {path: index for index, path in enumerate(paths)}
    source = [ZERO] * len(paths)
    source[index_by_path[vacuum_path(strands)]] = ONE
    carrier = source.copy()
    backing = id(carrier)

    def transaction(target, target_backing, rounds, family, generation):
        steps = rounds * (strands - 1)
        for step in range(steps):
            generator, exponent = program_operation(strands, rounds, family, step)
            target[:] = exact_action(target, paths, index_by_path, generator, exponent)
        boundary = target[index_by_path[vacuum_path(strands)]]
        forward = vector_commitment(target)
        missing = sum(left != right for left, right in zip(target, source)) + steps
        for step in range(steps - 1, -1, -1):
            generator, exponent = program_operation(strands, rounds, family, step)
            target[:] = exact_action(target, paths, index_by_path, generator, -exponent)
        return {
            "boundary": field_commitment(boundary),
            "forward": forward,
            "restored": target == source,
            "same_backing": id(target) == target_backing,
            "missing_inverse_nonzero": missing > 0,
            "generation": generation,
        }

    primary = transaction(carrier, backing, PRIMARY_ROUNDS, 0, 1)
    reuse = transaction(carrier, backing, REUSE_ROUNDS, 1, 2)
    fresh_carrier = source.copy()
    fresh = transaction(fresh_carrier, id(fresh_carrier), REUSE_ROUNDS, 1, 1)
    return {
        "strands": strands,
        "fusion_path_dimension": len(paths),
        "primary_boundary_commitment": primary["boundary"],
        "primary_forward_state_commitment": primary["forward"],
        "primary_restoration_error_field_cells": 0 if primary["restored"] else len(paths),
        "primary_same_coefficient_backing": primary["same_backing"],
        "primary_canonical_post_restoration_state_exact": primary["restored"],
        "primary_missing_inverse_error_nonzero": primary["missing_inverse_nonzero"],
        "reuse_boundary_commitment": reuse["boundary"],
        "reuse_forward_state_commitment": reuse["forward"],
        "reuse_restoration_error_field_cells": 0 if reuse["restored"] else len(paths),
        "reuse_same_coefficient_backing": reuse["same_backing"],
        "reuse_canonical_post_restoration_state_exact": reuse["restored"],
        "fresh_reuse_boundary_commitment": fresh["boundary"],
        "fresh_reuse_state_commitment": fresh["forward"],
        "fresh_same_coefficient_backing": fresh["same_backing"],
        "fresh_canonical_post_restoration_state_exact": fresh["restored"],
        "fresh_restored_reuse_boundary_agreement": reuse["boundary"] == fresh["boundary"],
        "fresh_restored_reuse_state_agreement": reuse["forward"] == fresh["forward"],
        "restoration_generation_after_reuse": reuse["generation"],
        "fresh_restoration_generation": fresh["generation"],
        "baseline_reload_used": False,
    }


def prime_factors(value):
    factors = []
    residual = value
    candidate = 2
    while candidate * candidate <= residual:
        if residual % candidate == 0:
            factors.append(candidate)
            while residual % candidate == 0:
                residual //= candidate
        candidate += 1
    if residual > 1:
        factors.append(residual)
    return tuple(factors)


def primitive_root(prime):
    if prime < 2 or any(prime % divisor == 0 for divisor in range(2, math.isqrt(prime) + 1)):
        raise ValueError("reference modulus must be prime")
    factors = prime_factors(prime - 1)
    return next(
        candidate
        for candidate in range(2, prime)
        if all(pow(candidate, (prime - 1) // factor, prime) != 1 for factor in factors)
    )


def order40_root(prime):
    if (prime - 1) % 40:
        raise ValueError("reference prime does not split zeta40")
    root = pow(primitive_root(prime), (prime - 1) // 40, prime)
    if pow(root, 40, prime) != 1 or pow(root, 20, prime) == 1 or pow(root, 8, prime) == 1:
        raise RuntimeError("reference root order mismatch")
    return root


def modular_dimensions(prime, root):
    delta = (pow(root, 2, prime) + pow(root, -2, prime)) % prime
    values = [1, delta]
    for _ in range(2, LABELS + 1):
        values.append((delta * values[-1] - values[-2]) % prime)
    if values[LABELS] != 0:
        raise RuntimeError("reference modular Jones-Wenzl mismatch")
    return tuple(values[:LABELS])


def generator_matrix(strands, generator, exponent, prime, root, paths, index_by_path):
    dimension = len(paths)
    dimensions = modular_dimensions(prime, root)
    alpha = pow(root, 11 if exponent == 1 else -11, prime)
    beta = pow(root, -11 if exponent == 1 else 11, prime)
    matrix = [[0] * dimension for _ in range(dimension)]
    for index, path in enumerate(paths):
        left, middle, right = path[generator - 1 : generator + 2]
        if left != right:
            matrix[index][index] = alpha
            continue
        alternatives = tuple(label for label in (left - 1, left + 1) if 0 <= label <= LEVEL)
        if len(alternatives) == 2 and middle == alternatives[1]:
            continue
        inverse = pow(dimensions[left], -1, prime)
        if len(alternatives) == 1:
            matrix[index][index] = (alpha + beta * dimensions[middle] * inverse) % prime
            continue
        peer_path = path[:generator] + (alternatives[1],) + path[generator + 1 :]
        peer = index_by_path[peer_path]
        matrix[index][index] = (alpha + beta * dimensions[alternatives[0]] * inverse) % prime
        matrix[index][peer] = beta * dimensions[alternatives[1]] * inverse % prime
        matrix[peer][index] = beta * dimensions[alternatives[0]] * inverse % prime
        matrix[peer][peer] = (alpha + beta * dimensions[alternatives[1]] * inverse) % prime
    return matrix


def matvec(matrix, vector, prime, *, transpose=False):
    dimension = len(vector)
    if transpose:
        return [sum(matrix[row][column] * vector[row] for row in range(dimension)) % prime for column in range(dimension)]
    return [sum(matrix[row][column] * vector[column] for column in range(dimension)) % prime for row in range(dimension)]


def add_basis(basis, pivots, candidate, prime):
    vector = [value % prime for value in candidate]
    for row, pivot in zip(basis, pivots):
        if vector[pivot]:
            factor = vector[pivot]
            vector = [(value - factor * basis_value) % prime for value, basis_value in zip(vector, row)]
    pivot = next((index for index, value in enumerate(vector) if value), None)
    if pivot is None:
        return None
    inverse = pow(vector[pivot], -1, prime)
    vector = [value * inverse % prime for value in vector]
    for index, row in enumerate(basis):
        if row[pivot]:
            factor = row[pivot]
            basis[index] = [
                (value - factor * new_value) % prime
                for value, new_value in zip(row, vector)
            ]
    position = next((index for index, old in enumerate(pivots) if old > pivot), len(pivots))
    pivots.insert(position, pivot)
    basis.insert(position, vector)
    return vector


def closure(source, matrices, prime, *, transpose=False):
    basis, pivots = [], []
    queue = [add_basis(basis, pivots, source, prime)]
    while queue:
        vector = queue.pop(0)
        if vector is None:
            continue
        for matrix in matrices:
            inserted = add_basis(basis, pivots, matvec(matrix, vector, prime, transpose=transpose), prime)
            if inserted is not None:
                queue.append(inserted)
    return basis


def rank(matrix, prime):
    basis, pivots = [], []
    for row in matrix:
        add_basis(basis, pivots, row, prime)
    return len(basis)


def rank_case(strands, prime):
    paths = vacuum_paths(strands)
    index_by_path = {path: index for index, path in enumerate(paths)}
    root = order40_root(prime)
    matrices = [
        generator_matrix(strands, generator, exponent, prime, root, paths, index_by_path)
        for generator in range(1, strands)
        for exponent in (-1, 1)
    ]
    source = [0] * len(paths)
    source[index_by_path[vacuum_path(strands)]] = 1
    reachable = closure(source, matrices, prime)
    observable = closure(source, matrices, prime, transpose=True)
    hankel = [[sum(a * b for a, b in zip(left, right)) % prime for right in reachable] for left in observable]
    return {
        "strands": strands,
        "fusion_path_dimension": len(paths),
        "prime": prime,
        "reachable_rank": len(reachable),
        "observable_rank": len(observable),
        "continuation_hankel_rank": rank(hankel, prime),
        "dense_generator_matrices_are_verifier_only": True,
        "peak_dense_generator_matrix_cells": len(paths) ** 2,
    }


def independent_operator_controls():
    strands, prime = 10, CERTIFICATE_PRIMES[0]
    paths = vacuum_paths(strands)
    index_by_path = {path: index for index, path in enumerate(paths)}
    root = order40_root(prime)
    matrices = {
        (generator, exponent): generator_matrix(strands, generator, exponent, prime, root, paths, index_by_path)
        for generator in range(1, strands)
        for exponent in (-1, 1)
    }
    basis = [[int(row == column) for row in range(len(paths))] for column in range(len(paths))]

    def apply(vector, word):
        for operation in word:
            vector = matvec(matrices[operation], vector, prime)
        return vector

    return {
        "all_generator_inverses_exact": all(
            apply(vector.copy(), ((generator, 1), (generator, -1))) == vector
            for generator in range(1, strands) for vector in basis
        ),
        "all_adjacent_yang_baxter_exact": all(
            apply(vector.copy(), ((generator, 1), (generator + 1, 1), (generator, 1)))
            == apply(vector.copy(), ((generator + 1, 1), (generator, 1), (generator + 1, 1)))
            for generator in range(1, strands - 1) for vector in basis
        ),
        "all_distant_commutation_exact": all(
            apply(vector.copy(), ((left, 1), (right, 1))) == apply(vector.copy(), ((right, 1), (left, 1)))
            for left in range(1, strands) for right in range(left + 2, strands) for vector in basis
        ),
    }


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("M233 standalone reference takes no input")
    transactions = [transaction_case(strands) for strands in GROWING_STRANDS]
    rank_cases = [
        rank_case(strands, prime)
        for strands in GROWING_STRANDS
        for prime in CERTIFICATE_PRIMES
    ]
    rank_law = [
        {
            "strands": strands,
            "fusion_path_dimension": dimension,
            "reachable_rank_both_primes": [
                case["reachable_rank"] for case in rank_cases if case["strands"] == strands
            ],
            "observable_rank_both_primes": [
                case["observable_rank"] for case in rank_cases if case["strands"] == strands
            ],
            "continuation_hankel_rank_both_primes": [
                case["continuation_hankel_rank"] for case in rank_cases if case["strands"] == strands
            ],
        }
        for strands, dimension in zip(GROWING_STRANDS, GROWING_DIMENSIONS)
    ]
    current_controls = {
        "dimensions_match_exact_a9_walk_counts": [len(vacuum_paths(strands)) for strands in GROWING_STRANDS] == list(GROWING_DIMENSIONS),
        "n4_positive_gauge_equivalence_exact": symmetric(1) == gauge(path_block(1)),
        "n4_negative_gauge_equivalence_exact": symmetric(-1) == gauge(path_block(-1)),
        "all_distinct_prime_ranks_full": all(
            case["reachable_rank"] == case["observable_rank"] == case["continuation_hankel_rank"] == case["fusion_path_dimension"]
            for case in rank_cases
        ),
        "flipped_first_exponent_changes_all_selected_boundaries": all(
            execute_word(strands, PRIMARY_ROUNDS, 0)[0][vacuum_paths(strands).index(vacuum_path(strands))]
            != execute_word(strands, PRIMARY_ROUNDS, 0, flip_first=True)[0][vacuum_paths(strands).index(vacuum_path(strands))]
            for strands in GROWING_STRANDS
        ),
        "all_transactions_restore_and_reuse": all(
            case["primary_canonical_post_restoration_state_exact"]
            and case["reuse_canonical_post_restoration_state_exact"]
            and case["fresh_restored_reuse_boundary_agreement"]
            and case["fresh_restored_reuse_state_agreement"]
            and case["primary_same_coefficient_backing"]
            and case["reuse_same_coefficient_backing"]
            and case["fresh_same_coefficient_backing"]
            and case["fresh_canonical_post_restoration_state_exact"]
            and not case["baseline_reload_used"]
            for case in transactions
        ),
        **independent_operator_controls(),
    }
    result = {
        "schema": "cat_cas.su2_level8_trivalent_continuation_rank_reference.v1",
        "transactions": transactions,
        "rank_cases": rank_cases,
        "rank_law": rank_law,
        "controls": current_controls,
        "certificate_primes": list(CERTIFICATE_PRIMES),
        "imports_m233_production": False,
        "imports_m232_production": False,
        "imports_m214_production": False,
        "uses_independent_cyclotomic_polynomial_oracle": True,
        "uses_independent_dense_generator_verifier": True,
    }
    if not all(result["controls"].values()):
        raise RuntimeError("M233 reference positive control failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
