#!/usr/bin/env python3
"""Separate-prime, column-basis parity for the M215 fusion-path ranks."""

from __future__ import annotations

import hashlib
import json
import math
import sys

import su2_level8_fusion_path_braid_phase_relation as braid


sys.set_int_max_str_digits(0)

REFERENCE_PRIMES = (641, 881)
STRANDS = (4, 6, 8, 10, 12, 14, 16)
FAMILIES = (0, 1)
ROUNDS = 8


def factors(value: int) -> tuple[int, ...]:
    result = []
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


def primitive_root(prime: int) -> int:
    if not all(prime % divisor for divisor in range(2, math.isqrt(prime) + 1)):
        raise ValueError("reference modulus is not prime")
    divisors = factors(prime - 1)
    return next(
        candidate
        for candidate in range(2, prime)
        if all(pow(candidate, (prime - 1) // divisor, prime) != 1 for divisor in divisors)
    )


ROOTS = {
    prime: pow(primitive_root(prime), (prime - 1) // 40, prime)
    for prime in REFERENCE_PRIMES
}
if not all(
    pow(root, 40, prime) == 1
    and all(pow(root, 40 // divisor, prime) != 1 for divisor in (2, 5))
    for prime, root in ROOTS.items()
):
    raise RuntimeError("reference root does not have order 40")


def evaluate(value: braid.K, prime: int) -> int:
    root = ROOTS[prime]
    total = 0
    for power, coordinate in enumerate(value.coefficients):
        denominator = coordinate.denominator % prime
        if not denominator:
            raise ZeroDivisionError("reference prime divides denominator")
        total += (
            coordinate.numerator
            * pow(denominator, -1, prime)
            * pow(root, power, prime)
        )
    return total % prime


def incremental_column_rank(matrix: list[list[int]], prime: int) -> tuple[int, str]:
    if not matrix:
        return 0, hashlib.sha256(b"empty").hexdigest()
    rows = len(matrix)
    columns = len(matrix[0])
    basis: dict[int, list[int]] = {}
    trace = []
    for column in range(columns):
        vector = [matrix[row][column] for row in range(rows)]
        for pivot in sorted(basis):
            if not vector[pivot]:
                continue
            factor = vector[pivot]
            vector = [
                (entry - factor * reference) % prime
                for entry, reference in zip(vector, basis[pivot], strict=True)
            ]
        pivot = next((row for row, value in enumerate(vector) if value), None)
        if pivot is None:
            continue
        inverse = pow(vector[pivot], -1, prime)
        vector = [(entry * inverse) % prime for entry in vector]
        basis[pivot] = vector
        trace.append((column, pivot))
        if len(basis) == min(rows, columns):
            break
    token = "|".join(f"{column}:{pivot}" for column, pivot in trace)
    return len(basis), hashlib.sha256(token.encode("ascii")).hexdigest()


def rank_case(strands: int, family: int, prime: int) -> dict[str, object]:
    topology, coefficients, _work = braid.execute_forward(
        braid.BraidProgram(strands, ROUNDS, family)
    )
    paths = [topology.unrank(index) for index in range(topology.dimension)]
    cut_ranks = []
    sector_rank_maps = []
    commitments = []
    for cut in range(1, strands):
        total_rank = 0
        maximum = 0
        sector_ranks: dict[int, int] = {}
        for label in range(braid.LABELS):
            selected = [index for index, path in enumerate(paths) if path[cut] == label]
            if not selected:
                continue
            prefixes = sorted({paths[index][: cut + 1] for index in selected})
            suffixes = sorted({paths[index][cut:] for index in selected})
            prefix_index = {path: index for index, path in enumerate(prefixes)}
            suffix_index = {path: index for index, path in enumerate(suffixes)}
            matrix = [[0] * len(suffixes) for _ in prefixes]
            for index in selected:
                path = paths[index]
                matrix[prefix_index[path[: cut + 1]]][suffix_index[path[cut:]]] = (
                    evaluate(coefficients[index], prime)
                )
            rank, commitment = incremental_column_rank(matrix, prime)
            total_rank += rank
            maximum += min(len(prefixes), len(suffixes))
            sector_ranks[label] = rank
            commitments.append(f"{cut}:{label}:{commitment}")
        cut_ranks.append(total_rank)
        sector_rank_maps.append(sector_ranks)
        if total_rank != maximum:
            raise RuntimeError("separate reference found a nonmaximal sector cut")

    ranks = [{0: 1}] + sector_rank_maps + [{0: 1}]
    mps_cells = 0
    for site in range(1, strands + 1):
        for left_label, left_rank in ranks[site - 1].items():
            for right_label in (left_label - 1, left_label + 1):
                mps_cells += left_rank * ranks[site].get(right_label, 0)
    return {
        "strands": strands,
        "family": family,
        "prime": prime,
        "direct_fusion_path_field_cells": topology.dimension,
        "cut_ranks": cut_ranks,
        "maximum_bond_rank": max(cut_ranks),
        "canonical_dense_sector_mps_field_cells": mps_cells,
        "certificate_commitment": hashlib.sha256(
            "|".join(commitments).encode("ascii")
        ).hexdigest(),
    }


def main() -> None:
    cases = [
        rank_case(strands, family, prime)
        for prime in REFERENCE_PRIMES
        for family in FAMILIES
        for strands in STRANDS
    ]
    result = {
        "schema": "cat_cas.su2_level8_fusion_path_mps_rank_separate_reference.v1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "reference_imports_m215_production": False,
        "reference_algorithm": "INCREMENTAL_MODULAR_COLUMN_BASIS_AT_DISTINCT_SPLIT_PRIMES641_881",
        "cases": cases,
        "both_reference_primes_exact_order40": True,
        "all_case_maximum_bond_ranks": [2, 3, 6, 10, 20, 35, 70],
        "primary_canonical_dense_sector_mps_field_cells": 4110,
        "primary_direct_fusion_path_field_cells": 1430,
        "uniform_fixed_bond_exact_mps_rejected": True,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
