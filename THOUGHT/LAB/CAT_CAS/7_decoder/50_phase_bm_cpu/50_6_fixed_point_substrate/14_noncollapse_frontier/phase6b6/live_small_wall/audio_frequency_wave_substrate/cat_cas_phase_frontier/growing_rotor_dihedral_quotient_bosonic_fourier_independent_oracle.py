#!/usr/bin/env python3
"""Independent exact oracle for the Rotor-6 quotient Fourier diagnostic.

This file imports no CAT_CAS module.  It rebuilds the two symmetry sectors,
the direct bosonic kernel certificates, and the first Gaussian-elimination
Fourier gate from public constants.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math


GRID = 17
ROTORS = 6
PRIME = 103
ROOT = 72
Histogram = tuple[int, ...]
Matrix = list[list[int]]


def histograms(position: int = 0, remaining: int = ROTORS, prefix: tuple[int, ...] = ()):
    if position == GRID - 1:
        yield prefix + (remaining,)
        return
    for count in range(remaining + 1):
        yield from histograms(position + 1, remaining - count, prefix + (count,))


def rotate(item: Histogram, amount: int) -> Histogram:
    amount %= GRID
    return item[amount:] + item[:amount]


def reflect(item: Histogram) -> Histogram:
    return (item[0],) + tuple(reversed(item[1:]))


def orbit(item: Histogram) -> tuple[Histogram, ...]:
    reflected = reflect(item)
    return tuple(
        sorted(
            {
                *(rotate(item, amount) for amount in range(GRID)),
                *(rotate(reflected, amount) for amount in range(GRID)),
            }
        )
    )


def canonical(item: Histogram) -> Histogram:
    return min(orbit(item))


def particles(item: Histogram) -> tuple[int, ...]:
    return tuple(mode for mode, count in enumerate(item) for _ in range(count))


def factorial_product(item: Histogram) -> int:
    result = 1
    for count in item:
        result = result * math.factorial(count) % PRIME
    return result


def entry(row: int, column: int) -> int:
    return pow(ROOT, row * column % GRID, PRIME)


def permanent_assignments(source: Histogram, target: Histogram) -> int:
    rows = particles(source)
    columns = particles(target)
    result = 0
    for assignment in itertools.permutations(range(ROTORS)):
        term = 1
        for row, column in zip(rows, assignment, strict=True):
            term = term * entry(row, columns[column]) % PRIME
        result = (result + term) % PRIME
    return result


def permanent_ryser(source: Histogram, target: Histogram) -> int:
    rows = particles(source)
    columns = particles(target)
    result = 0
    for mask in range(1, 1 << ROTORS):
        product = 1
        for row in rows:
            row_sum = sum(
                entry(row, columns[column])
                for column in range(ROTORS)
                if mask & (1 << column)
            ) % PRIME
            product = product * row_sum % PRIME
        if (ROTORS - mask.bit_count()) % 2:
            result -= product
        else:
            result += product
    return result % PRIME


def quotient_kernel(source: Histogram, target: Histogram) -> int:
    return (
        len(orbit(source))
        * permanent_ryser(source, target)
        * pow(factorial_product(source), -1, PRIME)
    ) % PRIME


def multiply(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            sum(left[row][middle] * right[middle][column] for middle in range(GRID))
            % PRIME
            for column in range(GRID)
        ]
        for row in range(GRID)
    ]


def identity() -> Matrix:
    return [[int(row == column) for column in range(GRID)] for row in range(GRID)]


def row_shear(matrix: Matrix, target: int, source: int, factor: int) -> None:
    matrix[target] = [
        (left + factor * right) % PRIME
        for left, right in zip(matrix[target], matrix[source], strict=True)
    ]


def independently_derive_first_network_gate() -> tuple[str, int, int, int]:
    """Rebuild elimination, then reverse/invert it to obtain the forward network."""
    reduced = [
        [pow(ROOT, row * column % GRID, PRIME) for column in range(GRID)]
        for row in range(GRID)
    ]
    reductions: list[tuple[str, int, int, int]] = []
    for column in range(GRID):
        pivot = next(row for row in range(column, GRID) if reduced[row][column])
        if pivot != column:
            reduced[column], reduced[pivot] = reduced[pivot], reduced[column]
            reductions.append(("SWAP", column, pivot, 0))
        scale = pow(reduced[column][column], -1, PRIME)
        if scale != 1:
            reduced[column] = [scale * value % PRIME for value in reduced[column]]
            reductions.append(("SCALE", column, column, scale))
        for row in range(GRID):
            if row == column or reduced[row][column] == 0:
                continue
            factor = -reduced[row][column] % PRIME
            row_shear(reduced, row, column, factor)
            reductions.append(("SHEAR", row, column, factor))
    if reduced != identity():
        raise RuntimeError("independent Fourier elimination failed")
    kind, target, source, factor = reductions[-1]
    if kind == "SHEAR":
        factor = -factor % PRIME
    elif kind == "SCALE":
        factor = pow(factor, -1, PRIME)
    return kind, target, source, factor


def explicit_orbit_kernel(source: Histogram, target: Histogram) -> int:
    return sum(
        permanent_ryser(member, target)
        * pow(factorial_product(member), -1, PRIME)
        for member in orbit(source)
    ) % PRIME


def commitment(values: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, values)).encode()).hexdigest()


def main() -> None:
    occupations = tuple(histograms())
    bracelets = tuple(sorted({canonical(item) for item in occupations}))
    zero_total = tuple(
        item
        for item in occupations
        if sum(mode * count for mode, count in enumerate(item)) % GRID == 0
    )
    targets = tuple(sorted({min(item, reflect(item)) for item in zero_total}))
    if (len(occupations), len(bracelets), len(zero_total), len(targets)) != (
        74613,
        2277,
        4389,
        2277,
    ):
        raise RuntimeError("independent quotient count changed")

    origin = (ROTORS,) + (0,) * (GRID - 1)
    dense_row = [quotient_kernel(source, origin) for source in bracelets]
    dense_column = [quotient_kernel(origin, target) for target in targets]
    if any(value == 0 for value in dense_row) or dense_column != [GRID] * 2277:
        raise RuntimeError("independent dense row/column certificate changed")

    generic_source = (1, 1, 1, 1, 1, 1) + (0,) * (GRID - 6)
    generic_target = (1, 1, 1, 1, 1, 0, 0, 1) + (0,) * (GRID - 8)
    ryser = permanent_ryser(generic_source, generic_target)
    assignments = permanent_assignments(generic_source, generic_target)
    if ryser != assignments:
        raise RuntimeError("independent permanent formulas differ")

    first_gate = independently_derive_first_network_gate()
    if first_gate != ("SHEAR", 15, 16, 10):
        raise RuntimeError("independent first Fourier gate changed")
    mismatch = math.comb(ROTORS, 1) * first_gate[3] % PRIME
    wrong_sector = (ROTORS - 1, 1) + (0,) * (GRID - 2)
    wrong_kernel = explicit_orbit_kernel(origin, wrong_sector)
    if mismatch == 0 or wrong_kernel != 0:
        raise RuntimeError("independent symmetry controls failed")

    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "oracle_imports_cat_cas_modules": False,
        "topology": {
            "occupations": len(occupations),
            "momentum_bracelets": len(bracelets),
            "zero_total_position_occupations": len(zero_total),
            "reflection_closed_position_cells": len(targets),
        },
        "direct_kernel": {
            "dense_row_nonzero_entries": sum(value != 0 for value in dense_row),
            "dense_column_nonzero_entries": sum(value != 0 for value in dense_column),
            "dense_column_constant": dense_column[0],
            "dense_row_commitment": commitment(dense_row),
            "dense_column_commitment": commitment(dense_column),
            "generic_ryser_permanent": ryser,
            "generic_assignment_permanent": assignments,
        },
        "butterfly": {
            "independently_derived_first_gate": list(first_gate),
            "first_gate_orbit_coefficient_mismatch": mismatch,
        },
        "controls": {
            "wrong_sector_orbit_kernel": wrong_kernel,
            "wrong_sector_total_coordinate": 1,
            "drop_orbit_factor_changes_origin_column_from17_to1": True,
        },
        "universal_structured_transform_lower_bound_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
