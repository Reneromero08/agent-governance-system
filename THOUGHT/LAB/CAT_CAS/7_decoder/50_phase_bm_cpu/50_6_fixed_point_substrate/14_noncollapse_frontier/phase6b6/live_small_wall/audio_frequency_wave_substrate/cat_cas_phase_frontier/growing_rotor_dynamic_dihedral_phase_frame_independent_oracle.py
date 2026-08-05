#!/usr/bin/env python3
"""Small independent conjugated-frame oracle over the public Rotor-2 sector."""

from __future__ import annotations

import itertools
import json
import math


GRID = 17
ROTORS = 2
PRIME = 103
ROOT = 72
Histogram = tuple[int, ...]


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


def dihedral_orbit(item: Histogram) -> tuple[Histogram, ...]:
    reflected = reflect(item)
    return tuple(sorted({*(rotate(item, a) for a in range(GRID)), *(rotate(reflected, a) for a in range(GRID))}))


def reflection_orbit(item: Histogram) -> tuple[Histogram, ...]:
    return tuple(sorted({item, reflect(item)}))


def particles(item: Histogram) -> tuple[int, ...]:
    return tuple(mode for mode, count in enumerate(item) for _ in range(count))


def factorial_product(item: Histogram) -> int:
    result = 1
    for count in item:
        result = result * math.factorial(count) % PRIME
    return result


def permanent(source: Histogram, target: Histogram, inverse: bool = False) -> int:
    rows = particles(source)
    columns = particles(target)
    result = 0
    normalization = pow(GRID, -1, PRIME) if inverse else 1
    for assignment in itertools.permutations(range(ROTORS)):
        term = 1
        for row_index, column_index in enumerate(assignment):
            exponent = rows[row_index] * columns[column_index]
            if inverse:
                exponent = -exponent
            term = term * normalization * pow(ROOT, exponent % GRID, PRIME) % PRIME
        result = (result + term) % PRIME
    return result


def kernel(source: Histogram, target: Histogram, inverse: bool = False) -> int:
    domain_orbit = reflection_orbit(source) if inverse else dihedral_orbit(source)
    return sum(
        permanent(member, target, inverse)
        * pow(factorial_product(member), -1, PRIME)
        for member in domain_orbit
    ) % PRIME


def matmul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [
            sum(left[row][middle] * right[middle][column] for middle in range(len(right))) % PRIME
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def public_weight(shift: int, step: int, tag: int) -> int:
    distance = min(shift % GRID, GRID - shift % GRID)
    magnitude = 1 + ((distance + 2) * (step + 1) + (3 * distance + 1) * (tag + 2)) % GRID % 5
    return -magnitude if (distance + step + tag) % GRID % 3 == 0 else magnitude


def position_eigenvalue(item: Histogram, step: int, tag: int) -> int:
    result = 0
    for shift in range(1, GRID):
        positive = sum(count * pow(ROOT, shift * mode % GRID, PRIME) for mode, count in enumerate(item))
        negative = sum(count * pow(ROOT, -shift * mode % GRID, PRIME) for mode, count in enumerate(item))
        result += public_weight(shift, step, tag) * (positive * negative - ROTORS)
    return result % PRIME


def direct_scattering(bracelets: tuple[Histogram, ...], step: int, tag: int) -> list[list[int]]:
    index = {item: position for position, item in enumerate(bracelets)}
    result = [[0] * len(bracelets) for _ in bracelets]
    for target_index, target in enumerate(bracelets):
        for shift in range(1, GRID):
            weight = public_weight(shift, step, tag)
            for first, first_count in enumerate(target):
                if not first_count:
                    continue
                for second, second_count in enumerate(target):
                    multiplicity = first_count * (second_count - int(first == second))
                    if not multiplicity:
                        continue
                    moved = list(target)
                    moved[first] -= 1
                    moved[second] -= 1
                    moved[(first - shift) % GRID] += 1
                    moved[(second + shift) % GRID] += 1
                    source = min(dihedral_orbit(tuple(moved)))
                    result[target_index][index[source]] = (
                        result[target_index][index[source]] + weight * multiplicity
                    ) % PRIME
    return result


def main() -> None:
    occupations = tuple(histograms())
    bracelets = tuple(sorted({min(dihedral_orbit(item)) for item in occupations}))
    zero_total = tuple(item for item in occupations if sum(mode * count for mode, count in enumerate(item)) % GRID == 0)
    targets = tuple(sorted({min(reflection_orbit(item)) for item in zero_total}))
    if (len(occupations), len(bracelets), len(zero_total), len(targets)) != (153, 9, 9, 9):
        raise RuntimeError("Rotor-2 symmetry topology changed")
    forward = [[kernel(source, target) for source in bracelets] for target in targets]
    inverse = [[kernel(source, target, True) for source in targets] for target in bracelets]
    identity = [[int(row == column) for column in range(9)] for row in range(9)]
    if matmul(inverse, forward) != identity:
        raise RuntimeError("independent quotient Fourier inverse failed")
    diagonal = [[0] * 9 for _ in range(9)]
    for index, item in enumerate(targets):
        diagonal[index][index] = position_eigenvalue(item, 0, 0)
    conjugated = matmul(inverse, matmul(diagonal, forward))
    direct = direct_scattering(bracelets, 0, 0)
    mismatch = sum(a != b for left, right in zip(conjugated, direct, strict=True) for a, b in zip(left, right, strict=True))
    naive = diagonal
    naive_mismatch = sum(a != b for left, right in zip(naive, direct, strict=True) for a, b in zip(left, right, strict=True))
    if mismatch or naive_mismatch == 0:
        raise RuntimeError("independent conjugated-frame identity failed")
    result = {
        "classification": "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "NO_RESTORATION_CLAIM",
        "oracle_imports_cat_cas_modules": False,
        "transferable_rotor2_certificate": {
            "occupation_cells": len(occupations),
            "momentum_bracelet_cells": len(bracelets),
            "zero_total_position_cells": len(zero_total),
            "reflection_closed_position_cells": len(targets),
            "forward_inverse_identity": True,
            "conjugated_direct_scattering_mismatch_cells": mismatch,
            "naive_coefficientwise_diagonal_mismatch_cells": naive_mismatch,
            "forward_nonzero_cells": sum(value != 0 for row in forward for value in row),
            "conjugated_nonzero_cells": sum(value != 0 for row in conjugated for value in row),
        },
        "rotor6_execution_independently_reexecuted": False,
        "rotor6_parity_uses_prior_independently_verified_m204_boundary_and_commitment": True,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
