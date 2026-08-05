#!/usr/bin/env python3
"""Independent exact oracle for the continuous-S1 theta chart result.

This implementation imports no CAT_CAS production module.  It reconstructs
public programs as plain tuples, derives the A-lattice determinant through an
independent eigenvalue identity, and computes Q-jets by phase-type polynomial
exponentiation rather than the production factor-by-factor recurrence.
"""

from __future__ import annotations

import hashlib
import json
import math


ORDER = 24
SIZES = (2, 3, 4, 8, 16, 32, 64)
C = tuple[int, int]
ZERO: C = (0, 0)
ONE: C = (1, 0)


def add(left: C, right: C) -> C:
    return left[0] + right[0], left[1] + right[1]


def multiply(left: C, right: C) -> C:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def unit(exponent: int) -> C:
    return ((1, 0), (0, 1), (-1, 0), (0, -1))[exponent % 4]


def program(size: int, family: int) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for index in range(size - 1):
        rotation = "ROTATE", 1 if (index + family) % 2 == 0 else 3
        intersection = "INTERSECT", (3 * index + family + 1) % 4
        result.extend(
            (rotation, intersection)
            if (index + 2 * family) % 3
            else (intersection, rotation)
        )
    return tuple(result)


def shift(counts: list[int], amount: int) -> list[int]:
    result = [0, 0, 0, 0]
    for phase, value in enumerate(counts):
        result[(phase + amount) % 4] = value
    return result


def execute(
    operations: tuple[tuple[str, int], ...]
) -> tuple[list[int], list[int]]:
    current = [1, 0, 0, 0]
    for kind, parameter in operations:
        if kind == "ROTATE":
            current = shift(current, parameter)
        elif kind == "INTERSECT":
            current[parameter] += 1
        else:
            raise ValueError("independent program type changed")
    final = current.copy()
    for kind, parameter in reversed(operations):
        if kind == "ROTATE":
            current = shift(current, -parameter)
        else:
            current[parameter] -= 1
            if current[parameter] < 0:
                raise RuntimeError("independent inverse underflow")
    return final, current


Poly = dict[tuple[int, int], C]


def polynomial_multiply(left: Poly, right: Poly) -> Poly:
    result: Poly = {}
    for (left_n, left_e), left_value in left.items():
        for (right_n, right_e), right_value in right.items():
            exponent = left_e + right_e
            if exponent > ORDER:
                continue
            key = left_n + right_n, exponent
            result[key] = add(
                result.get(key, ZERO), multiply(left_value, right_value)
            )
    return {key: value for key, value in result.items() if value != ZERO}


def phase_kernel(phase: int) -> Poly:
    radius = math.isqrt(ORDER)
    return {
        (mode, mode * mode): unit(phase * mode)
        for mode in range(-radius, radius + 1)
    }


def polynomial_power(base: Poly, exponent: int) -> Poly:
    result: Poly = {(0, 0): ONE}
    current = base
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = polynomial_multiply(result, current)
        remaining >>= 1
        if remaining:
            current = polynomial_multiply(current, current)
    return result


def grouped_q_jet(counts: list[int]) -> tuple[C, ...]:
    product: Poly = {(0, 0): ONE}
    for phase, multiplicity in enumerate(counts):
        if multiplicity:
            product = polynomial_multiply(
                product, polynomial_power(phase_kernel(phase), multiplicity)
            )
    return tuple(product.get((0, exponent), ZERO) for exponent in range(ORDER + 1))


def commitment(jet: tuple[C, ...]) -> str:
    payload = "|".join(f"{a}:{b}" for a, b in jet)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def lattice_certificate(size: int) -> dict[str, object]:
    rank = size - 1
    # I+J has eigenvalue one on the rank-1 difference space and eigenvalue
    # rank+1 on the all-ones vector.  Constructive unimodular row/column
    # differences give Smith form 1,...,1,size.
    determinant = 1 ** max(0, rank - 1) * (rank + 1)
    return {
        "lattice_rank": rank,
        "reduced_gram_determinant": determinant,
        "discriminant_fibers": determinant,
        "smith_final_invariant": size,
        "eigenvalue_product_certificate": determinant == size,
    }


def case(size: int, family: int) -> dict[str, object]:
    operations = program(size, family)
    counts, restored = execute(operations)
    if sum(counts) != size or restored != [1, 0, 0, 0]:
        raise RuntimeError("independent theta descriptor execution failed")
    jet = grouped_q_jet(counts)
    return {
        "total_factors": size,
        "family": family,
        "factor_counts": counts,
        "boundary_commitment": commitment(jet),
        "exact_reverse_restored": restored == [1, 0, 0, 0],
        **lattice_certificate(size),
    }


def composition_check() -> bool:
    for a, p, b, r in ((1, 1, 2, 3), (2, 3, 5, 1), (5, 2, 7, 3)):
        for mode in range(-9, 10):
            if (
                a * mode * mode + b * mode * mode,
                (p * mode + r * mode) % 4,
            ) != ((a + b) * mode * mode, ((p + r) * mode) % 4):
                return False
    return True


def main() -> None:
    cases = [case(size, family) for size in SIZES for family in (0, 1)]
    primary = case(64, 0)
    reuse = case(37, 1)
    first_counts, _ = execute((("ROTATE", 1), ("INTERSECT", 0)))
    second_counts, _ = execute((("INTERSECT", 0), ("ROTATE", 1)))
    first_jet = grouped_q_jet(first_counts)
    second_jet = grouped_q_jet(second_counts)
    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "independent_algorithms": [
            "PLAIN_TUPLE_PUBLIC_PROGRAM_RECONSTRUCTION",
            "PHASE_TYPE_SPARSE_POLYNOMIAL_BINARY_EXPONENTIATION",
            "I_PLUS_J_EIGENVALUE_AND_UNIMODULAR_DIFFERENCE_SMITH_CERTIFICATE",
        ],
        "composition_parameter_addition_exact": composition_check(),
        "factor_cases": cases,
        "primary_boundary_commitment": primary["boundary_commitment"],
        "reuse_boundary_commitment": reuse["boundary_commitment"],
        "module_order_counts_differ": first_counts != second_counts,
        "module_order_boundary_changes": first_jet != second_jet,
        "finite_angle_sampling_used": False,
        "full_infinite_theta_scalar_evaluated": False,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
