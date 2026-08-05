#!/usr/bin/env python3
"""Independent exact oracle for the continuous-S1 Laurent relation result."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction


DEPTHS = (1, 2, 4, 8, 16, 32, 64)
G = tuple[Fraction, Fraction]
ZERO: G = (Fraction(0), Fraction(0))
ONE: G = (Fraction(1), Fraction(0))
SOURCE: G = (Fraction(12, 13), Fraction(5, 13))
POINT: G = (Fraction(20, 29), Fraction(21, 29))
ROTATIONS: tuple[G, ...] = (
    (Fraction(3, 5), Fraction(4, 5)),
    (Fraction(5, 13), Fraction(12, 13)),
)
FACTORS: tuple[G, ...] = (
    (Fraction(8, 17), Fraction(15, 17)),
    (Fraction(7, 25), Fraction(24, 25)),
)


def add(left: G, right: G) -> G:
    return left[0] + right[0], left[1] + right[1]


def subtract(left: G, right: G) -> G:
    return left[0] - right[0], left[1] - right[1]


def multiply(left: G, right: G) -> G:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def inverse(value: G) -> G:
    norm = value[0] * value[0] + value[1] * value[1]
    if not norm:
        raise ZeroDivisionError
    return value[0] / norm, -value[1] / norm


def power(value: G, exponent: int) -> G:
    if exponent < 0:
        return power(inverse(value), -exponent)
    result = ONE
    for _ in range(exponent):
        result = multiply(result, value)
    return result


def token(value: G) -> str:
    return (
        f"{value[0].numerator}/{value[0].denominator}:"
        f"{value[1].numerator}/{value[1].denominator}"
    )


def commitment(values: list[G]) -> str:
    payload = "|".join(token(value) for value in values)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def boundary_commitment(value: G) -> str:
    return hashlib.sha256(token(value).encode("ascii")).hexdigest()


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: list[G]) -> int:
    return sum(
        signed_bits(component.numerator) + component.denominator.bit_length()
        for value in values
        for component in value
    )


def program(depth: int, family: int) -> tuple[tuple[str, int], ...]:
    operations: list[tuple[str, int]] = []
    for index in range(depth):
        rotation = ("R", (index + family) % 2)
        factor = ("I", (3 * index + 2 * family + 1) % 2)
        operations.extend(
            (rotation, factor) if (index + family) % 3 else (factor, rotation)
        )
    return tuple(operations)


def polynomial_product(left: list[G], right: list[G]) -> list[G]:
    result = [ZERO] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] = add(
                result[left_index + right_index],
                multiply(left_value, right_value),
            )
    return result


def forward(
    source: list[G], operations: tuple[tuple[str, int], ...]
) -> list[G]:
    coefficients = source.copy()
    for kind, parameter in operations:
        if kind == "R":
            rotation = ROTATIONS[parameter]
            coefficients = [
                multiply(value, power(rotation, index))
                for index, value in enumerate(coefficients)
            ]
        else:
            coefficients = polynomial_product(
                coefficients, [ONE, FACTORS[parameter]]
            )
    return coefficients


def exact_linear_factor_division_from_leading(
    coefficients: list[G], factor: G
) -> list[G]:
    quotient = [ZERO] * (len(coefficients) - 1)
    quotient[-1] = multiply(coefficients[-1], inverse(factor))
    for index in range(len(quotient) - 2, -1, -1):
        quotient[index] = multiply(
            subtract(coefficients[index + 1], quotient[index + 1]),
            inverse(factor),
        )
    if quotient[0] != coefficients[0]:
        raise RuntimeError("independent leading division has a remainder")
    if polynomial_product(quotient, [ONE, factor]) != coefficients:
        raise RuntimeError("independent factor reconstruction differs")
    return quotient


def reverse(
    coefficients: list[G], operations: tuple[tuple[str, int], ...]
) -> list[G]:
    current = coefficients.copy()
    for kind, parameter in reversed(operations):
        if kind == "R":
            rotation = inverse(ROTATIONS[parameter])
            current = [
                multiply(value, power(rotation, index))
                for index, value in enumerate(current)
            ]
        else:
            current = exact_linear_factor_division_from_leading(
                current, FACTORS[parameter]
            )
    return current


def evaluate_by_powers(coefficients: list[G], point: G) -> G:
    result = ZERO
    for index, coefficient in enumerate(coefficients):
        result = add(result, multiply(coefficient, power(point, index)))
    return result


def scalar_factor_formula(
    operations: tuple[tuple[str, int], ...], point: G
) -> tuple[G, int, int]:
    suffix = ONE
    result = ONE
    for kind, parameter in reversed(operations):
        if kind == "R":
            suffix = multiply(suffix, ROTATIONS[parameter])
        else:
            result = multiply(
                result,
                add(ONE, multiply(multiply(FACTORS[parameter], suffix), point)),
            )
    result = multiply(
        result, add(ONE, multiply(multiply(SOURCE, suffix), point))
    )
    return result, 2, 4


def case(depth: int, family: int) -> dict[str, object]:
    operations = program(depth, family)
    coefficients = forward([ONE, SOURCE], operations)
    direct = evaluate_by_powers(coefficients, POINT)
    factored, resident_cells, named_peak = scalar_factor_formula(
        operations, POINT
    )
    restored = reverse(coefficients, operations)
    degree = len(coefficients) - 1
    if (
        direct != factored
        or restored != [ONE, SOURCE]
        or degree != depth + 1
        or coefficients[-1] == ZERO
    ):
        raise RuntimeError("independent continuous-S1 oracle failed")
    return {
        "depth": depth,
        "family": family,
        "reduced_rational_numerator_degree": degree,
        "reduced_rational_denominator_degree": 0,
        "finite_support_hankel_rank": degree + 1,
        "relation_coefficient_cells": len(coefficients),
        "nonzero_harmonic_coefficients": sum(
            value != ZERO for value in coefficients
        ),
        "relation_payload_bits": payload_bits(coefficients),
        "state_commitment": commitment(coefficients),
        "boundary_commitment": boundary_commitment(direct),
        "scalar_boundary_resident_gaussian_rational_cells": resident_cells,
        "scalar_boundary_warm_named_gaussian_rational_cell_peak": named_peak,
        "exact_reverse_restored": True,
    }


def main() -> None:
    cases = [
        case(depth, family)
        for depth in DEPTHS
        for family in (0, 1)
    ]
    initial = [ONE, SOURCE]
    first = forward(initial, (("R", 0), ("I", 0)))
    second = forward(initial, (("I", 0), ("R", 0)))
    noncommuting = sum(
        left != right for left, right in zip(first, second, strict=True)
    )
    if noncommuting != 2:
        raise RuntimeError("independent module-order control changed")
    primary = case(64, 0)
    reuse = case(37, 1)
    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "independent_algorithms": [
            "GENERIC_POLYNOMIAL_CONVOLUTION",
            "DIRECT_POWER_SUM_BOUNDARY_EVALUATION",
            "LEADING_COEFFICIENT_LONG_DIVISION",
            "FACTORED_SUFFIX_ROTATION_BOUNDARY_FORMULA",
        ],
        "depth_cases": cases,
        "primary_depth64_boundary_commitment": primary[
            "boundary_commitment"
        ],
        "reuse_depth37_boundary_commitment": reuse["boundary_commitment"],
        "module_order_noncommuting_mismatch_cells": noncommuting,
        "analytic_rank_certificate": (
            "DPLUS2_HANKEL_ANTI_DIAGONAL_IS_THE_NONZERO_FINAL_COEFFICIENT"
        ),
        "finite_angle_sampling_used": False,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
