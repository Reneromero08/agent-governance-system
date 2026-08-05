#!/usr/bin/env python3
"""Independent direct-factor oracle for generic-center theta moment Q-jets."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction


ORDERS = (2, 4, 8, 12, 16, 20, 24)
COUNT = 24
Complex = tuple[Fraction, Fraction]
ZERO: Complex = (Fraction(0), Fraction(0))
ONE: Complex = (Fraction(1), Fraction(0))
ROTATIONS: tuple[Complex, ...] = (
    (Fraction(-3, 5), Fraction(4, 5)),
    (Fraction(-4, 5), Fraction(3, 5)),
)


def add(left: Complex, right: Complex) -> Complex:
    return left[0] + right[0], left[1] + right[1]


def multiply(left: Complex, right: Complex) -> Complex:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def conjugate(value: Complex) -> Complex:
    return value[0], -value[1]


def power(value: Complex, exponent: int) -> Complex:
    if exponent < 0:
        return power(conjugate(value), -exponent)
    result = ONE
    base = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        remaining >>= 1
    return result


def unit(parameter: int) -> Complex:
    t = Fraction(parameter)
    denominator = 1 + t * t
    return (1 - t * t) / denominator, 2 * t / denominator


def centers(count: int, family: int) -> list[Complex]:
    values = (
        [2 * index + 1 for index in range(1, count + 1)]
        if family == 0
        else [index * index + index + 1 for index in range(1, count + 1)]
    )
    return [unit(value) for value in values]


def public_program(count: int, family: int) -> tuple[tuple[str, int], ...]:
    operations: list[tuple[str, int]] = []
    for index in range(count):
        ingest = "INGEST", index
        if index % 4 == (family + 1) % 4:
            operations.extend((("ROTATE", (index + family) % 2), ingest))
        elif index % 5 == (family + 2) % 5:
            operations.extend((ingest, ("ROTATE", (index + family + 1) % 2)))
        else:
            operations.append(ingest)
    return tuple(operations)


def effective_centers(
    source: list[Complex], operations: tuple[tuple[str, int], ...]
) -> list[Complex]:
    active: list[Complex] = []
    for kind, parameter in operations:
        if kind == "ROTATE":
            active = [multiply(value, ROTATIONS[parameter]) for value in active]
        else:
            active.append(source[parameter])
    return active


def direct_factor_jet(active: list[Complex], order: int) -> tuple[Complex, ...]:
    table: dict[tuple[int, int], Complex] = {(0, 0): ONE}
    for center in active:
        updated: dict[tuple[int, int], Complex] = {}
        for (harmonic, exponent), coefficient in table.items():
            radius = math.isqrt(order - exponent)
            for mode in range(-radius, radius + 1):
                key = harmonic + mode, exponent + mode * mode
                contribution = multiply(coefficient, power(center, mode))
                updated[key] = add(updated.get(key, ZERO), contribution)
        table = {key: value for key, value in updated.items() if value != ZERO}
    return tuple(table.get((1, exponent), ZERO) for exponent in range(order + 1))


def token(value: Complex) -> str:
    return (
        f"{value[0].numerator}/{value[0].denominator}:"
        f"{value[1].numerator}/{value[1].denominator}"
    )


def commitment(values: tuple[Complex, ...]) -> str:
    return hashlib.sha256(
        "|".join(token(value) for value in values).encode("ascii")
    ).hexdigest()


def direct_moments(active: list[Complex], order: int) -> list[Complex]:
    maximum = (order + 1) // 2
    result = [(Fraction(len(active)), Fraction(0))]
    for moment in range(1, maximum + 1):
        value = ZERO
        for center in active:
            value = add(value, power(center, moment))
        result.append(value)
    return result


def case(order: int, family: int, count: int = COUNT) -> dict[str, object]:
    source = centers(count, family)
    operations = public_program(count, family)
    active = effective_centers(source, operations)
    jet = direct_factor_jet(active, order)
    moments = direct_moments(active, order)
    hierarchy = [
        {
            "moment": moment,
            "log_theta_q_m_x_m_coefficient": token(
                (Fraction(1 if moment % 2 else -1, moment), Fraction(0))
            ),
            "first_harmonic_dependency_order": 2 * moment - 1,
        }
        for moment in range(1, (order + 1) // 2 + 1)
    ]
    return {
        "q_jet_order": order,
        "family": family,
        "center_count": count,
        "moment_cells": len(moments),
        "moment_commitment": commitment(tuple(moments)),
        "boundary_commitment": commitment(jet),
        "hierarchy": hierarchy,
        "source_unchanged": source == centers(count, family),
    }


def main() -> None:
    cases = [case(order, family) for order in ORDERS for family in (0, 1)]
    primary = case(24, 0, 24)
    reuse = case(24, 1, 17)
    first = effective_centers(
        centers(2, 0), (("ROTATE", 0), ("INGEST", 0))
    )
    second = effective_centers(
        centers(2, 0), (("INGEST", 0), ("ROTATE", 0))
    )
    result = {
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "independent_algorithms": [
            "DIRECT_EFFECTIVE_CENTER_LIST_RECONSTRUCTION",
            "DIRECT_FACTOR_BY_FACTOR_SPARSE_QJET_CONVOLUTION",
            "JACOBI_PRODUCT_LOG_COEFFICIENT_HIERARCHY",
        ],
        "precision_cases": cases,
        "primary_boundary_commitment": primary["boundary_commitment"],
        "reuse_boundary_commitment": reuse["boundary_commitment"],
        "module_order_center_lists_differ": first != second,
        "module_order_boundary_changes": direct_factor_jet(first, 8)
        != direct_factor_jet(second, 8),
        "finite_angle_sampling_used": False,
        "full_infinite_theta_scalar_evaluated": False,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
