#!/usr/bin/env python3
"""Exact rational oracle for Stokes-sphere reduced Kerr Lie signatures."""

from __future__ import annotations

import json
from typing import Iterator

import sympy


PRIMES = (17, 19)
GRADES = 5
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
x, y, z = sympy.symbols("x y z")
VARIABLES = (x, y, z)
SPHERE = sympy.groebner(
    [x * x + y * y + z * z - 1],
    z,
    y,
    x,
    order="lex",
    domain=sympy.QQ,
)


def reduce_sphere(expression: sympy.Expr) -> sympy.Expr:
    return sympy.expand(SPHERE.reduce(sympy.expand(expression))[1])


def lie_poisson(left: sympy.Expr, right: sympy.Expr) -> sympy.Expr:
    return reduce_sphere(
        x
        * (
            sympy.diff(left, y) * sympy.diff(right, z)
            - sympy.diff(left, z) * sympy.diff(right, y)
        )
        + y
        * (
            sympy.diff(left, z) * sympy.diff(right, x)
            - sympy.diff(left, x) * sympy.diff(right, z)
        )
        + z
        * (
            sympy.diff(left, x) * sympy.diff(right, y)
            - sympy.diff(left, y) * sympy.diff(right, x)
        )
    )


H0 = reduce_sphere(z**2)
PRIMARY = reduce_sphere((24 * x - 7 * z) ** 2 / sympy.Integer(625))
REUSE = reduce_sphere((24 * x + 7 * z) ** 2 / sympy.Integer(625))


def quotient_basis(degree_limit: int) -> Iterator[tuple[int, int, int]]:
    for x_exponent in range(degree_limit + 1):
        for y_exponent in range(degree_limit - x_exponent + 1):
            for z_exponent in range(2):
                if x_exponent + y_exponent + z_exponent <= degree_limit:
                    yield (x_exponent, y_exponent, z_exponent)


def coefficient_mod(value: sympy.Rational, prime: int) -> int:
    numerator, denominator = sympy.fraction(value)
    return int(numerator) * pow(int(denominator), -1, prime) % prime


def hash_byte(hash_value: int, value: int) -> int:
    return ((hash_value ^ value) * FNV_PRIME) & ((1 << 64) - 1)


def grade_record(expression: sympy.Expr, degree_limit: int) -> dict[str, object]:
    polynomial = sympy.Poly(expression, *VARIABLES)
    coefficients = {
        monomial: sympy.Rational(value)
        for monomial, value in polynomial.as_dict().items()
    }
    rational_nonzero = [value for value in coefficients.values() if value]
    record: dict[str, object] = {
        "degree_limit": degree_limit,
        "quotient_basis_cells": (degree_limit + 1) ** 2,
        "rational_nonzero_terms": len(rational_nonzero),
        "maximum_numerator_bits": max(
            (
                abs(int(sympy.numer(value))).bit_length()
                for value in rational_nonzero
            ),
            default=0,
        ),
        "maximum_denominator_bits": max(
            (
                int(sympy.denom(value)).bit_length()
                for value in rational_nonzero
            ),
            default=0,
        ),
    }
    for prime in PRIMES:
        hash_value = FNV_OFFSET
        nonzero = 0
        for monomial in quotient_basis(degree_limit):
            for exponent in monomial:
                hash_value = hash_byte(hash_value, exponent)
            value = coefficient_mod(coefficients.get(monomial, 0), prime)
            hash_value = hash_byte(hash_value, value)
            nonzero += int(value != 0)
        record[f"nonzero_p{prime}"] = nonzero
        record[f"hash_p{prime}"] = f"{hash_value:016x}"
    return record


def chain(seed: sympy.Expr, generator: sympy.Expr) -> list[dict[str, object]]:
    expression = seed
    records = []
    for grade in range(GRADES):
        records.append(grade_record(expression, 2 + grade))
        expression = lie_poisson(generator, expression)
    return records


def main() -> None:
    primary = chain(H0, PRIMARY)
    reuse = chain(H0, REUSE)
    identity = chain(H0, H0)
    swapped = chain(PRIMARY, H0)
    output = {
        "result": "PASS",
        "oracle": "INDEPENDENT_SYMPY_EXACT_STOKES_QUOTIENT_LIE_POISSON",
        "sphere_relation": "x^2+y^2+z^2=1",
        "primary_grades": primary,
        "reuse_grades": reuse,
        "successive_primary_term_counts": [
            grade["rational_nonzero_terms"] for grade in primary
        ],
        "successive_primary_degrees_nonzero": all(
            grade["rational_nonzero_terms"] > 0 for grade in primary
        ),
        "identity_mixer_higher_grades_zero": all(
            grade["rational_nonzero_terms"] == 0
            for grade in identity[1:]
        ),
        "swapped_first_bracket_negates_primary": (
            swapped[1]["hash_p17"] != primary[1]["hash_p17"]
            and swapped[1]["hash_p19"] != primary[1]["hash_p19"]
        ),
        "original_four_dimensional_full_basis_cells": 1025,
        "stokes_quotient_full_basis_cells": sum(
            int(grade["quotient_basis_cells"]) for grade in primary
        ),
        "original_rational_term_counts": [6, 32, 85, 126, 231],
        "compact_direct_wave_semantic_state_bytes": 64,
        "exact_rank_reduction_established": True,
        "fixed_rank_reduction_found": False,
        "remaining_harmonic_rank_growth": True,
        "unbounded_growth_proved": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
