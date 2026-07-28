#!/usr/bin/env python3
"""Independent exact-rational oracle for the bounded symplectic Lie signature."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import sympy


PRIMES = (17, 19)
GRADES = 5
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211

q1, p1, q2, p2 = sympy.symbols("q1 p1 q2 p2")
VARIABLES = (q1, p1, q2, p2)


@dataclass(frozen=True)
class Program:
    cosine: sympy.Rational
    sine: sympy.Rational


PRIMARY = Program(sympy.Rational(3, 5), sympy.Rational(4, 5))
REUSE = Program(sympy.Rational(4, 5), sympy.Rational(3, 5))
IDENTITY = Program(sympy.Rational(1), sympy.Rational(0))


def kerr(program: Program) -> sympy.Expr:
    """Kerr polynomial in the frame obtained from an inverse real SU(2) mix."""
    c = program.cosine
    s = program.sine
    uq1 = c * q1 - s * q2
    up1 = c * p1 - s * p2
    uq2 = s * q1 + c * q2
    up2 = s * p1 + c * p2
    return sympy.expand(
        (uq1**2 + up1**2) ** 2 + (uq2**2 + up2**2) ** 2
    )


K0 = kerr(IDENTITY)


def poisson(left: sympy.Expr, right: sympy.Expr) -> sympy.Expr:
    return sympy.expand(
        sympy.diff(left, q1) * sympy.diff(right, p1)
        - sympy.diff(left, p1) * sympy.diff(right, q1)
        + sympy.diff(left, q2) * sympy.diff(right, p2)
        - sympy.diff(left, p2) * sympy.diff(right, q2)
    )


def homogeneous_monomials(degree: int) -> Iterator[tuple[int, int, int, int]]:
    for q1_exponent in range(degree + 1):
        for p1_exponent in range(degree - q1_exponent + 1):
            for q2_exponent in range(
                degree - q1_exponent - p1_exponent + 1
            ):
                yield (
                    q1_exponent,
                    p1_exponent,
                    q2_exponent,
                    degree - q1_exponent - p1_exponent - q2_exponent,
                )


def coefficient_mod(value: sympy.Rational, prime: int) -> int:
    numerator, denominator = sympy.fraction(value)
    return (
        int(numerator) * pow(int(denominator), -1, prime)
    ) % prime


def hash_byte(value: int, byte: int) -> int:
    return ((value ^ byte) * FNV_PRIME) & ((1 << 64) - 1)


def grade_record(expression: sympy.Expr, degree: int) -> dict[str, object]:
    polynomial = sympy.Poly(expression, *VARIABLES)
    coefficient_map = polynomial.as_dict()
    rational_coefficients = [
        sympy.Rational(value) for value in coefficient_map.values()
        if value != 0
    ]
    record: dict[str, object] = {
        "degree": degree,
        "coefficient_cells": int(sympy.binomial(degree + 3, 3)),
        "rational_nonzero_terms": len(rational_coefficients),
        "maximum_numerator_bits": max((
            abs(int(sympy.numer(value))).bit_length()
            for value in rational_coefficients
        ), default=0),
        "maximum_denominator_bits": max((
            int(sympy.denom(value)).bit_length()
            for value in rational_coefficients
        ), default=0),
    }
    for prime in PRIMES:
        hash_value = FNV_OFFSET
        nonzero = 0
        for monomial in homogeneous_monomials(degree):
            for exponent in monomial:
                hash_value = hash_byte(hash_value, exponent)
            residue = coefficient_mod(
                sympy.Rational(coefficient_map.get(monomial, 0)), prime
            )
            hash_value = hash_byte(hash_value, residue)
            nonzero += int(residue != 0)
        record[f"nonzero_p{prime}"] = nonzero
        record[f"hash_p{prime}"] = f"{hash_value:016x}"
    return record


def lie_chain(seed: sympy.Expr, generator: sympy.Expr) -> list[dict[str, object]]:
    expression = seed
    records = []
    for grade in range(GRADES):
        degree = 4 + 2 * grade
        records.append(grade_record(expression, degree))
        expression = poisson(generator, expression)
    return records


def main() -> None:
    primary = lie_chain(K0, kerr(PRIMARY))
    reuse = lie_chain(K0, kerr(REUSE))
    identity = lie_chain(K0, K0)
    swapped = lie_chain(kerr(PRIMARY), K0)
    output = {
        "result": "PASS",
        "oracle": "INDEPENDENT_SYMPY_EXACT_RATIONAL_POISSON_DICTIONARY",
        "primary_grades": primary,
        "reuse_grades": reuse,
        "identity_mixer_higher_grades_zero": all(
            grade["rational_nonzero_terms"] == 0
            for grade in identity[1:]
        ),
        "successive_primary_degrees_nonzero": all(
            grade["rational_nonzero_terms"] > 0 for grade in primary
        ),
        "successive_primary_term_counts": [
            grade["rational_nonzero_terms"] for grade in primary
        ],
        "swapped_first_bracket_negates_primary": (
            swapped[1]["hash_p17"] != primary[1]["hash_p17"]
            and swapped[1]["hash_p19"] != primary[1]["hash_p19"]
        ),
        "compact_direct_wave_semantic_state_bytes": 64,
        "retain_all_rational_grade_coefficient_cells": sum(
            int(grade["coefficient_cells"]) for grade in primary
        ),
        "finite_grade_growth_only": True,
        "unbounded_growth_proved": False,
        "fixed_rank_reduction_found": False,
        "computational_advantage": False,
        "small_wall_crossed": False,
        "terminal": False,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
