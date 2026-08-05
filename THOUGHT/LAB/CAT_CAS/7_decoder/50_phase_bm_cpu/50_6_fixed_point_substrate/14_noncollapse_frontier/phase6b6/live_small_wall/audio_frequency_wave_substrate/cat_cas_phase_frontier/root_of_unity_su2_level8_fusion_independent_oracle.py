#!/usr/bin/env python3
"""Independent polynomial-quotient oracle for the SU(2)_8 fusion carrier.

This file imports no CAT_CAS module.  Production evolves nine simple-object
coefficients and reverses fusion with a tridiagonal pivot solve.  The oracle
instead evolves an ordinary polynomial in x modulo U_9(x/2), computes fusion
inverses by polynomial extended Euclid, and evaluates the boundary at the
exact root-of-unity quantum dimension x=zeta_40^2+zeta_40^-2.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)

FIELD_DEGREE = 16
ROOT_ORDER = 40
SIMPLE_OBJECTS = 9
DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
PRIMARY_DEPTH = 128
REUSE_DEPTH = 73


@dataclass(frozen=True)
class E:
    coordinates: tuple[Fraction, ...]

    @staticmethod
    def integer(value: int) -> "E":
        return E((Fraction(value),) + (Fraction(0),) * 15)

    @staticmethod
    def root(power: int) -> "E":
        exponent = power % ROOT_ORDER
        values = [Fraction(0)] * (exponent + 1)
        values[exponent] = Fraction(1)
        return E(reduce_root_polynomial(values))

    def __add__(self, other: "E") -> "E":
        return E(tuple(a + b for a, b in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other: "E") -> "E":
        return E(tuple(a - b for a, b in zip(self.coordinates, other.coordinates)))

    def __neg__(self) -> "E":
        return E(tuple(-value for value in self.coordinates))

    def __mul__(self, other: "E") -> "E":
        values = [Fraction(0)] * 31
        for i, left in enumerate(self.coordinates):
            for j, right in enumerate(other.coordinates):
                values[i + j] += left * right
        return E(reduce_root_polynomial(values))

    def inverse(self) -> "E":
        if self == ZERO:
            raise ZeroDivisionError("zero cyclotomic element")
        columns = []
        for index in range(FIELD_DEGREE):
            columns.append((self * E.root(index)).coordinates)
        matrix = [
            [columns[column][row] for column in range(FIELD_DEGREE)]
            + [Fraction(row == 0)]
            for row in range(FIELD_DEGREE)
        ]
        pivot_row = 0
        for column in range(FIELD_DEGREE):
            pivot = next(
                row
                for row in range(pivot_row, FIELD_DEGREE)
                if matrix[row][column]
            )
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            scale = matrix[pivot_row][column]
            matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
            for row in range(FIELD_DEGREE):
                if row == pivot_row or not matrix[row][column]:
                    continue
                factor = matrix[row][column]
                matrix[row] = [
                    value - factor * basis
                    for value, basis in zip(
                        matrix[row], matrix[pivot_row], strict=True
                    )
                ]
            pivot_row += 1
        return E(tuple(matrix[row][-1] for row in range(FIELD_DEGREE)))

    def __truediv__(self, other: "E") -> "E":
        return self * other.inverse()

    def token(self) -> str:
        return ":".join(
            f"{value.numerator}/{value.denominator}"
            for value in self.coordinates
        )


def reduce_root_polynomial(values: list[Fraction]) -> tuple[Fraction, ...]:
    work = values.copy()
    work.extend([Fraction(0)] * max(0, FIELD_DEGREE - len(work)))
    for degree in range(len(work) - 1, 15, -1):
        value = work[degree]
        work[degree] = Fraction(0)
        work[degree - 4] += value
        work[degree - 8] -= value
        work[degree - 12] += value
        work[degree - 16] -= value
    return tuple(work[:FIELD_DEGREE])


ZERO = E.integer(0)
ONE = E.integer(1)
SOURCE = E.root(5)
FACTORS = (E.root(3), E.root(7))
TWIST_POWERS = (1, 3)
Polynomial = list[E]


def trim(values: Polynomial) -> Polynomial:
    while values and values[-1] == ZERO:
        values.pop()
    return values


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    size = max(len(left), len(right))
    result = [ZERO] * size
    for index in range(size):
        result[index] = (
            left[index] if index < len(left) else ZERO
        ) + (right[index] if index < len(right) else ZERO)
    return trim(result)


def poly_subtract(left: Polynomial, right: Polynomial) -> Polynomial:
    size = max(len(left), len(right))
    result = [ZERO] * size
    for index in range(size):
        result[index] = (
            left[index] if index < len(left) else ZERO
        ) - (right[index] if index < len(right) else ZERO)
    return trim(result)


def poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    if not left or not right:
        return []
    result = [ZERO] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = result[i + j] + a * b
    return trim(result)


def poly_scale(values: Polynomial, scalar: E) -> Polynomial:
    return trim([value * scalar for value in values])


def poly_divmod(
    numerator: Polynomial, denominator: Polynomial
) -> tuple[Polynomial, Polynomial]:
    divisor = trim(denominator.copy())
    if not divisor:
        raise ZeroDivisionError("zero quotient polynomial")
    remainder = trim(numerator.copy())
    quotient = [ZERO] * max(1, len(remainder) - len(divisor) + 1)
    inverse_lead = divisor[-1].inverse()
    while remainder and len(remainder) >= len(divisor):
        degree = len(remainder) - len(divisor)
        factor = remainder[-1] * inverse_lead
        quotient[degree] = quotient[degree] + factor
        for index, coefficient in enumerate(divisor):
            remainder[index + degree] = (
                remainder[index + degree] - factor * coefficient
            )
        trim(remainder)
    return trim(quotient), remainder


def character_polynomials(maximum: int) -> list[Polynomial]:
    basis = [[ONE], [ZERO, ONE]]
    for index in range(1, maximum):
        basis.append(
            poly_subtract([ZERO] + basis[index], basis[index - 1])
        )
    return basis[: maximum + 1]


CHARACTERS = character_polynomials(SIMPLE_OBJECTS)
SIMPLE_BASIS = CHARACTERS[:SIMPLE_OBJECTS]
FUSION_MODULUS = CHARACTERS[SIMPLE_OBJECTS]


def quotient_reduce(polynomial: Polynomial) -> Polynomial:
    return poly_divmod(polynomial, FUSION_MODULUS)[1]


def modular_inverse(polynomial: Polynomial) -> Polynomial:
    old_r = FUSION_MODULUS.copy()
    r = polynomial.copy()
    old_s: Polynomial = []
    s: Polynomial = [ONE]
    while r:
        quotient, remainder = poly_divmod(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, poly_subtract(old_s, poly_multiply(quotient, s))
    if len(old_r) != 1 or old_r[0] == ZERO:
        raise ZeroDivisionError("fusion factor is not invertible")
    return quotient_reduce(poly_scale(old_s, old_r[0].inverse()))


FUSION_MULTIPLIERS = ([ONE, FACTORS[0]], [ONE, FACTORS[1]])
FUSION_INVERSES = tuple(
    modular_inverse(multiplier) for multiplier in FUSION_MULTIPLIERS
)


def simple_to_polynomial(coefficients: list[E]) -> Polynomial:
    result: Polynomial = []
    for coefficient, basis in zip(coefficients, SIMPLE_BASIS, strict=True):
        result = poly_add(result, poly_scale(basis, coefficient))
    return quotient_reduce(result)


def polynomial_to_simple(polynomial: Polynomial) -> list[E]:
    residual = polynomial.copy() + [ZERO] * (SIMPLE_OBJECTS - len(polynomial))
    coefficients = [ZERO] * SIMPLE_OBJECTS
    for degree in range(SIMPLE_OBJECTS - 1, -1, -1):
        coefficient = residual[degree]
        coefficients[degree] = coefficient
        for index, basis_coefficient in enumerate(SIMPLE_BASIS[degree]):
            residual[index] = residual[index] - coefficient * basis_coefficient
    if any(value != ZERO for value in residual):
        raise AssertionError("simple-object conversion left a residual")
    return coefficients


def evaluate(polynomial: Polynomial, point: E) -> E:
    result = ZERO
    for coefficient in reversed(polynomial):
        result = result * point + coefficient
    return result


DELTA = E.root(2) + E.root(-2)
if evaluate(FUSION_MODULUS, DELTA) != ZERO:
    raise RuntimeError("independent Jones-Wenzl root check failed")


@dataclass(frozen=True)
class Program:
    depth: int
    family: int

    @property
    def steps(self) -> int:
        return 2 * self.depth

    def operation(self, step: int) -> tuple[str, int]:
        index, position = divmod(step, 2)
        twist = "TWIST_CASIMIR", (index + self.family) % 2
        fusion = "FUSE_FUNDAMENTAL", (3 * index + self.family) % 2
        ordered = (twist, fusion) if (index + self.family) % 3 else (fusion, twist)
        return ordered[position]


def apply_operation(
    polynomial: Polynomial, operation: tuple[str, int], inverse: bool
) -> Polynomial:
    kind, parameter = operation
    if kind == "FUSE_FUNDAMENTAL":
        multiplier = (
            FUSION_INVERSES[parameter]
            if inverse
            else FUSION_MULTIPLIERS[parameter]
        )
        return quotient_reduce(poly_multiply(polynomial, multiplier))
    if kind == "TWIST_CASIMIR":
        coefficients = polynomial_to_simple(polynomial)
        power = TWIST_POWERS[parameter]
        updated = [
            coefficient
            * E.root(
                (-1 if inverse else 1)
                * power
                * simple_object
                * (simple_object + 2)
            )
            for simple_object, coefficient in enumerate(coefficients)
        ]
        return simple_to_polynomial(updated)
    raise ValueError("unknown independent SU2 level-8 operation")


def source_polynomial() -> Polynomial:
    return simple_to_polynomial([ONE, SOURCE] + [ZERO] * 7)


def execute(depth: int, family: int) -> Polynomial:
    polynomial = source_polynomial()
    program = Program(depth, family)
    for step in range(program.steps):
        polynomial = apply_operation(polynomial, program.operation(step), False)
    return polynomial


def restore(polynomial: Polynomial, depth: int, family: int) -> Polynomial:
    program = Program(depth, family)
    for step in range(program.steps - 1, -1, -1):
        polynomial = apply_operation(polynomial, program.operation(step), True)
    return polynomial


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload(values: list[E] | tuple[E, ...]) -> int:
    return sum(
        signed_bits(coordinate.numerator) + coordinate.denominator.bit_length()
        for value in values
        for coordinate in value.coordinates
    )


def state_commitment(coefficients: list[E]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in coefficients).encode("ascii")
    ).hexdigest()


def boundary_commitment(value: E) -> str:
    return hashlib.sha256(value.token().encode("ascii")).hexdigest()


def case(depth: int, family: int) -> dict[str, object]:
    polynomial = execute(depth, family)
    coefficients = polynomial_to_simple(polynomial)
    boundary = evaluate(polynomial, DELTA)
    restored = restore(polynomial.copy(), depth, family)
    return {
        "depth": depth,
        "family": family,
        "simple_object_field_cells": len(coefficients),
        "nonzero_simple_object_field_cells": sum(
            value != ZERO for value in coefficients
        ),
        "simple_object_payload_bits": payload(coefficients),
        "ordinary_polynomial_cells": len(polynomial),
        "ordinary_polynomial_payload_bits": payload(polynomial),
        "state_commitment": state_commitment(coefficients),
        "boundary_commitment": boundary_commitment(boundary),
        "boundary_payload_bits": payload([boundary]),
        "source_restored": restored == source_polynomial(),
    }


def main() -> None:
    cases = [case(depth, family) for depth in DEPTHS for family in (0, 1)]
    primary = case(PRIMARY_DEPTH, 0)
    reuse = case(REUSE_DEPTH, 1)
    source = source_polynomial()
    twist_then_fusion = apply_operation(
        apply_operation(source.copy(), ("TWIST_CASIMIR", 0), False),
        ("FUSE_FUNDAMENTAL", 0),
        False,
    )
    fusion_then_twist = apply_operation(
        apply_operation(source.copy(), ("FUSE_FUNDAMENTAL", 0), False),
        ("TWIST_CASIMIR", 0),
        False,
    )
    if not all(item["source_restored"] for item in cases + [reuse]):
        raise RuntimeError("independent SU2 level-8 restoration failed")
    if (
        primary["simple_object_field_cells"] != 9
        or primary["simple_object_payload_bits"] != 15330
        or primary["boundary_payload_bits"] != 1761
    ):
        raise RuntimeError("independent SU2 level-8 primary tuple drifted")
    result = {
        "schema": "cat_cas.root_of_unity_su2_level8_fusion_oracle.v1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "independent_algorithms": [
            "ORDINARY_POLYNOMIAL_QUOTIENT_BY_U9_X_OVER2",
            "FUSION_FACTOR_MODULAR_INVERSE_BY_POLYNOMIAL_EXTENDED_EUCLID",
            "QZETA40_INVERSION_BY_SIXTEEN_BY_SIXTEEN_RATIONAL_LINEAR_SOLVE",
            "QUANTUM_DIMENSION_BOUNDARY_BY_POLYNOMIAL_EVALUATION_AT_ZETA40_SQUARED_PLUS_INVERSE",
        ],
        "cases": cases,
        "primary": primary,
        "reuse": reuse,
        "module_order_polynomials_differ": twist_then_fusion
        != fusion_then_twist,
        "module_order_boundaries_differ": evaluate(twist_then_fusion, DELTA)
        != evaluate(fusion_then_twist, DELTA),
        "jones_wenzl_polynomial_vanishes_at_quantum_dimension": True,
        "fusion_inverse_polynomials_verified": all(
            quotient_reduce(poly_multiply(multiplier, inverse)) == [ONE]
            for multiplier, inverse in zip(
                FUSION_MULTIPLIERS, FUSION_INVERSES, strict=True
            )
        ),
        "primary_resource_tuple_reproduced": True,
        "path_or_group_element_enumeration_used": False,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
