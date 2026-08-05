#!/usr/bin/env python3
"""Independent ordinary-polynomial oracle for the SU(2) character carrier.

This file imports no CAT_CAS module.  Production evolves irreducible
character coefficients directly.  The oracle instead uses the representation
ring isomorphism chi_r = U_r(x/2), evolves ordinary polynomials in x=chi_1,
and performs triangular Chebyshev basis conversions only for Casimir phases
and sealed comparisons.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from fractions import Fraction


sys.set_int_max_str_digits(0)

DEPTHS = (1, 2, 4, 8, 16, 32)
PRIMARY_DEPTH = 32
REUSE_DEPTH = 19
Complex = tuple[Fraction, Fraction]
Polynomial = list[Complex]
Operation = tuple[str, int]
ZERO: Complex = (Fraction(0), Fraction(0))
ONE: Complex = (Fraction(1), Fraction(0))
SOURCE: Complex = (Fraction(12, 13), Fraction(5, 13))
PHASES: tuple[Complex, ...] = (
    (Fraction(3, 5), Fraction(4, 5)),
    (Fraction(5, 13), Fraction(12, 13)),
)
FACTORS: tuple[Complex, ...] = (
    (Fraction(8, 17), Fraction(15, 17)),
    (Fraction(7, 25), Fraction(24, 25)),
)


def add(left: Complex, right: Complex) -> Complex:
    return left[0] + right[0], left[1] + right[1]


def subtract(left: Complex, right: Complex) -> Complex:
    return left[0] - right[0], left[1] - right[1]


def multiply(left: Complex, right: Complex) -> Complex:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def scale(value: Complex, scalar: int | Fraction) -> Complex:
    factor = Fraction(scalar)
    return value[0] * factor, value[1] * factor


def inverse(value: Complex) -> Complex:
    norm = value[0] * value[0] + value[1] * value[1]
    if not norm:
        raise ZeroDivisionError("zero complex rational")
    return value[0] / norm, -value[1] / norm


def divide(left: Complex, right: Complex) -> Complex:
    return multiply(left, inverse(right))


def power(value: Complex, exponent: int) -> Complex:
    if exponent < 0:
        return power(inverse(value), -exponent)
    result = ONE
    factor = value
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = multiply(result, factor)
        factor = multiply(factor, factor)
        remaining >>= 1
    return result


def trim(polynomial: Polynomial) -> Polynomial:
    while len(polynomial) > 1 and polynomial[-1] == ZERO:
        polynomial.pop()
    return polynomial


def character_polynomials(maximum: int) -> list[Polynomial]:
    result: list[Polynomial] = [[ONE]]
    if maximum == 0:
        return result
    result.append([ZERO, ONE])
    for highest_weight in range(1, maximum):
        shifted = [ZERO] + result[highest_weight]
        previous = result[highest_weight - 1]
        size = max(len(shifted), len(previous))
        candidate = [ZERO] * size
        for index in range(size):
            left = shifted[index] if index < len(shifted) else ZERO
            right = previous[index] if index < len(previous) else ZERO
            candidate[index] = subtract(left, right)
        result.append(trim(candidate))
    return result


def characters_to_polynomial(coefficients: list[Complex]) -> Polynomial:
    basis = character_polynomials(len(coefficients) - 1)
    result = [ZERO] * len(coefficients)
    for coefficient, character in zip(coefficients, basis, strict=True):
        for degree, integer_coefficient in enumerate(character):
            result[degree] = add(
                result[degree], multiply(coefficient, integer_coefficient)
            )
    return trim(result)


def polynomial_to_characters(polynomial: Polynomial) -> list[Complex]:
    residual = polynomial.copy()
    basis = character_polynomials(len(residual) - 1)
    coefficients = [ZERO] * len(residual)
    for highest_weight in range(len(residual) - 1, -1, -1):
        coefficient = residual[highest_weight]
        coefficients[highest_weight] = coefficient
        for degree, integer_coefficient in enumerate(basis[highest_weight]):
            residual[degree] = subtract(
                residual[degree], multiply(coefficient, integer_coefficient)
            )
    if any(value != ZERO for value in residual):
        raise AssertionError("Chebyshev character conversion left a residual")
    return trim(coefficients)


def multiply_linear(polynomial: Polynomial, factor: Complex) -> Polynomial:
    result = [ZERO] * (len(polynomial) + 1)
    for degree, coefficient in enumerate(polynomial):
        result[degree] = add(result[degree], coefficient)
        result[degree + 1] = add(
            result[degree + 1], multiply(factor, coefficient)
        )
    return trim(result)


def divide_linear(polynomial: Polynomial, factor: Complex) -> Polynomial:
    if len(polynomial) < 2:
        raise ValueError("linear divisor absent")
    residual = polynomial.copy()
    quotient = [ZERO] * (len(polynomial) - 1)
    for degree in range(len(polynomial) - 1, 0, -1):
        coefficient = divide(residual[degree], factor)
        quotient[degree - 1] = coefficient
        residual[degree] = subtract(
            residual[degree], multiply(factor, coefficient)
        )
        residual[degree - 1] = subtract(residual[degree - 1], coefficient)
    if any(value != ZERO for value in residual):
        raise AssertionError("ordinary polynomial is not exactly divisible")
    return trim(quotient)


def public_program(depth: int, family: int) -> tuple[Operation, ...]:
    operations: list[Operation] = []
    for index in range(depth):
        composition = (
            "COMPOSE_CASIMIR_PHASE",
            (index + family) % len(PHASES),
        )
        intersection = (
            "INTERSECT_FUNDAMENTAL",
            (3 * index + 2 * family + 1) % len(FACTORS),
        )
        if (index + family) % 3:
            operations.extend((composition, intersection))
        else:
            operations.extend((intersection, composition))
    return tuple(operations)


def apply_polynomial_operation(
    polynomial: Polynomial, operation: Operation, reverse: bool = False
) -> Polynomial:
    kind, parameter = operation
    if kind == "INTERSECT_FUNDAMENTAL":
        return (
            divide_linear(polynomial, FACTORS[parameter])
            if reverse
            else multiply_linear(polynomial, FACTORS[parameter])
        )
    if kind == "COMPOSE_CASIMIR_PHASE":
        coefficients = polynomial_to_characters(polynomial)
        phase = inverse(PHASES[parameter]) if reverse else PHASES[parameter]
        updated = [
            multiply(
                coefficient,
                power(phase, highest_weight * (highest_weight + 2)),
            )
            for highest_weight, coefficient in enumerate(coefficients)
        ]
        return characters_to_polynomial(updated)
    raise ValueError("unknown operation")


def execute(
    source: list[Complex], operations: tuple[Operation, ...]
) -> tuple[Polynomial, list[Complex]]:
    polynomial = characters_to_polynomial(source)
    for operation in operations:
        polynomial = apply_polynomial_operation(polynomial, operation)
    return polynomial, polynomial_to_characters(polynomial)


def restore(
    polynomial: Polynomial, operations: tuple[Operation, ...]
) -> Polynomial:
    for operation in reversed(operations):
        polynomial = apply_polynomial_operation(polynomial, operation, True)
    return polynomial


def evaluate_polynomial(polynomial: Polynomial, point: int) -> Complex:
    result = ZERO
    for coefficient in reversed(polynomial):
        result = add(multiply(result, (Fraction(point), Fraction(0))), coefficient)
    return result


def signed_bits(value: int) -> int:
    return max(1, abs(value).bit_length() + 1)


def payload_bits(values: list[Complex] | tuple[Complex, ...]) -> int:
    return sum(
        signed_bits(component.numerator) + component.denominator.bit_length()
        for value in values
        for component in value
    )


def token(value: Complex) -> str:
    return (
        f"{value[0].numerator}/{value[0].denominator}:"
        f"{value[1].numerator}/{value[1].denominator}"
    )


def state_commitment(coefficients: list[Complex]) -> str:
    return hashlib.sha256(
        "|".join(token(value) for value in coefficients).encode("ascii")
    ).hexdigest()


def boundary_commitment(value: Complex) -> str:
    return hashlib.sha256(token(value).encode("ascii")).hexdigest()


@dataclass
class Case:
    depth: int
    family: int
    character_cells: int
    character_payload_bits: int
    polynomial_cells: int
    polynomial_payload_bits: int
    state_commitment: str
    boundary_commitment: str
    boundary_payload_bits: int
    source_restored: bool

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


def case(depth: int, family: int) -> Case:
    source = [ONE, SOURCE]
    operations = public_program(depth, family)
    polynomial, coefficients = execute(source, operations)
    boundary = evaluate_polynomial(polynomial, 2)
    restored = restore(polynomial.copy(), operations)
    return Case(
        depth=depth,
        family=family,
        character_cells=len(coefficients),
        character_payload_bits=payload_bits(coefficients),
        polynomial_cells=len(polynomial),
        polynomial_payload_bits=payload_bits(polynomial),
        state_commitment=state_commitment(coefficients),
        boundary_commitment=boundary_commitment(boundary),
        boundary_payload_bits=payload_bits([boundary]),
        source_restored=(restored == characters_to_polynomial(source)),
    )


def main() -> None:
    cases = [case(depth, family) for depth in DEPTHS for family in (0, 1)]
    primary = case(PRIMARY_DEPTH, 0)
    reuse = case(REUSE_DEPTH, 1)
    source_polynomial = characters_to_polynomial([ONE, SOURCE])
    compose_then_fuse = apply_polynomial_operation(
        apply_polynomial_operation(
            source_polynomial.copy(), ("COMPOSE_CASIMIR_PHASE", 0)
        ),
        ("INTERSECT_FUNDAMENTAL", 0),
    )
    fuse_then_compose = apply_polynomial_operation(
        apply_polynomial_operation(
            source_polynomial.copy(), ("INTERSECT_FUNDAMENTAL", 0)
        ),
        ("COMPOSE_CASIMIR_PHASE", 0),
    )
    if not all(item.source_restored for item in cases + [reuse]):
        raise AssertionError("independent SU2 polynomial restoration failed")
    if primary.character_cells != 34 or primary.character_payload_bits != 3728007:
        raise AssertionError("independent primary SU2 resource tuple drifted")
    result = {
        "schema": "cat_cas.continuous_su2_character_relation_oracle.v1",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "INDEPENDENT_ORACLE_REEXECUTION",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "oracle_imports_cat_cas_modules": False,
        "independent_algorithms": [
            "ORDINARY_POLYNOMIAL_REPRESENTATION_RING_EXECUTION",
            "TRIANGULAR_CHEBYSHEV_CHARACTER_BASIS_CONVERSION",
            "EXACT_LINEAR_POLYNOMIAL_MULTIPLY_AND_LONG_DIVISION",
            "IDENTITY_BOUNDARY_AS_ORDINARY_POLYNOMIAL_EVALUATION_AT_X2",
        ],
        "cases": [item.as_dict() for item in cases],
        "primary": primary.as_dict(),
        "reuse": reuse.as_dict(),
        "module_order_polynomials_differ": compose_then_fuse != fuse_then_compose,
        "module_order_boundaries_differ": (
            evaluate_polynomial(compose_then_fuse, 2)
            != evaluate_polynomial(fuse_then_compose, 2)
        ),
        "primary_resource_tuple_reproduced": True,
        "continuous_su2_group_element_enumeration_used": False,
        "finite_group_reduction_used": False,
        "distinct_phase_resource_established": False,
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
