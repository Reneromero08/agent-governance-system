"""Exact reference verifier for the Jacobian-holonomy Small Wall bridge.

This module verifies representation identities only. It deliberately uses explicit
finite enumeration for small reference cases and never presents that enumeration as the
missing native catalytic fiber pushforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isqrt
from typing import Iterable, Mapping, Sequence


Q = Fraction
Monomial = tuple[int, int, int]


@dataclass(frozen=True)
class Polynomial:
    """Sparse exact polynomial in x, y, z."""

    terms: Mapping[Monomial, Fraction]

    def __post_init__(self) -> None:
        cleaned = {
            monomial: Fraction(coefficient)
            for monomial, coefficient in self.terms.items()
            if coefficient
        }
        object.__setattr__(self, "terms", cleaned)

    @staticmethod
    def constant(value: int | Fraction) -> "Polynomial":
        value = Fraction(value)
        return Polynomial({(0, 0, 0): value} if value else {})

    @staticmethod
    def variable(index: int) -> "Polynomial":
        if index not in (0, 1, 2):
            raise ValueError("variable index must be 0, 1, or 2")
        exponent = [0, 0, 0]
        exponent[index] = 1
        return Polynomial({tuple(exponent): Fraction(1)})

    def __add__(self, other: object) -> "Polynomial":
        rhs = coerce_polynomial(other)
        result = dict(self.terms)
        for monomial, coefficient in rhs.terms.items():
            result[monomial] = result.get(monomial, Fraction(0)) + coefficient
            if not result[monomial]:
                del result[monomial]
        return Polynomial(result)

    def __radd__(self, other: object) -> "Polynomial":
        return self + other

    def __neg__(self) -> "Polynomial":
        return Polynomial(
            {monomial: -coefficient for monomial, coefficient in self.terms.items()}
        )

    def __sub__(self, other: object) -> "Polynomial":
        return self + (-coerce_polynomial(other))

    def __rsub__(self, other: object) -> "Polynomial":
        return coerce_polynomial(other) - self

    def __mul__(self, other: object) -> "Polynomial":
        rhs = coerce_polynomial(other)
        result: dict[Monomial, Fraction] = {}
        for left_monomial, left_coefficient in self.terms.items():
            for right_monomial, right_coefficient in rhs.terms.items():
                monomial = tuple(
                    left_monomial[index] + right_monomial[index]
                    for index in range(3)
                )
                result[monomial] = (
                    result.get(monomial, Fraction(0))
                    + left_coefficient * right_coefficient
                )
        return Polynomial(result)

    def __rmul__(self, other: object) -> "Polynomial":
        return self * other

    def __pow__(self, exponent: int) -> "Polynomial":
        if exponent < 0:
            raise ValueError("negative polynomial powers are unsupported")
        result = Polynomial.constant(1)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def derivative(self, variable_index: int) -> "Polynomial":
        result: dict[Monomial, Fraction] = {}
        for monomial, coefficient in self.terms.items():
            exponent = monomial[variable_index]
            if exponent == 0:
                continue
            reduced = list(monomial)
            reduced[variable_index] -= 1
            result[tuple(reduced)] = coefficient * exponent
        return Polynomial(result)

    def evaluate(
        self,
        x_value: int | Fraction,
        y_value: int | Fraction,
        z_value: int | Fraction,
    ) -> Fraction:
        values = (Fraction(x_value), Fraction(y_value), Fraction(z_value))
        total = Fraction(0)
        for monomial, coefficient in self.terms.items():
            term = coefficient
            for value, exponent in zip(values, monomial, strict=True):
                term *= value**exponent
            total += term
        return total


def coerce_polynomial(value: object) -> Polynomial:
    if isinstance(value, Polynomial):
        return value
    if isinstance(value, (int, Fraction)):
        return Polynomial.constant(value)
    return NotImplemented  # type: ignore[return-value]


def determinant_3x3(matrix: Sequence[Sequence[Polynomial]]) -> Polynomial:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("expected a 3 by 3 matrix")
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


X = Polynomial.variable(0)
Y = Polynomial.variable(1)
Z = Polynomial.variable(2)
ONE = Polynomial.constant(1)

A = (ONE + X * Y) ** 3 * Z + Y**2 * (ONE + X * Y) * (4 + 3 * X * Y)
B = Y + 3 * X * (ONE + X * Y) ** 2 * Z + 3 * X * Y**2 * (4 + 3 * X * Y)
C = 2 * X - 3 * X**2 * Y - X**3 * Z
PSI = (-Q(1, 2) * A, B, C)
TARGET = (Q(1, 8), Q(0), Q(0))

FIBER_POINTS = (
    (Q(-1), Q(3, 2), Q(13, 2)),
    (Q(0), Q(0), Q(-1, 4)),
    (Q(1), Q(-3, 2), Q(13, 2)),
)


def jacobian_determinant() -> Polynomial:
    matrix = tuple(
        tuple(component.derivative(variable) for variable in range(3))
        for component in PSI
    )
    return determinant_3x3(matrix)


def map_point(point: Sequence[int | Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    if len(point) != 3:
        raise ValueError("a carrier point must have three coordinates")
    return tuple(component.evaluate(*point) for component in PSI)  # type: ignore[return-value]


def groebner_fiber_parameterization(
    sheet: int | Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    """Return the point defined by the exact fiber basis."""

    x_value = Fraction(sheet)
    if x_value**3 != x_value:
        raise ValueError("sheet must satisfy x^3 - x = 0")
    return (
        x_value,
        -Q(3, 2) * x_value,
        Q(27, 4) * x_value**2 - Q(1, 4),
    )


def sheet_selectors(sheet: int | Fraction) -> tuple[Fraction, Fraction, Fraction]:
    """Return null, false, and true idempotent values."""

    x_value = Fraction(sheet)
    e_null = x_value * (x_value - 1) / 2
    e_false = 1 - x_value**2
    e_true = x_value * (x_value + 1) / 2
    return e_null, e_false, e_true


Literal = tuple[int, bool]
Clause = tuple[Literal, Literal, Literal]
Formula = tuple[Clause, ...]


def literal_selector(literal: Literal, sheet: int) -> Fraction:
    variable_index, positive = literal
    if variable_index < 0:
        raise ValueError("variable index must be nonnegative")
    _, e_false, e_true = sheet_selectors(sheet)
    return e_true if positive else e_false


def formula_weight(formula: Formula, sheets: Sequence[int], variable_count: int) -> Fraction:
    if len(sheets) != variable_count:
        raise ValueError("sheet count does not match variable count")
    valid = Fraction(1)
    for sheet in sheets:
        e_null, e_false, e_true = sheet_selectors(sheet)
        valid *= e_false + e_true
        if e_null + e_false + e_true != 1:
            raise AssertionError("selector partition failed")

    clauses = Fraction(1)
    for clause in formula:
        unsatisfied = Fraction(1)
        for literal in clause:
            variable_index, _ = literal
            if variable_index >= variable_count:
                raise ValueError("literal references an undeclared variable")
            unsatisfied *= 1 - literal_selector(literal, sheets[variable_index])
        clauses *= 1 - unsatisfied
    return valid * clauses


def assignment_satisfies(formula: Formula, assignment: Sequence[bool]) -> bool:
    for clause in formula:
        if not any(
            assignment[variable_index] if positive else not assignment[variable_index]
            for variable_index, positive in clause
        ):
            return False
    return True


def brute_force_sat_count(formula: Formula, variable_count: int) -> int:
    return sum(
        assignment_satisfies(formula, assignment)
        for assignment in product((False, True), repeat=variable_count)
    )


def reference_fiber_trace(formula: Formula, variable_count: int) -> int:
    """Explicit small-case reference only, never the native operator."""

    total = Fraction(0)
    for sheets in product((-1, 0, 1), repeat=variable_count):
        total += formula_weight(formula, sheets, variable_count)
    if total.denominator != 1:
        raise AssertionError("fiber trace must be integral")
    return total.numerator


def first_primes(count: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be nonnegative")
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        limit = isqrt(candidate)
        if all(candidate % prime for prime in primes if prime <= limit):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def modular_signature(value: int, variable_count: int) -> tuple[int, ...]:
    if value < 0 or value > 2**variable_count:
        raise ValueError("value is outside the SAT-count range")
    return tuple(value % prime for prime in first_primes(variable_count + 1))


def modular_sieve_is_nonzero(signature: Iterable[int]) -> bool:
    return any(residue != 0 for residue in signature)


def verify_exact_reference() -> dict[str, object]:
    determinant_ok = jacobian_determinant() == Polynomial.constant(1)
    fiber_ok = all(map_point(point) == TARGET for point in FIBER_POINTS)
    parameterization_ok = tuple(
        groebner_fiber_parameterization(sheet) for sheet in (-1, 0, 1)
    ) == FIBER_POINTS

    selector_rows = tuple(sheet_selectors(sheet) for sheet in (-1, 0, 1))
    selectors_ok = selector_rows == (
        (Q(1), Q(0), Q(0)),
        (Q(0), Q(1), Q(0)),
        (Q(0), Q(0), Q(1)),
    )

    example_formula: Formula = (
        ((0, True), (1, True), (1, True)),
        ((0, False), (1, True), (1, True)),
    )
    count = brute_force_sat_count(example_formula, 2)
    trace = reference_fiber_trace(example_formula, 2)
    sieve_zero_ok = not modular_sieve_is_nonzero(modular_signature(0, 2))
    sieve_positive_ok = all(
        modular_sieve_is_nonzero(modular_signature(value, 2))
        for value in range(1, 2**2 + 1)
    )

    return {
        "jacobian_unit": determinant_ok,
        "three_fiber_points": fiber_ok,
        "fiber_parameterization": parameterization_ok,
        "sheet_idempotents": selectors_ok,
        "example_sat_count": count,
        "example_fiber_trace": trace,
        "fiber_trace_matches": count == trace,
        "prime_sieve_zero": sieve_zero_ok,
        "prime_sieve_positive": sieve_positive_ok,
        "claim_ceiling": "NATIVE_CATALYTIC_FIBER_PUSHFORWARD_NOT_ESTABLISHED",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(verify_exact_reference(), indent=2, sort_keys=True))
