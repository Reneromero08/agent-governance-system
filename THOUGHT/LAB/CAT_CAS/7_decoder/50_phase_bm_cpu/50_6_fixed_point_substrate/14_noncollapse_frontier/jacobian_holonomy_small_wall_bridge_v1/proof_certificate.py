"""Exact proof certificates for the Jacobian holonomy bridge.

The functions here verify algebraic identities used by the written case split. They do
not implement the native catalytic fiber pushforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

from .bridge_model import (
    A,
    B,
    C,
    FIBER_POINTS,
    Polynomial,
    TARGET,
    X,
    Y,
    Z,
    map_point,
)


Q = Fraction
ZERO = Polynomial.constant(0)
ONE = Polynomial.constant(1)


@dataclass(frozen=True)
class FiberAlgebraCertificate:
    """Exact identities supporting completeness of the selected target fiber."""

    c_factorization: bool
    b_invariant_lift: bool
    a_invariant_lift: bool
    b_reduction_on_l_zero: bool
    a_reduction_on_l_zero: bool
    listed_points_hit_target: bool
    x_zero_branch_point: tuple[Fraction, Fraction, Fraction]
    x_nonzero_branch_points: tuple[
        tuple[Fraction, Fraction, Fraction],
        tuple[Fraction, Fraction, Fraction],
    ]
    claim_ceiling: str

    @property
    def passed(self) -> bool:
        """Return whether every exact identity in the certificate passed."""

        return all(
            (
                self.c_factorization,
                self.b_invariant_lift,
                self.a_invariant_lift,
                self.b_reduction_on_l_zero,
                self.a_reduction_on_l_zero,
                self.listed_points_hit_target,
            )
        )


def fiber_algebra_certificate() -> FiberAlgebraCertificate:
    """Verify the polynomial identities used in the exact fiber case split.

    Set u = x*y, v = x^2*z, and L = 2 - 3*u - v. The third target equation is
    c = x*L. On the x != 0 branch, c = 0 gives L = 0. Multiplying the second and
    first source coordinates by x and x^2 respectively removes denominators after
    passing to u and v.
    """

    u = X * Y
    v = X**2 * Z
    l_value = 2 - 3 * u - v

    b_numerator = u + 3 * (1 + u) ** 2 * v + 3 * u**2 * (4 + 3 * u)
    a_numerator = (1 + u) ** 3 * v + u**2 * (1 + u) * (4 + 3 * u)

    c_factorization = C - X * l_value
    b_invariant_lift = X * B - b_numerator
    a_invariant_lift = X**2 * A - a_numerator

    # These identities reduce the target equations when L = 0.
    b_reduction = (
        b_numerator
        - 2 * (2 * u + 3)
        + 3 * (1 + u) ** 2 * l_value
    )
    a_reduction = (
        a_numerator
        - (u + 1) * (u + 2)
        + (u + 1) ** 3 * l_value
    )

    return FiberAlgebraCertificate(
        c_factorization=c_factorization == ZERO,
        b_invariant_lift=b_invariant_lift == ZERO,
        a_invariant_lift=a_invariant_lift == ZERO,
        b_reduction_on_l_zero=b_reduction == ZERO,
        a_reduction_on_l_zero=a_reduction == ZERO,
        listed_points_hit_target=all(map_point(point) == TARGET for point in FIBER_POINTS),
        x_zero_branch_point=FIBER_POINTS[1],
        x_nonzero_branch_points=(FIBER_POINTS[0], FIBER_POINTS[2]),
        claim_ceiling="EXACT_FIBER_CERTIFICATE_ONLY__NATIVE_PUSHFORWARD_NOT_ESTABLISHED",
    )


def target_fiber_case_split() -> dict[str, object]:
    """Return the exact logical case split proving that no target sheets are omitted."""

    certificate = fiber_algebra_certificate()
    if not certificate.passed:
        raise AssertionError("fiber algebra certificate failed")

    return {
        "target": TARGET,
        "case_x_zero": {
            "reason": "b=0 gives y=0, then normalized first target gives z=-1/4",
            "point": certificate.x_zero_branch_point,
        },
        "case_x_nonzero": {
            "reason": (
                "c=xL=0 gives L=0; b=0 gives u=-3/2; "
                "the first target gives x^2=1"
            ),
            "points": certificate.x_nonzero_branch_points,
        },
        "complete_fiber": FIBER_POINTS,
        "complete": certificate.passed,
        "claim_ceiling": certificate.claim_ceiling,
    }


@dataclass(frozen=True)
class FormulaCircuitShape:
    """A conservative shared-circuit size ledger for the sheet weight."""

    variable_count: int
    clause_count: int
    literal_count: int
    selector_nodes: int
    multiplication_nodes: int
    addition_or_subtraction_nodes: int

    @property
    def total_nodes(self) -> int:
        """Return the total counted arithmetic-circuit nodes."""

        return (
            self.selector_nodes
            + self.multiplication_nodes
            + self.addition_or_subtraction_nodes
        )


def formula_circuit_shape(variable_count: int, clause_count: int) -> FormulaCircuitShape:
    """Return a linear shared-circuit bound without expanding formula products."""

    if variable_count < 0 or clause_count < 0:
        raise ValueError("counts must be nonnegative")

    literal_count = 3 * clause_count
    selector_nodes = 3 * variable_count
    valid_product_multiplications = max(variable_count - 1, 0)
    clause_local_multiplications = 2 * clause_count
    clause_product_multiplications = max(clause_count - 1, 0)
    final_multiplication = 1 if variable_count and clause_count else 0
    multiplication_nodes = (
        valid_product_multiplications
        + clause_local_multiplications
        + clause_product_multiplications
        + final_multiplication
    )
    addition_or_subtraction_nodes = variable_count + 4 * clause_count

    return FormulaCircuitShape(
        variable_count=variable_count,
        clause_count=clause_count,
        literal_count=literal_count,
        selector_nodes=selector_nodes,
        multiplication_nodes=multiplication_nodes,
        addition_or_subtraction_nodes=addition_or_subtraction_nodes,
    )


def logarithmic_residue_is_unit(root: int | Fraction) -> bool:
    """Verify that p'(r)/p'(r) is one at a simple root of p(t)=t^3-t."""

    root_value = Fraction(root)
    if root_value**3 != root_value:
        raise ValueError("root must satisfy t^3-t=0")
    derivative = 3 * root_value**2 - 1
    if derivative == 0:
        raise AssertionError("selected fiber root is not simple")
    return derivative / derivative == 1


def _is_prime(value: int) -> bool:
    """Return whether ``value`` is prime by exact trial division."""

    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    for divisor in range(3, isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def prime_sieve_bound_certificate(variable_count: int, primes: tuple[int, ...]) -> bool:
    """Check primality, distinctness, and the exact zero-sieve product bound."""

    if variable_count < 0:
        raise ValueError("variable_count must be nonnegative")
    if len(primes) != variable_count + 1:
        return False
    if len(set(primes)) != len(primes):
        return False
    if not all(_is_prime(prime) for prime in primes):
        return False

    product_value = 1
    for prime in primes:
        product_value *= prime
    return product_value > 2**variable_count
