from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ClauseRelation, ConstraintHolo

Monomial = frozenset[str]
Polynomial = dict[Monomial, int]


def _add(left: Polynomial, right: Mapping[Monomial, int]) -> Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, 0) + coefficient
        if result[monomial] == 0:
            del result[monomial]
    return result


def _multiply(left: Mapping[Monomial, int], right: Mapping[Monomial, int]) -> Polynomial:
    result: Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            monomial = left_monomial | right_monomial
            result[monomial] = (
                result.get(monomial, 0)
                + left_coefficient * right_coefficient
            )
            if result[monomial] == 0:
                del result[monomial]
    return result


def clause_violation_polynomial(clause: ClauseRelation) -> Polynomial:
    """Expand the diagonal violation projector in Boolean occupation variables.

    Occupations obey n_i^2 = n_i, so monomials are represented as variable sets.
    A positive literal is violated by `(1-n_i)`. A negative literal is violated by
    `n_i`. A clause is violated only when every literal is violated.
    """

    polynomial: Polynomial = {frozenset(): 1}
    for literal in clause.literals:
        factor: Polynomial
        if literal.positive:
            factor = {
                frozenset(): 1,
                frozenset((literal.variable,)): -1,
            }
        else:
            factor = {frozenset((literal.variable,)): 1}
        polynomial = _multiply(polynomial, factor)
    return polynomial


def clause_hamiltonian_polynomial(holo: ConstraintHolo) -> Polynomial:
    result: Polynomial = {}
    for clause in holo.clauses:
        result = _add(result, clause_violation_polynomial(clause))
    return result


@dataclass(frozen=True)
class FermionicInteractionAudit:
    public_variables: int
    public_clauses: int
    local_max_degree: int
    combined_max_degree: int
    combined_term_count: int
    quadratic_gaussian_closed: bool
    interaction_order: str
    determinant_method_status: str
    auxiliary_field_status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_fermionic_interactions(holo: ConstraintHolo) -> FermionicInteractionAudit:
    local_polynomials = tuple(
        clause_violation_polynomial(clause) for clause in holo.clauses
    )
    combined = clause_hamiltonian_polynomial(holo)
    local_max = max(
        (len(monomial) for polynomial in local_polynomials for monomial in polynomial),
        default=0,
    )
    combined_max = max((len(monomial) for monomial in combined), default=0)
    gaussian = local_max <= 2 and combined_max <= 2

    return FermionicInteractionAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        local_max_degree=local_max,
        combined_max_degree=combined_max,
        combined_term_count=len(combined),
        quadratic_gaussian_closed=gaussian,
        interaction_order=(
            "quadratic_or_lower" if gaussian else f"degree_{max(local_max, combined_max)}_interacting"
        ),
        determinant_method_status=(
            "FREE_FERMION_DETERMINANT_METHOD_NOT_REJECTED_BY_DEGREE"
            if gaussian
            else "GENERIC_GAUSSIAN_DETERMINANT_CLOSURE_BROKEN_BY_INTERACTIONS"
        ),
        auxiliary_field_status=(
            "NOT_REQUIRED_BY_DEGREE"
            if gaussian
            else "EXACT_AUXILIARY_FIELD_SUM_OR_NON_GAUSSIAN_NATIVE_OPERATOR_REQUIRED"
        ),
    )
