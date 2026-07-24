from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from .catalytic_existential_trace import reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, ConstraintHoloError, Literal


class BoundaryDecision(str, Enum):
    VALID_SAT = "VALID_SAT"
    VALID_UNSAT = "VALID_UNSAT"
    INVALID_CARRIER = "INVALID_CARRIER"


DecisionBoundary = Callable[[ConstraintHolo], BoundaryDecision]


@dataclass(frozen=True)
class RestrictedConstraintHolo:
    contradiction: bool
    holo: ConstraintHolo | None


@dataclass(frozen=True)
class WitnessExtractionResult:
    valid: bool
    satisfiable: bool | None
    witness: Mapping[str, bool] | None
    decision_calls: int
    witness_verified: bool
    boundary_status: BoundaryDecision
    extraction_scope: str
    conditional_theorem: str


def restrict_public_relation(
    holo: ConstraintHolo,
    fixed: Mapping[str, bool],
) -> RestrictedConstraintHolo:
    if not set(fixed).issubset(holo.variables):
        raise ConstraintHoloError("restriction references a variable outside the public boundary")

    clauses: list[ClauseRelation] = []
    for clause in holo.clauses:
        remaining: list[Literal] = []
        clause_satisfied = False
        for literal in clause.literals:
            if literal.variable in fixed:
                value = bool(fixed[literal.variable])
                literal_value = value if literal.positive else not value
                if literal_value:
                    clause_satisfied = True
                    break
            else:
                remaining.append(literal)
        if clause_satisfied:
            continue
        if not remaining:
            return RestrictedConstraintHolo(contradiction=True, holo=None)
        while len(remaining) < 3:
            remaining.append(remaining[-1])
        clauses.append(ClauseRelation(tuple(remaining)))  # type: ignore[arg-type]

    remaining_variables = tuple(variable for variable in holo.variables if variable not in fixed)
    return RestrictedConstraintHolo(
        contradiction=False,
        holo=ConstraintHolo.build(remaining_variables, clauses),
    )


def reference_decision_boundary(holo: ConstraintHolo) -> BoundaryDecision:
    result = reference_existential_trace(holo)
    if not result.valid:
        return BoundaryDecision.INVALID_CARRIER
    return BoundaryDecision.VALID_SAT if result.satisfiable else BoundaryDecision.VALID_UNSAT


def _read_boundary(
    holo: ConstraintHolo,
    decision_boundary: DecisionBoundary,
) -> BoundaryDecision:
    decision = decision_boundary(holo)
    if not isinstance(decision, BoundaryDecision):
        return BoundaryDecision.INVALID_CARRIER
    return decision


def _invalid_result(calls: int) -> WitnessExtractionResult:
    return WitnessExtractionResult(
        valid=False,
        satisfiable=None,
        witness=None,
        decision_calls=calls,
        witness_verified=False,
        boundary_status=BoundaryDecision.INVALID_CARRIER,
        extraction_scope="classical_post_boundary_self_reduction",
        conditional_theorem="UNIFORM_POLYNOMIAL_CET_DECISION_IMPLIES_3SAT_IN_P",
    )


def extract_witness_by_boundary_self_reduction(
    holo: ConstraintHolo,
    decision_boundary: DecisionBoundary,
) -> WitnessExtractionResult:
    """Render a conventional witness using only polynomially many exact decisions.

    The boundary is total. Any invalid or unrecognized result terminates extraction and
    can never be interpreted as SAT or UNSAT.
    """

    calls = 1
    initial_decision = _read_boundary(holo, decision_boundary)
    if initial_decision is BoundaryDecision.INVALID_CARRIER:
        return _invalid_result(calls)
    if initial_decision is BoundaryDecision.VALID_UNSAT:
        return WitnessExtractionResult(
            valid=True,
            satisfiable=False,
            witness=None,
            decision_calls=calls,
            witness_verified=False,
            boundary_status=BoundaryDecision.VALID_UNSAT,
            extraction_scope="classical_post_boundary_self_reduction",
            conditional_theorem="UNIFORM_POLYNOMIAL_CET_DECISION_IMPLIES_3SAT_IN_P",
        )

    fixed: dict[str, bool] = {}
    for variable in holo.variables:
        false_restriction = restrict_public_relation(holo, {**fixed, variable: False})
        false_decision = BoundaryDecision.VALID_UNSAT
        if not false_restriction.contradiction:
            assert false_restriction.holo is not None
            calls += 1
            false_decision = _read_boundary(false_restriction.holo, decision_boundary)
            if false_decision is BoundaryDecision.INVALID_CARRIER:
                return _invalid_result(calls)
        if false_decision is BoundaryDecision.VALID_SAT:
            fixed[variable] = False
            continue

        true_restriction = restrict_public_relation(holo, {**fixed, variable: True})
        if true_restriction.contradiction:
            raise ConstraintHoloError("exact decision boundary produced an inconsistent self-reduction")
        assert true_restriction.holo is not None
        calls += 1
        true_decision = _read_boundary(true_restriction.holo, decision_boundary)
        if true_decision is BoundaryDecision.INVALID_CARRIER:
            return _invalid_result(calls)
        if true_decision is not BoundaryDecision.VALID_SAT:
            raise ConstraintHoloError("exact decision boundary failed both restriction branches")
        fixed[variable] = True

    verified = holo.accepts(fixed)
    if not verified:
        raise ConstraintHoloError("rendered witness failed the public relation")
    return WitnessExtractionResult(
        valid=True,
        satisfiable=True,
        witness=fixed,
        decision_calls=calls,
        witness_verified=True,
        boundary_status=BoundaryDecision.VALID_SAT,
        extraction_scope="classical_post_boundary_self_reduction",
        conditional_theorem="UNIFORM_POLYNOMIAL_CET_DECISION_IMPLIES_3SAT_IN_P",
    )
