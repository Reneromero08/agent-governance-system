from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .catalytic_existential_trace import reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, ConstraintHoloError, Literal

DecisionBoundary = Callable[[ConstraintHolo], bool]


@dataclass(frozen=True)
class RestrictedConstraintHolo:
    contradiction: bool
    holo: ConstraintHolo | None


@dataclass(frozen=True)
class WitnessExtractionResult:
    satisfiable: bool
    witness: Mapping[str, bool] | None
    decision_calls: int
    witness_verified: bool
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


def reference_decision_boundary(holo: ConstraintHolo) -> bool:
    return reference_existential_trace(holo).satisfiable


def extract_witness_by_boundary_self_reduction(
    holo: ConstraintHolo,
    decision_boundary: DecisionBoundary,
) -> WitnessExtractionResult:
    """Render a conventional witness using only polynomially many exact decisions.

    This routine belongs outside the native core. It proves that an exact polynomial
    decision boundary is sufficient for the conventional witness required by the Wall.
    """

    calls = 1
    if not decision_boundary(holo):
        return WitnessExtractionResult(
            satisfiable=False,
            witness=None,
            decision_calls=calls,
            witness_verified=False,
            extraction_scope="classical_post_boundary_self_reduction",
            conditional_theorem="UNIFORM_POLYNOMIAL_CET_DECISION_IMPLIES_3SAT_IN_P",
        )

    fixed: dict[str, bool] = {}
    for variable in holo.variables:
        false_restriction = restrict_public_relation(holo, {**fixed, variable: False})
        false_branch_satisfiable = False
        if not false_restriction.contradiction:
            assert false_restriction.holo is not None
            calls += 1
            false_branch_satisfiable = decision_boundary(false_restriction.holo)
        if false_branch_satisfiable:
            fixed[variable] = False
            continue

        true_restriction = restrict_public_relation(holo, {**fixed, variable: True})
        if true_restriction.contradiction:
            raise ConstraintHoloError("exact decision boundary produced an inconsistent self-reduction")
        assert true_restriction.holo is not None
        calls += 1
        if not decision_boundary(true_restriction.holo):
            raise ConstraintHoloError("exact decision boundary failed both restriction branches")
        fixed[variable] = True

    verified = holo.accepts(fixed)
    if not verified:
        raise ConstraintHoloError("rendered witness failed the public relation")
    return WitnessExtractionResult(
        satisfiable=True,
        witness=fixed,
        decision_calls=calls,
        witness_verified=True,
        extraction_scope="classical_post_boundary_self_reduction",
        conditional_theorem="UNIFORM_POLYNOMIAL_CET_DECISION_IMPLIES_3SAT_IN_P",
    )
