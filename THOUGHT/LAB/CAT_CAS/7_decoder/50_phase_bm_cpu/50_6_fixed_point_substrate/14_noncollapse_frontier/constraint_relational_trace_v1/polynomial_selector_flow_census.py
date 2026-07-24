from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .polynomial_selector_flow import integrate_polynomial_selector_flow


@dataclass(frozen=True)
class PolynomialSelectorFlowCensusFailure:
    clause_mask: int
    satisfying_assignments: int
    final_max_selector_constraint: float
    steps_executed: int


@dataclass(frozen=True)
class ThreeVariablePolynomialSelectorFlowCensus:
    total_formulae: int
    satisfiable_formulae: int
    unsatisfiable_formulae: int
    satisfiable_converged: int
    satisfiable_failed: int
    unsat_false_solution_count: int
    maximum_solution_steps: int
    failures: tuple[PolynomialSelectorFlowCensusFailure, ...]
    status: str


def _clause(sign_mask: int) -> ClauseRelation:
    variables = ("x", "y", "z")
    return ClauseRelation(
        tuple(
            Literal(variable, bool((sign_mask >> index) & 1))
            for index, variable in enumerate(variables)
        )
    )  # type: ignore[arg-type]


def run_three_variable_polynomial_selector_flow_census(
    step_size: float = 1.0e-3,
    max_steps: int = 100_000,
) -> ThreeVariablePolynomialSelectorFlowCensus:
    clause_basis = tuple(_clause(mask) for mask in range(8))
    satisfiable_formulae = 0
    unsatisfiable_formulae = 0
    satisfiable_converged = 0
    unsat_false_solutions = 0
    maximum_solution_steps = 0
    failures: list[PolynomialSelectorFlowCensusFailure] = []

    for formula_mask in range(1 << len(clause_basis)):
        clauses = tuple(
            clause
            for index, clause in enumerate(clause_basis)
            if (formula_mask >> index) & 1
        )
        holo = ConstraintHolo.build(("x", "y", "z"), clauses)
        reference = reference_existential_trace(holo)
        run = integrate_polynomial_selector_flow(
            holo,
            step_size=step_size,
            max_steps=max_steps,
        )
        if reference.satisfiable:
            satisfiable_formulae += 1
            if run.converged_to_public_solution:
                satisfiable_converged += 1
                maximum_solution_steps = max(
                    maximum_solution_steps,
                    run.steps_executed,
                )
            else:
                failures.append(
                    PolynomialSelectorFlowCensusFailure(
                        clause_mask=formula_mask,
                        satisfying_assignments=reference.witness_count,
                        final_max_selector_constraint=run.final_max_selector_constraint,
                        steps_executed=run.steps_executed,
                    )
                )
        else:
            unsatisfiable_formulae += 1
            if run.converged_to_public_solution:
                unsat_false_solutions += 1

    return ThreeVariablePolynomialSelectorFlowCensus(
        total_formulae=1 << len(clause_basis),
        satisfiable_formulae=satisfiable_formulae,
        unsatisfiable_formulae=unsatisfiable_formulae,
        satisfiable_converged=satisfiable_converged,
        satisfiable_failed=satisfiable_formulae - satisfiable_converged,
        unsat_false_solution_count=unsat_false_solutions,
        maximum_solution_steps=maximum_solution_steps,
        failures=tuple(failures),
        status=(
            "EXHAUSTIVE_THREE_VARIABLE_POLYNOMIAL_SELECTOR_FLOW_CENSUS_PASS"
            if satisfiable_converged == satisfiable_formulae
            and unsat_false_solutions == 0
            else "EXHAUSTIVE_THREE_VARIABLE_POLYNOMIAL_SELECTOR_FLOW_CENSUS_EXPOSED_FAILURES"
        ),
    )
