from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .self_organizing_clause_flow import integrate_reference_until_solution


@dataclass(frozen=True)
class FlowCensusFailure:
    clause_mask: int
    satisfying_assignments: int
    final_max_clause_constraint: float
    steps_executed: int


@dataclass(frozen=True)
class ThreeVariableFlowCensus:
    total_formulae: int
    satisfiable_formulae: int
    unsatisfiable_formulae: int
    satisfiable_converged: int
    satisfiable_failed: int
    unsat_false_solution_count: int
    maximum_solution_steps: int
    convergence_by_witness_count: tuple[tuple[int, int, int], ...]
    failures: tuple[FlowCensusFailure, ...]
    status: str


def _full_three_variable_clause(sign_mask: int) -> ClauseRelation:
    variables = ("x", "y", "z")
    return ClauseRelation(
        tuple(
            Literal(variable, bool((sign_mask >> index) & 1))
            for index, variable in enumerate(variables)
        )
    )  # type: ignore[arg-type]


def run_three_variable_flow_census(
    step_size: float = 2.0e-3,
    max_steps: int = 20_000,
) -> ThreeVariableFlowCensus:
    """Exhaust every conjunction of the eight complete 3-variable clauses.

    Each clause sign pattern forbids exactly one Boolean assignment. The 256 formulae
    therefore cover witness counts from eight down to zero without random generation.
    Reference enumeration supplies truth only after the public flow run is defined.
    """

    clause_basis = tuple(_full_three_variable_clause(mask) for mask in range(8))
    satisfiable_formulae = 0
    unsatisfiable_formulae = 0
    satisfiable_converged = 0
    unsat_false_solutions = 0
    max_solution_steps = 0
    failures: list[FlowCensusFailure] = []
    witness_bins: dict[int, list[int]] = {}

    for formula_mask in range(1 << len(clause_basis)):
        clauses = tuple(
            clause
            for index, clause in enumerate(clause_basis)
            if (formula_mask >> index) & 1
        )
        holo = ConstraintHolo.build(("x", "y", "z"), clauses)
        reference = reference_existential_trace(holo)
        witness_count = reference.witness_count
        run = integrate_reference_until_solution(
            holo,
            step_size=step_size,
            max_steps=max_steps,
        )
        bucket = witness_bins.setdefault(witness_count, [0, 0])
        bucket[0] += 1

        if reference.satisfiable:
            satisfiable_formulae += 1
            if run.converged_to_public_solution:
                satisfiable_converged += 1
                bucket[1] += 1
                max_solution_steps = max(max_solution_steps, run.steps_executed)
            else:
                failures.append(
                    FlowCensusFailure(
                        clause_mask=formula_mask,
                        satisfying_assignments=witness_count,
                        final_max_clause_constraint=run.final_max_clause_constraint,
                        steps_executed=run.steps_executed,
                    )
                )
        else:
            unsatisfiable_formulae += 1
            if run.converged_to_public_solution:
                unsat_false_solutions += 1

    return ThreeVariableFlowCensus(
        total_formulae=1 << len(clause_basis),
        satisfiable_formulae=satisfiable_formulae,
        unsatisfiable_formulae=unsatisfiable_formulae,
        satisfiable_converged=satisfiable_converged,
        satisfiable_failed=satisfiable_formulae - satisfiable_converged,
        unsat_false_solution_count=unsat_false_solutions,
        maximum_solution_steps=max_solution_steps,
        convergence_by_witness_count=tuple(
            (witness_count, totals[0], totals[1])
            for witness_count, totals in sorted(witness_bins.items())
        ),
        failures=tuple(failures),
        status=(
            "EXHAUSTIVE_THREE_VARIABLE_FLOW_CENSUS_PASS"
            if satisfiable_converged == satisfiable_formulae
            and unsat_false_solutions == 0
            else "EXHAUSTIVE_THREE_VARIABLE_FLOW_CENSUS_EXPOSED_FAILURES"
        ),
    )
