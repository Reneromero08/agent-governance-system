from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.polynomial_clause_flow import (  # noqa: E402
    audit_polynomial_clause_flow,
    integrate_polynomial_flow_until_solution,
    polynomial_clause_flow_derivative,
    public_polynomial_initial_state,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    exact_three_parity_cycle_holo,
    exact_three_unique_solution_holo,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def test_polynomial_flow_is_a_public_bounded_polynomial_ode() -> None:
    holo = exact_three_unique_solution_holo(3)
    audit = audit_polynomial_clause_flow(holo)

    assert audit.state_coordinates == len(holo.variables) + 5 * len(holo.clauses)
    assert audit.polynomial_degree_upper_bound <= 6
    assert audit.selector_simplex_preserved
    assert audit.bounded_voltage_barrier
    assert audit.bounded_memory_barriers
    assert audit.polynomial_ode_normal_form_status == (
        "POLYNOMIAL_VECTOR_FIELD_WITH_PUBLIC_RATIONAL_PARAMETERS"
    )
    assert audit.nonsatisfying_boolean_stationary_count == 0


def test_polynomial_flow_moves_from_public_seed() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    state = public_polynomial_initial_state(holo)
    derivative = polynomial_clause_flow_derivative(holo, state)

    assert derivative.max_abs() > 0.0
    assert abs(sum(derivative.selector_weights[0])) < 1.0e-12


def test_polynomial_flow_reaches_unique_solution_reference() -> None:
    holo = exact_three_unique_solution_holo(3)
    run = integrate_polynomial_flow_until_solution(
        holo,
        step_size=1.0e-3,
        max_steps=50_000,
    )

    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))


def test_polynomial_flow_reaches_small_parity_cycle_reference() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    run = integrate_polynomial_flow_until_solution(
        holo,
        step_size=1.0e-3,
        max_steps=100_000,
    )

    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))
