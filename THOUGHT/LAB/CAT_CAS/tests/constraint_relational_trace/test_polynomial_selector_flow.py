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

from constraint_relational_trace_v1.adaptive_polynomial_selector_flow import (  # noqa: E402
    integrate_adaptive_polynomial_selector_flow,
)
from constraint_relational_trace_v1.catalytic_existential_trace import (  # noqa: E402
    reference_existential_trace,
)
from constraint_relational_trace_v1.polynomial_selector_flow import (  # noqa: E402
    audit_polynomial_selector_flow,
    integrate_polynomial_selector_flow,
    polynomial_selector_flow_derivative,
    public_selector_initial_state,
    selector_euler_step,
)
from constraint_relational_trace_v1.polynomial_selector_flow_census import (  # noqa: E402
    run_three_variable_polynomial_selector_flow_census,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    complete_graph_edges,
    cycle_graph_edges,
    exact_three_graph_coloring_holo,
    exact_three_parity_cycle_holo,
    exact_three_pigeonhole_holo,
    exact_three_unique_solution_holo,
)


def test_selector_flow_is_a_public_polynomial_ode() -> None:
    holo = exact_three_unique_solution_holo(3)
    audit = audit_polynomial_selector_flow(holo)

    assert audit.state_coordinates == len(holo.variables) + 11 * len(holo.clauses)
    assert audit.polynomial_degree_upper_bound <= 6
    assert audit.clause_selector_mass_preserved
    assert audit.pair_selector_mass_preserved
    assert audit.public_rational_initial_state
    assert audit.wrong_boolean_corner_release_present
    assert audit.polynomial_ode_status == "PUBLIC_RATIONAL_POLYNOMIAL_VECTOR_FIELD"


def test_selector_native_derivative_preserves_each_selector_mass() -> None:
    holo = exact_three_unique_solution_holo(2)
    state = public_selector_initial_state(holo)
    derivative = polynomial_selector_flow_derivative(holo, state)

    assert derivative.max_abs() > 0.0
    for triple in derivative.clause_selector:
        assert abs(sum(triple)) < 1.0e-12
    for sextuple in derivative.pair_selector:
        for pair_index in range(3):
            start = 2 * pair_index
            assert abs(sum(sextuple[start : start + 2])) < 1.0e-12


def test_selector_euler_chart_preserves_three_pair_simplexes() -> None:
    holo = exact_three_unique_solution_holo(2)
    state = public_selector_initial_state(holo)
    advanced = selector_euler_step(holo, state, step_size=1.0e-3)

    for triple in advanced.clause_selector:
        assert abs(sum(triple) - 1.0) < 1.0e-12
        assert min(triple) >= 0.0
    for sextuple in advanced.pair_selector:
        for pair_index in range(3):
            start = 2 * pair_index
            pair = sextuple[start : start + 2]
            assert abs(sum(pair) - 1.0) < 1.0e-12
            assert min(pair) >= 0.0


def test_selector_flow_reaches_unique_solution_reference() -> None:
    holo = exact_three_unique_solution_holo(3)
    run = integrate_polynomial_selector_flow(
        holo,
        step_size=1.0e-3,
        max_steps=100_000,
    )

    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))


def test_adaptive_selector_flow_preserves_native_simplexes() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    run = integrate_adaptive_polynomial_selector_flow(
        holo,
        maximum_time=10.0,
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-9,
        maximum_step=5.0e-2,
    )

    assert run.solver_success
    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))
    assert run.maximum_clause_selector_mass_error < 1.0e-7
    assert run.maximum_pair_selector_mass_error < 1.0e-7
    assert run.maximum_voltage_magnitude < 1.1


def test_exhaustive_three_variable_selector_flow_census() -> None:
    census = run_three_variable_polynomial_selector_flow_census()

    assert census.total_formulae == 256
    assert census.satisfiable_formulae == 255
    assert census.unsatisfiable_formulae == 1
    assert census.satisfiable_converged == 255
    assert census.satisfiable_failed == 0
    assert census.unsat_false_solution_count == 0
    assert not census.failures
    assert census.status == (
        "EXHAUSTIVE_THREE_VARIABLE_POLYNOMIAL_SELECTOR_FLOW_CENSUS_PASS"
    )


def assert_selector_sat(holo, max_steps: int = 100_000) -> None:
    run = integrate_polynomial_selector_flow(
        holo,
        step_size=1.0e-3,
        max_steps=max_steps,
    )
    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))


def assert_selector_unsat_no_false_witness(holo, max_steps: int = 40_000) -> None:
    run = integrate_polynomial_selector_flow(
        holo,
        step_size=1.0e-3,
        max_steps=max_steps,
    )
    assert not run.converged_to_public_solution
    assert run.status == (
        "POLYNOMIAL_SELECTOR_FLOW_STEP_CAP_REACHED__NO_UNSAT_CONCLUSION"
    )


def test_selector_flow_parity_sat_control() -> None:
    assert_selector_sat(exact_three_parity_cycle_holo(4, total_charge=0))


def test_selector_flow_pigeonhole_sat_control() -> None:
    assert_selector_sat(exact_three_pigeonhole_holo(2, 2))


def test_selector_flow_coloring_sat_control() -> None:
    assert_selector_sat(
        exact_three_graph_coloring_holo(5, cycle_graph_edges(5)),
        max_steps=160_000,
    )


def test_selector_flow_parity_unsat_control() -> None:
    assert_selector_unsat_no_false_witness(
        exact_three_parity_cycle_holo(4, total_charge=1)
    )


def test_selector_flow_pigeonhole_unsat_control() -> None:
    assert_selector_unsat_no_false_witness(exact_three_pigeonhole_holo(3, 2))


def test_selector_flow_coloring_unsat_control() -> None:
    assert_selector_unsat_no_false_witness(
        exact_three_graph_coloring_holo(4, complete_graph_edges(4))
    )
