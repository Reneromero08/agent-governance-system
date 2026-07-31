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
from constraint_relational_trace_v1.polynomial_phase_selector_flow import (  # noqa: E402
    audit_polynomial_phase_selector_flow,
    boolean_phase_state,
    integrate_polynomial_phase_selector_flow,
    phase_circle_tangent_residual,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    exact_three_parity_cycle_holo,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def test_phase_carrier_is_tangent_to_every_circle() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    state = public_phase_selector_initial_state(holo)
    derivative = polynomial_phase_selector_flow_derivative(holo, state)
    assert phase_circle_tangent_residual(state, derivative) < 1.0e-12


def test_satisfying_boolean_phase_section_is_invariant() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    state = boolean_phase_state(
        holo,
        {"a": True, "b": False, "c": False},
    )
    derivative = polynomial_phase_selector_flow_derivative(holo, state)
    assert max(abs(value) for value in derivative.phase_cosine) == 0.0
    assert max(abs(value) for value in derivative.phase_sine) == 0.0


def test_violated_boolean_corner_is_released() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    state = boolean_phase_state(
        holo,
        {"a": False, "b": False, "c": False},
    )
    derivative = polynomial_phase_selector_flow_derivative(holo, state)
    assert max(abs(value) for value in derivative.phase_sine) > 0.0


def test_phase_carrier_audit_is_public_polynomial_geometry() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    audit = audit_polynomial_phase_selector_flow(holo)
    assert audit.state_coordinates == 2 * len(holo.variables) + 11 * len(holo.clauses)
    assert audit.polynomial_degree_upper_bound <= 6
    assert audit.circle_tangent_identity_exact
    assert audit.exact_clause_truth_channel
    assert audit.satisfying_boolean_sections_invariant
    assert audit.wrong_boolean_corner_release_present
    assert audit.global_convergence_status == "NOT_ESTABLISHED"


def test_phase_reference_reaches_small_parity_solution() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    run = integrate_polynomial_phase_selector_flow(
        holo,
        step_size=1.0e-3,
        max_steps=100_000,
    )
    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))
    assert run.maximum_circle_residual < 1.0e-10
