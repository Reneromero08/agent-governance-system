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
from constraint_relational_trace_v1.cotangent_flow_lift import (  # noqa: E402
    audit_cotangent_flow_lift,
)
from constraint_relational_trace_v1.self_organizing_clause_flow import (  # noqa: E402
    SelfOrganizingFlowState,
    audit_self_organizing_clause_flow,
    clause_constraint_values,
    integrate_reference_until_solution,
    projected_euler_step,
    public_perturbed_initial_state,
)
from constraint_relational_trace_v1.thermal_zero_mode_latch import (  # noqa: E402
    audit_thermal_zero_mode_latch,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def unique_solution(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    return ConstraintHolo.build(
        variables,
        tuple(
            clause(Literal(variable), Literal(variable), Literal(variable))
            for variable in variables
        ),
    )


def contradiction(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    return ConstraintHolo.build(
        variables,
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(
                Literal("x1", False),
                Literal("x1", False),
                Literal("x1", False),
            ),
        ),
    )


def test_self_organizing_flow_has_only_solution_boolean_equilibria() -> None:
    audit = audit_self_organizing_clause_flow(unique_solution(5))

    assert audit.polynomial_state_coordinates == 15
    assert audit.local_literal_couplings == 15
    assert audit.satisfying_boolean_corners == 1
    assert audit.satisfying_corners_stationary
    assert audit.nonsatisfying_stationary_corners == 0
    assert audit.all_boolean_stationary_points_are_solutions
    assert audit.worst_case_convergence_status == "NOT_ESTABLISHED"


def test_unsat_instance_has_no_boolean_equilibrium() -> None:
    audit = audit_self_organizing_clause_flow(contradiction(4))

    assert audit.satisfying_boolean_corners == 0
    assert audit.nonsatisfying_boolean_corners == 16
    assert audit.nonsatisfying_stationary_corners == 0
    assert audit.unsat_total_boundary_status == "NOT_ESTABLISHED"


def test_terminal_agnostic_clause_flow_reduces_an_active_clause_defect() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    initial = SelfOrganizingFlowState(
        voltages=(0.0, 0.0, 0.0),
        short_memory=(1.0,),
        long_memory=(1.0,),
    )

    before = clause_constraint_values(holo, initial)[0]
    after_state = projected_euler_step(holo, initial, step_size=0.1)
    after = clause_constraint_values(holo, after_state)[0]

    assert before == 0.5
    assert after < before
    assert all(value > 0 for value in after_state.voltages)


def test_public_perturbation_breaks_neutral_symmetry_without_a_witness() -> None:
    holo = ConstraintHolo.build(
        ("x", "y"),
        (
            clause(Literal("x"), Literal("y"), Literal("y")),
            clause(
                Literal("x", False),
                Literal("y", False),
                Literal("y", False),
            ),
        ),
    )

    run = integrate_reference_until_solution(
        holo,
        initial_state=public_perturbed_initial_state(holo),
        step_size=2.0e-3,
        max_steps=10_000,
    )

    assert run.converged_to_public_solution
    assert run.steps_executed < 10_000
    assert run.final_max_clause_constraint < 0.5
    assert holo.accepts(dict(run.final_assignment))


def test_thermal_latch_has_constant_unique_witness_population() -> None:
    audit = audit_thermal_zero_mode_latch(unique_solution(10))

    assert audit.physical_binary_coordinates == 10
    assert audit.inverse_temperature_scaling == "LINEAR_IN_PUBLIC_VARIABLE_COUNT"
    assert audit.guaranteed_sat_zero_population_lower_bound >= 0.8
    assert audit.exact_zero_energy_population_reference_only >= 0.8
    assert audit.one_sided_sample_count_for_99_percent_detection <= 3
    assert audit.constant_population_margin_established
    assert not audit.polynomial_total_resources_established


def test_thermal_latch_never_reports_zero_energy_for_unsat() -> None:
    audit = audit_thermal_zero_mode_latch(contradiction(8))

    assert audit.exact_zero_energy_population_reference_only == 0.0
    assert audit.guaranteed_sat_zero_population_lower_bound == 0.0
    assert audit.unsat_zero_population == 0.0
    assert audit.constant_population_margin_established


def test_cotangent_lift_recovers_smooth_flow_but_exposes_compensation() -> None:
    holo = unique_solution(7)
    audit = audit_cotangent_flow_lift(holo)

    assert audit.primal_state_coordinates == 21
    assert audit.lifted_state_coordinates == 42
    assert audit.primal_flow_recovered
    assert audit.smooth_region_phase_volume_preserved
    assert audit.exact_negative_time_inverse_on_smooth_regions
    assert "COTANGENT_EXPANSION" in audit.attractor_contraction_compensation
    assert not audit.polynomial_total_resources_established
