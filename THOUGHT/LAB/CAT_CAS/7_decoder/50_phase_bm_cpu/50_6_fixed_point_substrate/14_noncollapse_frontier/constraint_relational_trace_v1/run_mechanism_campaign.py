from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from constraint_relational_trace_v1.adaptive_polynomial_selector_flow import integrate_adaptive_polynomial_selector_flow
    from constraint_relational_trace_v1.clause_hamiltonian import audit_clause_hamiltonian
    from constraint_relational_trace_v1.constraint_holo import ClauseRelation, ConstraintHolo, Literal
    from constraint_relational_trace_v1.cotangent_flow_lift import audit_cotangent_flow_lift
    from constraint_relational_trace_v1.data_processing_amplifier_audit import audit_phase_oracle_data_processing
    from constraint_relational_trace_v1.exceptional_point_root_latch import audit_exceptional_point_root_latch
    from constraint_relational_trace_v1.fermionic_interaction_audit import audit_fermionic_interactions
    from constraint_relational_trace_v1.instanton_deadline_audit import audit_instanton_deadline_argument
    from constraint_relational_trace_v1.polynomial_selector_flow import (
        audit_polynomial_selector_flow,
        integrate_polynomial_selector_flow,
    )
    from constraint_relational_trace_v1.rank_one_resolvent_audit import audit_rank_one_resolvent_sensor
    from constraint_relational_trace_v1.self_organizing_clause_flow import (
        audit_self_organizing_clause_flow,
        integrate_reference_until_solution,
    )
    from constraint_relational_trace_v1.supersymmetric_index_compensation import audit_supersymmetric_index_compensation
    from constraint_relational_trace_v1.structured_clause_families import exact_three_parity_cycle_holo
    from constraint_relational_trace_v1.thermal_zero_mode_latch import audit_thermal_zero_mode_latch
    from constraint_relational_trace_v1.zero_mode_amplifier_audit import audit_ideal_zero_mode_amplifier
else:
    from .adaptive_polynomial_selector_flow import integrate_adaptive_polynomial_selector_flow
    from .clause_hamiltonian import audit_clause_hamiltonian
    from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
    from .cotangent_flow_lift import audit_cotangent_flow_lift
    from .data_processing_amplifier_audit import audit_phase_oracle_data_processing
    from .exceptional_point_root_latch import audit_exceptional_point_root_latch
    from .fermionic_interaction_audit import audit_fermionic_interactions
    from .instanton_deadline_audit import audit_instanton_deadline_argument
    from .polynomial_selector_flow import (
        audit_polynomial_selector_flow,
        integrate_polynomial_selector_flow,
    )
    from .rank_one_resolvent_audit import audit_rank_one_resolvent_sensor
    from .self_organizing_clause_flow import (
        audit_self_organizing_clause_flow,
        integrate_reference_until_solution,
    )
    from .supersymmetric_index_compensation import audit_supersymmetric_index_compensation
    from .structured_clause_families import exact_three_parity_cycle_holo
    from .thermal_zero_mode_latch import audit_thermal_zero_mode_latch
    from .zero_mode_amplifier_audit import audit_ideal_zero_mode_amplifier


def _clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def _unique_solution(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    return ConstraintHolo.build(
        variables,
        tuple(
            _clause(Literal(variable), Literal(variable), Literal(variable))
            for variable in variables
        ),
    )


def _unsat(variable_count: int) -> ConstraintHolo:
    variables = tuple(f"x{index}" for index in range(1, variable_count + 1))
    return ConstraintHolo.build(
        variables,
        (
            _clause(Literal("x1"), Literal("x1"), Literal("x1")),
            _clause(Literal("x1", False), Literal("x1", False), Literal("x1", False)),
        ),
    )


def _genuine_three_body() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("a", "b", "c"),
        (_clause(Literal("a"), Literal("b"), Literal("c")),),
    )


def build_mechanism_record(variable_count: int = 10) -> dict[str, object]:
    sat_holo = _unique_solution(variable_count)
    unsat_holo = _unsat(variable_count)

    sat_hamiltonian = audit_clause_hamiltonian(sat_holo)
    unsat_hamiltonian = audit_clause_hamiltonian(unsat_holo)
    sat_gain = audit_ideal_zero_mode_amplifier(sat_hamiltonian)
    sat_resolvent = audit_rank_one_resolvent_sensor(sat_hamiltonian)
    sat_ep = audit_exceptional_point_root_latch(sat_holo)
    unsat_ep = audit_exceptional_point_root_latch(unsat_holo)
    data_processing = audit_phase_oracle_data_processing(variable_count, 1)
    interactions = audit_fermionic_interactions(_genuine_three_body())
    sat_flow = audit_self_organizing_clause_flow(sat_holo)
    unsat_flow = audit_self_organizing_clause_flow(unsat_holo)
    sat_flow_run = integrate_reference_until_solution(
        sat_holo,
        step_size=0.01,
        max_steps=5_000,
    )
    sat_thermal = audit_thermal_zero_mode_latch(sat_holo)
    unsat_thermal = audit_thermal_zero_mode_latch(unsat_holo)
    cotangent_lift = audit_cotangent_flow_lift(sat_holo)
    supersymmetric_index = audit_supersymmetric_index_compensation(sat_holo)
    instanton_deadline = audit_instanton_deadline_argument(sat_holo)
    selector_holo = exact_three_parity_cycle_holo(4, total_charge=0)
    selector_audit = audit_polynomial_selector_flow(selector_holo)
    selector_reference_run = integrate_polynomial_selector_flow(
        selector_holo,
        step_size=1.0e-3,
        max_steps=100_000,
    )
    selector_adaptive_run = integrate_adaptive_polynomial_selector_flow(
        selector_holo,
        maximum_time=10.0,
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-9,
        maximum_step=5.0e-2,
    )

    gates = {
        "constant_energy_margin": (
            sat_hamiltonian.ground_energy == 0
            and unsat_hamiltonian.ground_energy >= 1
        ),
        "unique_zero_mode_weight_exact": (
            sat_hamiltonian.normalized_zero_mode_weight == 1 / (1 << variable_count)
        ),
        "gain_resource_transfer_exposed": (
            sat_gain.required_weight_gain == (1 << variable_count) - 1
            and not sat_gain.polynomial_energy_established
        ),
        "rank_one_probe_tradeoff_exposed": (
            sat_resolvent.normalized_zero_pole_residue == 1 / (1 << variable_count)
            and sat_resolvent.unnormalized_probe_norm_squared == (1 << variable_count)
        ),
        "ep_root_identity": (
            sat_ep.spectral_radius == 1.0
            and unsat_ep.spectral_radius == 0.0
            and sat_ep.symbolic_sensor_modes == variable_count
        ),
        "ep_resources_fail_closed": (
            not sat_ep.polynomial_total_resources_established
            and sat_ep.deterministic_noiseless_gain_status == "NOT_ESTABLISHED"
        ),
        "deterministic_quantum_amplifier_rejected": (
            data_processing.postselection_or_nonstandard_dynamics_required
            and not data_processing.constant_one_shot_separation_possible_under_cptp
        ),
        "gaussian_fermion_shortcut_rejected": (
            not interactions.quadratic_gaussian_closed
            and interactions.local_max_degree == 3
        ),
        "terminal_agnostic_clause_flow_compiled": (
            sat_flow.polynomial_state_coordinates == variable_count + 2 * variable_count
            and sat_flow.satisfying_corners_stationary
            and sat_flow.all_boolean_stationary_points_are_solutions
            and unsat_flow.satisfying_boolean_corners == 0
            and unsat_flow.nonsatisfying_stationary_corners == 0
        ),
        "answer_blind_flow_reaches_reference_solution": (
            sat_flow_run.converged_to_public_solution
            and sat_flow_run.steps_executed < 5_000
        ),
        "thermal_zero_mode_constant_margin": (
            sat_thermal.guaranteed_sat_zero_population_lower_bound >= 0.8
            and sat_thermal.exact_zero_energy_population_reference_only >= 0.8
            and unsat_thermal.exact_zero_energy_population_reference_only == 0.0
        ),
        "thermal_preparation_resources_fail_closed": (
            not sat_thermal.polynomial_total_resources_established
            and sat_thermal.worst_case_mixing_status == "POLYNOMIAL_WORST_CASE_MIXING_NOT_ESTABLISHED"
        ),
        "cotangent_lift_restoration_candidate": (
            cotangent_lift.primal_flow_recovered
            and cotangent_lift.exact_negative_time_inverse_on_smooth_regions
            and not cotangent_lift.polynomial_total_resources_established
        ),
        "finite_supersymmetric_index_shortcut_rejected": (
            supersymmetric_index.satisfying_assignments == 1
            and supersymmetric_index.bosonic_zero_modes == 1
            and supersymmetric_index.fermionic_zero_modes == 1
            and supersymmetric_index.finite_witten_index == 0
            and not supersymmetric_index.formula_dependent_index_detected
        ),
        "published_instanton_deadline_gap_exposed": (
            instanton_deadline.maximum_index_descent_steps
            == instanton_deadline.phase_space_dimension
            and "NO_EXPLICIT_UNIFORM_NUMERIC_BOUND"
            in instanton_deadline.uniform_instanton_width_status
            and instanton_deadline.polynomial_precision_status == "NOT_ESTABLISHED"
            and instanton_deadline.ordinary_p_equals_np_status == "NOT_ESTABLISHED"
        ),
        "polynomial_selector_dilation_compiled": (
            selector_audit.state_coordinates
            == len(selector_holo.variables) + 11 * len(selector_holo.clauses)
            and selector_audit.polynomial_degree_upper_bound <= 6
            and selector_audit.clause_selector_mass_preserved
            and selector_audit.pair_selector_mass_preserved
            and selector_audit.public_rational_initial_state
            and selector_audit.wrong_boolean_corner_release_present
        ),
        "polynomial_selector_reference_reaches_solution": (
            selector_reference_run.converged_to_public_solution
            and selector_holo.accepts(dict(selector_reference_run.final_assignment))
        ),
        "polynomial_selector_smooth_chart_reaches_solution": (
            selector_adaptive_run.solver_success
            and selector_adaptive_run.converged_to_public_solution
            and selector_adaptive_run.maximum_clause_selector_mass_error < 1.0e-7
            and selector_adaptive_run.maximum_pair_selector_mass_error < 1.0e-7
        ),
        "polynomial_selector_deadline_fails_closed": (
            selector_audit.global_convergence_status == "NOT_ESTABLISHED"
        ),
    }

    return {
        "schema": "CONSTRAINT_RELATIONAL_TRACE_MECHANISM_CAMPAIGN_V1",
        "status": (
            "MECHANISM_CAMPAIGN_PASS__POLYNOMIAL_SELECTOR_DILATION_THERMAL_LATCH_CANDIDATE__CET_NOT_ESTABLISHED"
            if all(gates.values())
            else "MECHANISM_CAMPAIGN_FAILED"
        ),
        "public_variable_count": variable_count,
        "gates": gates,
        "sat_clause_hamiltonian": asdict(sat_hamiltonian),
        "unsat_clause_hamiltonian": asdict(unsat_hamiltonian),
        "sat_zero_mode_gain": asdict(sat_gain),
        "sat_rank_one_resolvent": asdict(sat_resolvent),
        "sat_ep_root_latch": asdict(sat_ep),
        "unsat_ep_root_latch": asdict(unsat_ep),
        "data_processing_control": asdict(data_processing),
        "fermionic_interaction_control": asdict(interactions),
        "sat_self_organizing_flow": asdict(sat_flow),
        "unsat_self_organizing_flow": asdict(unsat_flow),
        "sat_reference_flow_run": asdict(sat_flow_run),
        "sat_thermal_zero_mode_latch": asdict(sat_thermal),
        "unsat_thermal_zero_mode_latch": asdict(unsat_thermal),
        "cotangent_flow_lift": asdict(cotangent_lift),
        "supersymmetric_index_compensation": asdict(supersymmetric_index),
        "instanton_deadline_audit": asdict(instanton_deadline),
        "polynomial_selector_flow_audit": asdict(selector_audit),
        "polynomial_selector_reference_run": asdict(selector_reference_run),
        "polynomial_selector_adaptive_run": asdict(selector_adaptive_run),
        "decision": {
            "strongest_candidate": (
                "POLYNOMIAL_SELECTOR_DILATION_OF_TERMINAL_AGNOSTIC_CLAUSE_FLOW_"
                "WITH_THERMAL_ZERO_MODE_LATCH"
            ),
            "mathematical_sat_margin": "CONSTANT_NORMALIZED_ZERO_MODE_POPULATION",
            "native_clean_port": "DIRECT_PUBLIC_CLAUSE_HAMILTONIAN",
            "deterministic_noiseless_gain": "NOT_REQUIRED_BY_THERMAL_CANDIDATE",
            "worst_case_native_preparation": (
                "UNIFORM_POLYNOMIAL_TRAJECTORY_BOUND_NOT_ESTABLISHED"
            ),
            "deterministic_exact_boundary": "NOT_ESTABLISHED",
            "complete_environment_restoration": "SMOOTH_REGION_ONLY__GLOBAL_NOT_ESTABLISHED",
            "polynomial_total_resources": "NOT_ESTABLISHED",
            "standard_model_transfer": "NOT_ESTABLISHED",
            "p_equals_np": "NOT_PROVEN",
        },
    }


def main() -> int:
    record = build_mechanism_record()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mechanism_campaign.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "output": str(output_path)}, sort_keys=True))
    return 0 if record["status"].startswith("MECHANISM_CAMPAIGN_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
