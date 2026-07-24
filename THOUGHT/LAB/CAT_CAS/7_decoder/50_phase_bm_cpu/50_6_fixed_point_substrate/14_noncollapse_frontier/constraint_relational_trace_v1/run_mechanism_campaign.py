from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from constraint_relational_trace_v1.clause_hamiltonian import audit_clause_hamiltonian
    from constraint_relational_trace_v1.constraint_holo import ClauseRelation, ConstraintHolo, Literal
    from constraint_relational_trace_v1.data_processing_amplifier_audit import audit_phase_oracle_data_processing
    from constraint_relational_trace_v1.exceptional_point_root_latch import audit_exceptional_point_root_latch
    from constraint_relational_trace_v1.fermionic_interaction_audit import audit_fermionic_interactions
    from constraint_relational_trace_v1.rank_one_resolvent_audit import audit_rank_one_resolvent_sensor
    from constraint_relational_trace_v1.zero_mode_amplifier_audit import audit_ideal_zero_mode_amplifier
else:
    from .clause_hamiltonian import audit_clause_hamiltonian
    from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
    from .data_processing_amplifier_audit import audit_phase_oracle_data_processing
    from .exceptional_point_root_latch import audit_exceptional_point_root_latch
    from .fermionic_interaction_audit import audit_fermionic_interactions
    from .rank_one_resolvent_audit import audit_rank_one_resolvent_sensor
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
    }

    return {
        "schema": "CONSTRAINT_RELATIONAL_TRACE_MECHANISM_CAMPAIGN_V1",
        "status": (
            "MECHANISM_CAMPAIGN_PASS__EP_ROOT_LATCH_REFERENCE_CANDIDATE__CET_NOT_ESTABLISHED"
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
        "decision": {
            "strongest_candidate": "ORDER_N_EXCEPTIONAL_POINT_ROOT_LATCH",
            "mathematical_sat_margin": "CONSTANT",
            "native_clean_port": "NOT_ESTABLISHED",
            "deterministic_noiseless_gain": "NOT_ESTABLISHED",
            "complete_environment_restoration": "NOT_ESTABLISHED",
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
