from __future__ import annotations

from math import isfinite
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

from constraint_relational_trace_v1.adaptive_phase_logit_flow import (  # noqa: E402
    audit_phase_logit_coordinate_identity,
    integrate_adaptive_phase_logit_flow,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    exact_three_parity_cycle_holo,
)


def test_phase_logit_chart_round_trip_is_exact() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    audit = audit_phase_logit_coordinate_identity(holo)
    assert audit.maximum_coordinate_residual < 1.0e-12
    assert audit.coordinate_identity_status == (
        "ANGLE_LOGIT_CHART_MATCHES_NATIVE_POLYNOMIAL_PHASE_FIELD"
    )


def test_fixed_deadline_phase_chart_emits_terminal_witness() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    run = integrate_adaptive_phase_logit_flow(
        holo,
        fixed_deadline=3.0,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-8,
        maximum_step=5.0e-2,
        solver_method="BDF",
    )
    assert run.reached_fixed_deadline
    assert run.terminal_solution_verified
    assert run.status == "TERMINAL_WITNESS_VERIFIED"
    assert run.first_passage_observed
    assert run.first_passage_time is not None
    assert holo.accepts(dict(run.terminal_assignment))
    assert run.terminal_clause_satisfaction_margin > 0.0


def test_unsat_phase_chart_never_emits_false_terminal_witness() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=1)
    run = integrate_adaptive_phase_logit_flow(
        holo,
        fixed_deadline=2.0,
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-8,
        maximum_step=2.0e-2,
        solver_method="DOP853",
    )
    assert run.reached_fixed_deadline
    assert not run.terminal_solution_verified
    assert run.status == "TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED"


def test_phase_resource_receipts_are_finite() -> None:
    holo = exact_three_parity_cycle_holo(4, total_charge=0)
    run = integrate_adaptive_phase_logit_flow(holo, fixed_deadline=3.0)
    values = (
        run.maximum_long_memory,
        run.maximum_unwrapped_phase_displacement,
        run.maximum_short_logit_magnitude,
        run.maximum_long_logit_magnitude,
        run.maximum_clause_log_ratio_magnitude,
        run.maximum_pair_log_ratio_magnitude,
        run.chart_trajectory_length_lower_bound,
        run.phase_trajectory_length_lower_bound,
        run.native_trajectory_length_lower_bound,
        run.maximum_chart_speed,
    )
    assert all(isfinite(value) and value >= 0.0 for value in values)
    assert 0.0 < run.minimum_clause_selector_weight <= 1.0
    assert 0.0 < run.minimum_pair_selector_weight <= 1.0
