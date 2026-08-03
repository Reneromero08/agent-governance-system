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

from constraint_relational_trace_v1.phase_stationary_point_audit import (  # noqa: E402
    audit_symmetric_sat_stationary_point,
    symmetric_non_solution_stationary_state,
    symmetric_sat_stationary_holo,
)
from constraint_relational_trace_v1.polynomial_phase_selector_flow import (  # noqa: E402
    phase_threshold_assignment,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)


def test_satisfiable_relation_has_non_solution_stationary_carrier_state() -> None:
    audit = audit_symmetric_sat_stationary_point()

    assert audit.witness_count == 2
    assert audit.stationary_phase_energy == 1.0
    assert audit.stationary_derivative_max_abs == 0.0
    assert not audit.stationary_threshold_is_witness
    assert audit.arbitrary_state_global_convergence_falsified
    assert audit.status == (
        "SAT_NON_SOLUTION_STATIONARY_POINT_ESTABLISHED__"
        "PUBLIC_SEED_SPECIFIC_THEOREM_REQUIRED"
    )


def test_stationary_obstruction_has_exact_positive_phase_eigenvalue() -> None:
    audit = audit_symmetric_sat_stationary_point()

    assert audit.phase_tangent_trace < 0.0
    assert audit.phase_tangent_determinant < 0.0
    assert audit.phase_positive_eigenvalue > 0.0
    assert audit.stationary_point_has_phase_saddle
    assert audit.phase_tangent_jacobian[0][1] == audit.phase_tangent_jacobian[1][0]


def test_public_low_discrepancy_seed_is_already_witness_for_obstruction_formula() -> None:
    holo = symmetric_sat_stationary_holo()
    stationary_state = symmetric_non_solution_stationary_state(holo)
    public_seed = public_phase_selector_initial_state(holo)
    public_assignment = phase_threshold_assignment(holo, public_seed)
    audit = audit_symmetric_sat_stationary_point()

    assert polynomial_phase_selector_flow_derivative(
        holo,
        stationary_state,
    ).max_abs() == 0.0
    assert polynomial_phase_selector_flow_derivative(
        holo,
        public_seed,
    ).max_abs() > 0.0
    assert not holo.accepts(phase_threshold_assignment(holo, stationary_state))
    assert public_assignment == {"x": True, "y": False}
    assert holo.accepts(public_assignment)
    assert audit.public_seed_threshold_is_witness
    assert audit.all_two_variable_renaming_gauges_seed_satisfy
    assert audit.public_seed_convergence_status == (
        "PUBLIC_SEED_ALREADY_TERMINAL_WITNESS_ON_THIS_OBSTRUCTION_FORMULA__"
        "GLOBAL_PUBLIC_SEED_THEOREM_UNRESOLVED"
    )
