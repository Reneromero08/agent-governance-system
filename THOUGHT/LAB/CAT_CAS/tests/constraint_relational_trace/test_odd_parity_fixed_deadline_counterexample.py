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

from constraint_relational_trace_v1.odd_parity_fixed_deadline_counterexample import (  # noqa: E402
    audit_odd_parity_fixed_deadline_counterexample,
    audit_odd_parity_stationary_manifold,
    odd_parity_three_variable_holo,
)


def test_odd_parity_midpoint_is_exact_normally_attracting_non_solution_manifold() -> None:
    holo = odd_parity_three_variable_holo()
    audit = audit_odd_parity_stationary_manifold()

    assert audit.semantic_digest == holo.semantic_digest()
    assert audit.presentation_digest == holo.presentation_digest()
    assert audit.reference_witness_count == 4
    assert dict(audit.public_seed_assignment) == {
        "x1": True,
        "x2": False,
        "x3": True,
    }
    assert not audit.public_seed_is_witness
    assert dict(audit.stationary_assignment) == {
        "x1": False,
        "x2": False,
        "x3": False,
    }
    assert not audit.stationary_is_witness
    assert audit.stationary_derivative_max_abs == 0.0
    assert audit.clause_violation == 0.5
    assert audit.phase_tangent_eigenvalues == (-20.0, -20.0, -20.0)
    assert audit.short_one_normal_exponent == -5.0
    assert audit.long_cap_normal_exponent == -2.25
    assert audit.selector_center_dimension == 8
    assert audit.normally_attracting_transverse_to_selector_centers
    assert audit.status == (
        "ODD_PARITY_NON_SOLUTION_MIDPOINT_MANIFOLD_NORMALLY_ATTRACTING"
    )


def test_deadline_three_fails_after_transient_witness_across_solver_families() -> None:
    audit = audit_odd_parity_fixed_deadline_counterexample()

    assert audit.fixed_deadline == 3.0
    assert {control.solver_method for control in audit.solver_controls} == {
        "DOP853",
        "Radau",
    }
    assert audit.every_solver_observed_transient_witness
    assert audit.every_solver_failed_terminal_verification
    assert audit.every_solver_approached_midpoint
    assert audit.fixed_deadline_three_completeness_falsified
    assert not audit.stronger_all_polynomial_deadlines_falsified
    assert audit.status == (
        "FIXED_DEADLINE_THREE_PUBLIC_SEED_COMPLETENESS_FALSIFIED__"
        "TRANSIENT_WITNESS_ONLY__ALL_POLYNOMIAL_DEADLINES_UNRESOLVED"
    )

    for control in audit.solver_controls:
        assert control.reached_fixed_deadline
        assert control.first_passage_observed
        assert control.first_passage_time is not None
        assert control.first_passage_time < 0.3
        assert not control.terminal_solution_verified
        assert dict(control.terminal_assignment) == {
            "x1": True,
            "x2": False,
            "x3": True,
        }
        assert control.terminal_clause_satisfaction_margin < 0.0
        assert control.terminal_phase_cosine_norm < 1.0e-8
        assert control.status == "TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED"
