from __future__ import annotations

from pathlib import Path
import sys

import pytest

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ConstraintHoloError,
)
from constraint_relational_trace_v1.odd_parity_hybrid_event_trajectory import (  # noqa: E402
    audit_odd_parity_hybrid_event_trajectory,
)


def test_odd_parity_first_hybrid_event_is_cross_solver_stable() -> None:
    audit = audit_odd_parity_hybrid_event_trajectory()

    assert audit.every_solver_found_entry_and_exit
    assert audit.first_entry_cross_solver_agreement
    assert audit.first_event_has_positive_transverse_speed
    assert audit.first_event_has_small_but_positive_active_set_gap
    assert audit.first_interval_has_long_dwell_but_small_guard_margin
    assert not audit.asymptotic_inverse_polynomial_event_bound_established
    assert audit.entry_time_spread is not None
    assert audit.entry_time_spread < 1.0e-6
    assert audit.entry_transverse_speed_spread is not None
    assert audit.entry_transverse_speed_spread < 1.0e-8
    assert audit.entry_active_set_gap_spread is not None
    assert audit.entry_active_set_gap_spread < 1.0e-10
    assert audit.maximum_guard_margin_spread is not None
    assert audit.maximum_guard_margin_spread < 1.0e-8
    assert audit.first_dwell_time_minimum is not None
    assert audit.first_dwell_time_minimum > 1.0
    assert audit.first_dwell_time_maximum is not None
    assert audit.first_dwell_time_maximum < 1.7
    assert audit.status == (
        "ODD_PARITY_HYBRID_EVENT_GEOMETRY_CROSS_SOLVER_REFERENCE_PASS__"
        "UNIFORM_PUBLIC_SEED_LOWER_BOUND_NOT_ESTABLISHED"
    )


def test_each_solver_resolves_same_small_margin_transverse_event() -> None:
    audit = audit_odd_parity_hybrid_event_trajectory()
    assert {control.solver_method for control in audit.solver_controls} == {
        "DOP853",
        "Radau",
    }

    for control in audit.solver_controls:
        assert control.solver_success
        assert control.status == "ODD_PARITY_FIRST_HYBRID_WITNESS_INTERVAL_RESOLVED"
        assert control.first_entry_time is not None
        assert 0.20 < control.first_entry_time < 0.22
        assert control.first_exit_time is not None
        assert 1.70 < control.first_exit_time < 1.80
        assert control.first_witness_dwell_time is not None
        assert 1.4 < control.first_witness_dwell_time < 1.7
        assert control.first_entry_guard_directional_derivative is not None
        assert 2.0e-3 < control.first_entry_guard_directional_derivative < 4.0e-3
        assert control.first_entry_active_set_gap is not None
        assert 1.0e-8 < control.first_entry_active_set_gap < 1.0e-5
        assert control.maximum_first_interval_guard_margin is not None
        assert 1.0e-4 < control.maximum_first_interval_guard_margin < 2.0e-4
        assert control.maximum_first_interval_guard_time is not None
        assert 0.8 < control.maximum_first_interval_guard_time < 1.2
        assert not control.terminal_solution_verified
        assert control.terminal_guard_margin <= 1.0e-8


@pytest.mark.parametrize(
    "kwargs",
    (
        {"fixed_deadline": 0.0},
        {"relative_tolerance": 0.0},
        {"solver_methods": ()},
        {"solver_methods": ("unsupported",)},
        {"sample_count": 16},
    ),
)
def test_hybrid_event_trajectory_rejects_invalid_controls(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConstraintHoloError):
        audit_odd_parity_hybrid_event_trajectory(**kwargs)
