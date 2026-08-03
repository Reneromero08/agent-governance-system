from __future__ import annotations

from math import isclose
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
from constraint_relational_trace_v1.hybrid_event_geometry_audit import (  # noqa: E402
    audit_hybrid_guard_event_geometry,
    conditional_hybrid_event_resource_transfer,
)
from constraint_relational_trace_v1.odd_parity_fixed_deadline_counterexample import (  # noqa: E402
    odd_parity_three_variable_holo,
)


def test_simple_odd_parity_boundary_has_smooth_transverse_entry() -> None:
    audit = audit_hybrid_guard_event_geometry(
        odd_parity_three_variable_holo(),
        (0.0, -0.5, -0.5),
        (1.0, 0.0, 0.0),
    )

    assert audit.event_surface
    assert audit.guard_margin == 0.0
    assert audit.active_clause_indices == (3,)
    assert audit.active_literal_indices[3] == (0,)
    assert audit.unique_classical_derivative
    assert audit.guard_directional_derivative == 1.0
    assert audit.active_clause_gap == 0.5
    assert audit.active_literal_gap == 0.5
    assert audit.active_set_gap == 0.5
    assert audit.classification == "SMOOTH_TRANSVERSE_WITNESS_ENTRY"
    assert not audit.inverse_polynomial_transversality_established


def test_simple_boundary_direction_reversal_is_smooth_exit() -> None:
    audit = audit_hybrid_guard_event_geometry(
        odd_parity_three_variable_holo(),
        (0.0, -0.5, -0.5),
        (-1.0, 0.0, 0.0),
    )

    assert audit.guard_directional_derivative == -1.0
    assert audit.classification == "SMOOTH_TRANSVERSE_WITNESS_EXIT"


def test_zero_normal_velocity_is_grazing_or_higher_order() -> None:
    audit = audit_hybrid_guard_event_geometry(
        odd_parity_three_variable_holo(),
        (0.0, -0.5, -0.5),
        (0.0, 1.0, 0.0),
    )

    assert audit.unique_classical_derivative
    assert audit.guard_directional_derivative == 0.0
    assert audit.classification == "SMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT"


def test_tied_min_max_geometry_is_nonsmooth_and_direction_dependent() -> None:
    holo = odd_parity_three_variable_holo()
    entry = audit_hybrid_guard_event_geometry(
        holo,
        (0.0, 0.0, -0.5),
        (1.0, -1.0, 0.0),
    )
    exit_event = audit_hybrid_guard_event_geometry(
        holo,
        (0.0, 0.0, -0.5),
        (1.0, 1.0, 0.0),
    )

    assert entry.active_clause_indices == (2, 3)
    assert entry.active_literal_indices[2] == (0, 1)
    assert entry.active_literal_indices[3] == (0, 1)
    assert not entry.unique_classical_derivative
    assert entry.active_set_gap == 0.0
    assert entry.guard_directional_derivative == 1.0
    assert entry.classification == "NONSMOOTH_DIRECTIONAL_WITNESS_ENTRY"

    assert exit_event.guard_directional_derivative == -1.0
    assert exit_event.classification == "NONSMOOTH_DIRECTIONAL_WITNESS_EXIT"


def test_conditional_transverse_bounds_transfer_to_margin_and_dwell() -> None:
    result = conditional_hybrid_event_resource_transfer(
        transverse_speed_lower_bound=0.25,
        guard_acceleration_upper_bound=2.0,
        active_set_gap=0.5,
        coordinate_speed_upper_bound=1.0,
    )

    assert isclose(result.active_set_stability_time, 0.125)
    assert isclose(result.derivative_stability_time, 0.0625)
    assert isclose(result.guaranteed_witness_dwell_time, 0.0625)
    assert isclose(result.guaranteed_guard_margin, 0.0078125)
    assert result.conditional_transfer_established
    assert not result.unconditional_polynomial_event_resources_established
    assert result.status == (
        "CONDITIONAL_TRANSVERSE_EVENT_MARGIN_AND_DWELL_TRANSFER_ESTABLISHED__"
        "UNIFORM_PUBLIC_SEED_BOUNDS_NOT_ESTABLISHED"
    )


@pytest.mark.parametrize(
    ("cosines", "velocity"),
    (
        ((0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, float("nan")), (0.0, 0.0, 0.0)),
    ),
)
def test_geometry_audit_fails_closed_on_bad_vectors(
    cosines: tuple[float, ...],
    velocity: tuple[float, ...],
) -> None:
    with pytest.raises(ConstraintHoloError):
        audit_hybrid_guard_event_geometry(
            odd_parity_three_variable_holo(),
            cosines,
            velocity,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "transverse_speed_lower_bound": 0.0,
            "guard_acceleration_upper_bound": 1.0,
            "active_set_gap": 1.0,
            "coordinate_speed_upper_bound": 1.0,
        },
        {
            "transverse_speed_lower_bound": 1.0,
            "guard_acceleration_upper_bound": -1.0,
            "active_set_gap": 1.0,
            "coordinate_speed_upper_bound": 1.0,
        },
        {
            "transverse_speed_lower_bound": 1.0,
            "guard_acceleration_upper_bound": 1.0,
            "active_set_gap": 0.0,
            "coordinate_speed_upper_bound": 1.0,
        },
    ),
)
def test_conditional_transfer_rejects_nonpositive_or_invalid_bounds(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ConstraintHoloError):
        conditional_hybrid_event_resource_transfer(**kwargs)
