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

from constraint_relational_trace_v1.native_hybrid_event_geometry_audit import (  # noqa: E402
    audit_native_hybrid_event_geometry,
    audit_odd_parity_simple_native_grazing_control,
    odd_parity_simple_native_grazing_state,
)


def test_current_native_law_has_exact_simple_guard_grazing_state() -> None:
    audit = audit_odd_parity_simple_native_grazing_control()
    geometry = audit.event_geometry

    assert geometry.event_surface
    assert geometry.guard_margin == 0.0
    assert geometry.active_clause_indices == (3,)
    assert geometry.active_literal_indices[3] == (0,)
    assert geometry.unique_classical_derivative
    assert geometry.active_set_gap == 0.5
    assert geometry.guard_directional_derivative == 0.0
    assert geometry.classification == "SMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT"

    assert audit.native_derivative_max_abs > 0.0
    assert not audit.full_carrier_stationary
    assert audit.simple_guard_surface
    assert audit.smooth_native_grazing_event
    assert not audit.structural_transversality_established
    assert audit.public_seed_transversality_status == (
        "NOT_DECIDED_BY_ARBITRARY_STATE_CONTROL"
    )
    assert audit.status == (
        "NATIVE_SIMPLE_GUARD_GRAZING_STATE_ESTABLISHED__"
        "PUBLIC_SEED_TRANSVERSALITY_REQUIRES_SEPARATE_THEOREM"
    )


def test_grazing_control_uses_valid_phase_circles_and_zero_active_velocity() -> None:
    holo, state = odd_parity_simple_native_grazing_state()
    audit = audit_native_hybrid_event_geometry(holo, state)

    for cosine, sine in zip(
        state.phase_cosine,
        state.phase_sine,
        strict=True,
    ):
        assert abs(cosine * cosine + sine * sine - 1.0) <= 1.0e-15

    assert audit.event_geometry.clause_margins == (0.5, 0.5, 0.5, 0.0)
    assert audit.event_geometry.clause_directional_derivatives[3] == 0.0
