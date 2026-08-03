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

from constraint_relational_trace_v1.hybrid_event_acceleration_bound import (  # noqa: E402
    derive_hybrid_event_polynomial_acceleration_bound,
)
from constraint_relational_trace_v1.hybrid_event_geometry_audit import (  # noqa: E402
    conditional_hybrid_event_resource_transfer,
)
from constraint_relational_trace_v1.odd_parity_fixed_deadline_counterexample import (  # noqa: E402
    odd_parity_three_variable_holo,
)
from constraint_relational_trace_v1.reduced_phase_resource_bound import (  # noqa: E402
    derive_reduced_phase_polynomial_resource_bound,
)


def test_hybrid_event_acceleration_bound_is_positive_and_polynomially_derived() -> None:
    holo = odd_parity_three_variable_holo()
    base = derive_reduced_phase_polynomial_resource_bound(holo)
    audit = derive_hybrid_event_polynomial_acceleration_bound(holo)

    assert audit.public_variables == 3
    assert audit.public_clauses == 4
    assert audit.maximum_phase_cosine_speed == base.maximum_angular_speed
    assert audit.maximum_clause_violation_derivative > 0.0
    assert audit.maximum_exact_gradient_derivative > 0.0
    assert audit.maximum_rigidity_derivative > 0.0
    assert audit.maximum_force_per_occurrence_derivative > 0.0
    assert audit.maximum_variable_force_derivative > 0.0
    assert audit.maximum_incident_violation_derivative > 0.0
    assert audit.maximum_angular_acceleration > 0.0
    assert audit.maximum_phase_cosine_acceleration > 0.0
    assert isfinite(audit.maximum_simple_guard_branch_acceleration)
    assert audit.polynomial_upper_bound_established
    assert not audit.inverse_polynomial_crossing_speed_established
    assert not audit.inverse_polynomial_active_set_gap_established
    assert audit.status == (
        "HYBRID_EVENT_POLYNOMIAL_SPEED_AND_ACCELERATION_UPPER_BOUNDS_ESTABLISHED__"
        "TRANSVERSE_LOWER_BOUNDS_NOT_ESTABLISHED"
    )


def test_acceleration_upper_bound_can_feed_conditional_dwell_transfer() -> None:
    audit = derive_hybrid_event_polynomial_acceleration_bound(
        odd_parity_three_variable_holo()
    )
    transfer = conditional_hybrid_event_resource_transfer(
        transverse_speed_lower_bound=0.1,
        guard_acceleration_upper_bound=(
            audit.maximum_simple_guard_branch_acceleration
        ),
        active_set_gap=0.25,
        coordinate_speed_upper_bound=audit.maximum_phase_cosine_speed,
    )

    assert transfer.conditional_transfer_established
    assert transfer.guaranteed_witness_dwell_time > 0.0
    assert transfer.guaranteed_guard_margin > 0.0
    assert not transfer.unconditional_polynomial_event_resources_established


def test_bound_scales_monotonically_on_duplicated_clause_presentation() -> None:
    holo = odd_parity_three_variable_holo()
    duplicated = holo.with_duplicate_clause(0)
    left = derive_hybrid_event_polynomial_acceleration_bound(holo)
    right = derive_hybrid_event_polynomial_acceleration_bound(duplicated)

    assert right.public_clauses == left.public_clauses + 1
    assert right.maximum_phase_cosine_speed >= left.maximum_phase_cosine_speed
    assert (
        right.maximum_simple_guard_branch_acceleration
        >= left.maximum_simple_guard_branch_acceleration
    )
