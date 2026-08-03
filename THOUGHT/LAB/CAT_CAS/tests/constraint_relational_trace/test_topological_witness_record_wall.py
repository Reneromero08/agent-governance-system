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

from constraint_relational_trace_v1.topological_witness_record_wall import (  # noqa: E402
    audit_topological_witness_record_wall,
)


def test_contractible_excursion_visits_witness_without_topological_change() -> None:
    audit = audit_topological_witness_record_wall()

    assert not audit.base_assignment_is_witness
    assert audit.visited_assignment_is_witness
    assert audit.excursion_visits_witness_orthant
    assert not audit.constant_loop_visits_witness_orthant
    assert audit.endpoint_fixed_homotopy_equivalent
    assert audit.net_quarter_turn_displacement == (0, 0, 0)
    assert audit.winding_vector == (0, 0, 0)
    assert audit.oriented_threshold_crossings == (1, -1)
    assert audit.net_oriented_threshold_crossing == 0


def test_pure_topological_recorder_cannot_persist_transient_witness_visit() -> None:
    audit = audit_topological_witness_record_wall()

    assert not audit.winding_or_homotopy_recorder_distinguishes_visit
    assert not audit.unsigned_crossing_count_is_topological_invariant
    assert not audit.pure_topological_visit_persistence_possible
    assert audit.required_non_topological_resource == (
        "GEOMETRIC_OR_HYBRID_EVENT_HISTORY_WITH_EXPLICIT_PRECISION_AND_RESTORATION"
    )
    assert audit.status == (
        "PURE_TOPOLOGICAL_WITNESS_VISIT_RECORDER_REJECTED__"
        "TRANSIENT_ORTHANT_ENTRY_IS_NOT_HOMOTOPY_INVARIANT"
    )


def test_excursion_is_exact_path_followed_by_its_inverse() -> None:
    audit = audit_topological_witness_record_wall()
    waypoints = audit.excursion_quarter_turn_waypoints

    assert waypoints[0] == waypoints[-1] == (2, 2, 2)
    assert waypoints[1] == waypoints[-2] == (1, 2, 2)
    assert waypoints[2] == (0, 2, 2)
