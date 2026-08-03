from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .odd_parity_fixed_deadline_counterexample import odd_parity_three_variable_holo


@dataclass(frozen=True)
class TopologicalWitnessRecordWallAudit:
    semantic_digest: str
    presentation_digest: str
    base_assignment: tuple[tuple[str, bool], ...]
    visited_assignment: tuple[tuple[str, bool], ...]
    base_assignment_is_witness: bool
    visited_assignment_is_witness: bool
    excursion_quarter_turn_waypoints: tuple[tuple[int, int, int], ...]
    net_quarter_turn_displacement: tuple[int, int, int]
    winding_vector: tuple[int, int, int]
    oriented_threshold_crossings: tuple[int, int]
    net_oriented_threshold_crossing: int
    excursion_visits_witness_orthant: bool
    constant_loop_visits_witness_orthant: bool
    endpoint_fixed_homotopy_equivalent: bool
    winding_or_homotopy_recorder_distinguishes_visit: bool
    unsigned_crossing_count_is_topological_invariant: bool
    pure_topological_visit_persistence_possible: bool
    required_non_topological_resource: str
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_topological_witness_record_wall() -> TopologicalWitnessRecordWallAudit:
    """Show that transient witness visitation is not a homotopy invariant.

    Angles are represented in quarter turns. ``2`` means phase angle ``pi`` and
    therefore cosine ``-1``; ``0`` means phase angle ``0`` and cosine ``+1``; ``1``
    is the threshold surface where cosine is zero.

    The excursion starts at the non-witness assignment ``FFF``, enters the odd-parity
    witness orthant ``TFF``, and retraces the same arc. It is a path followed by its
    inverse, hence endpoint-fixed null-homotopic with zero winding. The constant loop at
    ``FFF`` has the same endpoints and homotopy class but never visits a witness.

    Consequently no recorder depending only on path homotopy, winding, or oriented net
    crossing can persist exact witness visitation. Counting unsigned crossings would
    distinguish these paths, but unsigned crossing number changes under an arbitrarily
    small birth/death of a crossing pair and is therefore geometric/hybrid history, not
    a topological invariant.
    """

    holo = odd_parity_three_variable_holo()
    base = {"x1": False, "x2": False, "x3": False}
    visited = {"x1": True, "x2": False, "x3": False}

    waypoints = (
        (2, 2, 2),
        (1, 2, 2),
        (0, 2, 2),
        (1, 2, 2),
        (2, 2, 2),
    )
    increments = tuple(
        tuple(right[index] - left[index] for index in range(3))
        for left, right in zip(waypoints, waypoints[1:], strict=True)
    )
    net_displacement = tuple(
        sum(increment[index] for increment in increments) for index in range(3)
    )
    winding = tuple(value // 4 for value in net_displacement)
    oriented_crossings = (1, -1)
    net_crossing = sum(oriented_crossings)

    base_is_witness = holo.accepts(base)
    visited_is_witness = holo.accepts(visited)
    null_homotopic = net_displacement == (0, 0, 0)
    topological_distinguishes = False

    established = (
        not base_is_witness
        and visited_is_witness
        and null_homotopic
        and winding == (0, 0, 0)
        and net_crossing == 0
        and not topological_distinguishes
    )

    return TopologicalWitnessRecordWallAudit(
        semantic_digest=holo.semantic_digest(),
        presentation_digest=holo.presentation_digest(),
        base_assignment=tuple(sorted(base.items())),
        visited_assignment=tuple(sorted(visited.items())),
        base_assignment_is_witness=base_is_witness,
        visited_assignment_is_witness=visited_is_witness,
        excursion_quarter_turn_waypoints=waypoints,
        net_quarter_turn_displacement=net_displacement,
        winding_vector=winding,
        oriented_threshold_crossings=oriented_crossings,
        net_oriented_threshold_crossing=net_crossing,
        excursion_visits_witness_orthant=True,
        constant_loop_visits_witness_orthant=False,
        endpoint_fixed_homotopy_equivalent=null_homotopic,
        winding_or_homotopy_recorder_distinguishes_visit=topological_distinguishes,
        unsigned_crossing_count_is_topological_invariant=False,
        pure_topological_visit_persistence_possible=False,
        required_non_topological_resource=(
            "GEOMETRIC_OR_HYBRID_EVENT_HISTORY_WITH_EXPLICIT_PRECISION_AND_RESTORATION"
        ),
        status=(
            "PURE_TOPOLOGICAL_WITNESS_VISIT_RECORDER_REJECTED__"
            "TRANSIENT_ORTHANT_ENTRY_IS_NOT_HOMOTOPY_INVARIANT"
            if established
            else "TOPOLOGICAL_WITNESS_RECORD_WALL_NOT_ESTABLISHED"
        ),
    )
