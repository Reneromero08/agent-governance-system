from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError
from .hybrid_event_geometry_audit import audit_hybrid_guard_event_geometry
from .odd_parity_fixed_deadline_counterexample import odd_parity_three_variable_holo


@dataclass(frozen=True)
class HybridNonsmoothWitnessConeAudit:
    boundary_phase_cosine: tuple[float, float, float]
    witness_direction: tuple[float, float, float]
    boundary_active_clause_count: int
    boundary_maximum_active_literal_count: int
    boundary_active_set_gap: float
    boundary_guard_directional_derivative: float
    boundary_classification: str
    sampled_epsilons: tuple[float, ...]
    sampled_guard_margins: tuple[float, ...]
    sampled_directional_derivatives: tuple[float, ...]
    every_sample_is_verified_witness: bool
    every_sample_has_zero_active_set_gap: bool
    positive_directional_entry_without_active_set_gap: bool
    simple_event_gap_is_sufficient_not_necessary: bool
    uniform_nonsmooth_neighborhood_lower_bound_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_odd_parity_nonsmooth_witness_cone(
    epsilons: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625),
) -> HybridNonsmoothWitnessConeAudit:
    """Audit an exact tied witness cone for odd three-bit parity.

    The ray ``(epsilon,-epsilon,-epsilon)`` lies in the `TFF` witness orthant for every
    positive epsilon. At the origin all four clause margins tie at zero and several
    literal maxima tie, but the radial direction `(1,-1,-1)` gives every active clause
    directional derivative one. Thus the guard has positive directional derivative even
    though the active-set gap is exactly zero.
    """

    if not epsilons or not all(isfinite(value) and value > 0.0 for value in epsilons):
        raise ConstraintHoloError("nonsmooth witness-cone epsilons must be positive")

    holo = odd_parity_three_variable_holo()
    direction = (1.0, -1.0, -1.0)
    boundary = audit_hybrid_guard_event_geometry(
        holo,
        (0.0, 0.0, 0.0),
        direction,
    )

    margins: list[float] = []
    derivatives: list[float] = []
    all_witnesses = True
    all_zero_gap = True
    for epsilon in epsilons:
        point = (epsilon, -epsilon, -epsilon)
        geometry = audit_hybrid_guard_event_geometry(holo, point, direction)
        assignment = {
            variable: point[index] > 0.0
            for index, variable in enumerate(holo.variables)
        }
        margins.append(geometry.guard_margin)
        derivatives.append(geometry.guard_directional_derivative)
        all_witnesses = all_witnesses and holo.accepts(assignment)
        all_zero_gap = all_zero_gap and geometry.active_set_gap == 0.0

    positive_without_gap = (
        boundary.event_surface
        and boundary.active_set_gap == 0.0
        and boundary.guard_directional_derivative == 1.0
        and boundary.classification == "NONSMOOTH_DIRECTIONAL_WITNESS_ENTRY"
        and all_witnesses
        and all_zero_gap
        and all(
            abs(margin - epsilon) <= 1.0e-15
            for margin, epsilon in zip(margins, epsilons, strict=True)
        )
        and all(value == 1.0 for value in derivatives)
    )

    return HybridNonsmoothWitnessConeAudit(
        boundary_phase_cosine=(0.0, 0.0, 0.0),
        witness_direction=direction,
        boundary_active_clause_count=len(boundary.active_clause_indices),
        boundary_maximum_active_literal_count=max(
            (len(active) for active in boundary.active_literal_indices),
            default=0,
        ),
        boundary_active_set_gap=boundary.active_set_gap,
        boundary_guard_directional_derivative=(
            boundary.guard_directional_derivative
        ),
        boundary_classification=boundary.classification,
        sampled_epsilons=epsilons,
        sampled_guard_margins=tuple(margins),
        sampled_directional_derivatives=tuple(derivatives),
        every_sample_is_verified_witness=all_witnesses,
        every_sample_has_zero_active_set_gap=all_zero_gap,
        positive_directional_entry_without_active_set_gap=positive_without_gap,
        simple_event_gap_is_sufficient_not_necessary=positive_without_gap,
        uniform_nonsmooth_neighborhood_lower_bound_established=False,
        status=(
            "NONSMOOTH_WITNESS_CONE_WITH_POSITIVE_DIRECTIONAL_ENTRY_ESTABLISHED__"
            "ACTIVE_SET_GAP_NOT_NECESSARY"
            if positive_without_gap
            else "NONSMOOTH_WITNESS_CONE_AUDIT_NOT_ESTABLISHED"
        ),
    )
