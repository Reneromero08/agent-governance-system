from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class HybridGuardEventGeometryAudit:
    guard_margin: float
    clause_margins: tuple[float, ...]
    active_clause_indices: tuple[int, ...]
    active_literal_indices: tuple[tuple[int, ...], ...]
    clause_directional_derivatives: tuple[float, ...]
    guard_directional_derivative: float
    unique_classical_derivative: bool
    active_clause_gap: float
    active_literal_gap: float
    active_set_gap: float
    event_surface: bool
    classification: str
    inverse_polynomial_transversality_established: bool
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class ConditionalHybridEventResourceTransfer:
    transverse_speed_lower_bound: float
    guard_acceleration_upper_bound: float
    active_set_gap: float
    coordinate_speed_upper_bound: float
    active_set_stability_time: float
    derivative_stability_time: float
    guaranteed_witness_dwell_time: float
    guaranteed_guard_margin: float
    conditional_transfer_established: bool
    unconditional_polynomial_event_resources_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def _validate_vector(
    name: str,
    values: tuple[float, ...],
    expected: int,
) -> None:
    if len(values) != expected:
        raise ConstraintHoloError(f"{name} dimension does not match public variables")
    if not all(isfinite(value) for value in values):
        raise ConstraintHoloError(f"{name} must contain only finite values")


def audit_hybrid_guard_event_geometry(
    holo: ConstraintHolo,
    phase_cosine: tuple[float, ...],
    phase_velocity: tuple[float, ...],
    *,
    tolerance: float = 1.0e-12,
) -> HybridGuardEventGeometryAudit:
    """Audit the exact directional geometry of the semialgebraic witness guard.

    For literal margins ``a_jr = q_jr c_i`` and clause margins

        M_j = max_r a_jr,

    the witness guard is ``G = min_j M_j``. Its one-sided directional derivative is

        D G(c; dc) = min_{j in argmin M} max_{r in argmax a_j} q_jr dc_i.

    A classical derivative exists only when the active clause and its active literal are
    both unique. Tied min/max sets remain directionally differentiable but nonsmooth.
    """

    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ConstraintHoloError("hybrid event tolerance must be positive and finite")
    variable_count = len(holo.variables)
    _validate_vector("phase cosine", phase_cosine, variable_count)
    _validate_vector("phase velocity", phase_velocity, variable_count)

    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    clause_margins: list[float] = []
    active_literals: list[tuple[int, ...]] = []
    clause_directional: list[float] = []
    literal_gaps: list[float] = []

    for clause in holo.clauses:
        margins: list[float] = []
        directional: list[float] = []
        for literal in clause.literals:
            index = variable_index[literal.variable]
            sign = 1.0 if literal.positive else -1.0
            margins.append(sign * phase_cosine[index])
            directional.append(sign * phase_velocity[index])
        clause_margin = max(margins)
        active = tuple(
            index
            for index, value in enumerate(margins)
            if abs(value - clause_margin) <= tolerance
        )
        clause_margins.append(clause_margin)
        active_literals.append(active)
        clause_directional.append(max(directional[index] for index in active))
        if len(active) == 1:
            winner = active[0]
            runner_up = max(
                value for index, value in enumerate(margins) if index != winner
            )
            literal_gaps.append(clause_margin - runner_up)
        else:
            literal_gaps.append(0.0)

    guard_margin = min(clause_margins, default=1.0)
    active_clauses = tuple(
        index
        for index, value in enumerate(clause_margins)
        if abs(value - guard_margin) <= tolerance
    )
    guard_directional = min(
        (clause_directional[index] for index in active_clauses),
        default=0.0,
    )
    unique_derivative = (
        len(active_clauses) == 1
        and len(active_literals[active_clauses[0]]) == 1
    )

    if len(active_clauses) == 1 and len(clause_margins) > 1:
        active_clause = active_clauses[0]
        clause_gap = min(
            value
            for index, value in enumerate(clause_margins)
            if index != active_clause
        ) - guard_margin
    elif len(holo.clauses) <= 1:
        clause_gap = float("inf")
    else:
        clause_gap = 0.0

    active_literal_gap = (
        literal_gaps[active_clauses[0]] if len(active_clauses) == 1 else 0.0
    )
    active_set_gap = min(clause_gap, active_literal_gap)
    event_surface = abs(guard_margin) <= tolerance

    if guard_margin > tolerance:
        classification = "STRICT_WITNESS_REGION"
    elif guard_margin < -tolerance:
        classification = "STRICT_NON_WITNESS_REGION"
    elif guard_directional > tolerance:
        classification = (
            "SMOOTH_TRANSVERSE_WITNESS_ENTRY"
            if unique_derivative
            else "NONSMOOTH_DIRECTIONAL_WITNESS_ENTRY"
        )
    elif guard_directional < -tolerance:
        classification = (
            "SMOOTH_TRANSVERSE_WITNESS_EXIT"
            if unique_derivative
            else "NONSMOOTH_DIRECTIONAL_WITNESS_EXIT"
        )
    else:
        classification = (
            "SMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT"
            if unique_derivative
            else "NONSMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT"
        )

    return HybridGuardEventGeometryAudit(
        guard_margin=guard_margin,
        clause_margins=tuple(clause_margins),
        active_clause_indices=active_clauses,
        active_literal_indices=tuple(active_literals),
        clause_directional_derivatives=tuple(clause_directional),
        guard_directional_derivative=guard_directional,
        unique_classical_derivative=unique_derivative,
        active_clause_gap=clause_gap,
        active_literal_gap=active_literal_gap,
        active_set_gap=active_set_gap,
        event_surface=event_surface,
        classification=classification,
        inverse_polynomial_transversality_established=False,
    )


def conditional_hybrid_event_resource_transfer(
    *,
    transverse_speed_lower_bound: float,
    guard_acceleration_upper_bound: float,
    active_set_gap: float,
    coordinate_speed_upper_bound: float,
) -> ConditionalHybridEventResourceTransfer:
    """Transfer local geometric bounds into witness margin and dwell guarantees.

    Assume a simple event at ``G(0)=0`` with ``G_dot(0) >= kappa``. If the active
    clause/literal gap is ``Delta``, coordinate speed is at most ``V``, and the active
    branch acceleration is at most ``A``, then the same active branch persists for at
    least ``Delta/(4V)`` and its derivative stays at least ``kappa/2`` for at least
    ``kappa/(2A)``. On the smaller interval, ``G(t) >= kappa t / 2``.

    This is a conditional local lemma. The current campaign has not proved uniform
    inverse-polynomial ``kappa`` or ``Delta`` on every public-seed SAT trajectory.
    """

    values = (
        transverse_speed_lower_bound,
        guard_acceleration_upper_bound,
        active_set_gap,
        coordinate_speed_upper_bound,
    )
    if not all(isfinite(value) for value in values):
        raise ConstraintHoloError("hybrid event resource bounds must be finite")
    if transverse_speed_lower_bound <= 0.0:
        raise ConstraintHoloError("transverse speed lower bound must be positive")
    if guard_acceleration_upper_bound < 0.0:
        raise ConstraintHoloError("guard acceleration upper bound must be nonnegative")
    if active_set_gap <= 0.0:
        raise ConstraintHoloError("active-set gap must be positive")
    if coordinate_speed_upper_bound <= 0.0:
        raise ConstraintHoloError("coordinate speed upper bound must be positive")

    active_time = active_set_gap / (4.0 * coordinate_speed_upper_bound)
    derivative_time = (
        active_time
        if guard_acceleration_upper_bound == 0.0
        else transverse_speed_lower_bound
        / (2.0 * guard_acceleration_upper_bound)
    )
    dwell = min(active_time, derivative_time)
    margin = 0.5 * transverse_speed_lower_bound * dwell

    return ConditionalHybridEventResourceTransfer(
        transverse_speed_lower_bound=transverse_speed_lower_bound,
        guard_acceleration_upper_bound=guard_acceleration_upper_bound,
        active_set_gap=active_set_gap,
        coordinate_speed_upper_bound=coordinate_speed_upper_bound,
        active_set_stability_time=active_time,
        derivative_stability_time=derivative_time,
        guaranteed_witness_dwell_time=dwell,
        guaranteed_guard_margin=margin,
        conditional_transfer_established=dwell > 0.0 and margin > 0.0,
        unconditional_polynomial_event_resources_established=False,
        status=(
            "CONDITIONAL_TRANSVERSE_EVENT_MARGIN_AND_DWELL_TRANSFER_ESTABLISHED__"
            "UNIFORM_PUBLIC_SEED_BOUNDS_NOT_ESTABLISHED"
        ),
    )
