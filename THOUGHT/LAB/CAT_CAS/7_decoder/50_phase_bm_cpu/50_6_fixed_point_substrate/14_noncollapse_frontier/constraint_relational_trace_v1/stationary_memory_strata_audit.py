from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class StationaryMemoryStrataAudit:
    short_threshold: float
    long_threshold: float
    both_memory_coordinates_interior_excluded: bool
    boundary_boundary_strata: tuple[str, ...]
    short_interior_strata: tuple[str, ...]
    long_interior_strata: tuple[str, ...]
    total_stationary_stratum_types: int
    status: str
    claim_ceiling: str = CLAIM_CEILING


def stationary_memory_strata(
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> tuple[str, ...]:
    """Return every memory stratum compatible with zero memory derivative.

    For one clause:

        s_dot = beta (C-gamma) s(1-s)
        l_dot = alpha (C-delta) l(1-l/L).

    Since gamma != delta, both memory coordinates cannot be interior at a stationary
    point. The returned labels enumerate every remaining algebraic possibility.
    """

    boundary_boundary = tuple(
        f"SHORT_{short}__LONG_{long}__C_ARBITRARY"
        for short in ("ZERO", "ONE")
        for long in ("ZERO", "CAP")
    )
    short_interior = tuple(
        f"SHORT_INTERIOR_C_EQ_GAMMA__LONG_{long}"
        for long in ("ZERO", "CAP")
    )
    long_interior = tuple(
        f"SHORT_{short}__LONG_INTERIOR_C_EQ_DELTA"
        for short in ("ZERO", "ONE")
    )
    return boundary_boundary + short_interior + long_interior


def audit_stationary_memory_strata(
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> StationaryMemoryStrataAudit:
    boundary_boundary = tuple(
        f"SHORT_{short}__LONG_{long}__C_ARBITRARY"
        for short in ("ZERO", "ONE")
        for long in ("ZERO", "CAP")
    )
    short_interior = tuple(
        f"SHORT_INTERIOR_C_EQ_GAMMA__LONG_{long}"
        for long in ("ZERO", "CAP")
    )
    long_interior = tuple(
        f"SHORT_{short}__LONG_INTERIOR_C_EQ_DELTA"
        for short in ("ZERO", "ONE")
    )
    both_interior_excluded = parameters.gamma != parameters.delta
    strata = stationary_memory_strata(parameters)

    return StationaryMemoryStrataAudit(
        short_threshold=parameters.gamma,
        long_threshold=parameters.delta,
        both_memory_coordinates_interior_excluded=both_interior_excluded,
        boundary_boundary_strata=boundary_boundary,
        short_interior_strata=short_interior,
        long_interior_strata=long_interior,
        total_stationary_stratum_types=len(strata),
        status=(
            "STATIONARY_MEMORY_STRATA_CLASSIFIED__BOTH_INTERIOR_EXCLUDED"
            if both_interior_excluded
            else "STATIONARY_MEMORY_STRATA_DEGENERATE_THRESHOLDS"
        ),
    )
