from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class MemoryLogitDriftAudit:
    short_threshold: float
    long_threshold: float
    normalized_drift_rate: float
    violation_independent: bool
    interior_periodic_orbits_excluded: bool
    bounded_interior_recurrence_excluded: bool
    omega_limit_boundary_stratum_required: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def memory_logit_derivative_identity(
    clause_violation: float,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> tuple[float, float, float]:
    """Return exact short, long, and normalized-difference logit derivatives.

    For short memory `s` and normalized long memory `r = l/cap`, define

        u = log(s/(1-s))
        v = log(r/(1-r)).

    On the interior carrier:

        u_dot = beta (C-gamma)
        v_dot = alpha (C-delta)
        d/dt(v/alpha-u/beta) = gamma-delta.
    """

    if clause_violation < 0.0:
        raise ConstraintHoloError("clause violation must be nonnegative")
    short_logit_derivative = parameters.beta * (
        clause_violation - parameters.gamma
    )
    long_logit_derivative = parameters.alpha * (
        clause_violation - parameters.delta
    )
    normalized_difference_derivative = (
        long_logit_derivative / parameters.alpha
        - short_logit_derivative / parameters.beta
    )
    return (
        short_logit_derivative,
        long_logit_derivative,
        normalized_difference_derivative,
    )


def audit_memory_logit_drift(
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> MemoryLogitDriftAudit:
    samples = (0.0, parameters.delta, parameters.gamma, 1.0, 4.0)
    observed = tuple(
        memory_logit_derivative_identity(value, parameters)[2]
        for value in samples
    )
    expected = parameters.gamma - parameters.delta
    independent = all(value == expected for value in observed)
    nonzero = expected != 0.0
    excludes_interior_recurrence = independent and nonzero

    return MemoryLogitDriftAudit(
        short_threshold=parameters.gamma,
        long_threshold=parameters.delta,
        normalized_drift_rate=expected,
        violation_independent=independent,
        interior_periodic_orbits_excluded=excludes_interior_recurrence,
        bounded_interior_recurrence_excluded=excludes_interior_recurrence,
        omega_limit_boundary_stratum_required=excludes_interior_recurrence,
        status=(
            "MEMORY_LOGIT_CLOCK_ESTABLISHED__INTERIOR_RECURRENCE_EXCLUDED"
            if excludes_interior_recurrence
            else "MEMORY_LOGIT_CLOCK_NOT_ESTABLISHED"
        ),
    )
