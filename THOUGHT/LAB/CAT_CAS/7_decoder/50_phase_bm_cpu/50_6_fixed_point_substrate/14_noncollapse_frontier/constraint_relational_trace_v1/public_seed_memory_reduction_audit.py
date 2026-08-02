from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters
from .stationary_memory_strata_audit import stationary_memory_strata


@dataclass(frozen=True)
class PublicSeedMemoryReductionAudit:
    alpha_over_beta: float
    public_clock_exponential_rate: float
    public_short_initial_odds: float
    excluded_stationary_strata: tuple[str, ...]
    forward_compatible_stationary_strata: tuple[str, ...]
    finite_time_boundary_contact_excluded: bool
    long_memory_reconstructible_from_short_and_time: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def public_seed_normalized_long_memory_from_short(
    short_memory: float,
    elapsed_time: float,
    long_memory_cap: float,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> float:
    """Reconstruct normalized long memory from short memory and elapsed time.

    This function uses the declared public initial condition ``s(0)=1/2`` and
    ``l(0)=1``. With ``r=l/L`` and the frozen memory equations,

        odds(r(t)) = exp(alpha(gamma-delta)t)/(L-1)
                     * odds(s(t))**(alpha/beta).

    The relation is exact in the native continuous equations. Floating-point
    evaluation here is reference instrumentation only.
    """

    values = (short_memory, elapsed_time, long_memory_cap)
    if not all(isfinite(value) for value in values):
        raise ConstraintHoloError("public-seed memory reduction requires finite inputs")
    if not 0.0 < short_memory < 1.0:
        raise ConstraintHoloError("short memory must be interior")
    if elapsed_time < 0.0:
        raise ConstraintHoloError("elapsed time must be nonnegative")
    if long_memory_cap <= 1.0:
        raise ConstraintHoloError("public long-memory cap must exceed the initial value")

    short_odds = short_memory / (1.0 - short_memory)
    long_odds = (
        exp(parameters.alpha * (parameters.gamma - parameters.delta) * elapsed_time)
        / (long_memory_cap - 1.0)
        * short_odds ** (parameters.alpha / parameters.beta)
    )
    return long_odds / (1.0 + long_odds)


def classify_public_seed_forward_compatible_stationary_strata(
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split stationary memory strata by compatibility with the public memory clock."""

    if parameters.gamma <= parameters.delta:
        raise ConstraintHoloError(
            "public-seed forward-stratum reduction requires gamma greater than delta"
        )

    excluded = (
        "SHORT_ONE__LONG_ZERO__C_ARBITRARY",
        "SHORT_INTERIOR_C_EQ_GAMMA__LONG_ZERO",
        "SHORT_ONE__LONG_INTERIOR_C_EQ_DELTA",
    )
    all_strata = stationary_memory_strata(parameters)
    forward_compatible = tuple(label for label in all_strata if label not in excluded)
    if len(forward_compatible) != 5:
        raise AssertionError("unexpected stationary memory stratum inventory")
    return excluded, forward_compatible


def audit_public_seed_memory_reduction(
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> PublicSeedMemoryReductionAudit:
    excluded, forward_compatible = (
        classify_public_seed_forward_compatible_stationary_strata(parameters)
    )
    alpha_over_beta = parameters.alpha / parameters.beta
    clock_rate = parameters.alpha * (parameters.gamma - parameters.delta)
    established = (
        parameters.gamma > parameters.delta
        and alpha_over_beta > 0.0
        and clock_rate > 0.0
        and len(excluded) == 3
        and len(forward_compatible) == 5
    )
    return PublicSeedMemoryReductionAudit(
        alpha_over_beta=alpha_over_beta,
        public_clock_exponential_rate=clock_rate,
        public_short_initial_odds=1.0,
        excluded_stationary_strata=excluded,
        forward_compatible_stationary_strata=forward_compatible,
        finite_time_boundary_contact_excluded=established,
        long_memory_reconstructible_from_short_and_time=established,
        status=(
            "PUBLIC_SEED_MEMORY_REDUCTION_ESTABLISHED__FIVE_FORWARD_STRATA_REMAIN"
            if established
            else "PUBLIC_SEED_MEMORY_REDUCTION_NOT_ESTABLISHED"
        ),
    )
