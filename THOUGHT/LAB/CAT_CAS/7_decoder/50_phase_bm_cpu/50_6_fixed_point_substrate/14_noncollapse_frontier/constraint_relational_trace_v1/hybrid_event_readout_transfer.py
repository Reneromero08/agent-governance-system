from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log2

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .hybrid_witness_recorder_contract import evaluate_hybrid_witness_guard


@dataclass(frozen=True)
class HybridGuardLipschitzAudit:
    left_guard_margin: float
    right_guard_margin: float
    phase_linf_distance: float
    guard_difference: float
    one_lipschitz_inequality_holds: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class ConditionalHybridEventReadoutTransfer:
    deadline_upper_bound: float
    witness_plateau_length_lower_bound: float
    witness_guard_margin_lower_bound: float
    sampling_step_upper_bound: float
    phase_error_upper_bound: float
    sample_count_upper_bound: int
    guard_precision_bits_upper_bound: int
    detected_guard_margin_lower_bound: float
    polynomial_sampling_if_inputs_inverse_polynomial: bool
    guard_readout_stability_established: bool
    state_simulation_cost_established: bool
    event_history_restoration_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_hybrid_guard_one_lipschitz(
    holo: ConstraintHolo,
    left_phase_cosine: tuple[float, ...],
    right_phase_cosine: tuple[float, ...],
    *,
    tolerance: float = 1.0e-12,
) -> HybridGuardLipschitzAudit:
    """Check the exact one-Lipschitz guard law in phase-cosine L-infinity norm.

    Each literal margin is one signed phase cosine. Maximum and minimum are both
    nonexpansive in the L-infinity norm, hence

        |G_F(c)-G_F(c')| <= ||c-c'||_infinity.

    The executable audit evaluates one pair while the theorem is algebraic for every
    public formula.
    """

    if len(left_phase_cosine) != len(holo.variables):
        raise ConstraintHoloError("left phase dimension does not match public variables")
    if len(right_phase_cosine) != len(holo.variables):
        raise ConstraintHoloError("right phase dimension does not match public variables")
    values = left_phase_cosine + right_phase_cosine
    if not all(isfinite(value) for value in values):
        raise ConstraintHoloError("hybrid guard Lipschitz inputs must be finite")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ConstraintHoloError("hybrid guard Lipschitz tolerance must be positive")

    left = evaluate_hybrid_witness_guard(holo, left_phase_cosine)
    right = evaluate_hybrid_witness_guard(holo, right_phase_cosine)
    distance = max(
        (
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_phase_cosine,
                right_phase_cosine,
                strict=True,
            )
        ),
        default=0.0,
    )
    difference = abs(left.guard_margin - right.guard_margin)
    holds = difference <= distance + tolerance

    return HybridGuardLipschitzAudit(
        left_guard_margin=left.guard_margin,
        right_guard_margin=right.guard_margin,
        phase_linf_distance=distance,
        guard_difference=difference,
        one_lipschitz_inequality_holds=holds,
        status=(
            "HYBRID_GUARD_ONE_LIPSCHITZ_AUDIT_PASS"
            if holds
            else "HYBRID_GUARD_ONE_LIPSCHITZ_AUDIT_FAILURE"
        ),
    )


def conditional_hybrid_event_readout_transfer(
    *,
    deadline_upper_bound: float,
    witness_plateau_length_lower_bound: float,
    witness_guard_margin_lower_bound: float,
) -> ConditionalHybridEventReadoutTransfer:
    """Transfer a robust witness plateau into a finite sampling/readout schedule.

    Assume that before time ``T`` there is an interval of length at least ``tau`` on
    which the exact guard is at least ``mu``. A uniform time grid with step at most
    ``tau/2`` contains a sample in that interval. If each sampled phase cosine is known
    within ``mu/2`` in L-infinity norm, the one-Lipschitz guard law gives an observed
    guard of at least ``mu/2``.

    This closes only the observation schedule. It does not prove that the continuous
    carrier can be simulated to the required state error with polynomial cost.
    """

    values = (
        deadline_upper_bound,
        witness_plateau_length_lower_bound,
        witness_guard_margin_lower_bound,
    )
    if not all(isfinite(value) and value > 0.0 for value in values):
        raise ConstraintHoloError("hybrid readout transfer inputs must be positive")

    sampling_step = witness_plateau_length_lower_bound / 2.0
    phase_error = witness_guard_margin_lower_bound / 2.0
    sample_count = ceil(deadline_upper_bound / sampling_step) + 1
    precision_bits = max(1, ceil(log2(2.0 / witness_guard_margin_lower_bound)))
    detected_margin = witness_guard_margin_lower_bound - phase_error

    return ConditionalHybridEventReadoutTransfer(
        deadline_upper_bound=deadline_upper_bound,
        witness_plateau_length_lower_bound=witness_plateau_length_lower_bound,
        witness_guard_margin_lower_bound=witness_guard_margin_lower_bound,
        sampling_step_upper_bound=sampling_step,
        phase_error_upper_bound=phase_error,
        sample_count_upper_bound=sample_count,
        guard_precision_bits_upper_bound=precision_bits,
        detected_guard_margin_lower_bound=detected_margin,
        polynomial_sampling_if_inputs_inverse_polynomial=True,
        guard_readout_stability_established=detected_margin > 0.0,
        state_simulation_cost_established=False,
        event_history_restoration_established=False,
        status=(
            "CONDITIONAL_HYBRID_EVENT_SAMPLING_AND_GUARD_READOUT_TRANSFER_ESTABLISHED__"
            "STATE_SIMULATION_COST_NOT_ESTABLISHED"
        ),
    )
