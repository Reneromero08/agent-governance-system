from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError
from .hybrid_witness_recorder_contract import evaluate_hybrid_witness_guard
from .odd_parity_fixed_deadline_counterexample import odd_parity_three_variable_holo


@dataclass(frozen=True)
class HybridEventPrecisionWallAudit:
    sampled_epsilons: tuple[float, ...]
    sampled_guard_margins: tuple[float, ...]
    all_samples_are_verified_witnesses: bool
    guard_margin_tracks_epsilon_exactly: bool
    witness_guard_infimum: float
    positive_formula_only_guard_lower_bound_exists: bool
    semantic_contract_implies_polynomial_event_precision: bool
    semantic_contract_implies_polynomial_witness_dwell_time: bool
    dynamic_transversality_theorem_required: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_hybrid_event_precision_wall(
    *,
    epsilons: tuple[float, ...] = (0.5, 0.1, 0.01, 0.001),
) -> HybridEventPrecisionWallAudit:
    """Show that witness semantics alone provide no positive guard-margin floor.

    For the fixed odd-parity formula, every state

        (c1,c2,c3) = (epsilon,-epsilon,-epsilon)

    with ``epsilon > 0`` is a verified witness and has hybrid guard margin exactly
    ``epsilon``. Hence the infimum of witness guard margins is zero even for one fixed
    three-variable formula.

    This does not prove that the qualified public trajectory realizes arbitrarily small
    margins. It proves that semantic correctness of the guard cannot by itself supply an
    inverse-polynomial event margin or dwell time. Those must come from a separate
    dynamic transversality theorem for the declared public seed.
    """

    if not epsilons:
        raise ConstraintHoloError("hybrid precision wall requires at least one epsilon")
    if not all(isfinite(value) and 0.0 < value < 1.0 for value in epsilons):
        raise ConstraintHoloError("hybrid precision wall epsilons must lie in (0,1)")

    holo = odd_parity_three_variable_holo()
    evaluations = tuple(
        evaluate_hybrid_witness_guard(holo, (epsilon, -epsilon, -epsilon))
        for epsilon in epsilons
    )
    margins = tuple(item.guard_margin for item in evaluations)
    all_witnesses = all(
        item.unambiguous_sign_assignment
        and item.public_verifier_accepts is True
        and item.event_enabled
        for item in evaluations
    )
    exact_tracking = all(
        abs(margin - epsilon) <= 1.0e-15
        for margin, epsilon in zip(margins, epsilons, strict=True)
    )
    established = all_witnesses and exact_tracking

    return HybridEventPrecisionWallAudit(
        sampled_epsilons=epsilons,
        sampled_guard_margins=margins,
        all_samples_are_verified_witnesses=all_witnesses,
        guard_margin_tracks_epsilon_exactly=exact_tracking,
        witness_guard_infimum=0.0,
        positive_formula_only_guard_lower_bound_exists=False,
        semantic_contract_implies_polynomial_event_precision=False,
        semantic_contract_implies_polynomial_witness_dwell_time=False,
        dynamic_transversality_theorem_required=True,
        status=(
            "HYBRID_WITNESS_EVENT_PRECISION_WALL_ESTABLISHED__"
            "DYNAMIC_MARGIN_AND_DWELL_THEOREM_REQUIRED"
            if established
            else "HYBRID_WITNESS_EVENT_PRECISION_WALL_NOT_ESTABLISHED"
        ),
    )
