from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError


@dataclass(frozen=True)
class ThresholdWitnessLatchWallAudit:
    interior_witness_epsilon: float
    threshold_assignment: tuple[tuple[str, bool], ...]
    odd_parity_clause_violations: tuple[float, float, float, float]
    all_threshold_clauses_satisfied: bool
    every_exact_product_channel_strictly_positive: bool
    exact_product_zero_matches_only_boolean_literal_truth: bool
    nonzero_polynomial_exact_write_enable_possible: bool
    nonzero_real_analytic_exact_write_enable_possible: bool
    naive_polynomial_threshold_latch_status: str
    allowed_escape_routes: tuple[str, ...]
    status: str
    claim_ceiling: str = CLAIM_CEILING


def _odd_parity_clause_signs() -> tuple[tuple[float, float, float], ...]:
    return (
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
    )


def _clause_violation(
    signs: tuple[float, float, float],
    cosines: tuple[float, float, float],
    truth_gain: float,
) -> float:
    defects = tuple(
        1.0 - sign * cosine
        for sign, cosine in zip(signs, cosines, strict=True)
    )
    return truth_gain * defects[0] * defects[1] * defects[2] / 8.0


def _clause_threshold_satisfied(
    signs: tuple[float, float, float],
    cosines: tuple[float, float, float],
) -> bool:
    return any(
        sign * cosine > 0.0
        for sign, cosine in zip(signs, cosines, strict=True)
    )


def audit_threshold_witness_latch_wall(
    *,
    epsilon: float = 0.1,
    truth_gain: float = 4.0,
) -> ThresholdWitnessLatchWallAudit:
    """Audit the mismatch between threshold witnesses and polynomial zero channels.

    A false-positive-free autonomous write-enable that is exactly inactive on every
    non-witness sign orthant and active on a witness orthant cannot be a nonzero
    polynomial. Each non-witness orthant is a nonempty open set; a polynomial vanishing
    there is the zero polynomial. The same identity principle applies to real-analytic
    write-enables on a connected phase chart.

    This theorem is about exact sign-gated activation. It does not rule out hybrid event
    systems, discontinuous boundaries, topological crossing records, or a different
    smooth mechanism whose correctness does not require compact support on witness
    orthants.
    """

    if not 0.0 < epsilon < 1.0:
        raise ConstraintHoloError("latch-wall epsilon must lie in (0,1)")
    if truth_gain <= 0.0:
        raise ConstraintHoloError("latch-wall truth gain must be positive")

    # (True, False, False) is an odd-parity witness, represented strictly inside its
    # sign orthant rather than at a Boolean phase corner.
    cosines = (epsilon, -epsilon, -epsilon)
    signs = _odd_parity_clause_signs()
    satisfied = tuple(
        _clause_threshold_satisfied(clause_signs, cosines)
        for clause_signs in signs
    )
    violations = tuple(
        _clause_violation(clause_signs, cosines, truth_gain)
        for clause_signs in signs
    )
    all_satisfied = all(satisfied)
    every_positive = all(value > 0.0 for value in violations)

    return ThresholdWitnessLatchWallAudit(
        interior_witness_epsilon=epsilon,
        threshold_assignment=(("x1", True), ("x2", False), ("x3", False)),
        odd_parity_clause_violations=violations,  # type: ignore[arg-type]
        all_threshold_clauses_satisfied=all_satisfied,
        every_exact_product_channel_strictly_positive=every_positive,
        exact_product_zero_matches_only_boolean_literal_truth=True,
        nonzero_polynomial_exact_write_enable_possible=False,
        nonzero_real_analytic_exact_write_enable_possible=False,
        naive_polynomial_threshold_latch_status=(
            "IMPOSSIBLE_WITH_EXACT_ORTHANT_SUPPORTED_POLYNOMIAL_WRITE_ENABLE"
        ),
        allowed_escape_routes=(
            "DISCONTINUOUS_OR_HYBRID_THRESHOLD_EVENT_WITH_EXPLICIT_RESOURCE_MODEL",
            "TOPOLOGICAL_CROSSING_RECORD_NOT_REQUIRING_OPEN_SET_SUPPORT",
            "BOOLEAN_SECTION_LATCH_AFTER_PROVED_POLYNOMIAL_BOOLEANIZATION",
            "DIFFERENT_SMOOTH_PERSISTENCE_MECHANISM_WITH_INDEPENDENT_FALSE_POSITIVE_PROOF",
        ),
        status=(
            "THRESHOLD_WITNESS_LATCH_WALL_ESTABLISHED__"
            "CURRENT_EXACT_PRODUCT_ZERO_CHANNEL_DOES_NOT_SEE_INTERIOR_SIGN_WITNESS"
            if all_satisfied and every_positive
            else "THRESHOLD_WITNESS_LATCH_WALL_AUDIT_NOT_ESTABLISHED"
        ),
    )
