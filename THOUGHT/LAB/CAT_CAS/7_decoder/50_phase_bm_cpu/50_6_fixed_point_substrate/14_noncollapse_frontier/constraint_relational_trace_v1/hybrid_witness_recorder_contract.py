from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .odd_parity_fixed_deadline_counterexample import odd_parity_three_variable_holo
from .threshold_witness_latch_wall import audit_threshold_witness_latch_wall


@dataclass(frozen=True)
class HybridWitnessGuardEvaluation:
    phase_cosine: tuple[float, ...]
    variable_boundary_margin: float
    clause_margins: tuple[float, ...]
    guard_margin: float
    unambiguous_sign_assignment: bool
    threshold_assignment: tuple[tuple[str, bool], ...] | None
    public_verifier_accepts: bool | None
    event_enabled: bool
    guard_matches_public_verifier: bool


@dataclass(frozen=True)
class HybridWitnessRecorderContractAudit:
    public_variables: int
    public_clauses: int
    sign_assignments_audited: int
    witness_assignments: int
    nonwitness_assignments: int
    false_positive_count: int
    false_negative_count: int
    interior_witness_guard_margin: float
    current_exact_product_channels_strictly_positive_at_interior_witness: bool
    guard_semantics_exact_off_boundary: bool
    boundary_states_fail_closed: bool
    public_answer_blind_guard: bool
    terminal_agnostic_event: bool
    recorder_discrete_lanes: int
    recorder_lane_count_polynomial: bool
    hybrid_or_discontinuous_boundary_required: bool
    polynomial_event_precision_established: bool
    polynomial_witness_dwell_time_established: bool
    reversible_event_history_restoration_established: bool
    deterministic_polynomial_simulation_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def evaluate_hybrid_witness_guard(
    holo: ConstraintHolo,
    phase_cosine: tuple[float, ...],
) -> HybridWitnessGuardEvaluation:
    """Evaluate the exact semialgebraic threshold-witness guard.

    For every clause, the clause margin is the largest signed literal cosine. The
    formula guard is the smallest clause margin:

        G_F(c) = min_j max_{literal r in clause j} q_r c_i.

    Away from every variable threshold ``c_i = 0``, ``G_F(c) > 0`` exactly when the
    sign assignment satisfies the public formula. The min/max and strict comparison are
    an explicit hybrid boundary; they are not part of the smooth polynomial carrier.

    States with any zero cosine fail closed because the complete threshold assignment is
    ambiguous, even when another literal would already satisfy every clause.
    """

    if len(phase_cosine) != len(holo.variables):
        raise ConstraintHoloError("hybrid witness guard phase dimension mismatch")
    if not all(isfinite(value) and -1.0 <= value <= 1.0 for value in phase_cosine):
        raise ConstraintHoloError("hybrid witness guard requires finite phase cosines")

    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    clause_margins = tuple(
        max(
            (1.0 if literal.positive else -1.0)
            * phase_cosine[variable_index[literal.variable]]
            for literal in clause.literals
        )
        for clause in holo.clauses
    )
    guard_margin = min(clause_margins, default=1.0)
    boundary_margin = min((abs(value) for value in phase_cosine), default=1.0)
    unambiguous = boundary_margin > 0.0

    assignment: tuple[tuple[str, bool], ...] | None = None
    verified: bool | None = None
    if unambiguous:
        assignment_map = {
            variable: phase_cosine[index] > 0.0
            for index, variable in enumerate(holo.variables)
        }
        assignment = tuple(sorted(assignment_map.items()))
        verified = holo.accepts(assignment_map)

    enabled = unambiguous and guard_margin > 0.0
    semantic_match = unambiguous and enabled == bool(verified)
    return HybridWitnessGuardEvaluation(
        phase_cosine=phase_cosine,
        variable_boundary_margin=boundary_margin,
        clause_margins=clause_margins,
        guard_margin=guard_margin,
        unambiguous_sign_assignment=unambiguous,
        threshold_assignment=assignment,
        public_verifier_accepts=verified,
        event_enabled=enabled,
        guard_matches_public_verifier=semantic_match,
    )


def audit_hybrid_witness_recorder_contract(
    *,
    interior_magnitude: float = 0.1,
) -> HybridWitnessRecorderContractAudit:
    """Audit semantic exactness of the smallest explicit hybrid recorder boundary.

    The proposed recorder has one latched-valid bit and ``n`` stored assignment bits.
    This is only a semantic and coordinate-count contract. It does not establish that a
    witness crossing has inverse-polynomial margin or dwell time, that event isolation
    can be simulated with polynomial precision, or that the event history can be
    restored catalytically.
    """

    if not isfinite(interior_magnitude) or not 0.0 < interior_magnitude < 1.0:
        raise ConstraintHoloError("hybrid recorder interior magnitude must lie in (0,1)")

    holo = odd_parity_three_variable_holo()
    evaluations = tuple(
        evaluate_hybrid_witness_guard(
            holo,
            tuple(interior_magnitude if bit else -interior_magnitude for bit in bits),
        )
        for bits in product((False, True), repeat=len(holo.variables))
    )
    witnesses = sum(bool(item.public_verifier_accepts) for item in evaluations)
    false_positives = sum(
        item.event_enabled and not bool(item.public_verifier_accepts)
        for item in evaluations
    )
    false_negatives = sum(
        bool(item.public_verifier_accepts) and not item.event_enabled
        for item in evaluations
    )
    exact = all(item.guard_matches_public_verifier for item in evaluations)

    witness_evaluation = evaluate_hybrid_witness_guard(
        holo,
        (interior_magnitude, -interior_magnitude, -interior_magnitude),
    )
    boundary_evaluation = evaluate_hybrid_witness_guard(
        holo,
        (0.0, -interior_magnitude, -interior_magnitude),
    )
    latch_wall = audit_threshold_witness_latch_wall(
        epsilon=interior_magnitude,
    )

    established = (
        exact
        and witnesses == 4
        and false_positives == 0
        and false_negatives == 0
        and witness_evaluation.event_enabled
        and not boundary_evaluation.event_enabled
        and not boundary_evaluation.unambiguous_sign_assignment
    )

    return HybridWitnessRecorderContractAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        sign_assignments_audited=len(evaluations),
        witness_assignments=witnesses,
        nonwitness_assignments=len(evaluations) - witnesses,
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
        interior_witness_guard_margin=witness_evaluation.guard_margin,
        current_exact_product_channels_strictly_positive_at_interior_witness=(
            latch_wall.every_exact_product_channel_strictly_positive
        ),
        guard_semantics_exact_off_boundary=exact,
        boundary_states_fail_closed=(
            not boundary_evaluation.event_enabled
            and not boundary_evaluation.unambiguous_sign_assignment
        ),
        public_answer_blind_guard=True,
        terminal_agnostic_event=True,
        recorder_discrete_lanes=len(holo.variables) + 1,
        recorder_lane_count_polynomial=True,
        hybrid_or_discontinuous_boundary_required=True,
        polynomial_event_precision_established=False,
        polynomial_witness_dwell_time_established=False,
        reversible_event_history_restoration_established=False,
        deterministic_polynomial_simulation_established=False,
        status=(
            "HYBRID_THRESHOLD_WITNESS_RECORDER_SEMANTIC_CONTRACT_ESTABLISHED__"
            "DYNAMIC_RESOURCE_CLOSURE_NOT_ESTABLISHED"
            if established
            else "HYBRID_THRESHOLD_WITNESS_RECORDER_CONTRACT_NOT_ESTABLISHED"
        ),
    )
