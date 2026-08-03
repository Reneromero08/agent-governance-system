from __future__ import annotations

from math import nan
from pathlib import Path
import sys

import pytest

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ConstraintHoloError,
)
from constraint_relational_trace_v1.hybrid_witness_recorder_contract import (  # noqa: E402
    audit_hybrid_witness_recorder_contract,
    evaluate_hybrid_witness_guard,
)
from constraint_relational_trace_v1.odd_parity_fixed_deadline_counterexample import (  # noqa: E402
    odd_parity_three_variable_holo,
)


def test_hybrid_guard_matches_public_verifier_on_all_odd_parity_orthants() -> None:
    audit = audit_hybrid_witness_recorder_contract()

    assert audit.sign_assignments_audited == 8
    assert audit.witness_assignments == 4
    assert audit.nonwitness_assignments == 4
    assert audit.false_positive_count == 0
    assert audit.false_negative_count == 0
    assert audit.guard_semantics_exact_off_boundary
    assert audit.boundary_states_fail_closed


def test_hybrid_guard_detects_interior_witness_missed_by_exact_product_zero() -> None:
    holo = odd_parity_three_variable_holo()
    evaluation = evaluate_hybrid_witness_guard(holo, (0.1, -0.1, -0.1))
    audit = audit_hybrid_witness_recorder_contract(interior_magnitude=0.1)

    assert evaluation.unambiguous_sign_assignment
    assert evaluation.threshold_assignment == (
        ("x1", True),
        ("x2", False),
        ("x3", False),
    )
    assert evaluation.public_verifier_accepts
    assert evaluation.event_enabled
    assert evaluation.guard_margin == pytest.approx(0.1)
    assert audit.current_exact_product_channels_strictly_positive_at_interior_witness


def test_hybrid_guard_rejects_nonwitness_and_fails_closed_on_boundary() -> None:
    holo = odd_parity_three_variable_holo()
    nonwitness = evaluate_hybrid_witness_guard(holo, (-0.1, -0.1, -0.1))
    boundary = evaluate_hybrid_witness_guard(holo, (0.0, -0.1, -0.1))

    assert nonwitness.unambiguous_sign_assignment
    assert not nonwitness.public_verifier_accepts
    assert not nonwitness.event_enabled
    assert nonwitness.guard_margin < 0.0

    assert not boundary.unambiguous_sign_assignment
    assert boundary.threshold_assignment is None
    assert boundary.public_verifier_accepts is None
    assert not boundary.event_enabled
    assert not boundary.guard_matches_public_verifier


def test_semantic_contract_does_not_claim_dynamic_resource_closure() -> None:
    audit = audit_hybrid_witness_recorder_contract()

    assert audit.public_answer_blind_guard
    assert audit.terminal_agnostic_event
    assert audit.recorder_discrete_lanes == audit.public_variables + 1
    assert audit.recorder_lane_count_polynomial
    assert audit.hybrid_or_discontinuous_boundary_required
    assert not audit.polynomial_event_precision_established
    assert not audit.polynomial_witness_dwell_time_established
    assert not audit.reversible_event_history_restoration_established
    assert not audit.deterministic_polynomial_simulation_established
    assert audit.status == (
        "HYBRID_THRESHOLD_WITNESS_RECORDER_SEMANTIC_CONTRACT_ESTABLISHED__"
        "DYNAMIC_RESOURCE_CLOSURE_NOT_ESTABLISHED"
    )


@pytest.mark.parametrize(
    "cosines",
    (
        (0.1, -0.1),
        (0.1, -0.1, nan),
        (1.1, -0.1, -0.1),
    ),
)
def test_hybrid_guard_rejects_invalid_phase_input(
    cosines: tuple[float, ...],
) -> None:
    with pytest.raises(ConstraintHoloError):
        evaluate_hybrid_witness_guard(odd_parity_three_variable_holo(), cosines)
