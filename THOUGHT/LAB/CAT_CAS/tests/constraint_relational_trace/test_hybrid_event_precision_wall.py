from __future__ import annotations

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
from constraint_relational_trace_v1.hybrid_event_precision_wall import (  # noqa: E402
    audit_hybrid_event_precision_wall,
)


def test_fixed_formula_has_verified_witnesses_with_arbitrarily_small_guard_samples() -> None:
    audit = audit_hybrid_event_precision_wall(
        epsilons=(0.5, 0.1, 0.01, 0.001),
    )

    assert audit.all_samples_are_verified_witnesses
    assert audit.guard_margin_tracks_epsilon_exactly
    assert audit.sampled_guard_margins == pytest.approx(audit.sampled_epsilons)
    assert audit.witness_guard_infimum == 0.0
    assert not audit.positive_formula_only_guard_lower_bound_exists


def test_semantic_guard_does_not_imply_polynomial_event_resources() -> None:
    audit = audit_hybrid_event_precision_wall()

    assert not audit.semantic_contract_implies_polynomial_event_precision
    assert not audit.semantic_contract_implies_polynomial_witness_dwell_time
    assert audit.dynamic_transversality_theorem_required
    assert audit.status == (
        "HYBRID_WITNESS_EVENT_PRECISION_WALL_ESTABLISHED__"
        "DYNAMIC_MARGIN_AND_DWELL_THEOREM_REQUIRED"
    )


@pytest.mark.parametrize(
    "epsilons",
    (
        (),
        (0.0,),
        (1.0,),
        (-0.1,),
    ),
)
def test_precision_wall_rejects_invalid_epsilon_inventory(
    epsilons: tuple[float, ...],
) -> None:
    with pytest.raises(ConstraintHoloError):
        audit_hybrid_event_precision_wall(epsilons=epsilons)
