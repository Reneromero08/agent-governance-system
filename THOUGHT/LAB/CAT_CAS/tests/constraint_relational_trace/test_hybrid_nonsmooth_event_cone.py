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
from constraint_relational_trace_v1.hybrid_nonsmooth_event_cone import (  # noqa: E402
    audit_odd_parity_nonsmooth_witness_cone,
)


def test_odd_parity_has_exact_nonsmooth_positive_witness_cone() -> None:
    audit = audit_odd_parity_nonsmooth_witness_cone()

    assert audit.boundary_phase_cosine == (0.0, 0.0, 0.0)
    assert audit.witness_direction == (1.0, -1.0, -1.0)
    assert audit.boundary_active_clause_count == 4
    assert audit.boundary_maximum_active_literal_count == 3
    assert audit.boundary_active_set_gap == 0.0
    assert audit.boundary_guard_directional_derivative == 1.0
    assert audit.boundary_classification == (
        "NONSMOOTH_DIRECTIONAL_WITNESS_ENTRY"
    )
    assert audit.sampled_guard_margins == audit.sampled_epsilons
    assert all(value == 1.0 for value in audit.sampled_directional_derivatives)
    assert audit.every_sample_is_verified_witness
    assert audit.every_sample_has_zero_active_set_gap
    assert audit.positive_directional_entry_without_active_set_gap
    assert audit.simple_event_gap_is_sufficient_not_necessary
    assert not audit.uniform_nonsmooth_neighborhood_lower_bound_established
    assert audit.status == (
        "NONSMOOTH_WITNESS_CONE_WITH_POSITIVE_DIRECTIONAL_ENTRY_ESTABLISHED__"
        "ACTIVE_SET_GAP_NOT_NECESSARY"
    )


def test_nonsmooth_cone_accepts_any_positive_finite_epsilon_sequence() -> None:
    audit = audit_odd_parity_nonsmooth_witness_cone((0.3, 0.03, 0.003))
    assert audit.sampled_guard_margins == (0.3, 0.03, 0.003)
    assert audit.every_sample_is_verified_witness


@pytest.mark.parametrize("epsilons", ((), (0.5, 0.0), (float("nan"),)))
def test_nonsmooth_cone_rejects_invalid_epsilon_sequences(
    epsilons: tuple[float, ...],
) -> None:
    with pytest.raises(ConstraintHoloError):
        audit_odd_parity_nonsmooth_witness_cone(epsilons)
