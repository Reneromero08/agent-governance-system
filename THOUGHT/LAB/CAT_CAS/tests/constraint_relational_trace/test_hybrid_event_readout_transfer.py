from __future__ import annotations

from math import isclose
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
from constraint_relational_trace_v1.hybrid_event_readout_transfer import (  # noqa: E402
    audit_hybrid_guard_one_lipschitz,
    conditional_hybrid_event_readout_transfer,
)
from constraint_relational_trace_v1.odd_parity_fixed_deadline_counterexample import (  # noqa: E402
    odd_parity_three_variable_holo,
)


def test_hybrid_guard_is_one_lipschitz_on_odd_parity_control() -> None:
    audit = audit_hybrid_guard_one_lipschitz(
        odd_parity_three_variable_holo(),
        (0.3, -0.3, -0.3),
        (0.2, -0.25, -0.35),
    )

    assert isclose(audit.left_guard_margin, 0.3)
    assert isclose(audit.phase_linf_distance, 0.1)
    assert audit.guard_difference <= audit.phase_linf_distance + 1.0e-12
    assert audit.one_lipschitz_inequality_holds
    assert audit.status == "HYBRID_GUARD_ONE_LIPSCHITZ_AUDIT_PASS"


def test_guard_lipschitz_bound_is_tight_on_witness_ray() -> None:
    audit = audit_hybrid_guard_one_lipschitz(
        odd_parity_three_variable_holo(),
        (0.4, -0.4, -0.4),
        (0.1, -0.1, -0.1),
    )

    assert isclose(audit.phase_linf_distance, 0.3)
    assert isclose(audit.guard_difference, 0.3)
    assert audit.one_lipschitz_inequality_holds


def test_inverse_polynomial_plateau_transfers_to_polynomial_sampling_schedule() -> None:
    result = conditional_hybrid_event_readout_transfer(
        deadline_upper_bound=10.0,
        witness_plateau_length_lower_bound=0.25,
        witness_guard_margin_lower_bound=0.125,
    )

    assert isclose(result.sampling_step_upper_bound, 0.125)
    assert isclose(result.phase_error_upper_bound, 0.0625)
    assert result.sample_count_upper_bound == 81
    assert result.guard_precision_bits_upper_bound == 4
    assert isclose(result.detected_guard_margin_lower_bound, 0.0625)
    assert result.polynomial_sampling_if_inputs_inverse_polynomial
    assert result.guard_readout_stability_established
    assert not result.state_simulation_cost_established
    assert not result.event_history_restoration_established
    assert result.status == (
        "CONDITIONAL_HYBRID_EVENT_SAMPLING_AND_GUARD_READOUT_TRANSFER_ESTABLISHED__"
        "STATE_SIMULATION_COST_NOT_ESTABLISHED"
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"deadline_upper_bound": 0.0, "witness_plateau_length_lower_bound": 1.0, "witness_guard_margin_lower_bound": 1.0},
        {"deadline_upper_bound": 1.0, "witness_plateau_length_lower_bound": 0.0, "witness_guard_margin_lower_bound": 1.0},
        {"deadline_upper_bound": 1.0, "witness_plateau_length_lower_bound": 1.0, "witness_guard_margin_lower_bound": 0.0},
    ),
)
def test_readout_transfer_rejects_nonpositive_inputs(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ConstraintHoloError):
        conditional_hybrid_event_readout_transfer(**kwargs)


def test_lipschitz_audit_rejects_bad_phase_vectors() -> None:
    holo = odd_parity_three_variable_holo()
    with pytest.raises(ConstraintHoloError):
        audit_hybrid_guard_one_lipschitz(holo, (0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ConstraintHoloError):
        audit_hybrid_guard_one_lipschitz(
            holo,
            (0.0, 0.0, float("nan")),
            (0.0, 0.0, 0.0),
        )
