from __future__ import annotations

from math import exp, isclose
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
from constraint_relational_trace_v1.public_seed_memory_reduction_audit import (  # noqa: E402
    audit_public_seed_memory_reduction,
    public_seed_normalized_long_memory_from_short,
)
from constraint_relational_trace_v1.self_organizing_clause_flow import (  # noqa: E402
    SelfOrganizingFlowParameters,
)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def test_public_seed_memory_reduction_has_five_forward_strata() -> None:
    audit = audit_public_seed_memory_reduction()

    assert audit.alpha_over_beta == 0.25
    assert audit.public_clock_exponential_rate == 1.0
    assert audit.public_short_initial_odds == 1.0
    assert len(audit.excluded_stationary_strata) == 3
    assert len(audit.forward_compatible_stationary_strata) == 5
    assert "SHORT_ONE__LONG_ZERO__C_ARBITRARY" in audit.excluded_stationary_strata
    assert (
        "SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP"
        in audit.forward_compatible_stationary_strata
    )
    assert audit.finite_time_boundary_contact_excluded
    assert audit.long_memory_reconstructible_from_short_and_time
    assert audit.status == (
        "PUBLIC_SEED_MEMORY_REDUCTION_ESTABLISHED__FIVE_FORWARD_STRATA_REMAIN"
    )


def test_public_seed_odds_relation_matches_direct_constant_violation_solution() -> None:
    parameters = SelfOrganizingFlowParameters()
    cap = 40_000.0
    elapsed = 1.75
    violation = 0.4

    short_logit = parameters.beta * (violation - parameters.gamma) * elapsed
    long_initial_logit = -__import__("math").log(cap - 1.0)
    long_logit = long_initial_logit + parameters.alpha * (
        violation - parameters.delta
    ) * elapsed

    short_memory = _sigmoid(short_logit)
    direct_normalized_long = _sigmoid(long_logit)
    reconstructed = public_seed_normalized_long_memory_from_short(
        short_memory,
        elapsed,
        cap,
        parameters,
    )

    assert isclose(reconstructed, direct_normalized_long, rel_tol=1.0e-12, abs_tol=1.0e-14)


def test_public_seed_reconstruction_recovers_initial_long_memory() -> None:
    cap = 20_000.0
    normalized_long = public_seed_normalized_long_memory_from_short(0.5, 0.0, cap)
    assert isclose(normalized_long, 1.0 / cap, rel_tol=1.0e-12, abs_tol=1.0e-15)


@pytest.mark.parametrize(
    ("short_memory", "elapsed_time", "cap"),
    (
        (0.0, 0.0, 10.0),
        (1.0, 0.0, 10.0),
        (0.5, -1.0, 10.0),
        (0.5, 0.0, 1.0),
    ),
)
def test_public_seed_memory_reduction_rejects_invalid_chart_inputs(
    short_memory: float,
    elapsed_time: float,
    cap: float,
) -> None:
    with pytest.raises(ConstraintHoloError):
        public_seed_normalized_long_memory_from_short(
            short_memory,
            elapsed_time,
            cap,
        )
