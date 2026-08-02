from __future__ import annotations

from pathlib import Path
import math
import sys

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.memory_logit_drift_audit import (  # noqa: E402
    audit_memory_logit_drift,
    memory_logit_derivative_identity,
)


def test_memory_logit_clock_is_independent_of_clause_violation() -> None:
    observed = tuple(
        memory_logit_derivative_identity(violation)[2]
        for violation in (0.0, 0.05, 0.25, 1.0, 4.0)
    )

    assert all(math.isclose(value, 0.2) for value in observed)


def test_memory_logit_clock_excludes_bounded_interior_recurrence() -> None:
    audit = audit_memory_logit_drift()

    assert audit.short_threshold == 0.25
    assert audit.long_threshold == 0.05
    assert math.isclose(audit.normalized_drift_rate, 0.2)
    assert audit.violation_independent
    assert audit.interior_periodic_orbits_excluded
    assert audit.bounded_interior_recurrence_excluded
    assert audit.omega_limit_boundary_stratum_required
    assert audit.status == (
        "MEMORY_LOGIT_CLOCK_ESTABLISHED__INTERIOR_RECURRENCE_EXCLUDED"
    )
