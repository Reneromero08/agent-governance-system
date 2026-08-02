from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.stationary_memory_strata_audit import (  # noqa: E402
    audit_stationary_memory_strata,
    stationary_memory_strata,
)


def test_stationary_memory_strata_exclude_both_interior_coordinates() -> None:
    audit = audit_stationary_memory_strata()

    assert audit.short_threshold == 0.25
    assert audit.long_threshold == 0.05
    assert audit.both_memory_coordinates_interior_excluded
    assert audit.total_stationary_stratum_types == 8
    assert len(audit.boundary_boundary_strata) == 4
    assert len(audit.short_interior_strata) == 2
    assert len(audit.long_interior_strata) == 2
    assert audit.status == (
        "STATIONARY_MEMORY_STRATA_CLASSIFIED__BOTH_INTERIOR_EXCLUDED"
    )


def test_stationary_memory_strata_are_complete_for_distinct_thresholds() -> None:
    strata = stationary_memory_strata()

    assert len(strata) == len(set(strata)) == 8
    assert all("SHORT_INTERIOR" not in label or "LONG_INTERIOR" not in label for label in strata)
