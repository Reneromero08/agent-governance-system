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
from constraint_relational_trace_v1.threshold_witness_latch_wall import (  # noqa: E402
    audit_threshold_witness_latch_wall,
)


def test_interior_odd_parity_witness_is_invisible_to_exact_product_zero_channel() -> None:
    audit = audit_threshold_witness_latch_wall()

    assert audit.threshold_assignment == (
        ("x1", True),
        ("x2", False),
        ("x3", False),
    )
    assert audit.all_threshold_clauses_satisfied
    assert audit.every_exact_product_channel_strictly_positive
    assert audit.exact_product_zero_requires_literal_phase_boundary
    assert all(value > 0.0 for value in audit.odd_parity_clause_violations)
    assert isclose(audit.odd_parity_clause_violations[0], 0.3645)
    assert all(
        isclose(value, 0.5445)
        for value in audit.odd_parity_clause_violations[1:]
    )
    assert audit.status == (
        "THRESHOLD_WITNESS_LATCH_WALL_ESTABLISHED__"
        "CURRENT_EXACT_PRODUCT_ZERO_CHANNEL_DOES_NOT_SEE_INTERIOR_SIGN_WITNESS"
    )


def test_exact_orthant_supported_polynomial_write_enable_is_rejected() -> None:
    audit = audit_threshold_witness_latch_wall()

    assert not audit.nonzero_polynomial_exact_write_enable_possible
    assert not audit.nonzero_real_analytic_exact_write_enable_possible
    assert audit.naive_polynomial_threshold_latch_status == (
        "IMPOSSIBLE_WITH_EXACT_ORTHANT_SUPPORTED_POLYNOMIAL_WRITE_ENABLE"
    )
    assert "TOPOLOGICAL_CROSSING_RECORD_NOT_REQUIRING_OPEN_SET_SUPPORT" in (
        audit.allowed_escape_routes
    )
    assert "LITERAL_PHASE_BOUNDARY_LATCH_AFTER_PROVED_POLYNOMIAL_BOOLEANIZATION" in (
        audit.allowed_escape_routes
    )


@pytest.mark.parametrize("epsilon", (0.0, 1.0, -0.1, 1.1))
def test_latch_wall_rejects_noninterior_control(epsilon: float) -> None:
    with pytest.raises(ConstraintHoloError, match="epsilon"):
        audit_threshold_witness_latch_wall(epsilon=epsilon)


def test_latch_wall_rejects_nonpositive_truth_gain() -> None:
    with pytest.raises(ConstraintHoloError, match="truth gain"):
        audit_threshold_witness_latch_wall(truth_gain=0.0)
