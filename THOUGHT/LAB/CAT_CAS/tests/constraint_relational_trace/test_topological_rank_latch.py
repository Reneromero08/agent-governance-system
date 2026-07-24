from __future__ import annotations

from pathlib import Path
import math
import sys

CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.topological_rank_latch import (  # noqa: E402
    audit_topological_rank_latch,
    phase_oracle_value,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def unique_solution() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x1", "x2"),
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(Literal("x2", False), Literal("x2", False), Literal("x2", False)),
        ),
    )


def test_unique_solution_produces_one_topological_winding() -> None:
    result = audit_topological_rank_latch(unique_solution())
    assert result.satisfying_rank == 1
    assert result.determinant_winding == 1
    assert result.presence_index == 1
    assert result.restoration_verified
    assert result.basis_dimension == 4
    assert math.isclose(result.minimum_unique_trace_gap, 0.5)
    assert "POLYNOMIAL_NATIVE_DETERMINANT_LINE_SENSOR_NOT_ESTABLISHED" in result.native_readout_status


def test_unsat_has_zero_winding_and_valid_restoration() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(Literal("x", False), Literal("x", False), Literal("x", False)),
        ),
    )
    result = audit_topological_rank_latch(holo)
    assert result.satisfying_rank == 0
    assert result.determinant_winding == 0
    assert result.presence_index == 0
    assert result.normalized_trace_at_pi == 1.0
    assert result.restoration_verified


def test_all_assignments_satisfying_wind_once_each() -> None:
    holo = ConstraintHolo.build(("a", "b", "c"), ())
    result = audit_topological_rank_latch(holo)
    assert result.satisfying_rank == 8
    assert result.determinant_winding == 8
    assert result.presence_index == 1
    assert result.normalized_trace_at_pi == -1.0
    assert result.materialized_carrier_coordinates == 8
    assert result.determinant_line_filled_modes == 8


def test_duplicate_clause_does_not_change_topological_index() -> None:
    holo = unique_solution()
    original = audit_topological_rank_latch(holo)
    duplicated = audit_topological_rank_latch(holo.with_duplicate_clause(0))
    assert original.determinant_winding == duplicated.determinant_winding == 1
    assert original.presence_index == duplicated.presence_index == 1


def test_phase_oracle_is_identity_on_violating_basis_state() -> None:
    holo = unique_solution()
    angle = 0.417
    assert phase_oracle_value(holo, {"x1": False, "x2": False}, angle) == 1.0 + 0.0j
    satisfying_value = phase_oracle_value(holo, {"x1": True, "x2": False}, angle)
    assert abs(satisfying_value - complex(math.cos(angle), math.sin(angle))) < 1e-12
