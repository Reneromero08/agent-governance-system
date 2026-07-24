from __future__ import annotations

from pathlib import Path
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
from constraint_relational_trace_v1.oracle_determinant_compensation import (  # noqa: E402
    audit_oracle_determinant_compensation,
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


def unsat_two_variables() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x1", "x2"),
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(Literal("x1", False), Literal("x1", False), Literal("x1", False)),
        ),
    )


def test_full_reversible_oracle_winding_is_formula_independent_at_equal_dimension() -> None:
    sat_result = audit_oracle_determinant_compensation(unique_solution(), ancillary_qubits=2)
    unsat_result = audit_oracle_determinant_compensation(
        unsat_two_variables(), ancillary_qubits=2
    )
    assert sat_result.total_dimension == unsat_result.total_dimension == 16
    assert sat_result.full_winding == unsat_result.full_winding == 8
    assert sat_result.full_winding == sat_result.total_dimension // 2
    assert unsat_result.full_winding == unsat_result.total_dimension // 2
    assert sat_result.clean_winding == 1
    assert unsat_result.clean_winding == 0
    assert sat_result.full_winding_formula_independent
    assert unsat_result.full_winding_formula_independent


def test_clean_sector_keeps_sat_index_and_dirty_sector_compensates() -> None:
    result = audit_oracle_determinant_compensation(unique_solution(), ancillary_qubits=1)
    assert result.clean_winding == result.satisfying_rank == 1
    assert result.full_winding == 4
    assert result.complementary_winding == 3
    assert result.clean_winding + result.complementary_winding == result.full_winding
    assert result.compensation_identity_verified
    assert result.restricted_sensor_required


def test_unsat_clean_sector_has_zero_index_but_full_oracle_still_winds() -> None:
    result = audit_oracle_determinant_compensation(
        unsat_two_variables(), ancillary_qubits=1
    )
    assert result.clean_winding == 0
    assert result.full_winding == 4
    assert result.complementary_winding == 4
    assert result.restricted_sensor_required
