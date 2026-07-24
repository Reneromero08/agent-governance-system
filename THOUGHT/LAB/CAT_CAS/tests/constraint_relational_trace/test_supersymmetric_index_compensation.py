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

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.supersymmetric_index_compensation import (  # noqa: E402
    audit_supersymmetric_index_compensation,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def test_finite_witten_index_cancels_satisfying_zero_modes() -> None:
    holo = ConstraintHolo.build(
        ("x", "y"),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(Literal("y"), Literal("y"), Literal("y")),
        ),
    )
    audit = audit_supersymmetric_index_compensation(holo)

    assert audit.satisfying_assignments == 1
    assert audit.bosonic_zero_modes == 1
    assert audit.fermionic_zero_modes == 1
    assert audit.finite_witten_index == 0
    assert not audit.formula_dependent_index_detected


def test_unsat_and_sat_have_same_square_pairing_index() -> None:
    sat = ConstraintHolo.build(
        ("x",),
        (clause(Literal("x"), Literal("x"), Literal("x")),),
    )
    unsat = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(
                Literal("x", False),
                Literal("x", False),
                Literal("x", False),
            ),
        ),
    )

    sat_audit = audit_supersymmetric_index_compensation(sat)
    unsat_audit = audit_supersymmetric_index_compensation(unsat)

    assert sat_audit.finite_witten_index == 0
    assert unsat_audit.finite_witten_index == 0
    assert sat_audit.satisfying_assignments == 1
    assert unsat_audit.satisfying_assignments == 0
