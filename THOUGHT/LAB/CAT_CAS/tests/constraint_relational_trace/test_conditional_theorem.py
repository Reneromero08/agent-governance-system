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

from constraint_relational_trace_v1.conditional_p_equals_np import (  # noqa: E402
    BoundaryDecision,
    extract_witness_by_boundary_self_reduction,
    reference_decision_boundary,
    restrict_public_relation,
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def satisfiable_holo() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x1", "x2", "x3"),
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(Literal("x2", False), Literal("x2", False), Literal("x2", False)),
            clause(Literal("x3"), Literal("x3", False), Literal("x1")),
        ),
    )


def test_restriction_preserves_exact_three_literal_relation_shape() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    restricted = restrict_public_relation(holo, {"a": False})
    assert not restricted.contradiction
    assert restricted.holo is not None
    assert restricted.holo.variables == ("b", "c")
    assert len(restricted.holo.clauses[0].literals) == 3
    assert reference_decision_boundary(restricted.holo) is BoundaryDecision.VALID_SAT


def test_boundary_self_reduction_renders_verified_witness() -> None:
    holo = satisfiable_holo()
    result = extract_witness_by_boundary_self_reduction(holo, reference_decision_boundary)
    assert result.valid
    assert result.satisfiable is True
    assert result.boundary_status is BoundaryDecision.VALID_SAT
    assert result.witness_verified
    assert result.witness is not None
    assert holo.accepts(result.witness)
    assert result.decision_calls <= 1 + 2 * len(holo.variables)
    assert result.conditional_theorem == "UNIFORM_POLYNOMIAL_CET_DECISION_IMPLIES_3SAT_IN_P"


def test_boundary_self_reduction_reports_unsat_without_witness() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(Literal("x", False), Literal("x", False), Literal("x", False)),
        ),
    )
    result = extract_witness_by_boundary_self_reduction(holo, reference_decision_boundary)
    assert result.valid
    assert result.satisfiable is False
    assert result.boundary_status is BoundaryDecision.VALID_UNSAT
    assert result.witness is None
    assert not result.witness_verified
    assert result.decision_calls == 1


def test_invalid_carrier_never_becomes_sat_or_unsat() -> None:
    def invalid_boundary(_: ConstraintHolo) -> BoundaryDecision:
        return BoundaryDecision.INVALID_CARRIER

    result = extract_witness_by_boundary_self_reduction(
        satisfiable_holo(), invalid_boundary
    )
    assert not result.valid
    assert result.satisfiable is None
    assert result.boundary_status is BoundaryDecision.INVALID_CARRIER
    assert result.witness is None
    assert not result.witness_verified
    assert result.decision_calls == 1


def test_unrecognized_truthy_boundary_value_fails_closed() -> None:
    def malformed_boundary(_: ConstraintHolo) -> BoundaryDecision:
        return "INVALID_CARRIER"  # type: ignore[return-value]

    result = extract_witness_by_boundary_self_reduction(
        satisfiable_holo(), malformed_boundary
    )
    assert not result.valid
    assert result.satisfiable is None
    assert result.boundary_status is BoundaryDecision.INVALID_CARRIER
    assert result.witness is None
