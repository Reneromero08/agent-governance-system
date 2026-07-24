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
from constraint_relational_trace_v1.fermionic_interaction_audit import (  # noqa: E402
    audit_fermionic_interactions,
    clause_hamiltonian_polynomial,
    clause_violation_polynomial,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def test_three_distinct_positive_literals_create_cubic_interaction() -> None:
    relation = clause(Literal("a"), Literal("b"), Literal("c"))
    polynomial = clause_violation_polynomial(relation)
    assert polynomial[frozenset()] == 1
    assert polynomial[frozenset(("a", "b", "c"))] == -1
    assert max(map(len, polynomial)) == 3


def test_repeated_literal_reduces_under_boolean_idempotence() -> None:
    relation = clause(Literal("x"), Literal("x"), Literal("x"))
    polynomial = clause_violation_polynomial(relation)
    assert polynomial == {
        frozenset(): 1,
        frozenset(("x",)): -1,
    }


def test_genuine_three_clause_breaks_quadratic_gaussian_closure() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (clause(Literal("a"), Literal("b"), Literal("c")),),
    )
    audit = audit_fermionic_interactions(holo)
    assert audit.local_max_degree == 3
    assert audit.combined_max_degree == 3
    assert not audit.quadratic_gaussian_closed
    assert audit.determinant_method_status == "GENERIC_GAUSSIAN_DETERMINANT_CLOSURE_BROKEN_BY_INTERACTIONS"


def test_padded_two_variable_relation_remains_quadratic_or_lower() -> None:
    holo = ConstraintHolo.build(
        ("a", "b"),
        (clause(Literal("a"), Literal("b"), Literal("b")),),
    )
    audit = audit_fermionic_interactions(holo)
    assert audit.local_max_degree == 2
    assert audit.combined_max_degree == 2
    assert audit.quadratic_gaussian_closed


def test_aggregate_cubic_cancellation_does_not_remove_local_interactions() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c"),
        (
            clause(Literal("a"), Literal("b"), Literal("c")),
            clause(Literal("a", False), Literal("b", False), Literal("c", False)),
        ),
    )
    polynomial = clause_hamiltonian_polynomial(holo)
    assert polynomial[frozenset()] == 1
    assert frozenset(("a", "b", "c")) not in polynomial
    assert max(map(len, polynomial)) == 2
    audit = audit_fermionic_interactions(holo)
    assert audit.local_max_degree == 3
    assert audit.combined_max_degree == 2
    assert not audit.quadratic_gaussian_closed
    assert audit.auxiliary_field_status == "EXACT_AUXILIARY_FIELD_SUM_OR_NON_GAUSSIAN_NATIVE_OPERATOR_REQUIRED"
