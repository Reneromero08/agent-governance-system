from __future__ import annotations

from pathlib import Path
import sys

import pytest

CAT_CAS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = (
    CAT_CAS_ROOT
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.catalytic_existential_trace import (  # noqa: E402
    CLAIM_CEILING,
    FactorizedProjectorCandidate,
    audit_reversible_dilation,
    reference_existential_trace,
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    ConstraintHoloError,
    Literal,
    audit_public_record,
)
from constraint_relational_trace_v1.parity_holonomy import (  # noqa: E402
    ParityConstraint,
    ParityInstance,
    Z2PhaseCarrier,
    calibrate_parity_holonomy,
    compile_z2_transport,
)
from constraint_relational_trace_v1.run_reference_campaign import (  # noqa: E402
    build_campaign_record,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def unique_solution_holo() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x1", "x2"),
        (
            clause(Literal("x1"), Literal("x1"), Literal("x1")),
            clause(Literal("x2", False), Literal("x2", False), Literal("x2", False)),
        ),
    )


def test_dimacs_compiler_builds_exact_public_relation() -> None:
    holo = ConstraintHolo.from_dimacs(
        """
        c exact unique solution
        p cnf 2 2
        1 1 1 0
        -2 -2 -2 0
        """
    )
    result = reference_existential_trace(holo)
    assert result.satisfiable
    assert result.witness_count == 1
    assert result.witness == {"x1": True, "x2": False}
    assert result.claim_ceiling == CLAIM_CEILING


def test_public_record_contains_local_relations_and_no_answer_fields() -> None:
    record = unique_solution_holo().public_record()
    audit_public_record(record)
    assert record["native_operator"] == "CATALYTIC_EXISTENTIAL_TRACE_NOT_ESTABLISHED"
    assert "local_relations" in record
    assert "equality_junctions" in record


def test_duplicate_clause_is_boolean_idempotent() -> None:
    original = unique_solution_holo()
    duplicated = original.with_duplicate_clause(0)
    original_result = reference_existential_trace(original)
    duplicated_result = reference_existential_trace(duplicated)
    assert original.semantic_digest() == duplicated.semantic_digest()
    assert original_result.witness_count == duplicated_result.witness_count == 1


def test_clause_and_literal_order_do_not_change_semantics() -> None:
    left = ConstraintHolo.build(
        ("a", "b", "c"),
        (
            clause(Literal("a"), Literal("b", False), Literal("c")),
            clause(Literal("c", False), Literal("a"), Literal("b")),
        ),
    )
    right = ConstraintHolo.build(
        ("a", "b", "c"),
        (
            clause(Literal("b"), Literal("a"), Literal("c", False)),
            clause(Literal("c"), Literal("b", False), Literal("a")),
        ),
    )
    assert left.semantic_digest() == right.semantic_digest()
    assert reference_existential_trace(left).witness_count == reference_existential_trace(right).witness_count


def test_explicit_presentation_gauge_preserves_solution_count() -> None:
    original = unique_solution_holo()
    renamed = original.renamed({"x1": "beta", "x2": "alpha"})
    assert reference_existential_trace(original).witness_count == 1
    renamed_result = reference_existential_trace(renamed)
    assert renamed_result.witness_count == 1
    assert renamed_result.witness == {"alpha": False, "beta": True}


def test_materialized_reference_refuses_large_boundary() -> None:
    variables = tuple(f"x{index}" for index in range(21))
    holo = ConstraintHolo.build(variables, ())
    with pytest.raises(ConstraintHoloError, match="exceeds the frozen limit"):
        reference_existential_trace(holo)


def test_pairwise_local_consistency_does_not_fake_global_closure() -> None:
    instance = ParityInstance.build(
        ("a", "b", "c"),
        (
            ParityConstraint("a", "b", 0),
            ParityConstraint("b", "c", 0),
            ParityConstraint("a", "c", 1),
        ),
    )
    result = calibrate_parity_holonomy(instance)
    assert result.pairwise_locally_compatible
    assert not result.consistent
    assert result.cycle_residues == (1,)
    assert result.cycle_holonomies == (-1,)
    assert result.obstruction_scope == "native_cycle_product_on_borrowed_vertex_phase_lanes"
    assert result.restored


def test_consistent_parity_cycle_has_trivial_holonomy() -> None:
    instance = ParityInstance.build(
        ("a", "b", "c"),
        (
            ParityConstraint("a", "b", 1),
            ParityConstraint("b", "c", 1),
            ParityConstraint("a", "c", 0),
        ),
    )
    result = calibrate_parity_holonomy(instance)
    assert result.consistent
    assert result.cycle_residues == (0,)
    assert result.cycle_holonomies == (1,)
    assert result.initial_carrier_digest == result.restored_carrier_digest
    assert result.terminal_carrier_digest != result.initial_carrier_digest


def test_parity_presentation_order_does_not_change_native_holonomy() -> None:
    left = ParityInstance.build(
        ("a", "b", "c"),
        (
            ParityConstraint("a", "b", 0),
            ParityConstraint("b", "c", 0),
            ParityConstraint("a", "c", 1),
        ),
    )
    right = ParityInstance.build(
        ("c", "a", "b"),
        (
            ParityConstraint("c", "a", 1),
            ParityConstraint("c", "b", 0),
            ParityConstraint("b", "a", 0),
        ),
    )
    left_result = calibrate_parity_holonomy(left)
    right_result = calibrate_parity_holonomy(right)
    assert left == right
    assert left_result.cycle_holonomies == right_result.cycle_holonomies == (-1,)
    assert left_result.terminal_carrier_digest == right_result.terminal_carrier_digest


def test_wrong_inverse_order_does_not_restore_transport_chain() -> None:
    instance = ParityInstance.build(
        ("a", "b", "c"),
        (
            ParityConstraint("a", "b", 1),
            ParityConstraint("b", "c", 1),
        ),
    )
    program = compile_z2_transport(instance)
    carrier = Z2PhaseCarrier(instance.vertices)
    initial_digest = carrier.digest()
    carrier.execute(instance, program)
    for operation in program.tree_transports:
        constraint = instance.constraints[operation.constraint_index]
        carrier.transport(operation.parent, operation.child, constraint.transport)
    assert carrier.digest() != initial_digest


def test_reversible_dilation_restores_but_does_not_claim_existential_trace() -> None:
    audit = audit_reversible_dilation(unique_solution_holo())
    assert audit.all_basis_states_restored
    assert audit.provenance_retained
    assert audit.existential_boundary_idempotence_required
    assert not audit.native_existential_trace_established
    assert audit.accepted_basis_states == 1


def test_factorized_projector_exposes_the_actual_missing_boundary() -> None:
    candidate = FactorizedProjectorCandidate(unique_solution_holo())
    contract = candidate.public_contract()
    assert contract["native_operator_status"] == "CATALYTIC_EXISTENTIAL_TRACE_NOT_ESTABLISHED"
    assert contract["unresolved_boundary"] == "EXACT_NONZERO_NORMALIZATION_WITHOUT_PROVENANCE_EXPANSION"
    assert candidate.apply_to_basis({"x1": True, "x2": False}) == 1
    assert candidate.apply_to_basis({"x1": False, "x2": False}) == 0


def test_reference_campaign_passes_without_promoting_p_equals_np() -> None:
    record = build_campaign_record()
    assert record["status"] == "REFERENCE_CAMPAIGN_PASS__CET_NATIVE_OPERATOR_NOT_ESTABLISHED"
    assert all(record["gates"].values())
    proof_state = record["proof_state"]
    assert proof_state["catalytic_existential_trace"] == "NOT_ESTABLISHED"
    assert proof_state["p_equals_np"] == "NOT_PROVEN"
