from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from constraint_relational_trace_v1.catalytic_existential_trace import (
        CLAIM_CEILING,
        FactorizedProjectorCandidate,
        audit_reversible_dilation,
        reference_existential_trace,
    )
    from constraint_relational_trace_v1.conditional_p_equals_np import (
        extract_witness_by_boundary_self_reduction,
        reference_decision_boundary,
    )
    from constraint_relational_trace_v1.constraint_holo import ClauseRelation, ConstraintHolo, Literal
    from constraint_relational_trace_v1.mpo_configuration_audit import (
        audit_control_symbol_mpo_projection,
    )
    from constraint_relational_trace_v1.oracle_determinant_compensation import (
        audit_oracle_determinant_compensation,
    )
    from constraint_relational_trace_v1.parity_holonomy import (
        ParityConstraint,
        ParityInstance,
        calibrate_parity_holonomy,
    )
    from constraint_relational_trace_v1.relational_width_audit import (
        audit_residual_relation_width,
        equality_relation_holo,
    )
    from constraint_relational_trace_v1.topological_rank_latch import (
        audit_topological_rank_latch,
    )
else:
    from .catalytic_existential_trace import (
        CLAIM_CEILING,
        FactorizedProjectorCandidate,
        audit_reversible_dilation,
        reference_existential_trace,
    )
    from .conditional_p_equals_np import (
        extract_witness_by_boundary_self_reduction,
        reference_decision_boundary,
    )
    from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
    from .mpo_configuration_audit import audit_control_symbol_mpo_projection
    from .oracle_determinant_compensation import audit_oracle_determinant_compensation
    from .parity_holonomy import ParityConstraint, ParityInstance, calibrate_parity_holonomy
    from .relational_width_audit import audit_residual_relation_width, equality_relation_holo
    from .topological_rank_latch import audit_topological_rank_latch


def _literal(name: str, positive: bool = True) -> Literal:
    return Literal(name, positive)


def _clause(*literals: Literal) -> ClauseRelation:
    if len(literals) != 3:
        raise ValueError("three literals required")
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def build_campaign_record() -> dict[str, object]:
    unique = ConstraintHolo.build(
        ("x1", "x2"),
        (
            _clause(_literal("x1"), _literal("x1"), _literal("x1")),
            _clause(_literal("x2", False), _literal("x2", False), _literal("x2", False)),
        ),
    )
    impossible = ConstraintHolo.build(
        ("x1",),
        (
            _clause(_literal("x1"), _literal("x1"), _literal("x1")),
            _clause(_literal("x1", False), _literal("x1", False), _literal("x1", False)),
        ),
    )
    impossible_equal_dimension = ConstraintHolo.build(
        ("x1", "x2"),
        (
            _clause(_literal("x1"), _literal("x1"), _literal("x1")),
            _clause(_literal("x1", False), _literal("x1", False), _literal("x1", False)),
        ),
    )

    unique_boundary = reference_existential_trace(unique)
    impossible_boundary = reference_existential_trace(impossible)
    duplicate_boundary = reference_existential_trace(unique.with_duplicate_clause(0))
    renamed_boundary = reference_existential_trace(
        unique.renamed({"x1": "renamed_b", "x2": "renamed_a"})
    )
    witness_boundary = extract_witness_by_boundary_self_reduction(
        unique, reference_decision_boundary
    )

    parity_false_closure = ParityInstance.build(
        ("a", "b", "c"),
        (
            ParityConstraint("a", "b", 0),
            ParityConstraint("b", "c", 0),
            ParityConstraint("a", "c", 1),
        ),
    )
    parity_result = calibrate_parity_holonomy(parity_false_closure)

    projector = FactorizedProjectorCandidate(unique)
    dilation = audit_reversible_dilation(unique)
    topological_latch = audit_topological_rank_latch(unique)
    topological_unsat = audit_topological_rank_latch(impossible)
    determinant_compensation_sat = audit_oracle_determinant_compensation(
        unique, ancillary_qubits=2
    )
    determinant_compensation_unsat = audit_oracle_determinant_compensation(
        impossible_equal_dimension, ancillary_qubits=2
    )

    mpo_projection = audit_control_symbol_mpo_projection(
        control_states=4,
        alphabet_symbols=2,
        bounded_tape_cells=8,
    )
    equality_pairs = 5
    equality_holo = equality_relation_holo(equality_pairs)
    grouped_order = tuple(f"x{index}" for index in range(1, equality_pairs + 1)) + tuple(
        f"y{index}" for index in range(1, equality_pairs + 1)
    )
    interleaved_order = tuple(
        variable
        for index in range(1, equality_pairs + 1)
        for variable in (f"x{index}", f"y{index}")
    )
    grouped_width = audit_residual_relation_width(equality_holo, grouped_order)
    interleaved_width = audit_residual_relation_width(equality_holo, interleaved_order)

    gates = {
        "exact_open_relation": unique_boundary.witness_count == 1,
        "unsat_reference": not impossible_boundary.satisfiable,
        "duplicate_clause_idempotence": (
            unique_boundary.satisfiable == duplicate_boundary.satisfiable
            and unique_boundary.witness_count == duplicate_boundary.witness_count
        ),
        "presentation_gauge": unique_boundary.witness_count == renamed_boundary.witness_count,
        "pairwise_false_closure_exposed": (
            parity_result.pairwise_locally_compatible and not parity_result.consistent
        ),
        "native_z2_cycle_obstruction": parity_result.cycle_holonomies == (-1,),
        "program_derived_restoration": parity_result.restored,
        "reversible_evaluation_dilation": dilation.all_basis_states_restored,
        "conditional_witness_boundary": (
            witness_boundary.valid and witness_boundary.witness_verified
        ),
        "topological_rank_latch_complete_reference": (
            topological_latch.determinant_winding == 1
            and topological_latch.presence_index == 1
            and topological_unsat.determinant_winding == 0
            and topological_unsat.presence_index == 0
        ),
        "topological_latch_restoration": (
            topological_latch.restoration_verified
            and topological_unsat.restoration_verified
        ),
        "full_oracle_determinant_compensation": (
            determinant_compensation_sat.total_dimension
            == determinant_compensation_unsat.total_dimension
            and determinant_compensation_sat.full_winding
            == determinant_compensation_unsat.full_winding
            and determinant_compensation_sat.clean_winding == 1
            and determinant_compensation_unsat.clean_winding == 0
            and determinant_compensation_sat.compensation_identity_verified
            and determinant_compensation_unsat.compensation_identity_verified
        ),
        "historical_mpo_projection_rejected": (
            not mpo_projection.exact_configuration_injective
            and mpo_projection.invariant_scope
            == "finite_control_symbol_transition_graph_only"
        ),
        "residual_relation_width_exposes_order_dependence": (
            grouped_width.maximum_width == 2**equality_pairs
            and interleaved_width.maximum_width <= 3
        ),
        "native_cet_not_smuggled": not dilation.native_existential_trace_established,
    }

    return {
        "schema": "CONSTRAINT_RELATIONAL_TRACE_REFERENCE_CAMPAIGN_V1",
        "claim_ceiling": CLAIM_CEILING,
        "status": (
            "REFERENCE_CAMPAIGN_PASS__CET_NATIVE_OPERATOR_NOT_ESTABLISHED"
            if all(gates.values())
            else "REFERENCE_CAMPAIGN_FAILED"
        ),
        "gates": gates,
        "unique_solution_boundary": asdict(unique_boundary),
        "unsat_boundary": asdict(impossible_boundary),
        "duplicate_boundary": asdict(duplicate_boundary),
        "renamed_boundary": asdict(renamed_boundary),
        "conditional_witness_boundary": asdict(witness_boundary),
        "parity_holonomy": asdict(parity_result),
        "reversible_dilation": asdict(dilation),
        "factorized_projector": projector.public_contract(),
        "topological_rank_latch_unique": asdict(topological_latch),
        "topological_rank_latch_unsat": asdict(topological_unsat),
        "oracle_determinant_compensation_sat": asdict(determinant_compensation_sat),
        "oracle_determinant_compensation_unsat": asdict(determinant_compensation_unsat),
        "historical_mpo_projection_audit": asdict(mpo_projection),
        "grouped_residual_width": asdict(grouped_width),
        "interleaved_residual_width": asdict(interleaved_width),
        "proof_state": {
            "semantic_object": "ESTABLISHED_REFERENCE",
            "parity_holonomy": "ESTABLISHED_NATIVE_Z2_CALIBRATION",
            "false_closure_control": "ESTABLISHED_CALIBRATION",
            "conditional_cet_implies_3sat_in_p": "ESTABLISHED_THEOREM",
            "conditional_witness_self_reduction": "ESTABLISHED_REFERENCE",
            "topological_determinant_winding": "ESTABLISHED_COMPLETE_REFERENCE_INVARIANT",
            "full_reversible_oracle_determinant": "ESTABLISHED_FORMULA_INDEPENDENT_NULL",
            "clean_subspace_determinant": "ESTABLISHED_RETAINS_SAT_INDEX",
            "historical_mpo_control_symbol_projection": "REJECTED_AS_CONFIGURATION_CARRIER",
            "residual_relation_width": "ESTABLISHED_PRESENTATION_SENSITIVE_CONTROL",
            "polynomial_restricted_determinant_sensor": "NOT_ESTABLISHED",
            "polynomial_native_determinant_line_sensor": "NOT_ESTABLISHED",
            "universal_polynomial_bond_dimension": "NOT_ESTABLISHED",
            "native_relation_valued_transport": "NOT_ESTABLISHED_FOR_GENERAL_CLAUSES",
            "catalytic_existential_trace": "NOT_ESTABLISHED",
            "polynomial_resource_theorem": "NOT_ESTABLISHED",
            "standard_model_transfer": "NOT_ESTABLISHED",
            "p_equals_np": "NOT_PROVEN",
        },
    }


def main() -> int:
    record = build_campaign_record()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reference_campaign.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "output": str(output_path)}, sort_keys=True))
    return 0 if record["status"] == "REFERENCE_CAMPAIGN_PASS__CET_NATIVE_OPERATOR_NOT_ESTABLISHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
