from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    FactorizedProjectorCandidate,
    audit_reversible_dilation,
    reference_existential_trace,
)
from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .parity_holonomy import ParityConstraint, ParityInstance, calibrate_parity_holonomy


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

    unique_boundary = reference_existential_trace(unique)
    impossible_boundary = reference_existential_trace(impossible)
    duplicate_boundary = reference_existential_trace(unique.with_duplicate_clause(0))
    renamed_boundary = reference_existential_trace(
        unique.renamed({"x1": "renamed_b", "x2": "renamed_a"})
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
        "program_derived_restoration": parity_result.restored,
        "reversible_evaluation_dilation": dilation.all_basis_states_restored,
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
        "parity_holonomy": asdict(parity_result),
        "reversible_dilation": asdict(dilation),
        "factorized_projector": projector.public_contract(),
        "proof_state": {
            "semantic_object": "ESTABLISHED_REFERENCE",
            "parity_holonomy": "ESTABLISHED_CALIBRATION",
            "false_closure_control": "ESTABLISHED_CALIBRATION",
            "native_relation_valued_transport": "NOT_ESTABLISHED",
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
