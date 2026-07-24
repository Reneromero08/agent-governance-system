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

from constraint_relational_trace_v1.conditional_flow_p_equals_np import (  # noqa: E402
    PolynomialFlowDeadline,
    conditional_flow_theorem,
    run_conditional_deadline_reference,
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def test_flow_deadline_implication_is_complete_but_conditional() -> None:
    holo = ConstraintHolo.build(
        ("x", "y"),
        (clause(Literal("x"), Literal("y"), Literal("y")),),
    )
    theorem = conditional_flow_theorem(holo)

    assert theorem.public_compiler_polynomial
    assert theorem.vector_field_coordinates_polynomial
    assert theorem.vector_field_evaluation_polynomial
    assert theorem.timeout_totalizes_unsat
    assert theorem.conditional_consequence == "3SAT_IN_P__THEREFORE_P_EQUALS_NP"
    assert theorem.convergence_theorem_status == "NOT_ESTABLISHED"
    assert theorem.precision_theorem_status == "NOT_ESTABLISHED"
    assert theorem.p_equals_np_status == "CONDITIONAL_ONLY__NOT_PROVEN"


def test_reached_witness_is_unconditional() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (clause(Literal("x"), Literal("x"), Literal("x")),),
    )
    result = run_conditional_deadline_reference(
        holo,
        PolynomialFlowDeadline(coefficient=100, exponent=2),
        step_size=0.01,
    )

    assert result.reference_run.converged_to_public_solution
    assert result.conditional_boundary == "VALID_SAT"
    assert result.boundary_is_unconditional


def test_timeout_is_never_promoted_without_deadline_theorem() -> None:
    holo = ConstraintHolo.build(
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
    result = run_conditional_deadline_reference(
        holo,
        PolynomialFlowDeadline(coefficient=1, exponent=1),
    )

    assert not result.reference_run.converged_to_public_solution
    assert result.conditional_boundary == (
        "CONDITIONAL_UNSAT_IF_UNIFORM_DEADLINE_THEOREM_HOLDS"
    )
    assert not result.boundary_is_unconditional
    assert result.status == "REFERENCE_FLOW_TIMEOUT__NO_UNSAT_PROMOTION"
