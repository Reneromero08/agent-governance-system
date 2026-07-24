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
from constraint_relational_trace_v1.instanton_deadline_audit import (  # noqa: E402
    audit_instanton_deadline_argument,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def test_instanton_deadline_audit_is_linear_at_fixed_density() -> None:
    variables = tuple(f"x{index}" for index in range(6))
    holo = ConstraintHolo.build(
        variables,
        tuple(
            clause(Literal(variable), Literal(variable), Literal(variable))
            for variable in variables
        ),
    )
    audit = audit_instanton_deadline_argument(holo)

    assert audit.phase_space_dimension == 18
    assert audit.clause_density == 1.0
    assert audit.maximum_index_descent_steps == 18
    assert audit.conditional_continuous_time_bound == (
        "T_PHYS_LE_N_TIMES_ONE_PLUS_TWO_DENSITY_TIMES_TMAX"
    )
    assert "NO_EXPLICIT_UNIFORM_NUMERIC_BOUND" in audit.uniform_instanton_width_status
    assert "NO_EXPLICIT_UNIFORM_NUMERIC_BOUND" in audit.uniform_critical_dwell_status
    assert audit.polynomial_precision_status == "NOT_ESTABLISHED"
    assert audit.ordinary_p_equals_np_status == "NOT_ESTABLISHED"


def test_published_solvable_scope_does_not_totalize_unsat() -> None:
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
    audit = audit_instanton_deadline_argument(holo)

    assert audit.solvable_instance_scope == (
        "PUBLISHED_PROPOSITION_APPLIES_TO_SOLVABLE_FIXED_DENSITY_3SAT"
    )
    assert "NOT_COVERED" in audit.unsat_deadline_status
    assert "NUMERICAL_INTEGRATION" in audit.numerical_discretization_transfer_status
