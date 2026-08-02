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
from constraint_relational_trace_v1.reduced_exact_product_phase_flow import (  # noqa: E402
    audit_reduced_exact_product_phase_flow,
    public_reduced_exact_product_state,
    reduced_exact_product_phase_derivative,
)


def test_reduced_carrier_matches_full_exact_product_active_field() -> None:
    holo = ConstraintHolo.build(
        ("a", "b", "c", "d"),
        (
            ClauseRelation((Literal("a"), Literal("b"), Literal("c"))),
            ClauseRelation(
                (
                    Literal("b", False),
                    Literal("c"),
                    Literal("d", False),
                )
            ),
        ),
    )
    audit = audit_reduced_exact_product_phase_flow(holo)

    assert audit.reduction_exact
    assert audit.active_derivative_residual == 0.0
    assert audit.native_state_coordinates == 18
    assert audit.logit_chart_coordinates == 12
    assert audit.removed_pair_selector_coordinates == 12
    assert audit.status == (
        "REDUCED_EXACT_PRODUCT_PHASE_CARRIER_MATCHES_QUALIFIED_ACTIVE_FIELD"
    )


def test_reduced_carrier_derivative_is_finite_on_public_seed() -> None:
    holo = ConstraintHolo.build(
        ("x", "y", "z"),
        (
            ClauseRelation((Literal("x"), Literal("y"), Literal("z"))),
        ),
    )
    derivative = reduced_exact_product_phase_derivative(
        holo,
        public_reduced_exact_product_state(holo),
    )

    assert derivative.max_abs() > 0.0
