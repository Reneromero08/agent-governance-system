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
from constraint_relational_trace_v1.exact_product_reduction_audit import (  # noqa: E402
    audit_exact_product_pair_selector_reduction,
)


def test_pair_selectors_are_dead_coordinates_in_exact_product_mode() -> None:
    holo = ConstraintHolo.build(
        ("x", "y", "z"),
        (
            ClauseRelation(
                (
                    Literal("x"),
                    Literal("y", False),
                    Literal("z"),
                )
            ),
            ClauseRelation(
                (
                    Literal("x", False),
                    Literal("y"),
                    Literal("z", False),
                )
            ),
        ),
    )
    audit = audit_exact_product_pair_selector_reduction(holo)

    assert audit.pair_selector_subsystem_decoupled
    assert audit.active_derivative_residual == 0.0
    assert audit.full_state_coordinates == 28
    assert audit.removable_pair_selector_coordinates == 12
    assert audit.reduced_state_coordinates == 16
    assert audit.reduction_status == (
        "EXACT_PRODUCT_PAIR_SELECTORS_REMOVABLE__ACTIVE_CARRIER_2N_PLUS_5M"
    )
