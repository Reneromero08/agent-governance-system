from __future__ import annotations

from pathlib import Path
import math
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
from constraint_relational_trace_v1.reduced_phase_resource_bound import (  # noqa: E402
    derive_reduced_phase_polynomial_resource_bound,
)


def test_polynomial_deadline_closes_forward_range_length_and_precision() -> None:
    holo = ConstraintHolo.build(
        ("x", "y", "z"),
        (
            ClauseRelation((Literal("x"), Literal("y"), Literal("z"))),
            ClauseRelation(
                (
                    Literal("x", False),
                    Literal("y"),
                    Literal("z", False),
                )
            ),
        ),
    )
    bound = derive_reduced_phase_polynomial_resource_bound(holo)

    assert bound.native_state_coordinates == 16
    assert bound.long_memory_cap == 20_000.0
    assert bound.maximum_clause_violation == 4.0
    assert bound.maximum_exact_gradient_coordinate == 2.0
    assert bound.maximum_clause_rigidity_coordinate == 1.0
    assert bound.maximum_variable_relational_force == 252_006.0
    assert bound.maximum_variable_incident_violation == 24.0
    assert bound.maximum_angular_speed == 252_246.0
    assert math.isclose(bound.maximum_short_memory_speed, 18.75)
    assert math.isclose(bound.maximum_long_memory_speed, 98_750.0)
    assert bound.maximum_clause_selector_speed == 40.0
    assert bound.maximum_native_l2_speed > bound.maximum_angular_speed
    assert bound.maximum_short_logit_speed == 75.0
    assert math.isclose(bound.maximum_long_logit_speed, 19.75)
    assert bound.maximum_clause_log_ratio_speed == 40.0
    assert bound.state_range_polynomial_without_deadline
    assert bound.trajectory_length_polynomial_if_deadline_polynomial
    assert bound.logit_range_polynomial_if_deadline_polynomial
    assert bound.forward_standard_model_transfer_status == (
        "POLYNOMIAL_DEADLINE_IMPLIES_POLYNOMIAL_FORWARD_RANGE_LENGTH_AND_PRECISION"
    )
    assert bound.remaining_obligations == (
        "formula_uniform_public_seed_deadline",
        "robust_terminal_boundary_margin",
        "deterministic_total_unsat_boundary",
        "cotangent_or_environmental_restoration_bound",
    )
