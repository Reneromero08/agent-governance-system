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

from constraint_relational_trace_v1.adaptive_phase_logit_flow import (  # noqa: E402
    integrate_adaptive_phase_logit_flow,
)
from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    Literal,
)
from constraint_relational_trace_v1.phase_transition_corpus import (  # noqa: E402
    certify_phase_transition_case,
)
from constraint_relational_trace_v1.polynomial_phase_selector_flow import (  # noqa: E402
    PolynomialPhaseSelectorFlowState,
    polynomial_phase_selector_flow_derivative,
)


def test_exact_product_default_repairs_unique_witness_seed_38() -> None:
    case = certify_phase_transition_case(12, 51, 38)
    assert case.expected_status == "SAT"
    assert case.witness_count_reference_only == 1

    run = integrate_adaptive_phase_logit_flow(
        case.holo,
        fixed_deadline=3.0,
        solver_method="DOP853",
        maximum_step=2.0e-2,
    )

    assert run.reached_fixed_deadline
    assert run.terminal_solution_verified
    assert run.status == "TERMINAL_WITNESS_VERIFIED"
    assert run.first_passage_time is not None
    assert run.first_passage_time < 1.5
    assert run.terminal_clause_satisfaction_margin > 0.9
    assert run.maximum_long_memory < 100.0
    assert run.native_trajectory_length_lower_bound < 500.0


def test_selector_min_remains_a_falsified_nondefault_calibration() -> None:
    case = certify_phase_transition_case(12, 51, 38)
    run = integrate_adaptive_phase_logit_flow(
        case.holo,
        fixed_deadline=3.0,
        solver_method="DOP853",
        maximum_step=2.0e-2,
        gradient_mode="selector_min",
    )

    assert run.reached_fixed_deadline
    assert not run.terminal_solution_verified
    assert run.status == "TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED"
    assert run.terminal_clause_satisfaction_margin < 0.0
    assert run.maximum_long_memory > 500.0


def test_exact_product_gradient_scales_with_truth_gain() -> None:
    holo = ConstraintHolo.build(
        ("x", "y", "z"),
        (
            ClauseRelation(
                (
                    Literal("x"),
                    Literal("y"),
                    Literal("z"),
                )
            ),
        ),
    )
    state = PolynomialPhaseSelectorFlowState(
        phase_cosine=(0.0, 0.0, 0.0),
        phase_sine=(1.0, 1.0, 1.0),
        short_memory=(1.0,),
        long_memory=(1.0,),
        clause_selector=((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),),
        pair_selector=((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),),
    )

    default_gain = polynomial_phase_selector_flow_derivative(
        holo,
        state,
        truth_gain=4.0,
    )
    doubled_gain = polynomial_phase_selector_flow_derivative(
        holo,
        state,
        truth_gain=8.0,
    )

    assert default_gain.phase_cosine == (0.5, 0.5, 0.5)
    assert doubled_gain.phase_cosine == (1.0, 1.0, 1.0)
    assert doubled_gain.phase_sine == (0.0, 0.0, 0.0)
