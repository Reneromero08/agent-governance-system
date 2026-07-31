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
from constraint_relational_trace_v1.phase_transition_corpus import (  # noqa: E402
    sealed_phase_transition_cases,
)
from constraint_relational_trace_v1.phase_transition_flow_campaign import (  # noqa: E402
    build_phase_transition_flow_record,
)


def test_sealed_phase_transition_labels_and_digests() -> None:
    sat_case, unsat_case = sealed_phase_transition_cases()
    assert sat_case.expected_status == "SAT"
    assert sat_case.semantic_digest == (
        "5c43f19a094aafc9ec0867652cd8a487b9400abdb0e857708ae5189fbc1fe96f"
    )
    assert unsat_case.expected_status == "UNSAT"
    assert unsat_case.semantic_digest == (
        "5dc4f43dadfbd2bba8c407f8f88ae73c4f6594ad10aeb667a3e0a6ac501b0de3"
    )


def test_sealed_pair_fixed_deadline_boundary() -> None:
    sat_case, unsat_case = sealed_phase_transition_cases()
    sat_run = integrate_adaptive_phase_logit_flow(
        sat_case.holo,
        fixed_deadline=3.0,
        solver_method="DOP853",
        maximum_step=2.0e-2,
    )
    unsat_run = integrate_adaptive_phase_logit_flow(
        unsat_case.holo,
        fixed_deadline=3.0,
        solver_method="DOP853",
        maximum_step=2.0e-2,
    )
    assert sat_run.terminal_solution_verified
    assert sat_run.status == "TERMINAL_WITNESS_VERIFIED"
    assert not unsat_run.terminal_solution_verified
    assert not unsat_run.status.startswith("INVALID_CARRIER")


def test_sixteen_seed_phase_transition_campaign_has_no_false_positive() -> None:
    record = build_phase_transition_flow_record(seed_count=16, fixed_deadline=3.0)
    assert record["satisfiable_cases"] == 14
    assert record["unsatisfiable_cases"] == 2
    assert record["unsat_false_positives"] == 0
    assert record["invalid_carriers"] == 0
    assert record["satisfiable_terminal_witnesses"] == 14
    assert record["satisfiable_misses"] == 0
    assert record["status"] == (
        "PHASE_TRANSITION_FLOW_CAMPAIGN_ALL_CLASSIFIED_AT_FIXED_DEADLINE"
    )
