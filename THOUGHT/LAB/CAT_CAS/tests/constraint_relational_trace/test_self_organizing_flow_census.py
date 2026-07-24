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

from constraint_relational_trace_v1.self_organizing_flow_census import (  # noqa: E402
    run_three_variable_flow_census,
)


def test_exhaustive_three_variable_flow_census() -> None:
    census = run_three_variable_flow_census()

    assert census.total_formulae == 256
    assert census.satisfiable_formulae == 255
    assert census.unsatisfiable_formulae == 1
    assert census.unsat_false_solution_count == 0
    assert census.satisfiable_converged == census.satisfiable_formulae
    assert census.satisfiable_failed == 0
    assert not census.failures
    assert census.status == "EXHAUSTIVE_THREE_VARIABLE_FLOW_CENSUS_PASS"
