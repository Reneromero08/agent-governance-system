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

from constraint_relational_trace_v1.adaptive_clause_flow import (  # noqa: E402
    integrate_adaptive_until_solution,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    exact_three_parity_cycle_holo,
    exact_three_unique_solution_holo,
)


def test_adaptive_flow_reaches_exact_three_unique_solution() -> None:
    holo = exact_three_unique_solution_holo(8)
    run = integrate_adaptive_until_solution(holo, maximum_time=5.0)

    assert run.converged_to_public_solution
    assert run.continuous_time < 5.0
    assert run.function_evaluations > 0
    assert holo.accepts(dict(run.final_assignment))


def test_adaptive_flow_reaches_exact_three_parity_cycle() -> None:
    holo = exact_three_parity_cycle_holo(16, total_charge=0)
    run = integrate_adaptive_until_solution(holo, maximum_time=10.0)

    assert run.converged_to_public_solution
    assert run.continuous_time < 10.0
    assert run.function_evaluations > 0
    assert holo.accepts(dict(run.final_assignment))
