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

from constraint_relational_trace_v1.catalytic_existential_trace import (  # noqa: E402
    reference_existential_trace,
)
from constraint_relational_trace_v1.self_organizing_clause_flow import (  # noqa: E402
    integrate_reference_until_solution,
)
from constraint_relational_trace_v1.structured_clause_families import (  # noqa: E402
    complete_graph_edges,
    cycle_graph_edges,
    graph_three_coloring_holo,
    parity_cycle_holo,
    pigeonhole_holo,
)


def assert_sat_flow_reaches_solution(holo, max_steps: int = 40_000) -> int:
    reference = reference_existential_trace(holo)
    assert reference.satisfiable
    run = integrate_reference_until_solution(
        holo,
        step_size=2.0e-3,
        max_steps=max_steps,
    )
    assert run.converged_to_public_solution
    assert holo.accepts(dict(run.final_assignment))
    return run.steps_executed


def assert_unsat_flow_never_emits_solution(holo, max_steps: int = 20_000) -> None:
    reference = reference_existential_trace(holo)
    assert not reference.satisfiable
    run = integrate_reference_until_solution(
        holo,
        step_size=2.0e-3,
        max_steps=max_steps,
    )
    assert not run.converged_to_public_solution
    assert run.status == "REFERENCE_FLOW_STEP_CAP_REACHED__NO_UNSAT_CONCLUSION"


def test_even_and_odd_parity_cycles() -> None:
    assert_sat_flow_reaches_solution(parity_cycle_holo(8, total_charge=0))
    assert_unsat_flow_never_emits_solution(parity_cycle_holo(8, total_charge=1))


def test_pigeonhole_sat_and_unsat_controls() -> None:
    assert_sat_flow_reaches_solution(pigeonhole_holo(2, 2))
    assert_unsat_flow_never_emits_solution(pigeonhole_holo(3, 2))


def test_graph_coloring_sat_and_unsat_controls() -> None:
    assert_sat_flow_reaches_solution(
        graph_three_coloring_holo(5, cycle_graph_edges(5))
    )
    assert_unsat_flow_never_emits_solution(
        graph_three_coloring_holo(4, complete_graph_edges(4))
    )
