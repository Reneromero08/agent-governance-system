from __future__ import annotations

from dataclasses import dataclass

from .constraint_holo import ConstraintHolo
from .self_organizing_clause_flow import integrate_reference_until_solution
from .structured_clause_families import (
    cycle_graph_edges,
    exact_three_graph_coloring_holo,
    exact_three_parity_cycle_holo,
    exact_three_unique_solution_holo,
)


@dataclass(frozen=True)
class FlowScalingPoint:
    family: str
    public_variables: int
    public_clauses: int
    state_coordinates: int
    integration_step: float
    steps_to_verified_solution: int
    continuous_time_to_solution: float
    maximum_long_memory: float
    final_max_clause_constraint: float
    status: str


@dataclass(frozen=True)
class FlowScalingCampaign:
    points: tuple[FlowScalingPoint, ...]
    all_reached_verified_solution: bool
    maximum_observed_steps: int
    maximum_observed_long_memory: float
    empirical_status: str
    theorem_status: str


def _run_point(
    family: str,
    holo: ConstraintHolo,
    step_size: float,
    max_steps: int,
) -> FlowScalingPoint:
    run = integrate_reference_until_solution(
        holo,
        step_size=step_size,
        max_steps=max_steps,
    )
    return FlowScalingPoint(
        family=family,
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        state_coordinates=len(holo.variables) + 2 * len(holo.clauses),
        integration_step=step_size,
        steps_to_verified_solution=run.steps_executed,
        continuous_time_to_solution=run.steps_executed * step_size,
        maximum_long_memory=max(run.final_state.long_memory, default=1.0),
        final_max_clause_constraint=run.final_max_clause_constraint,
        status=run.status,
    )


def run_structured_flow_scaling_campaign(
    sizes: tuple[int, ...] = (4, 8, 16, 32),
    step_size: float = 2.0e-3,
    max_steps: int = 100_000,
) -> FlowScalingCampaign:
    """Measure deterministic public-seed convergence on structured SAT families.

    This campaign never enumerates assignments. A point passes only after its threshold
    assignment is directly verified against the public relation. The resulting trend is
    empirical evidence, not a worst-case theorem.
    """

    points: list[FlowScalingPoint] = []
    for size in sizes:
        points.append(
            _run_point(
                "UNIT_UNIQUE_SOLUTION",
                exact_three_unique_solution_holo(size),
                step_size,
                max_steps,
            )
        )
        points.append(
            _run_point(
                "EVEN_PARITY_CYCLE",
                exact_three_parity_cycle_holo(size, total_charge=0),
                step_size,
                max_steps,
            )
        )
        if size <= 16:
            points.append(
                _run_point(
                    "THREE_COLOR_CYCLE",
                    exact_three_graph_coloring_holo(size, cycle_graph_edges(size)),
                    step_size,
                    max_steps,
                )
            )

    all_solved = all(point.status == "REFERENCE_FLOW_REACHED_PUBLIC_SOLUTION" for point in points)
    return FlowScalingCampaign(
        points=tuple(points),
        all_reached_verified_solution=all_solved,
        maximum_observed_steps=max((point.steps_to_verified_solution for point in points), default=0),
        maximum_observed_long_memory=max((point.maximum_long_memory for point in points), default=1.0),
        empirical_status=(
            "STRUCTURED_FLOW_SCALING_CAMPAIGN_ALL_VERIFIED"
            if all_solved
            else "STRUCTURED_FLOW_SCALING_CAMPAIGN_EXPOSED_TIMEOUT"
        ),
        theorem_status=(
            "FINITE_STRUCTURED_EVIDENCE_ONLY__UNIFORM_WORST_CASE_DEADLINE_NOT_ESTABLISHED"
        ),
    )
