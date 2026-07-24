from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import (
    SelfOrganizingFlowParameters,
    SelfOrganizingFlowState,
    public_perturbed_initial_state,
    threshold_assignment,
)


@dataclass(frozen=True)
class AdaptiveClauseFlowRun:
    converged_to_public_solution: bool
    continuous_time: float
    function_evaluations: int
    accepted_internal_steps: int
    relative_tolerance: float
    absolute_tolerance: float
    maximum_step: float
    final_assignment: tuple[tuple[str, bool], ...]
    final_max_clause_constraint: float
    maximum_long_memory: float
    final_state: SelfOrganizingFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


def integrate_adaptive_until_solution(
    holo: ConstraintHolo,
    initial_state: SelfOrganizingFlowState | None = None,
    maximum_time: float = 20.0,
    relative_tolerance: float = 1.0e-7,
    absolute_tolerance: float = 1.0e-9,
    maximum_step: float = 5.0e-2,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> AdaptiveClauseFlowRun:
    """Instrument the clause flow with a vectorized adaptive RK solver.

    The vector field is still compiled only from public clause incidence, literal signs,
    and memory coordinates. Adaptive integration is an external chart used to test the
    continuous candidate. Its function-evaluation count is not native compute evidence.
    """

    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - environment contract
        raise ConstraintHoloError("adaptive clause flow requires NumPy and SciPy") from exc

    scalars = (
        maximum_time,
        relative_tolerance,
        absolute_tolerance,
        maximum_step,
    )
    if not all(isfinite(value) and value > 0 for value in scalars):
        raise ConstraintHoloError("adaptive integration controls must be positive and finite")

    variable_count = len(holo.variables)
    clause_count = len(holo.clauses)
    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    occurrence_indices = np.asarray(
        [
            [variable_index[literal.variable] for literal in clause.literals]
            for clause in holo.clauses
        ],
        dtype=np.int64,
    )
    occurrence_signs = np.asarray(
        [
            [1.0 if literal.positive else -1.0 for literal in clause.literals]
            for clause in holo.clauses
        ],
        dtype=np.float64,
    )
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, clause_count))

    state = initial_state or public_perturbed_initial_state(holo)
    if len(state.voltages) != variable_count:
        raise ConstraintHoloError("adaptive initial voltage count mismatch")
    if len(state.short_memory) != clause_count or len(state.long_memory) != clause_count:
        raise ConstraintHoloError("adaptive initial memory count mismatch")

    initial_vector = np.asarray(
        state.voltages + state.short_memory + state.long_memory,
        dtype=np.float64,
    )

    def unpack(vector):
        voltage = np.clip(vector[:variable_count], -1.0, 1.0)
        short_memory = np.clip(
            vector[variable_count : variable_count + clause_count],
            0.0,
            1.0,
        )
        long_memory = np.clip(vector[variable_count + clause_count :], 1.0, cap)
        return voltage, short_memory, long_memory

    def clause_values(voltage):
        if clause_count == 0:
            return np.empty(0, dtype=np.float64)
        defects = 1.0 - occurrence_signs * voltage[occurrence_indices]
        return 0.5 * np.min(defects, axis=1)

    def vector_field(_time, vector):
        voltage, short_memory, long_memory = unpack(vector)
        if clause_count == 0:
            return np.zeros_like(vector)

        local_voltage = voltage[occurrence_indices]
        defects = 1.0 - occurrence_signs * local_voltage
        minimum_defect = np.min(defects, axis=1)
        other_minimum = np.stack(
            (
                np.minimum(defects[:, 1], defects[:, 2]),
                np.minimum(defects[:, 0], defects[:, 2]),
                np.minimum(defects[:, 0], defects[:, 1]),
            ),
            axis=1,
        )
        gradient = 0.5 * occurrence_signs * other_minimum
        rigidity = 0.5 * (occurrence_signs - local_voltage) * np.isclose(
            defects,
            minimum_defect[:, None],
            rtol=0.0,
            atol=1.0e-12,
        )
        contributions = (
            (long_memory * short_memory)[:, None] * gradient
            + (
                (1.0 + parameters.zeta * long_memory)
                * (1.0 - short_memory)
            )[:, None]
            * rigidity
        )
        voltage_derivative = np.zeros(variable_count, dtype=np.float64)
        np.add.at(voltage_derivative, occurrence_indices.reshape(-1), contributions.reshape(-1))

        constraints = 0.5 * minimum_defect
        short_derivative = (
            parameters.beta
            * (short_memory + parameters.epsilon)
            * (constraints - parameters.gamma)
        )
        long_derivative = parameters.alpha * (constraints - parameters.delta)

        voltage_derivative[(voltage <= -1.0) & (voltage_derivative < 0.0)] = 0.0
        voltage_derivative[(voltage >= 1.0) & (voltage_derivative > 0.0)] = 0.0
        short_derivative[(short_memory <= 0.0) & (short_derivative < 0.0)] = 0.0
        short_derivative[(short_memory >= 1.0) & (short_derivative > 0.0)] = 0.0
        long_derivative[(long_memory <= 1.0) & (long_derivative < 0.0)] = 0.0
        long_derivative[(long_memory >= cap) & (long_derivative > 0.0)] = 0.0

        return np.concatenate((voltage_derivative, short_derivative, long_derivative))

    def solution_event(_time, vector):
        voltage, _short_memory, _long_memory = unpack(vector)
        constraints = clause_values(voltage)
        return float(np.max(constraints, initial=-1.0) - (0.5 - 1.0e-8))

    solution_event.terminal = True
    solution_event.direction = -1.0

    initial_voltage, initial_short, initial_long = unpack(initial_vector)
    initial_state_clipped = SelfOrganizingFlowState(
        tuple(float(value) for value in initial_voltage),
        tuple(float(value) for value in initial_short),
        tuple(float(value) for value in initial_long),
    )
    initial_assignment = threshold_assignment(holo, initial_state_clipped)
    initial_constraints = clause_values(initial_voltage)
    if holo.accepts(initial_assignment) and float(np.max(initial_constraints, initial=-1.0)) < 0.5:
        return AdaptiveClauseFlowRun(
            converged_to_public_solution=True,
            continuous_time=0.0,
            function_evaluations=0,
            accepted_internal_steps=0,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
            final_assignment=tuple(sorted(initial_assignment.items())),
            final_max_clause_constraint=float(np.max(initial_constraints, initial=0.0)),
            maximum_long_memory=max(initial_state_clipped.long_memory, default=1.0),
            final_state=initial_state_clipped,
            status="ADAPTIVE_FLOW_INITIAL_STATE_IS_PUBLIC_SOLUTION",
        )

    solution = solve_ivp(
        vector_field,
        (0.0, maximum_time),
        initial_vector,
        method="RK45",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
        events=solution_event,
    )
    final_vector = solution.y[:, -1]
    final_voltage, final_short, final_long = unpack(final_vector)
    final_state = SelfOrganizingFlowState(
        tuple(float(value) for value in final_voltage),
        tuple(float(value) for value in final_short),
        tuple(float(value) for value in final_long),
    )
    assignment = threshold_assignment(holo, final_state)
    constraints = clause_values(final_voltage)
    max_constraint = float(np.max(constraints, initial=0.0))
    converged = max_constraint < 0.5 and holo.accepts(assignment)

    return AdaptiveClauseFlowRun(
        converged_to_public_solution=converged,
        continuous_time=float(solution.t[-1]),
        function_evaluations=int(solution.nfev),
        accepted_internal_steps=max(0, len(solution.t) - 1),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
        final_assignment=tuple(sorted(assignment.items())),
        final_max_clause_constraint=max_constraint,
        maximum_long_memory=float(np.max(final_long, initial=1.0)),
        final_state=final_state,
        status=(
            "ADAPTIVE_FLOW_REACHED_PUBLIC_SOLUTION"
            if converged
            else "ADAPTIVE_FLOW_TIME_CAP_REACHED__NO_UNSAT_CONCLUSION"
        ),
    )
