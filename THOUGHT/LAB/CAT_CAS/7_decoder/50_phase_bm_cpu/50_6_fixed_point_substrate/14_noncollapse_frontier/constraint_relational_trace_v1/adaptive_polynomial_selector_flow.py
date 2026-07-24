from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .polynomial_selector_flow import (
    PolynomialSelectorFlowState,
    public_selector_initial_state,
    selector_threshold_assignment,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class AdaptivePolynomialSelectorFlowRun:
    converged_to_public_solution: bool
    continuous_time: float
    function_evaluations: int
    accepted_internal_steps: int
    solver_success: bool
    solver_message: str
    relative_tolerance: float
    absolute_tolerance: float
    maximum_step: float
    final_assignment: tuple[tuple[str, bool], ...]
    final_max_selector_constraint: float
    maximum_long_memory: float
    maximum_voltage_magnitude: float
    maximum_clause_selector_mass_error: float
    maximum_pair_selector_mass_error: float
    final_state: PolynomialSelectorFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


def integrate_adaptive_polynomial_selector_flow(
    holo: ConstraintHolo,
    initial_state: PolynomialSelectorFlowState | None = None,
    maximum_time: float = 20.0,
    relative_tolerance: float = 1.0e-7,
    absolute_tolerance: float = 1.0e-9,
    maximum_step: float = 5.0e-2,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 1.0,
) -> AdaptivePolynomialSelectorFlowRun:
    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover
        raise ConstraintHoloError(
            "adaptive polynomial selector flow requires NumPy and SciPy"
        ) from exc

    controls = (
        maximum_time,
        relative_tolerance,
        absolute_tolerance,
        maximum_step,
        selector_rate,
        boundary_release_rate,
    )
    if not all(isfinite(value) and value > 0 for value in controls):
        raise ConstraintHoloError(
            "adaptive polynomial selector controls must be positive and finite"
        )

    n = len(holo.variables)
    m = len(holo.clauses)
    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    indices = np.asarray(
        [
            [variable_index[literal.variable] for literal in clause.literals]
            for clause in holo.clauses
        ],
        dtype=np.int64,
    )
    signs = np.asarray(
        [
            [1.0 if literal.positive else -1.0 for literal in clause.literals]
            for clause in holo.clauses
        ],
        dtype=np.float64,
    )
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, m))
    initial = initial_state or public_selector_initial_state(holo)
    vector0 = np.asarray(
        initial.voltages
        + initial.short_memory
        + initial.long_memory
        + tuple(value for triple in initial.clause_selector for value in triple)
        + tuple(value for sextuple in initial.pair_selector for value in sextuple),
        dtype=np.float64,
    )

    def unpack(vector):
        voltage = vector[:n]
        short = vector[n : n + m]
        long = vector[n + m : n + 2 * m]
        clause_start = n + 2 * m
        clause_stop = clause_start + 3 * m
        clause_weights = (
            vector[clause_start:clause_stop].reshape((m, 3))
            if m
            else np.empty((0, 3))
        )
        pair_weights = (
            vector[clause_stop:].reshape((m, 3, 2))
            if m
            else np.empty((0, 3, 2))
        )
        return voltage, short, long, clause_weights, pair_weights

    def constraint_values(voltage, clause_weights):
        if m == 0:
            return np.empty(0, dtype=np.float64)
        defects = 1.0 - signs * voltage[indices]
        return 0.5 * np.sum(clause_weights * defects, axis=1)

    def field(_time, vector):
        voltage, short, long, clause_weights, pair_weights = unpack(vector)
        if m == 0:
            return np.zeros_like(vector)

        local_voltage = voltage[indices]
        defects = 1.0 - signs * local_voltage

        clause_weighted_cost = np.sum(clause_weights * defects, axis=1)
        clause_mass = np.sum(clause_weights, axis=1)
        clause_weight_derivative = (
            selector_rate
            * clause_weights
            * (
                clause_weighted_cost[:, None]
                - clause_mass[:, None] * defects
            )
        )
        constraint = 0.5 * clause_weighted_cost

        pair_costs = np.stack(
            (
                np.stack((defects[:, 1], defects[:, 2]), axis=1),
                np.stack((defects[:, 0], defects[:, 2]), axis=1),
                np.stack((defects[:, 0], defects[:, 1]), axis=1),
            ),
            axis=1,
        )
        pair_weighted_cost = np.sum(pair_weights * pair_costs, axis=2)
        pair_mass = np.sum(pair_weights, axis=2)
        pair_weight_derivative = (
            selector_rate
            * pair_weights
            * (
                pair_weighted_cost[:, :, None]
                - pair_mass[:, :, None] * pair_costs
            )
        )

        gradient = 0.5 * signs * pair_weighted_cost
        rigidity = 0.5 * (signs - local_voltage) * clause_weights
        contributions = (
            (long * short)[:, None] * gradient
            + ((1.0 + parameters.zeta * long) * (1.0 - short))[:, None]
            * rigidity
        )
        raw_voltage = np.zeros(n, dtype=np.float64)
        incident_constraint = np.zeros(n, dtype=np.float64)
        np.add.at(raw_voltage, indices.reshape(-1), contributions.reshape(-1))
        np.add.at(
            incident_constraint,
            indices.reshape(-1),
            np.repeat(constraint, 3),
        )
        voltage_derivative = (
            raw_voltage * (1.0 - voltage**2)
            - boundary_release_rate * voltage * incident_constraint
        )
        short_derivative = (
            parameters.beta
            * (constraint - parameters.gamma)
            * short
            * (1.0 - short)
        )
        long_derivative = (
            parameters.alpha
            * (constraint - parameters.delta)
            * long
            * (1.0 - long / cap)
        )
        return np.concatenate(
            (
                voltage_derivative,
                short_derivative,
                long_derivative,
                clause_weight_derivative.reshape(-1),
                pair_weight_derivative.reshape(-1),
            )
        )

    solution = solve_ivp(
        field,
        (0.0, maximum_time),
        vector0,
        method="RK45",
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )

    selected_index = len(solution.t) - 1
    selected_assignment: dict[str, bool] | None = None
    for index in range(len(solution.t)):
        voltage, short, long, clause_weights, pair_weights = unpack(
            solution.y[:, index]
        )
        state = PolynomialSelectorFlowState(
            voltages=tuple(float(value) for value in voltage),
            short_memory=tuple(float(value) for value in short),
            long_memory=tuple(float(value) for value in long),
            clause_selector=tuple(
                tuple(float(value) for value in triple)  # type: ignore[misc]
                for triple in clause_weights
            ),
            pair_selector=tuple(
                tuple(float(value) for value in sextuple.reshape(-1))  # type: ignore[misc]
                for sextuple in pair_weights
            ),
        )
        assignment = selector_threshold_assignment(holo, state)
        if holo.accepts(assignment):
            selected_index = index
            selected_assignment = assignment
            break

    voltage, short, long, clause_weights, pair_weights = unpack(
        solution.y[:, selected_index]
    )
    final_state = PolynomialSelectorFlowState(
        voltages=tuple(float(value) for value in voltage),
        short_memory=tuple(float(value) for value in short),
        long_memory=tuple(float(value) for value in long),
        clause_selector=tuple(
            tuple(float(value) for value in triple)  # type: ignore[misc]
            for triple in clause_weights
        ),
        pair_selector=tuple(
            tuple(float(value) for value in sextuple.reshape(-1))  # type: ignore[misc]
            for sextuple in pair_weights
        ),
    )
    assignment = selected_assignment or selector_threshold_assignment(holo, final_state)
    constraints = constraint_values(voltage, clause_weights)
    converged = holo.accepts(assignment)

    all_clause_mass = np.sum(
        solution.y[n + 2 * m : n + 5 * m, :].reshape((m, 3, -1)),
        axis=1,
    ) if m else np.empty((0, len(solution.t)))
    all_pair_mass = np.sum(
        solution.y[n + 5 * m :, :].reshape((m, 3, 2, -1)),
        axis=2,
    ) if m else np.empty((0, 3, len(solution.t)))

    return AdaptivePolynomialSelectorFlowRun(
        converged_to_public_solution=converged,
        continuous_time=float(solution.t[selected_index]),
        function_evaluations=int(solution.nfev),
        accepted_internal_steps=selected_index,
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
        final_assignment=tuple(sorted(assignment.items())),
        final_max_selector_constraint=float(np.max(constraints, initial=0.0)),
        maximum_long_memory=float(np.max(solution.y[n + m : n + 2 * m, :], initial=1.0)),
        maximum_voltage_magnitude=float(np.max(np.abs(solution.y[:n, :]), initial=0.0)),
        maximum_clause_selector_mass_error=float(
            np.max(np.abs(all_clause_mass - 1.0), initial=0.0)
        ),
        maximum_pair_selector_mass_error=float(
            np.max(np.abs(all_pair_mass - 1.0), initial=0.0)
        ),
        final_state=final_state,
        status=(
            "ADAPTIVE_POLYNOMIAL_SELECTOR_FLOW_REACHED_VERIFIED_PUBLIC_SOLUTION"
            if converged
            else (
                "ADAPTIVE_POLYNOMIAL_SELECTOR_FLOW_TIME_CAP_REACHED__NO_UNSAT_CONCLUSION"
                if solution.success
                else "ADAPTIVE_POLYNOMIAL_SELECTOR_FLOW_SOLVER_FAILED__NO_UNSAT_CONCLUSION"
            )
        ),
    )
