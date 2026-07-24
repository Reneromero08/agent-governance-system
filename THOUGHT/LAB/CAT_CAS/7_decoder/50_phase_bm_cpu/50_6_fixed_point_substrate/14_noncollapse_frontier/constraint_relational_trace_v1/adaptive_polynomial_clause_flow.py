from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .polynomial_clause_flow import (
    PolynomialClauseFlowParameters,
    PolynomialClauseFlowState,
    polynomial_threshold_assignment,
    public_polynomial_initial_state,
)


@dataclass(frozen=True)
class AdaptivePolynomialFlowRun:
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
    final_max_clause_violation: float
    maximum_long_memory: float
    final_state: PolynomialClauseFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


def integrate_adaptive_polynomial_flow(
    holo: ConstraintHolo,
    initial_state: PolynomialClauseFlowState | None = None,
    maximum_time: float = 20.0,
    relative_tolerance: float = 1.0e-7,
    absolute_tolerance: float = 1.0e-9,
    maximum_step: float = 5.0e-2,
    parameters: PolynomialClauseFlowParameters = PolynomialClauseFlowParameters(),
) -> AdaptivePolynomialFlowRun:
    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover
        raise ConstraintHoloError("adaptive polynomial flow requires NumPy and SciPy") from exc

    controls = (maximum_time, relative_tolerance, absolute_tolerance, maximum_step)
    if not all(isfinite(value) and value > 0 for value in controls):
        raise ConstraintHoloError("adaptive polynomial controls must be positive and finite")

    n = len(holo.variables)
    m = len(holo.clauses)
    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    indices = np.asarray(
        [[variable_index[literal.variable] for literal in clause.literals] for clause in holo.clauses],
        dtype=np.int64,
    )
    signs = np.asarray(
        [[1.0 if literal.positive else -1.0 for literal in clause.literals] for clause in holo.clauses],
        dtype=np.float64,
    )
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, m))
    initial = initial_state or public_polynomial_initial_state(holo)
    vector0 = np.asarray(
        initial.voltages
        + initial.short_memory
        + initial.long_memory
        + tuple(value for triple in initial.selector_weights for value in triple),
        dtype=np.float64,
    )

    def unpack(vector):
        voltage = vector[:n]
        short = vector[n : n + m]
        long = vector[n + m : n + 2 * m]
        weights = vector[n + 2 * m :].reshape((m, 3)) if m else np.empty((0, 3))
        return voltage, short, long, weights

    def violations(voltage):
        if m == 0:
            return np.empty(0, dtype=np.float64)
        defects = 1.0 - signs * voltage[indices]
        return np.prod(defects, axis=1) / 8.0

    def field(_time, vector):
        voltage, short, long, weights = unpack(vector)
        if m == 0:
            return np.zeros_like(vector)
        local_voltage = voltage[indices]
        defects = 1.0 - signs * local_voltage
        violation = np.prod(defects, axis=1) / 8.0
        weighted_defect = np.sum(weights * defects, axis=1)
        selector_mass = np.sum(weights, axis=1)
        weight_derivative = (
            parameters.selector_rate
            * weights
            * (weighted_defect[:, None] - selector_mass[:, None] * defects)
        )

        other_products = np.stack(
            (
                defects[:, 1] * defects[:, 2],
                defects[:, 0] * defects[:, 2],
                defects[:, 0] * defects[:, 1],
            ),
            axis=1,
        )
        gradient = signs * other_products / 4.0
        rigidity = 0.5 * (signs - local_voltage) * weights
        contributions = (
            (long * short)[:, None] * gradient
            + ((1.0 + parameters.zeta * long) * (1.0 - short))[:, None]
            * rigidity
        )
        raw_voltage = np.zeros(n, dtype=np.float64)
        incident_violation = np.zeros(n, dtype=np.float64)
        np.add.at(raw_voltage, indices.reshape(-1), contributions.reshape(-1))
        np.add.at(
            incident_violation,
            indices.reshape(-1),
            np.repeat(violation, 3),
        )
        voltage_derivative = (
            raw_voltage * (1.0 - voltage**2)
            - parameters.boundary_release_rate * voltage * incident_violation
        )
        short_derivative = (
            parameters.beta
            * (violation - parameters.gamma)
            * short
            * (1.0 - short)
        )
        long_derivative = (
            parameters.alpha
            * (violation - parameters.delta)
            * long
            * (1.0 - long / cap)
        )
        return np.concatenate(
            (
                voltage_derivative,
                short_derivative,
                long_derivative,
                weight_derivative.reshape(-1),
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
        voltage, short, long, weights = unpack(solution.y[:, index])
        state = PolynomialClauseFlowState(
            voltages=tuple(float(value) for value in voltage),
            short_memory=tuple(float(value) for value in short),
            long_memory=tuple(float(value) for value in long),
            selector_weights=tuple(
                tuple(float(value) for value in triple)  # type: ignore[misc]
                for triple in weights
            ),
        )
        assignment = polynomial_threshold_assignment(holo, state)
        if holo.accepts(assignment):
            selected_index = index
            selected_assignment = assignment
            break

    voltage, short, long, weights = unpack(solution.y[:, selected_index])
    final_state = PolynomialClauseFlowState(
        voltages=tuple(float(value) for value in voltage),
        short_memory=tuple(float(value) for value in short),
        long_memory=tuple(float(value) for value in long),
        selector_weights=tuple(
            tuple(float(value) for value in triple)  # type: ignore[misc]
            for triple in weights
        ),
    )
    assignment = selected_assignment or polynomial_threshold_assignment(holo, final_state)
    clause_violation = violations(voltage)
    converged = holo.accepts(assignment)

    return AdaptivePolynomialFlowRun(
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
        final_max_clause_violation=float(np.max(clause_violation, initial=0.0)),
        maximum_long_memory=float(np.max(long, initial=1.0)),
        final_state=final_state,
        status=(
            "ADAPTIVE_POLYNOMIAL_FLOW_REACHED_VERIFIED_PUBLIC_SOLUTION"
            if converged
            else (
                "ADAPTIVE_POLYNOMIAL_FLOW_TIME_CAP_REACHED__NO_UNSAT_CONCLUSION"
                if solution.success
                else "ADAPTIVE_POLYNOMIAL_FLOW_SOLVER_FAILED__NO_UNSAT_CONCLUSION"
            )
        ),
    )
