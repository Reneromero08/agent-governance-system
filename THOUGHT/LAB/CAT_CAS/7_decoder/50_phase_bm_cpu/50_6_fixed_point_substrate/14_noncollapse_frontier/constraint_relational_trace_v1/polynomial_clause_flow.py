from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class PolynomialClauseFlowParameters:
    alpha: float = 5.0
    beta: float = 20.0
    gamma: float = 0.25
    delta: float = 0.05
    zeta: float = 0.1
    selector_rate: float = 20.0
    boundary_release_rate: float = 1.0
    long_memory_cap_factor: float = 1.0e4

    def __post_init__(self) -> None:
        values = (
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.zeta,
            self.selector_rate,
            self.boundary_release_rate,
            self.long_memory_cap_factor,
        )
        if not all(isfinite(value) and value > 0 for value in values):
            raise ConstraintHoloError("polynomial flow parameters must be positive and finite")
        if not self.delta < self.gamma < 1:
            raise ConstraintHoloError("polynomial flow thresholds are inconsistent")


@dataclass(frozen=True)
class PolynomialClauseFlowState:
    voltages: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    selector_weights: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class PolynomialClauseFlowDerivative:
    voltages: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    selector_weights: tuple[tuple[float, float, float], ...]

    def max_abs(self) -> float:
        values = list(self.voltages + self.short_memory + self.long_memory)
        for triple in self.selector_weights:
            values.extend(triple)
        return max((abs(value) for value in values), default=0.0)


@dataclass(frozen=True)
class PolynomialClauseFlowAudit:
    public_variables: int
    public_clauses: int
    state_coordinates: int
    literal_couplings: int
    polynomial_degree_upper_bound: int
    selector_simplex_preserved: bool
    bounded_voltage_barrier: bool
    bounded_memory_barriers: bool
    satisfying_boolean_voltage_outward_force_zero: bool
    nonsatisfying_boolean_stationary_count: int
    polynomial_ode_normal_form_status: str
    global_convergence_status: str
    standard_model_transfer_status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class PolynomialClauseFlowRun:
    steps_executed: int
    converged_to_public_solution: bool
    final_assignment: tuple[tuple[str, bool], ...]
    final_max_clause_violation: float
    final_state: PolynomialClauseFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


def public_polynomial_initial_state(
    holo: ConstraintHolo,
    perturbation: float = 1.0e-2,
) -> PolynomialClauseFlowState:
    if not isfinite(perturbation) or not 0 < perturbation < 1:
        raise ConstraintHoloError("polynomial flow perturbation must lie in (0,1)")
    count = max(1, len(holo.variables))
    voltage = tuple(
        (1.0 if index % 2 == 0 else -1.0)
        * perturbation
        * (index + 1)
        / count
        for index, _variable in enumerate(holo.variables)
    )
    return PolynomialClauseFlowState(
        voltages=voltage,
        short_memory=tuple(0.5 for _ in holo.clauses),
        long_memory=tuple(1.0 for _ in holo.clauses),
        selector_weights=tuple((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) for _ in holo.clauses),
    )


def _validate_state(holo: ConstraintHolo, state: PolynomialClauseFlowState) -> None:
    if len(state.voltages) != len(holo.variables):
        raise ConstraintHoloError("polynomial flow voltage count mismatch")
    if len(state.short_memory) != len(holo.clauses):
        raise ConstraintHoloError("polynomial flow short-memory count mismatch")
    if len(state.long_memory) != len(holo.clauses):
        raise ConstraintHoloError("polynomial flow long-memory count mismatch")
    if len(state.selector_weights) != len(holo.clauses):
        raise ConstraintHoloError("polynomial flow selector count mismatch")


def polynomial_clause_violation_values(
    holo: ConstraintHolo,
    state: PolynomialClauseFlowState,
) -> tuple[float, ...]:
    voltage = dict(zip(holo.variables, state.voltages, strict=True))
    values: list[float] = []
    for clause in holo.clauses:
        defects = tuple(
            1.0 - (1.0 if literal.positive else -1.0) * voltage[literal.variable]
            for literal in clause.literals
        )
        values.append(defects[0] * defects[1] * defects[2] / 8.0)
    return tuple(values)


def polynomial_clause_flow_derivative(
    holo: ConstraintHolo,
    state: PolynomialClauseFlowState,
    parameters: PolynomialClauseFlowParameters = PolynomialClauseFlowParameters(),
) -> PolynomialClauseFlowDerivative:
    _validate_state(holo, state)
    voltage_index = {variable: index for index, variable in enumerate(holo.variables)}
    voltage_derivative = [0.0 for _ in holo.variables]
    incident_violation = [0.0 for _ in holo.variables]
    short_derivative: list[float] = []
    long_derivative: list[float] = []
    selector_derivative: list[tuple[float, float, float]] = []
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))

    for clause_index, clause in enumerate(holo.clauses):
        indices = tuple(voltage_index[literal.variable] for literal in clause.literals)
        signs = tuple(1.0 if literal.positive else -1.0 for literal in clause.literals)
        local_voltage = tuple(state.voltages[index] for index in indices)
        defects = tuple(1.0 - sign * value for sign, value in zip(signs, local_voltage, strict=True))
        violation = defects[0] * defects[1] * defects[2] / 8.0
        weights = state.selector_weights[clause_index]
        weighted_defect = sum(
            weight * defect for weight, defect in zip(weights, defects, strict=True)
        )
        selector_mass = sum(weights)
        selector_derivative.append(
            tuple(
                parameters.selector_rate
                * weight
                * (weighted_defect - selector_mass * defect)
                for weight, defect in zip(weights, defects, strict=True)
            )  # type: ignore[arg-type]
        )

        short_memory = state.short_memory[clause_index]
        long_memory = state.long_memory[clause_index]
        for variable_index in indices:
            incident_violation[variable_index] += violation

        for literal_index, variable_index in enumerate(indices):
            other = [defects[index] for index in range(3) if index != literal_index]
            gradient = signs[literal_index] * other[0] * other[1] / 4.0
            rigidity = 0.5 * (signs[literal_index] - local_voltage[literal_index]) * weights[literal_index]
            voltage_derivative[variable_index] += (
                long_memory * short_memory * gradient
                + (1.0 + parameters.zeta * long_memory)
                * (1.0 - short_memory)
                * rigidity
            )

        short_derivative.append(
            parameters.beta
            * (violation - parameters.gamma)
            * short_memory
            * (1.0 - short_memory)
        )
        long_derivative.append(
            parameters.alpha
            * (violation - parameters.delta)
            * long_memory
            * (1.0 - long_memory / cap)
        )

    voltage_derivative = [
        derivative * (1.0 - state.voltages[index] ** 2)
        - parameters.boundary_release_rate
        * state.voltages[index]
        * incident_violation[index]
        for index, derivative in enumerate(voltage_derivative)
    ]
    return PolynomialClauseFlowDerivative(
        voltages=tuple(voltage_derivative),
        short_memory=tuple(short_derivative),
        long_memory=tuple(long_derivative),
        selector_weights=tuple(selector_derivative),
    )


def polynomial_euler_step(
    holo: ConstraintHolo,
    state: PolynomialClauseFlowState,
    step_size: float,
    parameters: PolynomialClauseFlowParameters = PolynomialClauseFlowParameters(),
) -> PolynomialClauseFlowState:
    if not isfinite(step_size) or step_size <= 0:
        raise ConstraintHoloError("polynomial flow step must be positive and finite")
    derivative = polynomial_clause_flow_derivative(holo, state, parameters)
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))
    selectors: list[tuple[float, float, float]] = []
    for weights, deltas in zip(state.selector_weights, derivative.selector_weights, strict=True):
        raw = tuple(max(0.0, weight + step_size * delta) for weight, delta in zip(weights, deltas, strict=True))
        total = sum(raw)
        selectors.append(tuple(value / total for value in raw))  # type: ignore[arg-type]

    return PolynomialClauseFlowState(
        voltages=tuple(
            max(-1.0, min(1.0, value + step_size * delta))
            for value, delta in zip(state.voltages, derivative.voltages, strict=True)
        ),
        short_memory=tuple(
            max(0.0, min(1.0, value + step_size * delta))
            for value, delta in zip(state.short_memory, derivative.short_memory, strict=True)
        ),
        long_memory=tuple(
            max(0.0, min(cap, value + step_size * delta))
            for value, delta in zip(state.long_memory, derivative.long_memory, strict=True)
        ),
        selector_weights=tuple(selectors),
    )


def polynomial_threshold_assignment(
    holo: ConstraintHolo,
    state: PolynomialClauseFlowState,
) -> dict[str, bool]:
    return {
        variable: state.voltages[index] > 0.0
        for index, variable in enumerate(holo.variables)
    }


def integrate_polynomial_flow_until_solution(
    holo: ConstraintHolo,
    initial_state: PolynomialClauseFlowState | None = None,
    step_size: float = 1.0e-3,
    max_steps: int = 100_000,
    parameters: PolynomialClauseFlowParameters = PolynomialClauseFlowParameters(),
) -> PolynomialClauseFlowRun:
    if max_steps < 1:
        raise ConstraintHoloError("polynomial flow step cap must be positive")
    state = initial_state or public_polynomial_initial_state(holo)
    for step in range(max_steps + 1):
        assignment = polynomial_threshold_assignment(holo, state)
        violations = polynomial_clause_violation_values(holo, state)
        if holo.accepts(assignment):
            return PolynomialClauseFlowRun(
                steps_executed=step,
                converged_to_public_solution=True,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_clause_violation=max(violations, default=0.0),
                final_state=state,
                status="POLYNOMIAL_FLOW_REACHED_VERIFIED_PUBLIC_SOLUTION",
            )
        if step == max_steps:
            return PolynomialClauseFlowRun(
                steps_executed=step,
                converged_to_public_solution=False,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_clause_violation=max(violations, default=0.0),
                final_state=state,
                status="POLYNOMIAL_FLOW_STEP_CAP_REACHED__NO_UNSAT_CONCLUSION",
            )
        state = polynomial_euler_step(holo, state, step_size, parameters)
    raise AssertionError("unreachable polynomial flow termination")


def audit_polynomial_clause_flow(holo: ConstraintHolo) -> PolynomialClauseFlowAudit:
    variable_count = len(holo.variables)
    clause_count = len(holo.clauses)
    state_coordinates = variable_count + 2 * clause_count + 3 * clause_count
    nonsatisfying_stationary = 0
    if variable_count <= 12:
        for bits in range(1 << variable_count):
            assignment = {
                variable: bool((bits >> index) & 1)
                for index, variable in enumerate(holo.variables)
            }
            if holo.accepts(assignment):
                continue
            state = PolynomialClauseFlowState(
                voltages=tuple(1.0 if assignment[variable] else -1.0 for variable in holo.variables),
                short_memory=tuple(0.5 for _ in holo.clauses),
                long_memory=tuple(1.0 for _ in holo.clauses),
                selector_weights=tuple((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) for _ in holo.clauses),
            )
            nonsatisfying_stationary += int(
                polynomial_clause_flow_derivative(holo, state).max_abs() <= 1.0e-12
            )

    return PolynomialClauseFlowAudit(
        public_variables=variable_count,
        public_clauses=clause_count,
        state_coordinates=state_coordinates,
        literal_couplings=3 * clause_count,
        polynomial_degree_upper_bound=6,
        selector_simplex_preserved=True,
        bounded_voltage_barrier=True,
        bounded_memory_barriers=True,
        satisfying_boolean_voltage_outward_force_zero=True,
        nonsatisfying_boolean_stationary_count=nonsatisfying_stationary,
        polynomial_ode_normal_form_status="POLYNOMIAL_VECTOR_FIELD_WITH_PUBLIC_RATIONAL_PARAMETERS",
        global_convergence_status="NOT_ESTABLISHED",
        standard_model_transfer_status="POLYNOMIAL_LENGTH_THEOREM_APPLICABLE_IF_DEADLINE_AND_ROBUSTNESS_HOLD",
    )
