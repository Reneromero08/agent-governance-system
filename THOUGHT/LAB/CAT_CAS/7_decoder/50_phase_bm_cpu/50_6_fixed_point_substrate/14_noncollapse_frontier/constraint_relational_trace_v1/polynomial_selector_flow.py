from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class PolynomialSelectorFlowState:
    voltages: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    clause_selector: tuple[tuple[float, float, float], ...]
    pair_selector: tuple[tuple[float, float, float, float, float, float], ...]


@dataclass(frozen=True)
class PolynomialSelectorFlowDerivative:
    voltages: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    clause_selector: tuple[tuple[float, float, float], ...]
    pair_selector: tuple[tuple[float, float, float, float, float, float], ...]

    def max_abs(self) -> float:
        values = list(self.voltages + self.short_memory + self.long_memory)
        for triple in self.clause_selector:
            values.extend(triple)
        for sextuple in self.pair_selector:
            values.extend(sextuple)
        return max((abs(value) for value in values), default=0.0)


@dataclass(frozen=True)
class PolynomialSelectorFlowRun:
    steps_executed: int
    converged_to_public_solution: bool
    final_assignment: tuple[tuple[str, bool], ...]
    final_max_selector_constraint: float
    final_state: PolynomialSelectorFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class PolynomialSelectorFlowAudit:
    public_variables: int
    public_clauses: int
    state_coordinates: int
    polynomial_degree_upper_bound: int
    clause_selector_mass_preserved: bool
    pair_selector_mass_preserved: bool
    public_rational_initial_state: bool
    wrong_boolean_corner_release_present: bool
    min_selector_limit_status: str
    polynomial_ode_status: str
    global_convergence_status: str
    claim_ceiling: str = CLAIM_CEILING


def public_selector_initial_state(
    holo: ConstraintHolo,
    perturbation: float = 1.0e-2,
) -> PolynomialSelectorFlowState:
    if not isfinite(perturbation) or not 0 < perturbation < 1:
        raise ConstraintHoloError("selector-flow perturbation must lie in (0,1)")
    count = max(1, len(holo.variables))
    voltages = tuple(
        (1.0 if index % 2 == 0 else -1.0)
        * perturbation
        * (index + 1)
        / count
        for index, _variable in enumerate(holo.variables)
    )
    return PolynomialSelectorFlowState(
        voltages=voltages,
        short_memory=tuple(0.5 for _ in holo.clauses),
        long_memory=tuple(1.0 for _ in holo.clauses),
        clause_selector=tuple((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) for _ in holo.clauses),
        pair_selector=tuple((0.5, 0.5, 0.5, 0.5, 0.5, 0.5) for _ in holo.clauses),
    )


def selector_clause_values(
    holo: ConstraintHolo,
    state: PolynomialSelectorFlowState,
) -> tuple[float, ...]:
    voltage = dict(zip(holo.variables, state.voltages, strict=True))
    values: list[float] = []
    for clause_index, clause in enumerate(holo.clauses):
        defects = tuple(
            1.0 - (1.0 if literal.positive else -1.0) * voltage[literal.variable]
            for literal in clause.literals
        )
        weights = state.clause_selector[clause_index]
        values.append(0.5 * sum(weight * defect for weight, defect in zip(weights, defects, strict=True)))
    return tuple(values)


def _replicator_derivative(
    weights: tuple[float, ...],
    costs: tuple[float, ...],
    rate: float,
) -> tuple[float, ...]:
    weighted_cost = sum(weight * cost for weight, cost in zip(weights, costs, strict=True))
    mass = sum(weights)
    return tuple(
        rate * weight * (weighted_cost - mass * cost)
        for weight, cost in zip(weights, costs, strict=True)
    )


def polynomial_selector_flow_derivative(
    holo: ConstraintHolo,
    state: PolynomialSelectorFlowState,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 1.0,
) -> PolynomialSelectorFlowDerivative:
    if not isfinite(selector_rate) or selector_rate <= 0:
        raise ConstraintHoloError("selector rate must be positive and finite")
    if not isfinite(boundary_release_rate) or boundary_release_rate <= 0:
        raise ConstraintHoloError("boundary release rate must be positive and finite")
    n = len(holo.variables)
    m = len(holo.clauses)
    if len(state.voltages) != n or len(state.short_memory) != m or len(state.long_memory) != m:
        raise ConstraintHoloError("selector-flow state dimension mismatch")
    if len(state.clause_selector) != m or len(state.pair_selector) != m:
        raise ConstraintHoloError("selector-flow selector dimension mismatch")

    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    voltage_derivative = [0.0 for _ in holo.variables]
    incident_constraint = [0.0 for _ in holo.variables]
    short_derivative: list[float] = []
    long_derivative: list[float] = []
    clause_selector_derivative: list[tuple[float, float, float]] = []
    pair_selector_derivative: list[tuple[float, float, float, float, float, float]] = []
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, m))

    for clause_index, clause in enumerate(holo.clauses):
        indices = tuple(variable_index[literal.variable] for literal in clause.literals)
        signs = tuple(1.0 if literal.positive else -1.0 for literal in clause.literals)
        local_voltage = tuple(state.voltages[index] for index in indices)
        defects = tuple(1.0 - sign * value for sign, value in zip(signs, local_voltage, strict=True))
        global_weights = state.clause_selector[clause_index]
        pair_weights = state.pair_selector[clause_index]
        constraint = 0.5 * sum(
            weight * defect for weight, defect in zip(global_weights, defects, strict=True)
        )
        clause_selector_derivative.append(
            _replicator_derivative(global_weights, defects, selector_rate)  # type: ignore[arg-type]
        )

        pair_derivatives: list[float] = []
        pair_minima: list[float] = []
        for literal_index in range(3):
            others = tuple(index for index in range(3) if index != literal_index)
            weights = (
                pair_weights[2 * literal_index],
                pair_weights[2 * literal_index + 1],
            )
            costs = (defects[others[0]], defects[others[1]])
            pair_derivatives.extend(_replicator_derivative(weights, costs, selector_rate))
            pair_minima.append(sum(weight * cost for weight, cost in zip(weights, costs, strict=True)))
        pair_selector_derivative.append(tuple(pair_derivatives))  # type: ignore[arg-type]

        short_memory = state.short_memory[clause_index]
        long_memory = state.long_memory[clause_index]
        for variable_index_local in indices:
            incident_constraint[variable_index_local] += constraint
        for literal_index, variable_index_local in enumerate(indices):
            gradient = 0.5 * signs[literal_index] * pair_minima[literal_index]
            rigidity = (
                0.5
                * (signs[literal_index] - local_voltage[literal_index])
                * global_weights[literal_index]
            )
            voltage_derivative[variable_index_local] += (
                long_memory * short_memory * gradient
                + (1.0 + parameters.zeta * long_memory)
                * (1.0 - short_memory)
                * rigidity
            )

        short_derivative.append(
            parameters.beta
            * (constraint - parameters.gamma)
            * short_memory
            * (1.0 - short_memory)
        )
        long_derivative.append(
            parameters.alpha
            * (constraint - parameters.delta)
            * long_memory
            * (1.0 - long_memory / cap)
        )

    voltage_derivative = [
        derivative * (1.0 - state.voltages[index] ** 2)
        - boundary_release_rate * state.voltages[index] * incident_constraint[index]
        for index, derivative in enumerate(voltage_derivative)
    ]
    return PolynomialSelectorFlowDerivative(
        voltages=tuple(voltage_derivative),
        short_memory=tuple(short_derivative),
        long_memory=tuple(long_derivative),
        clause_selector=tuple(clause_selector_derivative),
        pair_selector=tuple(pair_selector_derivative),
    )


def selector_euler_step(
    holo: ConstraintHolo,
    state: PolynomialSelectorFlowState,
    step_size: float,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 1.0,
) -> PolynomialSelectorFlowState:
    if not isfinite(step_size) or step_size <= 0:
        raise ConstraintHoloError("selector-flow step must be positive and finite")
    derivative = polynomial_selector_flow_derivative(
        holo,
        state,
        parameters,
        selector_rate,
        boundary_release_rate,
    )
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))

    def updated_simplex(weights, deltas):
        raw = tuple(
            max(0.0, weight + step_size * delta)
            for weight, delta in zip(weights, deltas, strict=True)
        )
        total = sum(raw)
        if total <= 0.0:
            raise ConstraintHoloError("selector simplex lost all mass")
        return tuple(value / total for value in raw)

    def updated_pair_selectors(weights, deltas):
        pairs: list[float] = []
        for pair_index in range(3):
            start = 2 * pair_index
            pair = updated_simplex(
                weights[start : start + 2],
                deltas[start : start + 2],
            )
            pairs.extend(pair)
        return tuple(pairs)

    return PolynomialSelectorFlowState(
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
        clause_selector=tuple(
            updated_simplex(weights, deltas)  # type: ignore[arg-type]
            for weights, deltas in zip(state.clause_selector, derivative.clause_selector, strict=True)
        ),
        pair_selector=tuple(
            updated_pair_selectors(weights, deltas)  # type: ignore[arg-type]
            for weights, deltas in zip(
                state.pair_selector,
                derivative.pair_selector,
                strict=True,
            )
        ),
    )


def selector_threshold_assignment(
    holo: ConstraintHolo,
    state: PolynomialSelectorFlowState,
) -> dict[str, bool]:
    return {
        variable: state.voltages[index] > 0.0
        for index, variable in enumerate(holo.variables)
    }


def integrate_polynomial_selector_flow(
    holo: ConstraintHolo,
    initial_state: PolynomialSelectorFlowState | None = None,
    step_size: float = 1.0e-3,
    max_steps: int = 100_000,
) -> PolynomialSelectorFlowRun:
    state = initial_state or public_selector_initial_state(holo)
    for step in range(max_steps + 1):
        assignment = selector_threshold_assignment(holo, state)
        constraints = selector_clause_values(holo, state)
        if holo.accepts(assignment):
            return PolynomialSelectorFlowRun(
                steps_executed=step,
                converged_to_public_solution=True,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_selector_constraint=max(constraints, default=0.0),
                final_state=state,
                status="POLYNOMIAL_SELECTOR_FLOW_REACHED_VERIFIED_PUBLIC_SOLUTION",
            )
        if step == max_steps:
            return PolynomialSelectorFlowRun(
                steps_executed=step,
                converged_to_public_solution=False,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_selector_constraint=max(constraints, default=0.0),
                final_state=state,
                status="POLYNOMIAL_SELECTOR_FLOW_STEP_CAP_REACHED__NO_UNSAT_CONCLUSION",
            )
        state = selector_euler_step(holo, state, step_size)
    raise AssertionError("unreachable selector-flow termination")


def audit_polynomial_selector_flow(holo: ConstraintHolo) -> PolynomialSelectorFlowAudit:
    m = len(holo.clauses)
    return PolynomialSelectorFlowAudit(
        public_variables=len(holo.variables),
        public_clauses=m,
        state_coordinates=len(holo.variables) + 11 * m,
        polynomial_degree_upper_bound=6,
        clause_selector_mass_preserved=True,
        pair_selector_mass_preserved=True,
        public_rational_initial_state=True,
        wrong_boolean_corner_release_present=True,
        min_selector_limit_status="REPLICATOR_WEIGHTS_CONVERGE_TOWARD_LOCAL_MINIMA_WITHOUT_DIVISION",
        polynomial_ode_status="PUBLIC_RATIONAL_POLYNOMIAL_VECTOR_FIELD",
        global_convergence_status="NOT_ESTABLISHED",
    )
