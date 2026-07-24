from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    REFERENCE_VARIABLE_LIMIT,
    iter_boundary_assignments,
)
from .constraint_holo import ClauseRelation, ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class SelfOrganizingFlowParameters:
    """Public parameters for the terminal-agnostic clause flow.

    The equations are the clause-local memory-assisted dynamics introduced for
    self-organizing 3-SAT circuits. The software integrator is instrumentation. The
    candidate compute object is the continuous relational flow itself.
    """

    alpha: float = 5.0
    beta: float = 20.0
    gamma: float = 0.25
    delta: float = 0.05
    epsilon: float = 1.0e-3
    zeta: float = 0.1
    long_memory_cap_factor: float = 1.0e4

    def __post_init__(self) -> None:
        values = (
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.epsilon,
            self.zeta,
            self.long_memory_cap_factor,
        )
        if not all(isfinite(value) for value in values):
            raise ConstraintHoloError("self-organizing flow parameters must be finite")
        if self.alpha <= 0 or self.beta <= 0 or self.epsilon <= 0:
            raise ConstraintHoloError("flow rates and epsilon must be positive")
        if not 0 <= self.delta < self.gamma < 0.5:
            raise ConstraintHoloError("flow thresholds must satisfy 0 <= delta < gamma < 1/2")
        if self.zeta < 0 or self.long_memory_cap_factor < 1:
            raise ConstraintHoloError("flow gain parameters are outside the public domain")


@dataclass(frozen=True)
class SelfOrganizingFlowState:
    voltages: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]


@dataclass(frozen=True)
class SelfOrganizingFlowDerivative:
    voltages: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]

    def max_abs(self) -> float:
        values = self.voltages + self.short_memory + self.long_memory
        return max((abs(value) for value in values), default=0.0)


@dataclass(frozen=True)
class ReferenceClauseFlowRun:
    steps_executed: int
    converged_to_public_solution: bool
    final_assignment: tuple[tuple[str, bool], ...]
    final_max_clause_constraint: float
    final_state: SelfOrganizingFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class SelfOrganizingClauseFlowAudit:
    public_variables: int
    public_clauses: int
    polynomial_state_coordinates: int
    local_literal_couplings: int
    satisfying_boolean_corners: int
    nonsatisfying_boolean_corners: int
    satisfying_corners_stationary: bool
    nonsatisfying_stationary_corners: int
    all_boolean_stationary_points_are_solutions: bool
    bounded_public_coordinates: bool
    terminal_agnostic_relation_status: str
    nonboolean_equilibrium_exclusion_status: str
    worst_case_convergence_status: str
    unsat_total_boundary_status: str
    native_restoration_status: str
    standard_model_transfer_status: str
    claim_ceiling: str = CLAIM_CEILING


def _literal_sign(clause: ClauseRelation, literal_index: int) -> float:
    return 1.0 if clause.literals[literal_index].positive else -1.0


def _validate_state(
    holo: ConstraintHolo,
    state: SelfOrganizingFlowState,
    parameters: SelfOrganizingFlowParameters,
) -> None:
    if len(state.voltages) != len(holo.variables):
        raise ConstraintHoloError("voltage coordinate count does not match public variables")
    if len(state.short_memory) != len(holo.clauses):
        raise ConstraintHoloError("short-memory coordinate count does not match clauses")
    if len(state.long_memory) != len(holo.clauses):
        raise ConstraintHoloError("long-memory coordinate count does not match clauses")
    if not all(-1.0 <= value <= 1.0 for value in state.voltages):
        raise ConstraintHoloError("voltage coordinates must remain in [-1, 1]")
    if not all(0.0 <= value <= 1.0 for value in state.short_memory):
        raise ConstraintHoloError("short-memory coordinates must remain in [0, 1]")
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))
    if not all(1.0 <= value <= cap for value in state.long_memory):
        raise ConstraintHoloError("long-memory coordinates exceed their public bounds")


def clause_constraint_values(
    holo: ConstraintHolo,
    state: SelfOrganizingFlowState,
) -> tuple[float, ...]:
    voltage = dict(zip(holo.variables, state.voltages, strict=True))
    values: list[float] = []
    for clause in holo.clauses:
        literal_defects = tuple(
            1.0
            - (1.0 if literal.positive else -1.0) * voltage[literal.variable]
            for literal in clause.literals
        )
        values.append(0.5 * min(literal_defects))
    return tuple(values)


def _project_outward(value: float, derivative: float, lower: float, upper: float) -> float:
    if value <= lower and derivative < 0:
        return 0.0
    if value >= upper and derivative > 0:
        return 0.0
    return derivative


def self_organizing_flow_derivative(
    holo: ConstraintHolo,
    state: SelfOrganizingFlowState,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> SelfOrganizingFlowDerivative:
    """Evaluate the public clause-local vector field.

    Every term is compiled from one literal occurrence, a shared variable voltage, and
    two per-clause memory coordinates. No assignment, witness, score winner, or solver
    output enters the vector field.
    """

    _validate_state(holo, state, parameters)
    voltage_index = {variable: index for index, variable in enumerate(holo.variables)}
    constraints = clause_constraint_values(holo, state)
    voltage_derivative = [0.0 for _ in holo.variables]

    for clause_index, clause in enumerate(holo.clauses):
        literal_defects = tuple(
            1.0
            - _literal_sign(clause, literal_index)
            * state.voltages[voltage_index[literal.variable]]
            for literal_index, literal in enumerate(clause.literals)
        )
        minimum_defect = min(literal_defects)
        short_memory = state.short_memory[clause_index]
        long_memory = state.long_memory[clause_index]

        for literal_index, literal in enumerate(clause.literals):
            other_defects = tuple(
                defect
                for other_index, defect in enumerate(literal_defects)
                if other_index != literal_index
            )
            sign = _literal_sign(clause, literal_index)
            variable_index = voltage_index[literal.variable]
            voltage = state.voltages[variable_index]
            gradient_term = 0.5 * sign * min(other_defects)
            rigidity_term = (
                0.5 * (sign - voltage)
                if abs(literal_defects[literal_index] - minimum_defect) <= 1.0e-12
                else 0.0
            )
            voltage_derivative[variable_index] += (
                long_memory * short_memory * gradient_term
                + (1.0 + parameters.zeta * long_memory)
                * (1.0 - short_memory)
                * rigidity_term
            )

    short_derivative = tuple(
        parameters.beta
        * (state.short_memory[index] + parameters.epsilon)
        * (constraint - parameters.gamma)
        for index, constraint in enumerate(constraints)
    )
    long_derivative = tuple(
        parameters.alpha * (constraint - parameters.delta)
        for constraint in constraints
    )
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))

    return SelfOrganizingFlowDerivative(
        voltages=tuple(
            _project_outward(state.voltages[index], derivative, -1.0, 1.0)
            for index, derivative in enumerate(voltage_derivative)
        ),
        short_memory=tuple(
            _project_outward(state.short_memory[index], derivative, 0.0, 1.0)
            for index, derivative in enumerate(short_derivative)
        ),
        long_memory=tuple(
            _project_outward(state.long_memory[index], derivative, 1.0, cap)
            for index, derivative in enumerate(long_derivative)
        ),
    )


def projected_euler_step(
    holo: ConstraintHolo,
    state: SelfOrganizingFlowState,
    step_size: float,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> SelfOrganizingFlowState:
    if not isfinite(step_size) or step_size <= 0:
        raise ConstraintHoloError("reference integration step must be positive and finite")
    derivative = self_organizing_flow_derivative(holo, state, parameters)
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))

    return SelfOrganizingFlowState(
        voltages=tuple(
            min(1.0, max(-1.0, value + step_size * delta))
            for value, delta in zip(state.voltages, derivative.voltages, strict=True)
        ),
        short_memory=tuple(
            min(1.0, max(0.0, value + step_size * delta))
            for value, delta in zip(
                state.short_memory,
                derivative.short_memory,
                strict=True,
            )
        ),
        long_memory=tuple(
            min(cap, max(1.0, value + step_size * delta))
            for value, delta in zip(
                state.long_memory,
                derivative.long_memory,
                strict=True,
            )
        ),
    )


def public_perturbed_initial_state(
    holo: ConstraintHolo,
    perturbation: float = 1.0e-2,
) -> SelfOrganizingFlowState:
    """Create an answer-blind deterministic seed from the public boundary order.

    Exact symmetry can leave a continuous relation at a neutral saddle. The perturbation
    is a declared gauge choice, not hidden randomness or a witness. Renaming variables
    may select a different satisfying section, but must not change SAT/UNSAT truth.
    """

    if not isfinite(perturbation) or not 0 < perturbation < 1:
        raise ConstraintHoloError("public perturbation must lie strictly between zero and one")
    count = max(1, len(holo.variables))
    voltages = tuple(
        (1.0 if index % 2 == 0 else -1.0)
        * perturbation
        * (index + 1)
        / count
        for index, _variable in enumerate(holo.variables)
    )
    return SelfOrganizingFlowState(
        voltages=voltages,
        short_memory=tuple(1.0 for _ in holo.clauses),
        long_memory=tuple(1.0 for _ in holo.clauses),
    )


def threshold_assignment(
    holo: ConstraintHolo,
    state: SelfOrganizingFlowState,
) -> dict[str, bool]:
    _validate_state(holo, state, SelfOrganizingFlowParameters())
    return {
        variable: state.voltages[index] > 0.0
        for index, variable in enumerate(holo.variables)
    }


def integrate_reference_until_solution(
    holo: ConstraintHolo,
    initial_state: SelfOrganizingFlowState | None = None,
    step_size: float = 2.0e-3,
    max_steps: int = 20_000,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> ReferenceClauseFlowRun:
    """Instrument the continuous flow without promoting the integrator as ontology."""

    if max_steps < 1:
        raise ConstraintHoloError("reference flow step cap must be positive")
    state = initial_state or public_perturbed_initial_state(holo)
    _validate_state(holo, state, parameters)

    for step in range(max_steps + 1):
        constraints = clause_constraint_values(holo, state)
        assignment = threshold_assignment(holo, state)
        converged = (
            all(value < 0.5 for value in constraints)
            and holo.accepts(assignment)
        )
        if converged or step == max_steps:
            return ReferenceClauseFlowRun(
                steps_executed=step,
                converged_to_public_solution=converged,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_clause_constraint=max(constraints, default=0.0),
                final_state=state,
                status=(
                    "REFERENCE_FLOW_REACHED_PUBLIC_SOLUTION"
                    if converged
                    else "REFERENCE_FLOW_STEP_CAP_REACHED__NO_UNSAT_CONCLUSION"
                ),
            )
        state = projected_euler_step(holo, state, step_size, parameters)

    raise AssertionError("unreachable reference flow termination")


def boolean_corner_state(
    holo: ConstraintHolo,
    assignment: Mapping[str, bool],
) -> SelfOrganizingFlowState:
    if set(assignment) != set(holo.variables):
        raise ConstraintHoloError("assignment domain must equal the public boundary")
    return SelfOrganizingFlowState(
        voltages=tuple(1.0 if assignment[variable] else -1.0 for variable in holo.variables),
        short_memory=tuple(0.0 for _ in holo.clauses),
        long_memory=tuple(1.0 for _ in holo.clauses),
    )


def audit_self_organizing_clause_flow(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> SelfOrganizingClauseFlowAudit:
    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("self-organizing flow audit exceeds reference limit")

    satisfying_corners = 0
    nonsatisfying_corners = 0
    satisfying_stationary = True
    nonsatisfying_stationary = 0

    for assignment in iter_boundary_assignments(holo):
        accepted = holo.accepts(assignment)
        derivative = self_organizing_flow_derivative(
            holo,
            boolean_corner_state(holo, assignment),
        )
        stationary = derivative.max_abs() <= 1.0e-12
        if accepted:
            satisfying_corners += 1
            satisfying_stationary = satisfying_stationary and stationary
        else:
            nonsatisfying_corners += 1
            nonsatisfying_stationary += int(stationary)

    return SelfOrganizingClauseFlowAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        polynomial_state_coordinates=len(holo.variables) + 2 * len(holo.clauses),
        local_literal_couplings=3 * len(holo.clauses),
        satisfying_boolean_corners=satisfying_corners,
        nonsatisfying_boolean_corners=nonsatisfying_corners,
        satisfying_corners_stationary=satisfying_stationary,
        nonsatisfying_stationary_corners=nonsatisfying_stationary,
        all_boolean_stationary_points_are_solutions=(nonsatisfying_stationary == 0),
        bounded_public_coordinates=True,
        terminal_agnostic_relation_status=(
            "PUBLIC_CLAUSE_LOCAL_MEMORY_ASSISTED_FLOW_COMPILED_WITHOUT_WITNESS"
        ),
        nonboolean_equilibrium_exclusion_status="NOT_ESTABLISHED",
        worst_case_convergence_status="NOT_ESTABLISHED",
        unsat_total_boundary_status="NOT_ESTABLISHED",
        native_restoration_status=(
            "SMOOTH_REGION_COTANGENT_LIFT_AVAILABLE__"
            "PROJECTED_BOUNDARY_AND_SWITCHING_EVENTS_NOT_CLOSED"
        ),
        standard_model_transfer_status="NOT_ESTABLISHED",
    )
