from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class PolynomialPhaseSelectorFlowState:
    phase_cosine: tuple[float, ...]
    phase_sine: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    clause_selector: tuple[tuple[float, float, float], ...]
    pair_selector: tuple[tuple[float, float, float, float, float, float], ...]


@dataclass(frozen=True)
class PolynomialPhaseSelectorFlowDerivative:
    phase_cosine: tuple[float, ...]
    phase_sine: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    clause_selector: tuple[tuple[float, float, float], ...]
    pair_selector: tuple[tuple[float, float, float, float, float, float], ...]

    def max_abs(self) -> float:
        values = list(
            self.phase_cosine
            + self.phase_sine
            + self.short_memory
            + self.long_memory
        )
        for triple in self.clause_selector:
            values.extend(triple)
        for sextuple in self.pair_selector:
            values.extend(sextuple)
        return max((abs(value) for value in values), default=0.0)


@dataclass(frozen=True)
class PolynomialPhaseSelectorFlowRun:
    steps_executed: int
    converged_to_public_solution: bool
    final_assignment: tuple[tuple[str, bool], ...]
    final_max_clause_violation: float
    maximum_circle_residual: float
    final_state: PolynomialPhaseSelectorFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class PolynomialPhaseSelectorFlowAudit:
    public_variables: int
    public_clauses: int
    state_coordinates: int
    polynomial_degree_upper_bound: int
    circle_tangent_identity_exact: bool
    clause_selector_mass_preserved: bool
    pair_selector_mass_preserved: bool
    public_rational_initial_state: bool
    exact_clause_truth_channel: bool
    satisfying_boolean_sections_invariant: bool
    wrong_boolean_corner_release_present: bool
    native_carrier_status: str
    global_convergence_status: str
    claim_ceiling: str = CLAIM_CEILING


def _radical_inverse_base_three(index: int) -> float:
    if index < 1:
        raise ConstraintHoloError("phase low-discrepancy index must be positive")
    remaining = index
    numerator = 0
    denominator = 1
    while remaining:
        remaining, digit = divmod(remaining, 3)
        denominator *= 3
        numerator = numerator * 3 + digit
    return numerator / denominator


def public_phase_selector_initial_state(
    holo: ConstraintHolo,
    perturbation: float = 1.0e-2,
) -> PolynomialPhaseSelectorFlowState:
    """Construct one answer-blind rational point on each public phase circle."""

    if not isfinite(perturbation) or not 0.0 < perturbation < 0.25:
        raise ConstraintHoloError("phase perturbation must lie in (0,1/4)")
    count = max(1, len(holo.variables))
    cosines: list[float] = []
    sines: list[float] = []
    for index, _variable in enumerate(holo.variables):
        low_discrepancy = 2.0 * _radical_inverse_base_three(index + 1) - 1.0
        alternating = (1.0 if index % 2 == 0 else -1.0) * (index + 1) / count
        offset = perturbation * low_discrepancy + perturbation**2 * alternating
        # Rational parametrization of S^1 near the unresolved c=0 chart.
        parameter = 1.0 + offset
        denominator = 1.0 + parameter**2
        cosines.append((1.0 - parameter**2) / denominator)
        sines.append(2.0 * parameter / denominator)
    return PolynomialPhaseSelectorFlowState(
        phase_cosine=tuple(cosines),
        phase_sine=tuple(sines),
        short_memory=tuple(0.5 for _ in holo.clauses),
        long_memory=tuple(1.0 for _ in holo.clauses),
        clause_selector=tuple(
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) for _ in holo.clauses
        ),
        pair_selector=tuple(
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5) for _ in holo.clauses
        ),
    )


def boolean_phase_state(
    holo: ConstraintHolo,
    assignment: Mapping[str, bool],
    *,
    short_memory: float = 0.5,
    long_memory: float = 1.0,
) -> PolynomialPhaseSelectorFlowState:
    if set(assignment) != set(holo.variables):
        raise ConstraintHoloError("phase assignment domain must equal public variables")
    return PolynomialPhaseSelectorFlowState(
        phase_cosine=tuple(1.0 if assignment[v] else -1.0 for v in holo.variables),
        phase_sine=tuple(0.0 for _ in holo.variables),
        short_memory=tuple(short_memory for _ in holo.clauses),
        long_memory=tuple(long_memory for _ in holo.clauses),
        clause_selector=tuple(
            (0.6, 0.3, 0.1) for _ in holo.clauses
        ),
        pair_selector=tuple(
            (0.7, 0.3, 0.4, 0.6, 0.8, 0.2) for _ in holo.clauses
        ),
    )


def _literal_defects(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
) -> tuple[tuple[float, float, float], ...]:
    voltage = dict(zip(holo.variables, state.phase_cosine, strict=True))
    return tuple(
        tuple(
            1.0
            - (1.0 if literal.positive else -1.0) * voltage[literal.variable]
            for literal in clause.literals
        )
        for clause in holo.clauses
    )  # type: ignore[return-value]


def exact_clause_violation_from_defects(
    defects: tuple[float, float, float],
    truth_gain: float = 4.0,
) -> float:
    """Exact polynomial OR violation channel.

    At Boolean phase sections this is zero iff at least one literal is true. Selectors
    never participate in semantic truth; they only choose correction direction.
    """

    if not isfinite(truth_gain) or truth_gain <= 0.0:
        raise ConstraintHoloError("truth gain must be positive and finite")
    return truth_gain * defects[0] * defects[1] * defects[2] / 8.0


def phase_clause_violation_values(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
    truth_gain: float = 4.0,
) -> tuple[float, ...]:
    return tuple(
        exact_clause_violation_from_defects(defects, truth_gain)
        for defects in _literal_defects(holo, state)
    )


def _replicator_derivative(
    weights: tuple[float, ...],
    costs: tuple[float, ...],
    rate: float,
) -> tuple[float, ...]:
    weighted_cost = sum(
        weight * cost for weight, cost in zip(weights, costs, strict=True)
    )
    mass = sum(weights)
    return tuple(
        rate * weight * (weighted_cost - mass * cost)
        for weight, cost in zip(weights, costs, strict=True)
    )


def polynomial_phase_selector_flow_derivative(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> PolynomialPhaseSelectorFlowDerivative:
    controls = (selector_rate, boundary_release_rate, truth_gain)
    if not all(isfinite(value) and value > 0.0 for value in controls):
        raise ConstraintHoloError("phase flow controls must be positive and finite")
    n = len(holo.variables)
    m = len(holo.clauses)
    if (
        len(state.phase_cosine) != n
        or len(state.phase_sine) != n
        or len(state.short_memory) != m
        or len(state.long_memory) != m
        or len(state.clause_selector) != m
        or len(state.pair_selector) != m
    ):
        raise ConstraintHoloError("phase selector state dimension mismatch")

    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    relational_force = [0.0 for _ in holo.variables]
    incident_violation = [0.0 for _ in holo.variables]
    short_derivative: list[float] = []
    long_derivative: list[float] = []
    clause_selector_derivative: list[tuple[float, float, float]] = []
    pair_selector_derivative: list[
        tuple[float, float, float, float, float, float]
    ] = []
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, m))

    for clause_index, clause in enumerate(holo.clauses):
        indices = tuple(variable_index[literal.variable] for literal in clause.literals)
        signs = tuple(1.0 if literal.positive else -1.0 for literal in clause.literals)
        local_cosine = tuple(state.phase_cosine[index] for index in indices)
        defects = tuple(
            1.0 - sign * value
            for sign, value in zip(signs, local_cosine, strict=True)
        )
        violation = exact_clause_violation_from_defects(defects, truth_gain)  # type: ignore[arg-type]
        global_weights = state.clause_selector[clause_index]
        pair_weights = state.pair_selector[clause_index]
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
            pair_derivatives.extend(
                _replicator_derivative(weights, costs, selector_rate)
            )
            pair_minima.append(
                sum(
                    weight * cost
                    for weight, cost in zip(weights, costs, strict=True)
                )
            )
        pair_selector_derivative.append(tuple(pair_derivatives))  # type: ignore[arg-type]

        short_memory = state.short_memory[clause_index]
        long_memory = state.long_memory[clause_index]
        for local_index, variable_index_local in enumerate(indices):
            incident_violation[variable_index_local] += violation
            gradient = 0.5 * signs[local_index] * pair_minima[local_index]
            rigidity = (
                0.5
                * (signs[local_index] - local_cosine[local_index])
                * global_weights[local_index]
            )
            relational_force[variable_index_local] += (
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

    cosine_derivative: list[float] = []
    sine_derivative: list[float] = []
    for cosine, sine, force, violation in zip(
        state.phase_cosine,
        state.phase_sine,
        relational_force,
        incident_violation,
        strict=True,
    ):
        angular_velocity = (
            -sine * force + boundary_release_rate * cosine * violation
        )
        cosine_derivative.append(-sine * angular_velocity)
        sine_derivative.append(cosine * angular_velocity)

    return PolynomialPhaseSelectorFlowDerivative(
        phase_cosine=tuple(cosine_derivative),
        phase_sine=tuple(sine_derivative),
        short_memory=tuple(short_derivative),
        long_memory=tuple(long_derivative),
        clause_selector=tuple(clause_selector_derivative),
        pair_selector=tuple(pair_selector_derivative),
    )


def phase_circle_residual(state: PolynomialPhaseSelectorFlowState) -> float:
    return max(
        (
            abs(cosine**2 + sine**2 - 1.0)
            for cosine, sine in zip(
                state.phase_cosine, state.phase_sine, strict=True
            )
        ),
        default=0.0,
    )


def phase_circle_tangent_residual(
    state: PolynomialPhaseSelectorFlowState,
    derivative: PolynomialPhaseSelectorFlowDerivative,
) -> float:
    return max(
        (
            abs(cosine * dc + sine * ds)
            for cosine, sine, dc, ds in zip(
                state.phase_cosine,
                state.phase_sine,
                derivative.phase_cosine,
                derivative.phase_sine,
                strict=True,
            )
        ),
        default=0.0,
    )


def phase_selector_euler_step(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
    step_size: float,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> PolynomialPhaseSelectorFlowState:
    """Reference chart step; projection is instrumentation, not native evolution."""

    if not isfinite(step_size) or step_size <= 0.0:
        raise ConstraintHoloError("phase selector step must be positive and finite")
    derivative = polynomial_phase_selector_flow_derivative(
        holo,
        state,
        parameters=parameters,
        selector_rate=selector_rate,
        boundary_release_rate=boundary_release_rate,
        truth_gain=truth_gain,
    )
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, len(holo.clauses)))

    cosines: list[float] = []
    sines: list[float] = []
    for cosine, sine, dc, ds in zip(
        state.phase_cosine,
        state.phase_sine,
        derivative.phase_cosine,
        derivative.phase_sine,
        strict=True,
    ):
        raw_cosine = cosine + step_size * dc
        raw_sine = sine + step_size * ds
        norm = sqrt(raw_cosine**2 + raw_sine**2)
        cosines.append(raw_cosine / norm)
        sines.append(raw_sine / norm)

    def updated_simplex(
        weights: tuple[float, ...], deltas: tuple[float, ...]
    ) -> tuple[float, ...]:
        raw = tuple(
            max(0.0, weight + step_size * delta)
            for weight, delta in zip(weights, deltas, strict=True)
        )
        total = sum(raw)
        if total <= 0.0:
            raise ConstraintHoloError("selector chart lost positive simplex mass")
        return tuple(value / total for value in raw)

    pair_selectors: list[tuple[float, float, float, float, float, float]] = []
    for weights, deltas in zip(
        state.pair_selector, derivative.pair_selector, strict=True
    ):
        pairs: list[float] = []
        for index in range(3):
            pair = updated_simplex(
                weights[2 * index : 2 * index + 2],
                deltas[2 * index : 2 * index + 2],
            )
            pairs.extend(pair)
        pair_selectors.append(tuple(pairs))  # type: ignore[arg-type]

    return PolynomialPhaseSelectorFlowState(
        phase_cosine=tuple(cosines),
        phase_sine=tuple(sines),
        short_memory=tuple(
            max(0.0, min(1.0, value + step_size * delta))
            for value, delta in zip(
                state.short_memory, derivative.short_memory, strict=True
            )
        ),
        long_memory=tuple(
            max(0.0, min(cap, value + step_size * delta))
            for value, delta in zip(
                state.long_memory, derivative.long_memory, strict=True
            )
        ),
        clause_selector=tuple(
            updated_simplex(weights, deltas)  # type: ignore[arg-type]
            for weights, deltas in zip(
                state.clause_selector, derivative.clause_selector, strict=True
            )
        ),
        pair_selector=tuple(pair_selectors),
    )


def phase_threshold_assignment(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
) -> dict[str, bool]:
    return {
        variable: state.phase_cosine[index] > 0.0
        for index, variable in enumerate(holo.variables)
    }


def integrate_polynomial_phase_selector_flow(
    holo: ConstraintHolo,
    initial_state: PolynomialPhaseSelectorFlowState | None = None,
    step_size: float = 1.0e-3,
    max_steps: int = 100_000,
) -> PolynomialPhaseSelectorFlowRun:
    """Reference first-passage chart used only for falsification and controls."""

    state = initial_state or public_phase_selector_initial_state(holo)
    maximum_circle_residual = phase_circle_residual(state)
    for step in range(max_steps + 1):
        assignment = phase_threshold_assignment(holo, state)
        violations = phase_clause_violation_values(holo, state)
        if holo.accepts(assignment):
            return PolynomialPhaseSelectorFlowRun(
                steps_executed=step,
                converged_to_public_solution=True,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_clause_violation=max(violations, default=0.0),
                maximum_circle_residual=maximum_circle_residual,
                final_state=state,
                status="POLYNOMIAL_PHASE_SELECTOR_FLOW_REACHED_VERIFIED_PUBLIC_SOLUTION",
            )
        if step == max_steps:
            return PolynomialPhaseSelectorFlowRun(
                steps_executed=step,
                converged_to_public_solution=False,
                final_assignment=tuple(sorted(assignment.items())),
                final_max_clause_violation=max(violations, default=0.0),
                maximum_circle_residual=maximum_circle_residual,
                final_state=state,
                status=(
                    "POLYNOMIAL_PHASE_SELECTOR_FLOW_STEP_CAP_REACHED__"
                    "NO_UNSAT_CONCLUSION"
                ),
            )
        state = phase_selector_euler_step(holo, state, step_size)
        maximum_circle_residual = max(
            maximum_circle_residual, phase_circle_residual(state)
        )
    raise AssertionError("unreachable phase selector termination")


def audit_polynomial_phase_selector_flow(
    holo: ConstraintHolo,
) -> PolynomialPhaseSelectorFlowAudit:
    m = len(holo.clauses)
    return PolynomialPhaseSelectorFlowAudit(
        public_variables=len(holo.variables),
        public_clauses=m,
        state_coordinates=2 * len(holo.variables) + 11 * m,
        polynomial_degree_upper_bound=6,
        circle_tangent_identity_exact=True,
        clause_selector_mass_preserved=True,
        pair_selector_mass_preserved=True,
        public_rational_initial_state=True,
        exact_clause_truth_channel=True,
        satisfying_boolean_sections_invariant=True,
        wrong_boolean_corner_release_present=True,
        native_carrier_status="PUBLIC_RATIONAL_POLYNOMIAL_S1_PHASE_CARRIER",
        global_convergence_status="NOT_ESTABLISHED",
    )
