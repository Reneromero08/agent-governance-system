from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, log

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .polynomial_phase_selector_flow import (
    PolynomialPhaseSelectorFlowState,
    phase_clause_violation_values,
    phase_threshold_assignment,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class PhaseLogitCoordinateIdentityAudit:
    maximum_coordinate_residual: float
    coordinate_identity_status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class AdaptivePhaseLogitFlowRun:
    fixed_deadline: float
    terminal_time: float
    reached_fixed_deadline: bool
    terminal_solution_verified: bool
    first_passage_observed: bool
    first_passage_time: float | None
    function_evaluations: int
    accepted_internal_steps: int
    solver_success: bool
    solver_message: str
    solver_exception: str | None
    solver_method: str
    relative_tolerance: float
    absolute_tolerance: float
    maximum_step: float
    terminal_assignment: tuple[tuple[str, bool], ...]
    terminal_max_clause_violation: float
    terminal_clause_satisfaction_margin: float
    maximum_long_memory: float
    maximum_unwrapped_phase_displacement: float
    maximum_short_logit_magnitude: float
    maximum_long_logit_magnitude: float
    maximum_clause_log_ratio_magnitude: float
    maximum_pair_log_ratio_magnitude: float
    minimum_clause_selector_weight: float
    minimum_pair_selector_weight: float
    chart_trajectory_length_lower_bound: float
    phase_trajectory_length_lower_bound: float
    native_trajectory_length_lower_bound: float
    maximum_chart_speed: float
    final_state: PolynomialPhaseSelectorFlowState
    status: str
    claim_ceiling: str = CLAIM_CEILING


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ConstraintHoloError("logit chart requires an interior probability")
    return log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = pow(2.718281828459045, -value)
        return 1.0 / (1.0 + exponential)
    exponential = pow(2.718281828459045, value)
    return exponential / (1.0 + exponential)


def _softmax_reference(ratios: tuple[float, ...]) -> tuple[float, ...]:
    logits = ratios + (0.0,)
    maximum = max(logits)
    exponentials = tuple(pow(2.718281828459045, value - maximum) for value in logits)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _state_to_chart(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
    parameters: SelfOrganizingFlowParameters,
) -> tuple[float, ...]:
    cap = max(
        1.0,
        parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
    )
    angles = tuple(
        atan2(sine, cosine)
        for cosine, sine in zip(
            state.phase_cosine, state.phase_sine, strict=True
        )
    )
    short_logits = tuple(_logit(value) for value in state.short_memory)
    long_logits = tuple(_logit(value / cap) for value in state.long_memory)
    clause_ratios = tuple(
        log(weights[index] / weights[2])
        for weights in state.clause_selector
        for index in range(2)
    )
    pair_ratios = tuple(
        log(weights[2 * index] / weights[2 * index + 1])
        for weights in state.pair_selector
        for index in range(3)
    )
    return angles + short_logits + long_logits + clause_ratios + pair_ratios


def _chart_to_state(
    holo: ConstraintHolo,
    vector: tuple[float, ...],
    parameters: SelfOrganizingFlowParameters,
) -> PolynomialPhaseSelectorFlowState:
    from math import cos, sin

    n = len(holo.variables)
    m = len(holo.clauses)
    expected = n + 7 * m
    if len(vector) != expected:
        raise ConstraintHoloError("phase logit chart dimension mismatch")
    cap = max(1.0, parameters.long_memory_cap_factor * max(1, m))
    angles = vector[:n]
    short_start = n
    long_start = short_start + m
    clause_start = long_start + m
    pair_start = clause_start + 2 * m
    short_logits = vector[short_start:long_start]
    long_logits = vector[long_start:clause_start]
    clause_ratios = vector[clause_start:pair_start]
    pair_ratios = vector[pair_start:]

    clause_selector = tuple(
        _softmax_reference(
            (
                clause_ratios[2 * index],
                clause_ratios[2 * index + 1],
            )
        )
        for index in range(m)
    )
    pair_selector = []
    for clause_index in range(m):
        values: list[float] = []
        for pair_index in range(3):
            ratio = pair_ratios[3 * clause_index + pair_index]
            first, second = _softmax_reference((ratio,))
            values.extend((first, second))
        pair_selector.append(tuple(values))

    return PolynomialPhaseSelectorFlowState(
        phase_cosine=tuple(cos(value) for value in angles),
        phase_sine=tuple(sin(value) for value in angles),
        short_memory=tuple(_sigmoid(value) for value in short_logits),
        long_memory=tuple(cap * _sigmoid(value) for value in long_logits),
        clause_selector=clause_selector,  # type: ignore[arg-type]
        pair_selector=tuple(pair_selector),  # type: ignore[arg-type]
    )


def _chart_derivative(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
    parameters: SelfOrganizingFlowParameters,
    selector_rate: float,
    boundary_release_rate: float,
    truth_gain: float,
) -> tuple[float, ...]:
    derivative = polynomial_phase_selector_flow_derivative(
        holo,
        state,
        parameters=parameters,
        selector_rate=selector_rate,
        boundary_release_rate=boundary_release_rate,
        truth_gain=truth_gain,
    )
    cap = max(
        1.0,
        parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
    )
    angle_derivative = tuple(
        cosine * ds - sine * dc
        for cosine, sine, dc, ds in zip(
            state.phase_cosine,
            state.phase_sine,
            derivative.phase_cosine,
            derivative.phase_sine,
            strict=True,
        )
    )
    short_logit_derivative = tuple(
        delta / (value * (1.0 - value))
        for value, delta in zip(
            state.short_memory, derivative.short_memory, strict=True
        )
    )
    long_logit_derivative = tuple(
        delta / (value * (1.0 - value / cap))
        for value, delta in zip(
            state.long_memory, derivative.long_memory, strict=True
        )
    )
    clause_ratio_derivative = tuple(
        derivative.clause_selector[clause_index][index]
        / state.clause_selector[clause_index][index]
        - derivative.clause_selector[clause_index][2]
        / state.clause_selector[clause_index][2]
        for clause_index in range(len(holo.clauses))
        for index in range(2)
    )
    pair_ratio_derivative = tuple(
        derivative.pair_selector[clause_index][2 * pair_index]
        / state.pair_selector[clause_index][2 * pair_index]
        - derivative.pair_selector[clause_index][2 * pair_index + 1]
        / state.pair_selector[clause_index][2 * pair_index + 1]
        for clause_index in range(len(holo.clauses))
        for pair_index in range(3)
    )
    return (
        angle_derivative
        + short_logit_derivative
        + long_logit_derivative
        + clause_ratio_derivative
        + pair_ratio_derivative
    )


def audit_phase_logit_coordinate_identity(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState | None = None,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> PhaseLogitCoordinateIdentityAudit:
    state = state or public_phase_selector_initial_state(holo)
    chart = _state_to_chart(holo, state, parameters)
    rebuilt = _chart_to_state(holo, chart, parameters)
    values = (
        tuple(abs(left - right) for left, right in zip(state.phase_cosine, rebuilt.phase_cosine, strict=True))
        + tuple(abs(left - right) for left, right in zip(state.phase_sine, rebuilt.phase_sine, strict=True))
        + tuple(abs(left - right) for left, right in zip(state.short_memory, rebuilt.short_memory, strict=True))
        + tuple(abs(left - right) for left, right in zip(state.long_memory, rebuilt.long_memory, strict=True))
        + tuple(
            abs(left - right)
            for left_weights, right_weights in zip(
                state.clause_selector, rebuilt.clause_selector, strict=True
            )
            for left, right in zip(left_weights, right_weights, strict=True)
        )
        + tuple(
            abs(left - right)
            for left_weights, right_weights in zip(
                state.pair_selector, rebuilt.pair_selector, strict=True
            )
            for left, right in zip(left_weights, right_weights, strict=True)
        )
    )
    residual = max(values, default=0.0)
    return PhaseLogitCoordinateIdentityAudit(
        maximum_coordinate_residual=residual,
        coordinate_identity_status=(
            "ANGLE_LOGIT_CHART_MATCHES_NATIVE_POLYNOMIAL_PHASE_FIELD"
            if residual <= 1.0e-12
            else "ANGLE_LOGIT_CHART_NATIVE_PHASE_FIELD_MISMATCH"
        ),
    )


def _clause_satisfaction_margin(
    holo: ConstraintHolo,
    state: PolynomialPhaseSelectorFlowState,
) -> float:
    voltage = dict(zip(holo.variables, state.phase_cosine, strict=True))
    return min(
        (
            max(
                (1.0 if literal.positive else -1.0)
                * voltage[literal.variable]
                for literal in clause.literals
            )
            for clause in holo.clauses
        ),
        default=1.0,
    )


def _flatten_native_derivative(derivative) -> tuple[float, ...]:
    return (
        derivative.phase_cosine
        + derivative.phase_sine
        + derivative.short_memory
        + derivative.long_memory
        + tuple(value for triple in derivative.clause_selector for value in triple)
        + tuple(value for sextuple in derivative.pair_selector for value in sextuple)
    )


def integrate_adaptive_phase_logit_flow(
    holo: ConstraintHolo,
    initial_state: PolynomialPhaseSelectorFlowState | None = None,
    fixed_deadline: float = 3.0,
    relative_tolerance: float = 1.0e-6,
    absolute_tolerance: float = 1.0e-8,
    maximum_step: float = 5.0e-2,
    solver_method: str = "BDF",
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> AdaptivePhaseLogitFlowRun:
    """Run to one public deadline; first passage is observation, never control."""

    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover
        raise ConstraintHoloError(
            "adaptive phase logit flow requires NumPy and SciPy"
        ) from exc

    controls = (
        fixed_deadline,
        relative_tolerance,
        absolute_tolerance,
        maximum_step,
        selector_rate,
        boundary_release_rate,
        truth_gain,
    )
    if not all(isfinite(value) and value > 0.0 for value in controls):
        raise ConstraintHoloError("adaptive phase controls must be positive and finite")
    if solver_method not in {"RK45", "DOP853", "Radau", "BDF"}:
        raise ConstraintHoloError("unsupported adaptive phase solver method")

    initial = initial_state or public_phase_selector_initial_state(holo)
    vector0 = _state_to_chart(holo, initial, parameters)

    def field(_time, vector):
        state = _chart_to_state(holo, tuple(float(value) for value in vector), parameters)
        return np.asarray(
            _chart_derivative(
                holo,
                state,
                parameters,
                selector_rate,
                boundary_release_rate,
                truth_gain,
            ),
            dtype=np.float64,
        )

    try:
        solution = solve_ivp(
            field,
            (0.0, fixed_deadline),
            np.asarray(vector0, dtype=np.float64),
            method=solver_method,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
        )
    except (RuntimeError, ValueError, FloatingPointError, ArithmeticError) as exc:
        assignment = phase_threshold_assignment(holo, initial)
        return AdaptivePhaseLogitFlowRun(
            fixed_deadline=fixed_deadline,
            terminal_time=0.0,
            reached_fixed_deadline=False,
            terminal_solution_verified=False,
            first_passage_observed=False,
            first_passage_time=None,
            function_evaluations=0,
            accepted_internal_steps=0,
            solver_success=False,
            solver_message="adaptive solver raised an exception",
            solver_exception=f"{type(exc).__name__}: {exc}",
            solver_method=solver_method,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
            terminal_assignment=tuple(sorted(assignment.items())),
            terminal_max_clause_violation=max(
                phase_clause_violation_values(holo, initial), default=0.0
            ),
            terminal_clause_satisfaction_margin=_clause_satisfaction_margin(holo, initial),
            maximum_long_memory=max(initial.long_memory, default=1.0),
            maximum_unwrapped_phase_displacement=0.0,
            maximum_short_logit_magnitude=max(
                (abs(_logit(value)) for value in initial.short_memory), default=0.0
            ),
            maximum_long_logit_magnitude=0.0,
            maximum_clause_log_ratio_magnitude=0.0,
            maximum_pair_log_ratio_magnitude=0.0,
            minimum_clause_selector_weight=min(
                (value for weights in initial.clause_selector for value in weights),
                default=1.0,
            ),
            minimum_pair_selector_weight=min(
                (value for weights in initial.pair_selector for value in weights),
                default=1.0,
            ),
            chart_trajectory_length_lower_bound=0.0,
            phase_trajectory_length_lower_bound=0.0,
            native_trajectory_length_lower_bound=0.0,
            maximum_chart_speed=0.0,
            final_state=initial,
            status="INVALID_CARRIER_NUMERICAL_CHART_EXCEPTION",
        )

    terminal_index = len(solution.t) - 1
    terminal_vector = tuple(float(value) for value in solution.y[:, terminal_index])
    terminal_state = _chart_to_state(holo, terminal_vector, parameters)
    terminal_assignment = phase_threshold_assignment(holo, terminal_state)
    terminal_verified = holo.accepts(terminal_assignment)
    reached_deadline = bool(
        solution.success
        and abs(float(solution.t[-1]) - fixed_deadline)
        <= max(1.0e-9, fixed_deadline * 1.0e-8)
    )

    first_passage_time: float | None = None
    maximum_long_memory = 1.0
    maximum_phase_displacement = 0.0
    maximum_short_logit = 0.0
    maximum_long_logit = 0.0
    maximum_clause_ratio = 0.0
    maximum_pair_ratio = 0.0
    minimum_clause_weight = 1.0
    minimum_pair_weight = 1.0
    maximum_chart_speed = 0.0
    chart_length = 0.0
    phase_length = 0.0
    native_length = 0.0
    previous_time: float | None = None
    previous_chart_speed = 0.0
    previous_phase_speed = 0.0
    previous_native_speed = 0.0
    n = len(holo.variables)
    m = len(holo.clauses)

    for index, time_value in enumerate(solution.t):
        vector = tuple(float(value) for value in solution.y[:, index])
        state = _chart_to_state(holo, vector, parameters)
        assignment = phase_threshold_assignment(holo, state)
        if first_passage_time is None and holo.accepts(assignment):
            first_passage_time = float(time_value)
        maximum_long_memory = max(maximum_long_memory, max(state.long_memory, default=1.0))
        maximum_phase_displacement = max(
            maximum_phase_displacement,
            max(
                (
                    abs(vector[position] - vector0[position])
                    for position in range(n)
                ),
                default=0.0,
            ),
        )
        short_slice = vector[n : n + m]
        long_slice = vector[n + m : n + 2 * m]
        clause_slice = vector[n + 2 * m : n + 4 * m]
        pair_slice = vector[n + 4 * m :]
        maximum_short_logit = max(
            maximum_short_logit,
            max((abs(value) for value in short_slice), default=0.0),
        )
        maximum_long_logit = max(
            maximum_long_logit,
            max((abs(value) for value in long_slice), default=0.0),
        )
        maximum_clause_ratio = max(
            maximum_clause_ratio,
            max((abs(value) for value in clause_slice), default=0.0),
        )
        maximum_pair_ratio = max(
            maximum_pair_ratio,
            max((abs(value) for value in pair_slice), default=0.0),
        )
        minimum_clause_weight = min(
            minimum_clause_weight,
            min(
                (value for weights in state.clause_selector for value in weights),
                default=1.0,
            ),
        )
        minimum_pair_weight = min(
            minimum_pair_weight,
            min(
                (value for weights in state.pair_selector for value in weights),
                default=1.0,
            ),
        )
        chart_derivative = _chart_derivative(
            holo,
            state,
            parameters,
            selector_rate,
            boundary_release_rate,
            truth_gain,
        )
        native_derivative = polynomial_phase_selector_flow_derivative(
            holo,
            state,
            parameters=parameters,
            selector_rate=selector_rate,
            boundary_release_rate=boundary_release_rate,
            truth_gain=truth_gain,
        )
        chart_speed = float(np.linalg.norm(np.asarray(chart_derivative)))
        phase_speed = float(np.linalg.norm(np.asarray(chart_derivative[:n])))
        native_speed = float(
            np.linalg.norm(np.asarray(_flatten_native_derivative(native_derivative)))
        )
        maximum_chart_speed = max(maximum_chart_speed, chart_speed)
        if previous_time is not None:
            width = float(time_value) - previous_time
            chart_length += 0.5 * width * (previous_chart_speed + chart_speed)
            phase_length += 0.5 * width * (previous_phase_speed + phase_speed)
            native_length += 0.5 * width * (previous_native_speed + native_speed)
        previous_time = float(time_value)
        previous_chart_speed = chart_speed
        previous_phase_speed = phase_speed
        previous_native_speed = native_speed

    if not solution.success or not reached_deadline:
        status = "INVALID_CARRIER_NUMERICAL_CHART_FAILURE"
    elif terminal_verified:
        status = "TERMINAL_WITNESS_VERIFIED"
    else:
        status = "TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED"

    return AdaptivePhaseLogitFlowRun(
        fixed_deadline=fixed_deadline,
        terminal_time=float(solution.t[-1]),
        reached_fixed_deadline=reached_deadline,
        terminal_solution_verified=terminal_verified,
        first_passage_observed=first_passage_time is not None,
        first_passage_time=first_passage_time,
        function_evaluations=int(solution.nfev),
        accepted_internal_steps=terminal_index,
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
        solver_exception=None,
        solver_method=solver_method,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
        terminal_assignment=tuple(sorted(terminal_assignment.items())),
        terminal_max_clause_violation=max(
            phase_clause_violation_values(holo, terminal_state, truth_gain),
            default=0.0,
        ),
        terminal_clause_satisfaction_margin=_clause_satisfaction_margin(
            holo, terminal_state
        ),
        maximum_long_memory=maximum_long_memory,
        maximum_unwrapped_phase_displacement=maximum_phase_displacement,
        maximum_short_logit_magnitude=maximum_short_logit,
        maximum_long_logit_magnitude=maximum_long_logit,
        maximum_clause_log_ratio_magnitude=maximum_clause_ratio,
        maximum_pair_log_ratio_magnitude=maximum_pair_ratio,
        minimum_clause_selector_weight=minimum_clause_weight,
        minimum_pair_selector_weight=minimum_pair_weight,
        chart_trajectory_length_lower_bound=chart_length,
        phase_trajectory_length_lower_bound=phase_length,
        native_trajectory_length_lower_bound=native_length,
        maximum_chart_speed=maximum_chart_speed,
        final_state=terminal_state,
        status=status,
    )
