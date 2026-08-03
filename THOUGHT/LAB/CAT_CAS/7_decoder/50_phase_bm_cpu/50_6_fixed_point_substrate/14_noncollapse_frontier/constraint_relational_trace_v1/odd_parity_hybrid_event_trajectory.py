from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .adaptive_phase_logit_flow import (
    _chart_derivative,
    _chart_to_state,
    _state_to_chart,
)
from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError
from .hybrid_event_geometry_audit import audit_hybrid_guard_event_geometry
from .hybrid_witness_recorder_contract import evaluate_hybrid_witness_guard
from .odd_parity_fixed_deadline_counterexample import odd_parity_three_variable_holo
from .polynomial_phase_selector_flow import (
    phase_threshold_assignment,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class OddParityHybridEventSolverControl:
    solver_method: str
    solver_success: bool
    first_entry_time: float | None
    first_exit_time: float | None
    first_witness_dwell_time: float | None
    first_entry_guard_directional_derivative: float | None
    first_entry_active_set_gap: float | None
    maximum_first_interval_guard_margin: float | None
    maximum_first_interval_guard_time: float | None
    terminal_solution_verified: bool
    terminal_guard_margin: float
    status: str


@dataclass(frozen=True)
class OddParityHybridEventTrajectoryAudit:
    fixed_deadline: float
    solver_controls: tuple[OddParityHybridEventSolverControl, ...]
    every_solver_found_entry_and_exit: bool
    entry_time_spread: float | None
    entry_transverse_speed_spread: float | None
    entry_active_set_gap_spread: float | None
    maximum_guard_margin_spread: float | None
    first_dwell_time_minimum: float | None
    first_dwell_time_maximum: float | None
    first_entry_cross_solver_agreement: bool
    first_event_has_positive_transverse_speed: bool
    first_event_has_small_but_positive_active_set_gap: bool
    first_interval_has_long_dwell_but_small_guard_margin: bool
    asymptotic_inverse_polynomial_event_bound_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def _spread(values: tuple[float, ...]) -> float:
    return max(values) - min(values)


def _integrate_solver_control(
    solver_method: str,
    *,
    fixed_deadline: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
    sample_count: int,
    parameters: SelfOrganizingFlowParameters,
    selector_rate: float,
    boundary_release_rate: float,
    truth_gain: float,
) -> OddParityHybridEventSolverControl:
    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover
        raise ConstraintHoloError(
            "odd-parity hybrid event trajectory requires NumPy and SciPy"
        ) from exc

    holo = odd_parity_three_variable_holo()
    initial = public_phase_selector_initial_state(holo)
    vector0 = _state_to_chart(holo, initial, parameters)

    def field(_time, vector):
        state = _chart_to_state(
            holo,
            tuple(float(value) for value in vector),
            parameters,
        )
        return np.asarray(
            _chart_derivative(
                holo,
                state,
                parameters,
                selector_rate,
                boundary_release_rate,
                truth_gain,
                "exact_product",
            ),
            dtype=np.float64,
        )

    def guard_value(_time, vector) -> float:
        state = _chart_to_state(
            holo,
            tuple(float(value) for value in vector),
            parameters,
        )
        return evaluate_hybrid_witness_guard(
            holo,
            state.phase_cosine,
        ).guard_margin

    def entry_event(time, vector) -> float:
        return guard_value(time, vector)

    def exit_event(time, vector) -> float:
        return guard_value(time, vector)

    entry_event.direction = 1.0  # type: ignore[attr-defined]
    entry_event.terminal = False  # type: ignore[attr-defined]
    exit_event.direction = -1.0  # type: ignore[attr-defined]
    exit_event.terminal = False  # type: ignore[attr-defined]

    solution = solve_ivp(
        field,
        (0.0, fixed_deadline),
        np.asarray(vector0, dtype=np.float64),
        method=solver_method,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
        dense_output=True,
        events=(entry_event, exit_event),
    )

    entries = tuple(float(value) for value in solution.t_events[0])
    exits = tuple(float(value) for value in solution.t_events[1])
    first_entry = entries[0] if entries else None
    first_exit = (
        next((value for value in exits if first_entry is not None and value > first_entry), None)
        if exits
        else None
    )

    entry_derivative: float | None = None
    entry_gap: float | None = None
    maximum_margin: float | None = None
    maximum_margin_time: float | None = None
    dwell: float | None = None

    if first_entry is not None and first_exit is not None and solution.sol is not None:
        dwell = first_exit - first_entry
        entry_vector = tuple(float(value) for value in solution.sol(first_entry))
        entry_state = _chart_to_state(holo, entry_vector, parameters)
        native_derivative = polynomial_phase_selector_flow_derivative(
            holo,
            entry_state,
            parameters=parameters,
            selector_rate=selector_rate,
            boundary_release_rate=boundary_release_rate,
            truth_gain=truth_gain,
            gradient_mode="exact_product",
        )
        geometry = audit_hybrid_guard_event_geometry(
            holo,
            entry_state.phase_cosine,
            native_derivative.phase_cosine,
            tolerance=1.0e-8,
        )
        entry_derivative = geometry.guard_directional_derivative
        entry_gap = geometry.active_set_gap

        sample_times = np.linspace(first_entry, first_exit, sample_count)
        sampled_margins = []
        for time_value in sample_times:
            state = _chart_to_state(
                holo,
                tuple(float(value) for value in solution.sol(float(time_value))),
                parameters,
            )
            sampled_margins.append(
                evaluate_hybrid_witness_guard(
                    holo,
                    state.phase_cosine,
                ).guard_margin
            )
        maximum_index = int(np.argmax(np.asarray(sampled_margins)))
        maximum_margin = float(sampled_margins[maximum_index])
        maximum_margin_time = float(sample_times[maximum_index])

    terminal_vector = tuple(float(value) for value in solution.y[:, -1])
    terminal_state = _chart_to_state(holo, terminal_vector, parameters)
    terminal_assignment = phase_threshold_assignment(holo, terminal_state)
    terminal_verified = holo.accepts(terminal_assignment)
    terminal_guard = evaluate_hybrid_witness_guard(
        holo,
        terminal_state.phase_cosine,
    ).guard_margin

    complete = (
        solution.success
        and first_entry is not None
        and first_exit is not None
        and dwell is not None
        and entry_derivative is not None
        and entry_gap is not None
        and maximum_margin is not None
    )

    return OddParityHybridEventSolverControl(
        solver_method=solver_method,
        solver_success=bool(solution.success),
        first_entry_time=first_entry,
        first_exit_time=first_exit,
        first_witness_dwell_time=dwell,
        first_entry_guard_directional_derivative=entry_derivative,
        first_entry_active_set_gap=entry_gap,
        maximum_first_interval_guard_margin=maximum_margin,
        maximum_first_interval_guard_time=maximum_margin_time,
        terminal_solution_verified=terminal_verified,
        terminal_guard_margin=terminal_guard,
        status=(
            "ODD_PARITY_FIRST_HYBRID_WITNESS_INTERVAL_RESOLVED"
            if complete
            else "ODD_PARITY_HYBRID_WITNESS_INTERVAL_NOT_RESOLVED"
        ),
    )


def audit_odd_parity_hybrid_event_trajectory(
    *,
    fixed_deadline: float = 3.0,
    solver_methods: tuple[str, ...] = ("DOP853", "Radau"),
    relative_tolerance: float = 1.0e-9,
    absolute_tolerance: float = 1.0e-11,
    maximum_step: float = 2.0e-2,
    sample_count: int = 2049,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> OddParityHybridEventTrajectoryAudit:
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
        raise ConstraintHoloError("hybrid trajectory controls must be positive and finite")
    if not solver_methods or any(
        method not in {"RK45", "DOP853", "Radau", "BDF"}
        for method in solver_methods
    ):
        raise ConstraintHoloError("unsupported hybrid trajectory solver set")
    if not isinstance(sample_count, int) or sample_count < 33:
        raise ConstraintHoloError("hybrid trajectory sample count must be at least 33")

    solver_controls = tuple(
        _integrate_solver_control(
            method,
            fixed_deadline=fixed_deadline,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
            sample_count=sample_count,
            parameters=parameters,
            selector_rate=selector_rate,
            boundary_release_rate=boundary_release_rate,
            truth_gain=truth_gain,
        )
        for method in solver_methods
    )
    complete = bool(solver_controls) and all(
        control.status == "ODD_PARITY_FIRST_HYBRID_WITNESS_INTERVAL_RESOLVED"
        for control in solver_controls
    )

    if complete:
        entry_times = tuple(
            float(control.first_entry_time) for control in solver_controls
        )
        derivatives = tuple(
            float(control.first_entry_guard_directional_derivative)
            for control in solver_controls
        )
        gaps = tuple(
            float(control.first_entry_active_set_gap)
            for control in solver_controls
        )
        margins = tuple(
            float(control.maximum_first_interval_guard_margin)
            for control in solver_controls
        )
        dwells = tuple(
            float(control.first_witness_dwell_time)
            for control in solver_controls
        )
        entry_spread = _spread(entry_times)
        derivative_spread = _spread(derivatives)
        gap_spread = _spread(gaps)
        margin_spread = _spread(margins)
        dwell_minimum = min(dwells)
        dwell_maximum = max(dwells)
    else:
        entry_spread = None
        derivative_spread = None
        gap_spread = None
        margin_spread = None
        dwell_minimum = None
        dwell_maximum = None
        derivatives = ()
        gaps = ()
        margins = ()

    agreement = (
        complete
        and entry_spread is not None
        and entry_spread < 1.0e-6
        and derivative_spread is not None
        and derivative_spread < 1.0e-8
        and gap_spread is not None
        and gap_spread < 1.0e-10
        and margin_spread is not None
        and margin_spread < 1.0e-8
    )
    positive_transverse = complete and all(value > 2.0e-3 for value in derivatives)
    small_gap = complete and all(1.0e-8 < value < 1.0e-5 for value in gaps)
    long_small = (
        complete
        and dwell_minimum is not None
        and dwell_minimum > 1.0
        and all(1.0e-4 < value < 2.0e-4 for value in margins)
    )

    return OddParityHybridEventTrajectoryAudit(
        fixed_deadline=fixed_deadline,
        solver_controls=solver_controls,
        every_solver_found_entry_and_exit=complete,
        entry_time_spread=entry_spread,
        entry_transverse_speed_spread=derivative_spread,
        entry_active_set_gap_spread=gap_spread,
        maximum_guard_margin_spread=margin_spread,
        first_dwell_time_minimum=dwell_minimum,
        first_dwell_time_maximum=dwell_maximum,
        first_entry_cross_solver_agreement=agreement,
        first_event_has_positive_transverse_speed=positive_transverse,
        first_event_has_small_but_positive_active_set_gap=small_gap,
        first_interval_has_long_dwell_but_small_guard_margin=long_small,
        asymptotic_inverse_polynomial_event_bound_established=False,
        status=(
            "ODD_PARITY_HYBRID_EVENT_GEOMETRY_CROSS_SOLVER_REFERENCE_PASS__"
            "UNIFORM_PUBLIC_SEED_LOWER_BOUND_NOT_ESTABLISHED"
            if agreement and positive_transverse and small_gap and long_small
            else "ODD_PARITY_HYBRID_EVENT_GEOMETRY_REFERENCE_FAILURE"
        ),
    )
