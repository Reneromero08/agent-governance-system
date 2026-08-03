from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .adaptive_phase_logit_flow import integrate_adaptive_phase_logit_flow
from .catalytic_existential_trace import CLAIM_CEILING, reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .polynomial_phase_selector_flow import (
    PolynomialPhaseSelectorFlowState,
    phase_threshold_assignment,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class OddParityStationaryManifoldAudit:
    semantic_digest: str
    presentation_digest: str
    reference_witness_count: int
    public_seed_assignment: tuple[tuple[str, bool], ...]
    public_seed_is_witness: bool
    stationary_assignment: tuple[tuple[str, bool], ...]
    stationary_is_witness: bool
    stationary_derivative_max_abs: float
    clause_violation: float
    phase_tangent_eigenvalues: tuple[float, float, float]
    short_one_normal_exponent: float
    long_cap_normal_exponent: float
    selector_center_dimension: int
    normally_attracting_transverse_to_selector_centers: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


@dataclass(frozen=True)
class OddParitySolverControl:
    solver_method: str
    reached_fixed_deadline: bool
    first_passage_observed: bool
    first_passage_time: float | None
    terminal_solution_verified: bool
    terminal_assignment: tuple[tuple[str, bool], ...]
    terminal_clause_satisfaction_margin: float
    terminal_phase_cosine_norm: float
    status: str


@dataclass(frozen=True)
class OddParityFixedDeadlineCounterexampleAudit:
    fixed_deadline: float
    stationary: OddParityStationaryManifoldAudit
    solver_controls: tuple[OddParitySolverControl, ...]
    every_solver_observed_transient_witness: bool
    every_solver_failed_terminal_verification: bool
    every_solver_approached_midpoint: bool
    fixed_deadline_three_completeness_falsified: bool
    stronger_all_polynomial_deadlines_falsified: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def odd_parity_three_variable_holo() -> ConstraintHolo:
    """Return the exact four-clause CNF for odd parity on three variables."""

    return ConstraintHolo.build(
        ("x1", "x2", "x3"),
        (
            ClauseRelation(
                (
                    Literal("x1"),
                    Literal("x2", False),
                    Literal("x3", False),
                )
            ),
            ClauseRelation(
                (
                    Literal("x1", False),
                    Literal("x2"),
                    Literal("x3", False),
                )
            ),
            ClauseRelation(
                (
                    Literal("x1", False),
                    Literal("x2", False),
                    Literal("x3"),
                )
            ),
            ClauseRelation(
                (
                    Literal("x1"),
                    Literal("x2"),
                    Literal("x3"),
                )
            ),
        ),
    )


def odd_parity_midpoint_stationary_state(
    holo: ConstraintHolo,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> PolynomialPhaseSelectorFlowState:
    cap = max(
        1.0,
        parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
    )
    return PolynomialPhaseSelectorFlowState(
        phase_cosine=tuple(0.0 for _ in holo.variables),
        phase_sine=tuple(1.0 for _ in holo.variables),
        short_memory=tuple(1.0 for _ in holo.clauses),
        long_memory=tuple(cap for _ in holo.clauses),
        clause_selector=tuple(
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) for _ in holo.clauses
        ),
        pair_selector=tuple(
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5) for _ in holo.clauses
        ),
    )


def audit_odd_parity_stationary_manifold(
    *,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    boundary_release_rate: float = 10.0,
) -> OddParityStationaryManifoldAudit:
    holo = odd_parity_three_variable_holo()
    reference = reference_existential_trace(holo)
    public_seed = public_phase_selector_initial_state(holo)
    public_assignment = phase_threshold_assignment(holo, public_seed)
    stationary = odd_parity_midpoint_stationary_state(holo, parameters)
    stationary_assignment = phase_threshold_assignment(holo, stationary)
    stationary_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        stationary,
        parameters=parameters,
        boundary_release_rate=boundary_release_rate,
        gradient_mode="exact_product",
    )

    clause_violation = 0.5
    phase_eigenvalue = -2.0 * boundary_release_rate
    short_normal = -parameters.beta * (clause_violation - parameters.gamma)
    long_normal = -parameters.alpha * (clause_violation - parameters.delta)
    selector_center_dimension = 2 * len(holo.clauses)
    normally_attracting = (
        stationary_derivative.max_abs() == 0.0
        and phase_eigenvalue < 0.0
        and short_normal < 0.0
        and long_normal < 0.0
    )

    return OddParityStationaryManifoldAudit(
        semantic_digest=holo.semantic_digest(),
        presentation_digest=holo.presentation_digest(),
        reference_witness_count=reference.witness_count,
        public_seed_assignment=tuple(sorted(public_assignment.items())),
        public_seed_is_witness=holo.accepts(public_assignment),
        stationary_assignment=tuple(sorted(stationary_assignment.items())),
        stationary_is_witness=holo.accepts(stationary_assignment),
        stationary_derivative_max_abs=stationary_derivative.max_abs(),
        clause_violation=clause_violation,
        phase_tangent_eigenvalues=(
            phase_eigenvalue,
            phase_eigenvalue,
            phase_eigenvalue,
        ),
        short_one_normal_exponent=short_normal,
        long_cap_normal_exponent=long_normal,
        selector_center_dimension=selector_center_dimension,
        normally_attracting_transverse_to_selector_centers=normally_attracting,
        status=(
            "ODD_PARITY_NON_SOLUTION_MIDPOINT_MANIFOLD_NORMALLY_ATTRACTING"
            if normally_attracting
            else "ODD_PARITY_MIDPOINT_MANIFOLD_AUDIT_NOT_ESTABLISHED"
        ),
    )


def audit_odd_parity_fixed_deadline_counterexample(
    *,
    fixed_deadline: float = 3.0,
    solver_methods: tuple[str, ...] = ("DOP853", "Radau"),
    relative_tolerance: float = 1.0e-9,
    absolute_tolerance: float = 1.0e-11,
    maximum_step: float = 2.0e-2,
) -> OddParityFixedDeadlineCounterexampleAudit:
    """Cross-check the public trajectory at the branch's fixed deadline three.

    This audit falsifies the current deadline-three terminal boundary. It deliberately
    does not claim that every possible formula-uniform polynomial deadline is ruled out.
    """

    holo = odd_parity_three_variable_holo()
    stationary = audit_odd_parity_stationary_manifold()
    controls: list[OddParitySolverControl] = []
    for solver_method in solver_methods:
        run = integrate_adaptive_phase_logit_flow(
            holo,
            fixed_deadline=fixed_deadline,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
            solver_method=solver_method,
            gradient_mode="exact_product",
        )
        terminal_norm = sqrt(
            sum(value * value for value in run.final_state.phase_cosine)
        )
        controls.append(
            OddParitySolverControl(
                solver_method=solver_method,
                reached_fixed_deadline=run.reached_fixed_deadline,
                first_passage_observed=run.first_passage_observed,
                first_passage_time=run.first_passage_time,
                terminal_solution_verified=run.terminal_solution_verified,
                terminal_assignment=run.terminal_assignment,
                terminal_clause_satisfaction_margin=(
                    run.terminal_clause_satisfaction_margin
                ),
                terminal_phase_cosine_norm=terminal_norm,
                status=run.status,
            )
        )

    every_transient = bool(controls) and all(
        control.first_passage_observed for control in controls
    )
    every_terminal_failure = bool(controls) and all(
        control.reached_fixed_deadline
        and not control.terminal_solution_verified
        and control.status == "TERMINAL_NO_WITNESS__UNSAT_NOT_ESTABLISHED"
        for control in controls
    )
    every_midpoint = bool(controls) and all(
        control.terminal_phase_cosine_norm < 1.0e-8 for control in controls
    )
    deadline_three = abs(fixed_deadline - 3.0) <= 1.0e-12
    falsified = (
        deadline_three
        and stationary.reference_witness_count == 4
        and not stationary.public_seed_is_witness
        and not stationary.stationary_is_witness
        and stationary.normally_attracting_transverse_to_selector_centers
        and every_transient
        and every_terminal_failure
        and every_midpoint
    )

    return OddParityFixedDeadlineCounterexampleAudit(
        fixed_deadline=fixed_deadline,
        stationary=stationary,
        solver_controls=tuple(controls),
        every_solver_observed_transient_witness=every_transient,
        every_solver_failed_terminal_verification=every_terminal_failure,
        every_solver_approached_midpoint=every_midpoint,
        fixed_deadline_three_completeness_falsified=falsified,
        stronger_all_polynomial_deadlines_falsified=False,
        status=(
            "FIXED_DEADLINE_THREE_PUBLIC_SEED_COMPLETENESS_FALSIFIED__"
            "TRANSIENT_WITNESS_ONLY__ALL_POLYNOMIAL_DEADLINES_UNRESOLVED"
            if falsified
            else "ODD_PARITY_FIXED_DEADLINE_COUNTEREXAMPLE_NOT_ESTABLISHED"
        ),
    )
