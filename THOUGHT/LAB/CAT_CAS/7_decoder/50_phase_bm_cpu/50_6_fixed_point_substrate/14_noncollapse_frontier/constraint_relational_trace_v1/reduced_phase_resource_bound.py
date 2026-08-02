from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class ReducedPhasePolynomialResourceBound:
    public_variables: int
    public_clauses: int
    native_state_coordinates: int
    long_memory_cap: float
    maximum_clause_violation: float
    maximum_exact_gradient_coordinate: float
    maximum_clause_rigidity_coordinate: float
    maximum_variable_relational_force: float
    maximum_variable_incident_violation: float
    maximum_angular_speed: float
    maximum_short_memory_speed: float
    maximum_long_memory_speed: float
    maximum_clause_selector_speed: float
    maximum_native_l2_speed: float
    maximum_short_logit_speed: float
    maximum_long_logit_speed: float
    maximum_clause_log_ratio_speed: float
    state_range_polynomial_without_deadline: bool
    trajectory_length_polynomial_if_deadline_polynomial: bool
    logit_range_polynomial_if_deadline_polynomial: bool
    forward_standard_model_transfer_status: str
    remaining_obligations: tuple[str, ...]
    claim_ceiling: str = CLAIM_CEILING


def derive_reduced_phase_polynomial_resource_bound(
    holo: ConstraintHolo,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> ReducedPhasePolynomialResourceBound:
    """Derive formula-uniform forward resource bounds for the reduced carrier.

    The result is conditional only where it says so. It does not prove a polynomial
    deadline. It proves that if the declared public seed reaches the terminal boundary
    in polynomial continuous time, then state range, trajectory length, and logit
    precision of the reduced forward carrier are polynomial as well.
    """

    controls = (selector_rate, boundary_release_rate, truth_gain)
    if any(value <= 0.0 for value in controls):
        raise ConstraintHoloError("resource-bound controls must be positive")

    variables = len(holo.variables)
    clauses = len(holo.clauses)
    clause_scale = max(1, clauses)
    cap = parameters.long_memory_cap_factor * clause_scale

    # Every literal defect lies in [0,2] on S1 because its cosine lies in [-1,1].
    max_violation = truth_gain
    max_gradient = truth_gain / 2.0
    max_rigidity = 1.0

    # A variable can occur at most three times per public clause.
    max_occurrences = 3 * clauses
    max_force_per_occurrence = (
        cap * max_gradient
        + 1.0
        + parameters.zeta * cap
    )
    max_variable_force = max_occurrences * max_force_per_occurrence
    max_incident_violation = max_occurrences * max_violation
    max_angular_speed = (
        max_variable_force
        + boundary_release_rate * max_incident_violation
    )

    max_short_speed = (
        parameters.beta
        * max(parameters.gamma, truth_gain - parameters.gamma)
        / 4.0
    )
    max_long_speed = (
        parameters.alpha
        * max(parameters.delta, truth_gain - parameters.delta)
        * cap
        / 4.0
    )
    max_selector_speed = 2.0 * selector_rate

    native_l2_speed = sqrt(
        2 * variables * max_angular_speed**2
        + clauses * max_short_speed**2
        + clauses * max_long_speed**2
        + 3 * clauses * max_selector_speed**2
    )

    # In logit coordinates the memory equations lose their logistic factors.
    max_short_logit_speed = parameters.beta * max(
        parameters.gamma,
        truth_gain - parameters.gamma,
    )
    max_long_logit_speed = parameters.alpha * max(
        parameters.delta,
        truth_gain - parameters.delta,
    )
    # Replicator log-ratio derivative is r(cost_j-cost_i), with costs in [0,2].
    max_clause_log_ratio_speed = 2.0 * selector_rate

    return ReducedPhasePolynomialResourceBound(
        public_variables=variables,
        public_clauses=clauses,
        native_state_coordinates=2 * variables + 5 * clauses,
        long_memory_cap=cap,
        maximum_clause_violation=max_violation,
        maximum_exact_gradient_coordinate=max_gradient,
        maximum_clause_rigidity_coordinate=max_rigidity,
        maximum_variable_relational_force=max_variable_force,
        maximum_variable_incident_violation=max_incident_violation,
        maximum_angular_speed=max_angular_speed,
        maximum_short_memory_speed=max_short_speed,
        maximum_long_memory_speed=max_long_speed,
        maximum_clause_selector_speed=max_selector_speed,
        maximum_native_l2_speed=native_l2_speed,
        maximum_short_logit_speed=max_short_logit_speed,
        maximum_long_logit_speed=max_long_logit_speed,
        maximum_clause_log_ratio_speed=max_clause_log_ratio_speed,
        state_range_polynomial_without_deadline=True,
        trajectory_length_polynomial_if_deadline_polynomial=True,
        logit_range_polynomial_if_deadline_polynomial=True,
        forward_standard_model_transfer_status=(
            "POLYNOMIAL_DEADLINE_IMPLIES_POLYNOMIAL_FORWARD_RANGE_LENGTH_AND_PRECISION"
        ),
        remaining_obligations=(
            "formula_uniform_public_seed_deadline",
            "robust_terminal_boundary_margin",
            "deterministic_total_unsat_boundary",
            "cotangent_or_environmental_restoration_bound",
        ),
    )
