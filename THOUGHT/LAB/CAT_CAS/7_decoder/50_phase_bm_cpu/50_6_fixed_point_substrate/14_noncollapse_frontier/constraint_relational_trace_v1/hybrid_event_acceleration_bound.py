from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo
from .reduced_phase_resource_bound import (
    derive_reduced_phase_polynomial_resource_bound,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class HybridEventPolynomialAccelerationBound:
    public_variables: int
    public_clauses: int
    maximum_phase_cosine_speed: float
    maximum_clause_violation_derivative: float
    maximum_exact_gradient_derivative: float
    maximum_rigidity_derivative: float
    maximum_force_per_occurrence_derivative: float
    maximum_variable_force_derivative: float
    maximum_incident_violation_derivative: float
    maximum_angular_acceleration: float
    maximum_phase_cosine_acceleration: float
    maximum_simple_guard_branch_acceleration: float
    polynomial_upper_bound_established: bool
    inverse_polynomial_crossing_speed_established: bool
    inverse_polynomial_active_set_gap_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def derive_hybrid_event_polynomial_acceleration_bound(
    holo: ConstraintHolo,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> HybridEventPolynomialAccelerationBound:
    """Derive a coarse formula-uniform acceleration bound for simple guard branches.

    At a smooth event the active guard branch is one signed phase cosine ``q_i c_i``.
    The reduced exact-product resource audit already bounds state range and angular speed.
    Differentiating the explicit field term by term gives a polynomial upper bound on
    ``|c_i''|`` and therefore on the active guard-branch acceleration.

    The bound is intentionally coarse. Its role is theorem reduction: polynomial upper
    speed/acceleration is available, while inverse-polynomial lower bounds on crossing
    speed and active-set separation remain unproved.
    """

    base = derive_reduced_phase_polynomial_resource_bound(
        holo,
        parameters=parameters,
        selector_rate=selector_rate,
        boundary_release_rate=boundary_release_rate,
        truth_gain=truth_gain,
    )
    clauses = len(holo.clauses)
    occurrences = 3 * clauses
    cap = base.long_memory_cap
    angular_speed = base.maximum_angular_speed

    # C = truth_gain*d1*d2*d3/8, defects in [0,2], and |d_dot| <= |c_dot|.
    violation_derivative = 1.5 * truth_gain * angular_speed

    # g_i = truth_gain*q_i*d_j*d_k/8.
    gradient_derivative = 0.5 * truth_gain * angular_speed

    # R_i = 0.5*(q_i-c_i)*w_i and |w_i_dot| <= 2*selector_rate.
    rigidity_derivative = 0.5 * angular_speed + 2.0 * selector_rate

    gradient_bound = base.maximum_exact_gradient_coordinate
    short_speed = base.maximum_short_memory_speed
    long_speed = base.maximum_long_memory_speed
    weighted_gain = 1.0 + parameters.zeta * cap

    memory_gradient_derivative = (
        long_speed * gradient_bound
        + cap * short_speed * gradient_bound
        + cap * gradient_derivative
    )
    rigidity_channel_derivative = (
        parameters.zeta * long_speed
        + weighted_gain * short_speed
        + weighted_gain * rigidity_derivative
    )
    force_per_occurrence_derivative = (
        memory_gradient_derivative + rigidity_channel_derivative
    )
    variable_force_derivative = occurrences * force_per_occurrence_derivative
    incident_violation_derivative = occurrences * violation_derivative

    # omega = -s*F + rho*c*V.
    angular_acceleration = (
        angular_speed * base.maximum_variable_relational_force
        + variable_force_derivative
        + boundary_release_rate
        * (
            angular_speed * base.maximum_variable_incident_violation
            + incident_violation_derivative
        )
    )

    # c_dot = -s*omega, hence |c_ddot| <= |omega|^2 + |omega_dot|.
    phase_cosine_acceleration = angular_speed**2 + angular_acceleration

    return HybridEventPolynomialAccelerationBound(
        public_variables=len(holo.variables),
        public_clauses=clauses,
        maximum_phase_cosine_speed=angular_speed,
        maximum_clause_violation_derivative=violation_derivative,
        maximum_exact_gradient_derivative=gradient_derivative,
        maximum_rigidity_derivative=rigidity_derivative,
        maximum_force_per_occurrence_derivative=(
            force_per_occurrence_derivative
        ),
        maximum_variable_force_derivative=variable_force_derivative,
        maximum_incident_violation_derivative=incident_violation_derivative,
        maximum_angular_acceleration=angular_acceleration,
        maximum_phase_cosine_acceleration=phase_cosine_acceleration,
        maximum_simple_guard_branch_acceleration=phase_cosine_acceleration,
        polynomial_upper_bound_established=True,
        inverse_polynomial_crossing_speed_established=False,
        inverse_polynomial_active_set_gap_established=False,
        status=(
            "HYBRID_EVENT_POLYNOMIAL_SPEED_AND_ACCELERATION_UPPER_BOUNDS_ESTABLISHED__"
            "TRANSVERSE_LOWER_BOUNDS_NOT_ESTABLISHED"
        ),
    )
