from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    REFERENCE_VARIABLE_LIMIT,
    reference_existential_trace,
)
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class ExceptionalPointRootLatchResult:
    public_variables: int
    public_clauses: int
    assignment_dimension: int
    satisfying_rank: int
    clean_uniform_amplitude: float
    clean_uniform_intensity: float
    exceptional_point_order: int
    gain_per_transport: float
    cycle_gain: float
    effective_cycle_coupling: float
    spectral_radius: float
    presence_margin: float
    unique_amplitude_dynamic_range: float
    unique_intensity_dynamic_range: float
    symbolic_sensor_modes: int
    symbolic_description_size: int
    ep_identity_status: str
    deterministic_noiseless_gain_status: str
    clean_projection_status: str
    reversible_dilation_status: str
    physical_noise_floor_status: str
    polynomial_total_resources_established: bool
    claim_ceiling: str = CLAIM_CEILING


def audit_exceptional_point_root_latch(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
    gain_per_transport: float = 2.0,
) -> ExceptionalPointRootLatchResult:
    """Audit an order-n EP companion driven by the clean solution amplitude.

    The clean uniform matrix element is epsilon = #SAT / 2^n. An n-step companion
    cycle with gain two on each transport has cycle product 2^n * epsilon = #SAT.
    At epsilon zero the sensor is an n-th order nilpotent exceptional point. For any
    satisfying formula its spectral radius is (#SAT)^(1/n), at least one.

    This is an algebraic candidate, not an implemented physical or standard-model
    operation. The clean projection has exponentially small pre-gain intensity for a
    unique witness, and deterministic noiseless gain/restoration are unresolved.
    """

    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("EP root-latch audit exceeds reference limit")
    if gain_per_transport <= 1.0:
        raise ConstraintHoloError("root-latch gain must exceed one")

    boundary = reference_existential_trace(holo, variable_limit=variable_limit)
    variables = len(holo.variables)
    order = max(1, variables)
    dimension = 1 << variables
    satisfying_rank = boundary.witness_count
    epsilon = satisfying_rank / dimension
    intensity = epsilon * epsilon
    cycle_gain = gain_per_transport**order
    effective_coupling = cycle_gain * epsilon
    spectral_radius = (
        effective_coupling ** (1.0 / order)
        if effective_coupling > 0.0
        else 0.0
    )
    unique_amplitude_range = float(dimension)
    unique_intensity_range = float(dimension * dimension)

    return ExceptionalPointRootLatchResult(
        public_variables=variables,
        public_clauses=len(holo.clauses),
        assignment_dimension=dimension,
        satisfying_rank=satisfying_rank,
        clean_uniform_amplitude=epsilon,
        clean_uniform_intensity=intensity,
        exceptional_point_order=order,
        gain_per_transport=gain_per_transport,
        cycle_gain=cycle_gain,
        effective_cycle_coupling=effective_coupling,
        spectral_radius=spectral_radius,
        presence_margin=spectral_radius,
        unique_amplitude_dynamic_range=unique_amplitude_range,
        unique_intensity_dynamic_range=unique_intensity_range,
        symbolic_sensor_modes=order,
        symbolic_description_size=(
            variables + 3 * len(holo.clauses) + order
        ),
        ep_identity_status=(
            "ORDER_N_EP_ZERO_FOR_UNSAT__ROOT_RADIUS_AT_LEAST_ONE_FOR_SAT"
            if gain_per_transport == 2.0
            else "GENERAL_GAIN_EP_IDENTITY_ESTABLISHED_REFERENCE"
        ),
        deterministic_noiseless_gain_status="NOT_ESTABLISHED",
        clean_projection_status=(
            "POLYNOMIAL_CIRCUIT_DESCRIPTION__UNIQUE_PORT_INTENSITY_4^-n"
        ),
        reversible_dilation_status=(
            "LOSS_AND_GAIN_ENVIRONMENT_MUST_BE_RETAINED_AND_RESTORED"
        ),
        physical_noise_floor_status=(
            "NOISE_BELOW_UNIQUE_PRE_GAIN_SIGNAL_NOT_ESTABLISHED"
        ),
        polynomial_total_resources_established=False,
    )
