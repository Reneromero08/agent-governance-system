from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError


@dataclass(frozen=True)
class DataProcessingAmplifierAudit:
    public_variables: int
    basis_dimension: int
    marked_basis_states: int
    oracle_state_overlap: float
    oracle_trace_distance: float
    deterministic_cptp_output_distance_upper_bound: float
    constant_one_shot_separation_possible_under_cptp: bool
    postselection_or_nonstandard_dynamics_required: bool
    scope: str
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_phase_oracle_data_processing(
    public_variables: int,
    marked_basis_states: int,
) -> DataProcessingAmplifierAudit:
    """Bound one-shot deterministic amplification after a phase-marking oracle.

    Compare the uniform state with no marked phases to the state where `k` basis
    amplitudes receive phase pi. Their pure-state overlap is `1 - 2k/N`. Trace distance
    is contractive under every deterministic completely positive trace-preserving map.
    """

    if public_variables < 1:
        raise ConstraintHoloError("public variable count must be positive")
    dimension = 1 << public_variables
    if marked_basis_states < 0 or marked_basis_states > dimension:
        raise ConstraintHoloError("marked-state count outside the public basis")

    overlap = 1.0 - 2.0 * marked_basis_states / dimension
    trace_distance = sqrt(max(0.0, 1.0 - overlap * overlap))

    return DataProcessingAmplifierAudit(
        public_variables=public_variables,
        basis_dimension=dimension,
        marked_basis_states=marked_basis_states,
        oracle_state_overlap=overlap,
        oracle_trace_distance=trace_distance,
        deterministic_cptp_output_distance_upper_bound=trace_distance,
        constant_one_shot_separation_possible_under_cptp=(trace_distance >= 0.5),
        postselection_or_nonstandard_dynamics_required=(trace_distance < 0.5),
        scope=(
            "one_shot_formula_blind_post_oracle_deterministic_quantum_amplifier"
        ),
        status=(
            "TRACE_DISTANCE_CONTRACTIVITY_BLOCKS_CONSTANT_ONE_SHOT_AMPLIFICATION"
            if trace_distance < 0.5
            else "INPUT_ORACLE_STATES_ALREADY_CONSTANTLY_SEPARATED"
        ),
    )
