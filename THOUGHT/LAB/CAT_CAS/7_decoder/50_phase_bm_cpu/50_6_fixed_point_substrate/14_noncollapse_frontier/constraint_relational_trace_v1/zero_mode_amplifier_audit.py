from __future__ import annotations

from dataclasses import dataclass
from math import log

from .catalytic_existential_trace import CLAIM_CEILING
from .clause_hamiltonian import ClauseHamiltonianAudit
from .constraint_holo import ConstraintHoloError


@dataclass(frozen=True)
class ZeroModeAmplifierAudit:
    basis_dimension: int
    zero_mode_degeneracy: int
    initial_zero_mode_weight: float
    target_zero_mode_weight: float
    required_weight_gain: float | None
    ideal_exponential_gain_time: float | None
    minimum_output_to_input_energy_ratio: float | None
    mode_count_if_materialized: int
    polynomial_time_possible_under_ideal_gain: bool
    polynomial_energy_established: bool
    polynomial_mode_carrier_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_ideal_zero_mode_amplifier(
    hamiltonian: ClauseHamiltonianAudit,
    target_zero_mode_weight: float = 0.5,
    gain_rate: float = 1.0,
) -> ZeroModeAmplifierAudit:
    """Track where an ideal active zero-mode latch can hide the resource cost.

    This is not a universal lower bound on nonlinear substrates. It is a fail-closed
    ledger for the common proposal that a small initial zero-mode component is amplified
    until it becomes a macroscopic pointer.
    """

    if not 0.0 < target_zero_mode_weight < 1.0:
        raise ConstraintHoloError("target zero-mode weight must lie strictly between zero and one")
    if gain_rate <= 0.0:
        raise ConstraintHoloError("gain rate must be positive")

    initial = hamiltonian.normalized_zero_mode_weight
    if initial == 0.0:
        required_gain = None
        ideal_time = None
        energy_ratio = None
    else:
        required_gain = (
            target_zero_mode_weight * (1.0 - initial)
            / (initial * (1.0 - target_zero_mode_weight))
        )
        ideal_time = log(required_gain) / gain_rate if required_gain > 1.0 else 0.0
        energy_ratio = required_gain

    return ZeroModeAmplifierAudit(
        basis_dimension=hamiltonian.basis_dimension,
        zero_mode_degeneracy=hamiltonian.ground_degeneracy,
        initial_zero_mode_weight=initial,
        target_zero_mode_weight=target_zero_mode_weight,
        required_weight_gain=required_gain,
        ideal_exponential_gain_time=ideal_time,
        minimum_output_to_input_energy_ratio=energy_ratio,
        mode_count_if_materialized=hamiltonian.basis_dimension,
        polynomial_time_possible_under_ideal_gain=(
            ideal_time is not None and ideal_time <= hamiltonian.public_variables + 1
        ),
        polynomial_energy_established=False,
        polynomial_mode_carrier_established=False,
        status=(
            "IDEAL_GAIN_CAN_MOVE_EXPONENT_FROM_TIME_TO_GAIN_OR_MODE_RESOURCES__"
            "POLYNOMIAL_TOTAL_RESOURCE_LAW_NOT_ESTABLISHED"
        ),
    )
