from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHoloError


@dataclass(frozen=True)
class MPOConfigurationAudit:
    control_states: int
    alphabet_symbols: int
    bounded_tape_cells: int
    reported_bond_dimension: int
    full_bounded_configuration_count: int
    omitted_head_positions: int
    omitted_tape_configurations: int
    compression_ratio: float
    exact_configuration_injective: bool
    invariant_scope: str
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_control_symbol_mpo_projection(
    control_states: int,
    alphabet_symbols: int,
    bounded_tape_cells: int,
) -> MPOConfigurationAudit:
    """Compare the historical `(state, current symbol)` bond to real configurations."""

    if control_states < 1 or alphabet_symbols < 2 or bounded_tape_cells < 1:
        raise ConstraintHoloError("invalid bounded machine dimensions")

    reported_bond_dimension = control_states * alphabet_symbols
    tape_configurations = alphabet_symbols**bounded_tape_cells
    full_configurations = control_states * bounded_tape_cells * tape_configurations

    return MPOConfigurationAudit(
        control_states=control_states,
        alphabet_symbols=alphabet_symbols,
        bounded_tape_cells=bounded_tape_cells,
        reported_bond_dimension=reported_bond_dimension,
        full_bounded_configuration_count=full_configurations,
        omitted_head_positions=bounded_tape_cells,
        omitted_tape_configurations=tape_configurations,
        compression_ratio=full_configurations / reported_bond_dimension,
        exact_configuration_injective=(
            reported_bond_dimension >= full_configurations
        ),
        invariant_scope="finite_control_symbol_transition_graph_only",
        status=(
            "MPO_CONTROL_SYMBOL_PROJECTION_NOT_AN_EXACT_CONFIGURATION_CARRIER"
        ),
    )
