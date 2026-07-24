from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .clause_hamiltonian import ClauseHamiltonianAudit


@dataclass(frozen=True)
class RankOneResolventAudit:
    basis_dimension: int
    zero_mode_degeneracy: int
    normalized_probe_norm_squared: float
    normalized_zero_pole_residue: float
    unnormalized_probe_norm_squared: float
    unnormalized_zero_pole_residue: float
    unique_witness_normalized_residue: float
    constant_residue_requires_probe_norm_squared: float
    rank_one_update_description_size: int
    cached_resolvent_element_required: str
    polynomial_probe_energy_established: bool
    polynomial_resolvent_access_established: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_rank_one_resolvent_sensor(
    hamiltonian: ClauseHamiltonianAudit,
) -> RankOneResolventAudit:
    """Audit a uniform rank-one probe coupled to the clause Hamiltonian.

    For a normalized all-assignments probe |u>, the pole residue of
    <u|(z-H_F)^-1|u> at zero equals the zero-mode fraction k/N. An unnormalized
    all-ones probe has residue k, but its norm squared is N.
    """

    dimension = hamiltonian.basis_dimension
    degeneracy = hamiltonian.ground_degeneracy
    normalized_residue = degeneracy / dimension
    unnormalized_residue = float(degeneracy)

    return RankOneResolventAudit(
        basis_dimension=dimension,
        zero_mode_degeneracy=degeneracy,
        normalized_probe_norm_squared=1.0,
        normalized_zero_pole_residue=normalized_residue,
        unnormalized_probe_norm_squared=float(dimension),
        unnormalized_zero_pole_residue=unnormalized_residue,
        unique_witness_normalized_residue=1.0 / dimension,
        constant_residue_requires_probe_norm_squared=float(dimension),
        rank_one_update_description_size=1,
        cached_resolvent_element_required=(
            "<probe|(z-H_F)^-1|probe> across the complete unresolved relation"
        ),
        polynomial_probe_energy_established=False,
        polynomial_resolvent_access_established=False,
        status=(
            "RANK_ONE_DESCRIPTION_COMPACT__NORMALIZED_POLE_RESIDUE_EXPONENTIALLY_SMALL__"
            "UNNORMALIZED_CONSTANT_RESIDUE_REQUIRES_EXPONENTIAL_PROBE_NORM"
        ),
    )
