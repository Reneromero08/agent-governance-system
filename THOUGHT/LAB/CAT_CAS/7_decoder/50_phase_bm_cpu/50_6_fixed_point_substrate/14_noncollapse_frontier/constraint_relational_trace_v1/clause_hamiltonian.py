from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Mapping

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    REFERENCE_VARIABLE_LIMIT,
    iter_boundary_assignments,
)
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class ClauseHamiltonianAudit:
    public_variables: int
    public_clauses: int
    symbolic_local_terms: int
    basis_dimension: int
    ground_energy: int
    ground_degeneracy: int
    satisfiable: bool
    unsat_energy_margin: int
    normalized_zero_mode_weight: float
    inverse_participation_amplification: float | None
    unique_witness_log2_amplification: float | None
    phase_evolution_inverse: str
    local_description_status: str
    native_zero_mode_sensor_status: str
    claim_ceiling: str = CLAIM_CEILING


def clause_violation_energy(
    holo: ConstraintHolo,
    assignment: Mapping[str, bool],
) -> int:
    if set(assignment) != set(holo.variables):
        raise ConstraintHoloError("assignment domain must equal the public boundary")
    return sum(not clause.accepts(assignment) for clause in holo.clauses)


def audit_clause_hamiltonian(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> ClauseHamiltonianAudit:
    """Audit H_F = sum_j violation_projector(C_j) on a tiny reference basis.

    H_F is diagonal, positive semidefinite, commuting, and three-local in the public
    clause representation. Its minimum eigenvalue is an integer. Therefore SAT has
    ground energy zero and UNSAT has ground energy at least one.
    """

    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("clause-Hamiltonian audit exceeds reference limit")

    energies = tuple(
        clause_violation_energy(holo, assignment)
        for assignment in iter_boundary_assignments(holo)
    )
    dimension = len(energies)
    ground_energy = min(energies)
    ground_degeneracy = sum(energy == ground_energy for energy in energies)
    satisfying = ground_energy == 0
    zero_mode_weight = (
        ground_degeneracy / dimension if satisfying else 0.0
    )
    amplification = (
        dimension / ground_degeneracy if satisfying and ground_degeneracy else None
    )

    return ClauseHamiltonianAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        symbolic_local_terms=len(holo.clauses),
        basis_dimension=dimension,
        ground_energy=ground_energy,
        ground_degeneracy=ground_degeneracy if satisfying else 0,
        satisfiable=satisfying,
        unsat_energy_margin=(ground_energy if not satisfying else 1),
        normalized_zero_mode_weight=zero_mode_weight,
        inverse_participation_amplification=amplification,
        unique_witness_log2_amplification=(
            log2(amplification) if amplification is not None else None
        ),
        phase_evolution_inverse="exp(-i t H_F)^-1 = exp(+i t H_F)",
        local_description_status="COMMUTING_LOCAL_CLAUSE_HAMILTONIAN_EXACT",
        native_zero_mode_sensor_status=(
            "CONSTANT_ENERGY_MARGIN_ESTABLISHED__"
            "POLYNOMIAL_ZERO_MODE_POPULATION_OR_DETERMINANT_SENSOR_NOT_ESTABLISHED"
        ),
    )
