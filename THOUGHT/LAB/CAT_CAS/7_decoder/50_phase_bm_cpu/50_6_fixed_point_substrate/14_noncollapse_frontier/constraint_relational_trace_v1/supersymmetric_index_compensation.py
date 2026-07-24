from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    REFERENCE_VARIABLE_LIMIT,
    iter_boundary_assignments,
)
from .clause_hamiltonian import clause_violation_energy
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class SupersymmetricIndexCompensationAudit:
    public_variables: int
    public_clauses: int
    assignment_basis_dimension: int
    bosonic_dimension: int
    fermionic_dimension: int
    satisfying_assignments: int
    bosonic_zero_modes: int
    fermionic_zero_modes: int
    finite_witten_index: int
    formula_dependent_index_detected: bool
    square_pairing_compensation_status: str
    fredholm_escape_status: str
    physical_instanton_status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_supersymmetric_index_compensation(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> SupersymmetricIndexCompensationAudit:
    """Audit the finite graded pairing built from the clause Hamiltonian.

    Let B and F each have one basis state per assignment and define the odd supercharge

        Q |x,B> = sqrt(E_F(x)) |x,F>.

    Then H={Q,Q^dagger} has energy E_F(x) in both sectors. Every satisfying assignment
    creates one bosonic and one fermionic zero mode. Their Witten-index contributions
    cancel exactly. More generally, for a finite complex with fixed graded dimensions,
    the Euler/Witten index is dim(B)-dim(F), independent of the differential.

    A formula-dependent Fredholm index therefore requires an infinite or open boundary,
    formula-dependent graded dimensions, or a non-Fredholm boundary contribution. None
    is established by the finite clause carrier.
    """

    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("supersymmetric index audit exceeds reference limit")

    energies = tuple(
        clause_violation_energy(holo, assignment)
        for assignment in iter_boundary_assignments(holo)
    )
    dimension = len(energies)
    solutions = sum(energy == 0 for energy in energies)
    index = solutions - solutions

    return SupersymmetricIndexCompensationAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        assignment_basis_dimension=dimension,
        bosonic_dimension=dimension,
        fermionic_dimension=dimension,
        satisfying_assignments=solutions,
        bosonic_zero_modes=solutions,
        fermionic_zero_modes=solutions,
        finite_witten_index=index,
        formula_dependent_index_detected=False,
        square_pairing_compensation_status=(
            "SAT_ZERO_MODES_PAIR_ACROSS_GRADING__FINITE_WITTEN_INDEX_FORMULA_INDEPENDENT"
        ),
        fredholm_escape_status=(
            "OPEN_OR_INFINITE_BOUNDARY_INDEX_CANDIDATE_NOT_ESTABLISHED"
        ),
        physical_instanton_status=(
            "DYNAMICAL_INSTANTON_CHAIN_MAY_GUIDE_PREPARATION__"
            "DOES_NOT_SUPPLY_FINITE_INDEX_DECISION"
        ),
    )
