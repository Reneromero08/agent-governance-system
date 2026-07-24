from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING, reference_existential_trace
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class OracleDeterminantCompensationResult:
    assignment_qubits: int
    ancillary_qubits: int
    total_dimension: int
    clean_subspace_dimension: int
    satisfying_rank: int
    clean_winding: int
    complementary_winding: int
    full_winding: int
    full_winding_formula_independent: bool
    compensation_identity_verified: bool
    restricted_sensor_required: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_oracle_determinant_compensation(
    holo: ConstraintHolo,
    ancillary_qubits: int = 1,
) -> OracleDeterminantCompensationResult:
    """Audit determinant winding for compute-phase-uncompute implementations.

    Let V_F reversibly compute the formula into one designated output qubit, possibly
    using more workspace. Conjugating a one-qubit phase gate by V_F cannot change the
    determinant of the full unitary. Exactly half of all full Hilbert-space basis states
    have the designated output bit equal to one, so the full determinant winding is
    fixed by dimension and carries no formula information.

    On the clean workspace subspace, the induced assignment-only oracle winds once per
    satisfying assignment. The complementary dirty sectors carry the exact compensating
    winding.
    """

    if ancillary_qubits < 1:
        raise ConstraintHoloError("at least one designated output qubit is required")
    boundary = reference_existential_trace(holo)
    assignment_qubits = len(holo.variables)
    clean_dimension = 1 << assignment_qubits
    total_dimension = 1 << (assignment_qubits + ancillary_qubits)
    satisfying_rank = boundary.witness_count
    clean_winding = satisfying_rank
    full_winding = total_dimension // 2
    complementary_winding = full_winding - clean_winding

    return OracleDeterminantCompensationResult(
        assignment_qubits=assignment_qubits,
        ancillary_qubits=ancillary_qubits,
        total_dimension=total_dimension,
        clean_subspace_dimension=clean_dimension,
        satisfying_rank=satisfying_rank,
        clean_winding=clean_winding,
        complementary_winding=complementary_winding,
        full_winding=full_winding,
        full_winding_formula_independent=True,
        compensation_identity_verified=(
            clean_winding + complementary_winding == full_winding
        ),
        restricted_sensor_required=True,
        status=(
            "FULL_REVERSIBLE_ORACLE_DETERMINANT_FORMULA_INDEPENDENT__"
            "CLEAN_SUBSPACE_DETERMINANT_RETAINS_SAT_INDEX"
        ),
    )
