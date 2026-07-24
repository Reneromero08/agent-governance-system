from __future__ import annotations

from dataclasses import dataclass
from math import log2

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class ProvenBasinAccessAudit:
    public_variables: int
    public_clauses: int
    isolated_solution_coordinates: int
    gamma: float
    guaranteed_voltage_fraction: float
    guaranteed_voltage_log2_fraction: float
    short_memory_face_dimension: int
    short_memory_full_volume_fraction: float
    public_short_memory_zero_is_allowed: bool
    solution_orthant_requires_unknown_signs: bool
    unique_solution_voltage_fraction_status: str
    random_full_state_guarantee_status: str
    deterministic_public_seed_guarantee_status: str
    global_instanton_access_status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_proven_basin_access(
    holo: ConstraintHolo,
    isolated_solution_coordinates: int | None = None,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> ProvenBasinAccessAudit:
    """Audit what the published restricted-orthant basin theorem actually covers.

    For a solution with proper isolated index set I, the proven voltage basin contains

        J0 = product_{i in I} [solution_sign_i (1-2 gamma), solution_sign_i]
             times unrestricted remaining voltages.

    Relative to [-1,1]^n, each isolated coordinate contributes fraction gamma. The
    theorem also fixes all short-memory coordinates to zero. A unique isolated solution
    therefore has proven voltage fraction gamma^n, and the full subset has zero Lebesgue
    measure if short memories are sampled continuously rather than set deliberately.
    """

    variable_count = len(holo.variables)
    clause_count = len(holo.clauses)
    isolated = variable_count if isolated_solution_coordinates is None else isolated_solution_coordinates
    if not 0 <= isolated <= variable_count:
        raise ConstraintHoloError("isolated solution coordinate count is outside the public boundary")

    fraction = parameters.gamma**isolated
    return ProvenBasinAccessAudit(
        public_variables=variable_count,
        public_clauses=clause_count,
        isolated_solution_coordinates=isolated,
        gamma=parameters.gamma,
        guaranteed_voltage_fraction=fraction,
        guaranteed_voltage_log2_fraction=(
            isolated * log2(parameters.gamma) if isolated else 0.0
        ),
        short_memory_face_dimension=clause_count,
        short_memory_full_volume_fraction=(1.0 if clause_count == 0 else 0.0),
        public_short_memory_zero_is_allowed=True,
        solution_orthant_requires_unknown_signs=(isolated > 0),
        unique_solution_voltage_fraction_status=(
            "EXPONENTIALLY_SMALL_GAMMA_TO_N_FOR_ISOLATED_UNIQUE_SOLUTION"
            if isolated == variable_count and variable_count > 0
            else "DEPENDS_ON_ISOLATED_INDEX_SIZE"
        ),
        random_full_state_guarantee_status=(
            "PROVEN_BASIN_FACE_HAS_ZERO_FULL_VOLUME_WHEN_SHORT_MEMORY_IS_CONTINUOUSLY_RANDOM"
        ),
        deterministic_public_seed_guarantee_status=(
            "NOT_ESTABLISHED_WITHOUT_KNOWING_A_SOLUTION_ORTHANT"
        ),
        global_instanton_access_status=(
            "REQUIRES_A_SEPARATE_THEOREM_THAT_PUBLIC_SEED_REACHES_THE_INSTANTON_SOLUTION_CHAIN"
        ),
    )
