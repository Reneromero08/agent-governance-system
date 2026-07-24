"""Constraint Relational Trace proof-campaign package."""

from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .catalytic_existential_trace import (
    CLAIM_CEILING,
    FactorizedProjectorCandidate,
    ReferenceBoundaryResult,
    ReversibleDilationAudit,
    reference_existential_trace,
)
from .conditional_p_equals_np import (
    BoundaryDecision,
    WitnessExtractionResult,
    extract_witness_by_boundary_self_reduction,
    restrict_public_relation,
)
from .mpo_configuration_audit import (
    MPOConfigurationAudit,
    audit_control_symbol_mpo_projection,
)
from .oracle_determinant_compensation import (
    OracleDeterminantCompensationResult,
    audit_oracle_determinant_compensation,
)
from .parity_holonomy import (
    ParityConstraint,
    ParityInstance,
    Z2PhaseCarrier,
    Z2TransportProgram,
    calibrate_parity_holonomy,
    compile_z2_transport,
)
from .relational_width_audit import (
    RelationalWidthAudit,
    audit_residual_relation_width,
    equality_relation_holo,
)
from .topological_rank_latch import (
    MaterializedPhaseOracleCarrier,
    TopologicalRankLatchResult,
    audit_topological_rank_latch,
    phase_oracle_value,
)

__all__ = [
    "BoundaryDecision",
    "CLAIM_CEILING",
    "ClauseRelation",
    "ConstraintHolo",
    "FactorizedProjectorCandidate",
    "Literal",
    "MPOConfigurationAudit",
    "MaterializedPhaseOracleCarrier",
    "OracleDeterminantCompensationResult",
    "ParityConstraint",
    "ParityInstance",
    "ReferenceBoundaryResult",
    "RelationalWidthAudit",
    "ReversibleDilationAudit",
    "TopologicalRankLatchResult",
    "WitnessExtractionResult",
    "Z2PhaseCarrier",
    "Z2TransportProgram",
    "audit_control_symbol_mpo_projection",
    "audit_oracle_determinant_compensation",
    "audit_residual_relation_width",
    "audit_topological_rank_latch",
    "calibrate_parity_holonomy",
    "compile_z2_transport",
    "equality_relation_holo",
    "extract_witness_by_boundary_self_reduction",
    "phase_oracle_value",
    "reference_existential_trace",
    "restrict_public_relation",
]
