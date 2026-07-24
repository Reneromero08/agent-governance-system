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
from .parity_holonomy import (
    ParityConstraint,
    ParityInstance,
    Z2PhaseCarrier,
    Z2TransportProgram,
    calibrate_parity_holonomy,
    compile_z2_transport,
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
    "MaterializedPhaseOracleCarrier",
    "ParityConstraint",
    "ParityInstance",
    "ReferenceBoundaryResult",
    "ReversibleDilationAudit",
    "TopologicalRankLatchResult",
    "WitnessExtractionResult",
    "Z2PhaseCarrier",
    "Z2TransportProgram",
    "audit_topological_rank_latch",
    "calibrate_parity_holonomy",
    "compile_z2_transport",
    "extract_witness_by_boundary_self_reduction",
    "phase_oracle_value",
    "reference_existential_trace",
    "restrict_public_relation",
]
