from __future__ import annotations

from dataclasses import dataclass
import cmath
from fractions import Fraction
import math
from typing import Mapping

from .catalytic_existential_trace import (
    CLAIM_CEILING,
    REFERENCE_VARIABLE_LIMIT,
    iter_boundary_assignments,
    reference_existential_trace,
)
from .constraint_holo import ConstraintHolo, ConstraintHoloError


@dataclass(frozen=True)
class TopologicalRankLatchResult:
    public_variables: int
    public_clauses: int
    symbolic_projector_size: int
    basis_dimension: int
    satisfying_rank: int
    determinant_winding: int
    presence_index: int
    normalized_trace_at_pi: float
    minimum_unique_trace_gap: float
    materialized_carrier_coordinates: int
    determinant_line_filled_modes: int
    program_inverse: str
    restoration_verified: bool
    determinant_projection_error: float
    semantic_status: str
    native_readout_status: str
    claim_ceiling: str = CLAIM_CEILING


class MaterializedPhaseOracleCarrier:
    """Exponential exact-turn reference carrier for the winding identity only."""

    def __init__(self, holo: ConstraintHolo) -> None:
        if len(holo.variables) > REFERENCE_VARIABLE_LIMIT:
            raise ConstraintHoloError("materialized phase-oracle carrier exceeds reference limit")
        self._holo = holo
        self._assignments = tuple(iter_boundary_assignments(holo))
        self._turns = [Fraction(0, 1) for _ in self._assignments]

    @property
    def coordinate_count(self) -> int:
        return len(self._turns)

    def snapshot(self) -> tuple[Fraction, ...]:
        return tuple(self._turns)

    def apply_turns(self, turns: Fraction) -> None:
        if not isinstance(turns, Fraction):
            raise ConstraintHoloError("phase program must use exact rational turns")
        for index, assignment in enumerate(self._assignments):
            if self._holo.accepts(assignment):
                self._turns[index] += turns

    def determinant(self) -> complex:
        total_turns = sum(self._turns, start=Fraction(0, 1))
        angle = 2.0 * math.pi * float(total_turns)
        return cmath.exp(1j * angle)


def phase_oracle_value(
    holo: ConstraintHolo,
    assignment: Mapping[str, bool],
    angle: float,
) -> complex:
    """Public basis action of U_F(angle) = I + (exp(i angle)-1) Q_F."""

    return cmath.exp(1j * angle) if holo.accepts(assignment) else 1.0 + 0.0j


def audit_topological_rank_latch(
    holo: ConstraintHolo,
    variable_limit: int = REFERENCE_VARIABLE_LIMIT,
) -> TopologicalRankLatchResult:
    """Verify the complete index and expose the unresolved native readout.

    For the exact solution projector Q_F, define:

        U_F(theta) = I + (exp(i theta) - 1) Q_F.

    Every satisfying basis state winds once as theta traverses a full loop. Every
    violating basis state remains at one. Therefore determinant winding equals the
    rank of Q_F, which equals the number of satisfying assignments.
    """

    if len(holo.variables) > variable_limit:
        raise ConstraintHoloError("topological rank-latch audit exceeds reference limit")

    boundary = reference_existential_trace(holo, variable_limit=variable_limit)
    dimension = 1 << len(holo.variables)
    satisfying_rank = boundary.witness_count

    carrier = MaterializedPhaseOracleCarrier(holo)
    initial = carrier.snapshot()
    test_turns = Fraction(37, 101)
    test_angle = 2.0 * math.pi * float(test_turns)
    carrier.apply_turns(test_turns)
    forward_determinant = carrier.determinant()
    carrier.apply_turns(-test_turns)
    restored = carrier.snapshot() == initial

    expected_determinant = cmath.exp(1j * test_angle * satisfying_rank)
    projection_error = abs(forward_determinant - expected_determinant)
    tolerance = 1e-12 * max(1, satisfying_rank)
    if projection_error > tolerance:
        raise ConstraintHoloError("materialized determinant disagrees with projector rank")

    normalized_trace_at_pi = 1.0 - (2.0 * satisfying_rank / dimension)
    minimum_unique_gap = 2.0 / dimension

    return TopologicalRankLatchResult(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        symbolic_projector_size=len(holo.variables) + 3 * len(holo.clauses),
        basis_dimension=dimension,
        satisfying_rank=satisfying_rank,
        determinant_winding=satisfying_rank,
        presence_index=int(satisfying_rank > 0),
        normalized_trace_at_pi=normalized_trace_at_pi,
        minimum_unique_trace_gap=minimum_unique_gap,
        materialized_carrier_coordinates=carrier.coordinate_count,
        determinant_line_filled_modes=dimension,
        program_inverse="U_F(theta)^-1 = U_F(-theta)",
        restoration_verified=restored,
        determinant_projection_error=projection_error,
        semantic_status="TOPOLOGICAL_DETERMINANT_WINDING_COMPLETE_FOR_REFERENCE_RELATION",
        native_readout_status=(
            "POLYNOMIAL_NATIVE_DETERMINANT_LINE_SENSOR_NOT_ESTABLISHED__"
            "NORMALIZED_TRACE_HAS_EXPONENTIAL_UNIQUE_WITNESS_GAP"
        ),
    )
