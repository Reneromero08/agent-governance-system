from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING, reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .polynomial_phase_selector_flow import (
    PolynomialPhaseSelectorFlowState,
    phase_clause_violation_values,
    phase_threshold_assignment,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class SatStationaryPointAudit:
    public_variables: int
    public_clauses: int
    witness_count: int
    stationary_phase_energy: float
    stationary_derivative_max_abs: float
    stationary_threshold_is_witness: bool
    public_seed_derivative_max_abs: float
    arbitrary_state_global_convergence_falsified: bool
    public_seed_convergence_status: str
    status: str
    claim_ceiling: str = CLAIM_CEILING


def symmetric_sat_stationary_holo() -> ConstraintHolo:
    """Return a symmetric satisfiable relation with a non-solution stationary point."""

    return ConstraintHolo.build(
        ("x", "y"),
        (
            ClauseRelation(
                (
                    Literal("x"),
                    Literal("y"),
                    Literal("y"),
                )
            ),
            ClauseRelation(
                (
                    Literal("x", False),
                    Literal("y", False),
                    Literal("y", False),
                )
            ),
        ),
    )


def symmetric_non_solution_stationary_state(
    holo: ConstraintHolo,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> PolynomialPhaseSelectorFlowState:
    """Place the carrier on the exact unresolved symmetry manifold.

    The phase gradients from the two clauses cancel. Short memory is fixed at one,
    long memory is fixed at its public cap, and equal selector costs freeze every
    replicator simplex. The resulting full carrier state is stationary even though its
    threshold assignment is not a witness.
    """

    clause_count = len(holo.clauses)
    cap = max(
        1.0,
        parameters.long_memory_cap_factor * max(1, clause_count),
    )
    return PolynomialPhaseSelectorFlowState(
        phase_cosine=tuple(0.0 for _ in holo.variables),
        phase_sine=tuple(1.0 for _ in holo.variables),
        short_memory=tuple(1.0 for _ in holo.clauses),
        long_memory=tuple(cap for _ in holo.clauses),
        clause_selector=tuple(
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) for _ in holo.clauses
        ),
        pair_selector=tuple(
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5) for _ in holo.clauses
        ),
    )


def audit_symmetric_sat_stationary_point() -> SatStationaryPointAudit:
    """Establish the arbitrary-initial-state convergence obstruction.

    This audit does not refute convergence from the declared public seed. It proves that
    the theorem must be seed-specific: the native vector field has a satisfiable but
    non-solution stationary state on an exact symmetry manifold.
    """

    holo = symmetric_sat_stationary_holo()
    reference = reference_existential_trace(holo)
    stationary_state = symmetric_non_solution_stationary_state(holo)
    stationary_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        stationary_state,
    )
    stationary_assignment = phase_threshold_assignment(holo, stationary_state)
    stationary_energy = sum(phase_clause_violation_values(holo, stationary_state))

    public_seed = public_phase_selector_initial_state(holo)
    public_seed_derivative = polynomial_phase_selector_flow_derivative(holo, public_seed)

    stationary_max_abs = stationary_derivative.max_abs()
    public_seed_max_abs = public_seed_derivative.max_abs()
    threshold_is_witness = holo.accepts(stationary_assignment)
    obstruction_established = (
        reference.satisfiable
        and reference.witness_count == 2
        and stationary_energy > 0.0
        and stationary_max_abs == 0.0
        and not threshold_is_witness
        and public_seed_max_abs > 0.0
    )

    return SatStationaryPointAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        witness_count=reference.witness_count,
        stationary_phase_energy=stationary_energy,
        stationary_derivative_max_abs=stationary_max_abs,
        stationary_threshold_is_witness=threshold_is_witness,
        public_seed_derivative_max_abs=public_seed_max_abs,
        arbitrary_state_global_convergence_falsified=obstruction_established,
        public_seed_convergence_status="NOT_DECIDED_BY_STATIONARY_POINT_AUDIT",
        status=(
            "SAT_NON_SOLUTION_STATIONARY_POINT_ESTABLISHED__"
            "PUBLIC_SEED_SPECIFIC_THEOREM_REQUIRED"
            if obstruction_established
            else "STATIONARY_POINT_OBSTRUCTION_NOT_ESTABLISHED"
        ),
    )
