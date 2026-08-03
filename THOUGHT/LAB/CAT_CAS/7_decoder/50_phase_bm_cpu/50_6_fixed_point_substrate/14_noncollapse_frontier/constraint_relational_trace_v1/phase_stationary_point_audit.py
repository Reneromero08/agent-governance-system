from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

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
    public_seed_threshold_is_witness: bool
    all_two_variable_renaming_gauges_seed_satisfy: bool
    phase_tangent_jacobian: tuple[tuple[float, float], tuple[float, float]]
    phase_tangent_trace: float
    phase_tangent_determinant: float
    phase_positive_eigenvalue: float
    stationary_point_has_phase_saddle: bool
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


def _two_variable_renaming_gauges(holo: ConstraintHolo) -> tuple[ConstraintHolo, ...]:
    return (
        holo.renamed({"x": "a", "y": "b"}),
        holo.renamed({"x": "b", "y": "a"}),
    )


def _public_seed_threshold_is_witness(holo: ConstraintHolo) -> bool:
    public_seed = public_phase_selector_initial_state(holo)
    return holo.accepts(phase_threshold_assignment(holo, public_seed))


def _stationary_phase_tangent_jacobian(
    holo: ConstraintHolo,
    parameters: SelfOrganizingFlowParameters,
    boundary_release_rate: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the exact cosine-chart tangent block at the positive-sine obstruction.

    For the symmetric two-clause relation at ``c_x=c_y=0``, ``z_x=z_y=1``, short
    memories one, and long memories at the common cap ``L``, the exact-product phase
    linearization is

        [[-rho,       -2L],
         [-2L, -2L - 2rho]].

    Memory-to-phase couplings do not remove these eigenvalues: at the memory boundaries
    the memory equations have no first-order phase dependence, and exact-product phase
    force is selector-independent when short memory equals one.
    """

    cap = max(
        1.0,
        parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
    )
    rho = boundary_release_rate
    return (
        (-rho, -2.0 * cap),
        (-2.0 * cap, -2.0 * cap - 2.0 * rho),
    )


def audit_symmetric_sat_stationary_point(
    *,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    boundary_release_rate: float = 10.0,
) -> SatStationaryPointAudit:
    """Establish the arbitrary-state obstruction and separate it from the public seed.

    The audit still does not prove global convergence from the declared public seed. It
    proves three narrower facts:

    1. a satisfiable non-solution stationary carrier state exists;
    2. its phase tangent block is a saddle for the public cap ``L > rho``;
    3. the declared two-variable public seed already thresholds to a witness under both
       possible variable-renaming order gauges.
    """

    holo = symmetric_sat_stationary_holo()
    reference = reference_existential_trace(holo)
    stationary_state = symmetric_non_solution_stationary_state(holo, parameters)
    stationary_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        stationary_state,
        parameters=parameters,
        boundary_release_rate=boundary_release_rate,
    )
    stationary_assignment = phase_threshold_assignment(holo, stationary_state)
    stationary_energy = sum(phase_clause_violation_values(holo, stationary_state))

    public_seed = public_phase_selector_initial_state(holo)
    public_seed_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        public_seed,
        parameters=parameters,
        boundary_release_rate=boundary_release_rate,
    )
    public_seed_is_witness = holo.accepts(
        phase_threshold_assignment(holo, public_seed)
    )
    gauge_seed_satisfaction = tuple(
        _public_seed_threshold_is_witness(gauge)
        for gauge in _two_variable_renaming_gauges(holo)
    )

    jacobian = _stationary_phase_tangent_jacobian(
        holo,
        parameters,
        boundary_release_rate,
    )
    trace = jacobian[0][0] + jacobian[1][1]
    determinant = (
        jacobian[0][0] * jacobian[1][1]
        - jacobian[0][1] * jacobian[1][0]
    )
    discriminant = trace * trace - 4.0 * determinant
    positive_eigenvalue = (trace + sqrt(discriminant)) / 2.0
    phase_saddle = determinant < 0.0 and positive_eigenvalue > 0.0

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
        public_seed_threshold_is_witness=public_seed_is_witness,
        all_two_variable_renaming_gauges_seed_satisfy=all(
            gauge_seed_satisfaction
        ),
        phase_tangent_jacobian=jacobian,
        phase_tangent_trace=trace,
        phase_tangent_determinant=determinant,
        phase_positive_eigenvalue=positive_eigenvalue,
        stationary_point_has_phase_saddle=phase_saddle,
        arbitrary_state_global_convergence_falsified=obstruction_established,
        public_seed_convergence_status=(
            "PUBLIC_SEED_ALREADY_TERMINAL_WITNESS_ON_THIS_OBSTRUCTION_FORMULA__"
            "GLOBAL_PUBLIC_SEED_THEOREM_UNRESOLVED"
            if public_seed_is_witness and all(gauge_seed_satisfaction)
            else "PUBLIC_SEED_RELATION_TO_OBSTRUCTION_REQUIRES_FURTHER_AUDIT"
        ),
        status=(
            "SAT_NON_SOLUTION_STATIONARY_POINT_ESTABLISHED__"
            "PUBLIC_SEED_SPECIFIC_THEOREM_REQUIRED"
            if obstruction_established
            else "STATIONARY_POINT_OBSTRUCTION_NOT_ESTABLISHED"
        ),
    )
