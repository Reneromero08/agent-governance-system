from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ClauseRelation, ConstraintHolo, Literal
from .polynomial_phase_selector_flow import (
    PolynomialPhaseSelectorFlowState,
    exact_clause_violation_from_defects,
    polynomial_phase_selector_flow_derivative,
)


@dataclass(frozen=True)
class PhaseEnergyMonotonicityAudit:
    public_variables: int
    public_clauses: int
    clause_energy: float
    directional_energy_derivative: float
    energy_increases: bool
    release_term_present: bool
    clause_energy_lyapunov_status: str
    claim_ceiling: str = CLAIM_CEILING


def one_clause_positive_holo() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x", "y", "z"),
        (
            ClauseRelation(
                (
                    Literal("x"),
                    Literal("y"),
                    Literal("z"),
                )
            ),
        ),
    )


def phase_energy_growth_state() -> PolynomialPhaseSelectorFlowState:
    cosine = 0.5
    sine = sqrt(3.0) / 2.0
    return PolynomialPhaseSelectorFlowState(
        phase_cosine=(cosine, cosine, cosine),
        phase_sine=(sine, sine, sine),
        short_memory=(1.0,),
        long_memory=(1.0,),
        clause_selector=((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),),
        pair_selector=((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),),
    )


def audit_phase_energy_monotonicity(
    truth_gain: float = 4.0,
    boundary_release_rate: float = 10.0,
) -> PhaseEnergyMonotonicityAudit:
    """Show that exact clause energy is not a global Lyapunov function.

    The exact product gradient alone descends clause energy. The public boundary-release
    term is required to free wrong Boolean corners, but it can temporarily increase the
    same energy. This explicit one-clause state has positive dE/dt, so a convergence proof
    needs an augmented functional or a topological argument rather than raw clause energy.
    """

    holo = one_clause_positive_holo()
    state = phase_energy_growth_state()
    defects = tuple(1.0 - value for value in state.phase_cosine)
    energy = exact_clause_violation_from_defects(defects, truth_gain)  # type: ignore[arg-type]
    derivative = polynomial_phase_selector_flow_derivative(
        holo,
        state,
        truth_gain=truth_gain,
        boundary_release_rate=boundary_release_rate,
        gradient_mode="exact_product",
    )

    gradient_coefficient = truth_gain / 8.0
    partial_derivatives = tuple(
        -gradient_coefficient * defects[(index + 1) % 3] * defects[(index + 2) % 3]
        for index in range(3)
    )
    energy_rate = sum(
        partial * phase_rate
        for partial, phase_rate in zip(
            partial_derivatives,
            derivative.phase_cosine,
            strict=True,
        )
    )
    increases = energy_rate > 0.0

    return PhaseEnergyMonotonicityAudit(
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        clause_energy=energy,
        directional_energy_derivative=energy_rate,
        energy_increases=increases,
        release_term_present=boundary_release_rate > 0.0,
        clause_energy_lyapunov_status=(
            "CLAUSE_PRODUCT_ENERGY_NOT_GLOBAL_LYAPUNOV__AUGMENTED_FUNCTIONAL_REQUIRED"
            if increases
            else "CLAUSE_PRODUCT_ENERGY_COUNTEREXAMPLE_NOT_ESTABLISHED"
        ),
    )
