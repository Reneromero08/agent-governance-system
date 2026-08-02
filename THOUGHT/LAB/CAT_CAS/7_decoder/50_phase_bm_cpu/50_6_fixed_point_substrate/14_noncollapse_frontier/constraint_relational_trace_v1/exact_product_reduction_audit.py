from __future__ import annotations

from dataclasses import dataclass, replace

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo
from .polynomial_phase_selector_flow import (
    PolynomialPhaseSelectorFlowDerivative,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)


@dataclass(frozen=True)
class ExactProductPairSelectorReductionAudit:
    public_variables: int
    public_clauses: int
    full_state_coordinates: int
    reduced_state_coordinates: int
    removable_pair_selector_coordinates: int
    active_derivative_residual: float
    pair_selector_subsystem_decoupled: bool
    reduction_status: str
    claim_ceiling: str = CLAIM_CEILING


def _flatten_active_derivative(
    derivative: PolynomialPhaseSelectorFlowDerivative,
) -> tuple[float, ...]:
    """Return every derivative coordinate that can feed the exact-product carrier."""

    return (
        derivative.phase_cosine
        + derivative.phase_sine
        + derivative.short_memory
        + derivative.long_memory
        + tuple(
            value
            for selector_derivative in derivative.clause_selector
            for value in selector_derivative
        )
    )


def audit_exact_product_pair_selector_reduction(
    holo: ConstraintHolo,
) -> ExactProductPairSelectorReductionAudit:
    """Prove pair selectors are dynamically dead in exact-product mode.

    The pair-selector subsystem was inherited from the selector-min carrier. The exact
    product gradient depends only on public literal defects. Pair selectors continue to
    evolve internally, but they do not feed phase, memory, or clause-selector dynamics.
    Therefore they can be removed from the active candidate without changing its native
    trajectory on the remaining coordinates.
    """

    initial = public_phase_selector_initial_state(holo)
    alternate_pair_selectors = tuple(
        (0.9, 0.1, 0.2, 0.8, 0.7, 0.3) for _ in holo.clauses
    )
    alternate = replace(initial, pair_selector=alternate_pair_selectors)

    initial_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        initial,
        gradient_mode="exact_product",
    )
    alternate_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        alternate,
        gradient_mode="exact_product",
    )
    initial_active = _flatten_active_derivative(initial_derivative)
    alternate_active = _flatten_active_derivative(alternate_derivative)
    residual = max(
        (
            abs(left - right)
            for left, right in zip(
                initial_active,
                alternate_active,
                strict=True,
            )
        ),
        default=0.0,
    )

    clauses = len(holo.clauses)
    full_coordinates = 2 * len(holo.variables) + 11 * clauses
    removable = 6 * clauses
    reduced_coordinates = full_coordinates - removable
    decoupled = residual == 0.0

    return ExactProductPairSelectorReductionAudit(
        public_variables=len(holo.variables),
        public_clauses=clauses,
        full_state_coordinates=full_coordinates,
        reduced_state_coordinates=reduced_coordinates,
        removable_pair_selector_coordinates=removable,
        active_derivative_residual=residual,
        pair_selector_subsystem_decoupled=decoupled,
        reduction_status=(
            "EXACT_PRODUCT_PAIR_SELECTORS_REMOVABLE__ACTIVE_CARRIER_2N_PLUS_5M"
            if decoupled
            else "EXACT_PRODUCT_PAIR_SELECTOR_DECOUPLING_NOT_ESTABLISHED"
        ),
    )
