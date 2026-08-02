from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .polynomial_phase_selector_flow import (
    PolynomialPhaseSelectorFlowState,
    polynomial_phase_selector_flow_derivative,
    public_phase_selector_initial_state,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class ReducedExactProductPhaseState:
    phase_cosine: tuple[float, ...]
    phase_sine: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    clause_selector: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class ReducedExactProductPhaseDerivative:
    phase_cosine: tuple[float, ...]
    phase_sine: tuple[float, ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    clause_selector: tuple[tuple[float, float, float], ...]

    def max_abs(self) -> float:
        values = list(
            self.phase_cosine
            + self.phase_sine
            + self.short_memory
            + self.long_memory
        )
        for triple in self.clause_selector:
            values.extend(triple)
        return max((abs(value) for value in values), default=0.0)


@dataclass(frozen=True)
class ReducedExactProductPhaseAudit:
    public_variables: int
    public_clauses: int
    native_state_coordinates: int
    logit_chart_coordinates: int
    removed_pair_selector_coordinates: int
    active_derivative_residual: float
    reduction_exact: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def public_reduced_exact_product_state(
    holo: ConstraintHolo,
) -> ReducedExactProductPhaseState:
    full = public_phase_selector_initial_state(holo)
    return ReducedExactProductPhaseState(
        phase_cosine=full.phase_cosine,
        phase_sine=full.phase_sine,
        short_memory=full.short_memory,
        long_memory=full.long_memory,
        clause_selector=full.clause_selector,
    )


def _embed_reduced_state(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
) -> PolynomialPhaseSelectorFlowState:
    clauses = len(holo.clauses)
    if (
        len(state.phase_cosine) != len(holo.variables)
        or len(state.phase_sine) != len(holo.variables)
        or len(state.short_memory) != clauses
        or len(state.long_memory) != clauses
        or len(state.clause_selector) != clauses
    ):
        raise ConstraintHoloError("reduced exact-product phase state dimension mismatch")
    return PolynomialPhaseSelectorFlowState(
        phase_cosine=state.phase_cosine,
        phase_sine=state.phase_sine,
        short_memory=state.short_memory,
        long_memory=state.long_memory,
        clause_selector=state.clause_selector,
        pair_selector=tuple(
            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5) for _ in holo.clauses
        ),
    )


def reduced_exact_product_phase_derivative(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
) -> ReducedExactProductPhaseDerivative:
    """Evaluate the active exact-product carrier with no pair-selector state.

    A canonical pair-selector embedding is used only to reuse the qualified reference
    field. Exact-product pair selectors are proven dynamically decoupled, so the active
    derivative is independent of this embedding.
    """

    full_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        _embed_reduced_state(holo, state),
        parameters=parameters,
        selector_rate=selector_rate,
        boundary_release_rate=boundary_release_rate,
        truth_gain=truth_gain,
        gradient_mode="exact_product",
    )
    return ReducedExactProductPhaseDerivative(
        phase_cosine=full_derivative.phase_cosine,
        phase_sine=full_derivative.phase_sine,
        short_memory=full_derivative.short_memory,
        long_memory=full_derivative.long_memory,
        clause_selector=full_derivative.clause_selector,
    )


def _flatten_reduced_derivative(
    derivative: ReducedExactProductPhaseDerivative,
) -> tuple[float, ...]:
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


def audit_reduced_exact_product_phase_flow(
    holo: ConstraintHolo,
) -> ReducedExactProductPhaseAudit:
    reduced = public_reduced_exact_product_state(holo)
    reduced_derivative = reduced_exact_product_phase_derivative(holo, reduced)
    full = public_phase_selector_initial_state(holo)
    full_derivative = polynomial_phase_selector_flow_derivative(
        holo,
        full,
        gradient_mode="exact_product",
    )
    full_active = (
        full_derivative.phase_cosine
        + full_derivative.phase_sine
        + full_derivative.short_memory
        + full_derivative.long_memory
        + tuple(
            value
            for selector_derivative in full_derivative.clause_selector
            for value in selector_derivative
        )
    )
    reduced_active = _flatten_reduced_derivative(reduced_derivative)
    residual = max(
        (
            abs(left - right)
            for left, right in zip(full_active, reduced_active, strict=True)
        ),
        default=0.0,
    )
    clauses = len(holo.clauses)
    exact = residual == 0.0
    return ReducedExactProductPhaseAudit(
        public_variables=len(holo.variables),
        public_clauses=clauses,
        native_state_coordinates=2 * len(holo.variables) + 5 * clauses,
        logit_chart_coordinates=len(holo.variables) + 4 * clauses,
        removed_pair_selector_coordinates=6 * clauses,
        active_derivative_residual=residual,
        reduction_exact=exact,
        status=(
            "REDUCED_EXACT_PRODUCT_PHASE_CARRIER_MATCHES_QUALIFIED_ACTIVE_FIELD"
            if exact
            else "REDUCED_EXACT_PRODUCT_PHASE_CARRIER_MISMATCH"
        ),
    )
