from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo
from .hybrid_event_geometry_audit import (
    HybridGuardEventGeometryAudit,
    audit_hybrid_guard_event_geometry,
)
from .odd_parity_fixed_deadline_counterexample import odd_parity_three_variable_holo
from .reduced_exact_product_phase_flow import (
    ReducedExactProductPhaseState,
    reduced_exact_product_phase_derivative,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class NativeHybridEventGeometryAudit:
    event_geometry: HybridGuardEventGeometryAudit
    native_derivative_max_abs: float
    full_carrier_stationary: bool
    simple_guard_surface: bool
    smooth_native_grazing_event: bool
    structural_transversality_established: bool
    public_seed_transversality_status: str
    status: str
    claim_ceiling: str = CLAIM_CEILING


def odd_parity_simple_native_grazing_state(
    holo: ConstraintHolo | None = None,
) -> tuple[ConstraintHolo, ReducedExactProductPhaseState]:
    """Return an exact simple guard-boundary state with zero native normal velocity.

    The phase point ``(0,-1/2,-1/2)`` lies on the unique odd-parity witness-entry face
    controlled by ``x1`` in the all-positive clause. Setting short memory to one and long
    memory to zero suppresses both exact-gradient and rigidity force channels. At
    ``c1=0`` the boundary-release contribution also vanishes, so the active normal
    velocity is exactly zero while other phase coordinates can still move.
    """

    relation = holo or odd_parity_three_variable_holo()
    state = ReducedExactProductPhaseState(
        phase_cosine=(0.0, -0.5, -0.5),
        phase_sine=(1.0, sqrt(3.0) / 2.0, sqrt(3.0) / 2.0),
        short_memory=tuple(1.0 for _ in relation.clauses),
        long_memory=tuple(0.0 for _ in relation.clauses),
        clause_selector=tuple(
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            for _ in relation.clauses
        ),
    )
    return relation, state


def audit_native_hybrid_event_geometry(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
    *,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
    tolerance: float = 1.0e-12,
) -> NativeHybridEventGeometryAudit:
    derivative = reduced_exact_product_phase_derivative(
        holo,
        state,
        parameters=parameters,
        selector_rate=selector_rate,
        boundary_release_rate=boundary_release_rate,
        truth_gain=truth_gain,
    )
    geometry = audit_hybrid_guard_event_geometry(
        holo,
        state.phase_cosine,
        derivative.phase_cosine,
        tolerance=tolerance,
    )
    derivative_max = derivative.max_abs()
    simple_surface = geometry.event_surface and geometry.unique_classical_derivative
    smooth_grazing = (
        simple_surface
        and geometry.classification == "SMOOTH_GRAZING_OR_HIGHER_ORDER_EVENT"
    )

    return NativeHybridEventGeometryAudit(
        event_geometry=geometry,
        native_derivative_max_abs=derivative_max,
        full_carrier_stationary=derivative_max <= tolerance,
        simple_guard_surface=simple_surface,
        smooth_native_grazing_event=smooth_grazing,
        structural_transversality_established=False,
        public_seed_transversality_status="NOT_DECIDED_BY_ARBITRARY_STATE_CONTROL",
        status=(
            "NATIVE_SIMPLE_GUARD_GRAZING_STATE_ESTABLISHED__"
            "PUBLIC_SEED_TRANSVERSALITY_REQUIRES_SEPARATE_THEOREM"
            if smooth_grazing
            else "NATIVE_GUARD_GRAZING_CONTROL_NOT_ESTABLISHED"
        ),
    )


def audit_odd_parity_simple_native_grazing_control() -> NativeHybridEventGeometryAudit:
    holo, state = odd_parity_simple_native_grazing_state()
    return audit_native_hybrid_event_geometry(holo, state)
