from __future__ import annotations

from dataclasses import dataclass

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo


@dataclass(frozen=True)
class CotangentFlowLiftAudit:
    primal_state_coordinates: int
    lifted_state_coordinates: int
    local_description_polynomial: bool
    lifted_hamiltonian: str
    primal_flow_recovered: bool
    cotangent_transport_law: str
    smooth_region_phase_volume_preserved: bool
    exact_negative_time_inverse_on_smooth_regions: bool
    attractor_contraction_compensation: str
    projected_boundary_status: str
    switching_surface_status: str
    momentum_dynamic_range_status: str
    complete_environment_restoration_status: str
    polynomial_total_resources_established: bool
    claim_ceiling: str = CLAIM_CEILING


def audit_cotangent_flow_lift(holo: ConstraintHolo) -> CotangentFlowLiftAudit:
    """Audit the canonical reversible lift of a clause-flow vector field.

    For a smooth autonomous flow q_dot = f(q), the cotangent Hamiltonian

        H(q, p) = p dot f(q)

    gives

        q_dot = f(q)
        p_dot = -Df(q)^T p.

    The finite-time transport is (q, p) -> (Phi_t(q), D Phi_t(q)^(-T) p), so negative
    time is the program-derived inverse. This is a real reversible dilation of the
    interior vector field, not transcript replay.

    The current self-organizing reference flow also uses projected coordinate bounds
    and piecewise minimum selectors. Those event surfaces are not yet included in a
    globally reversible carrier. Stable contraction in q is compensated by expansion
    in p, so polynomial momentum range and precision remain proof obligations.
    """

    primal_dimension = len(holo.variables) + 2 * len(holo.clauses)
    return CotangentFlowLiftAudit(
        primal_state_coordinates=primal_dimension,
        lifted_state_coordinates=2 * primal_dimension,
        local_description_polynomial=True,
        lifted_hamiltonian="H(q,p)=p_dot_f_F(q)",
        primal_flow_recovered=True,
        cotangent_transport_law="p(t)=D_Phi_t(q0)^(-T) p(0)",
        smooth_region_phase_volume_preserved=True,
        exact_negative_time_inverse_on_smooth_regions=True,
        attractor_contraction_compensation=(
            "CONTRACTION_IN_PRIMAL_DIRECTIONS_REAPPEARS_AS_COTANGENT_EXPANSION"
        ),
        projected_boundary_status=(
            "HARD_PROJECTION_IS_MANY_TO_ONE__SMOOTH_REVERSIBLE_BOUNDARY_CARRIER_NOT_ESTABLISHED"
        ),
        switching_surface_status=(
            "MIN_SELECTOR_EVENT_ORDER_MUST_BE_CARRIED_OR_SMOOTHED_FOR_GLOBAL_INVERSE"
        ),
        momentum_dynamic_range_status="POLYNOMIAL_BOUND_NOT_ESTABLISHED",
        complete_environment_restoration_status="NOT_ESTABLISHED",
        polynomial_total_resources_established=False,
    )
