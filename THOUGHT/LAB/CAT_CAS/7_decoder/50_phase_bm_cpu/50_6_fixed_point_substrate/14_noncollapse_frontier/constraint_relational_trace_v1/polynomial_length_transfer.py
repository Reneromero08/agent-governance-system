from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


@dataclass(frozen=True)
class PolynomialLengthTransferAudit:
    public_variables: int
    public_clauses: int
    maximum_variable_occurrences: int
    assumed_physical_time: float
    long_memory_upper_bound: float
    voltage_speed_upper_bound: float
    short_memory_speed_upper_bound: float
    long_memory_speed_upper_bound: float
    euclidean_speed_upper_bound: float
    trajectory_length_upper_bound: float
    coordinate_status: str
    state_range_status: str
    speed_status: str
    length_status: str
    continuous_model_status: str
    selector_status: str
    boundary_status: str
    transfer_status: str
    claim_ceiling: str = CLAIM_CEILING


def audit_polynomial_length_transfer(
    holo: ConstraintHolo,
    assumed_physical_time: float,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> PolynomialLengthTransferAudit:
    if not isfinite(assumed_physical_time) or assumed_physical_time <= 0:
        raise ConstraintHoloError("physical time bound must be positive and finite")

    variable_count = len(holo.variables)
    clause_count = len(holo.clauses)
    occurrence_degree = {variable: 0 for variable in holo.variables}
    for clause in holo.clauses:
        for literal in clause.literals:
            occurrence_degree[literal.variable] += 1
    maximum_degree = max(occurrence_degree.values(), default=0)

    long_speed = parameters.alpha * max(parameters.delta, 1.0 - parameters.delta)
    short_speed = (
        parameters.beta
        * (1.0 + parameters.epsilon)
        * max(parameters.gamma, 1.0 - parameters.gamma)
    )
    long_bound = 1.0 + assumed_physical_time * long_speed
    voltage_speed = maximum_degree * (
        1.0 + (1.0 + parameters.zeta) * long_bound
    )
    euclidean_speed = sqrt(
        variable_count * voltage_speed**2
        + clause_count * short_speed**2
        + clause_count * long_speed**2
    )

    return PolynomialLengthTransferAudit(
        public_variables=variable_count,
        public_clauses=clause_count,
        maximum_variable_occurrences=maximum_degree,
        assumed_physical_time=assumed_physical_time,
        long_memory_upper_bound=long_bound,
        voltage_speed_upper_bound=voltage_speed,
        short_memory_speed_upper_bound=short_speed,
        long_memory_speed_upper_bound=long_speed,
        euclidean_speed_upper_bound=euclidean_speed,
        trajectory_length_upper_bound=assumed_physical_time * euclidean_speed,
        coordinate_status="N_PLUS_2M_COORDINATES",
        state_range_status=(
            "POLYNOMIAL_IF_PHYSICAL_TIME_IS_POLYNOMIAL_AND_THE_PUBLIC_CAP_EXCEEDS_THIS_BOUND"
        ),
        speed_status="POLYNOMIAL_IF_TIME_AND_PUBLIC_CLAUSE_COUNT_ARE_POLYNOMIAL",
        length_status="POLYNOMIAL_IF_PHYSICAL_TIME_IS_POLYNOMIAL",
        continuous_model_status="PIECEWISE_POLYNOMIAL_CARATHEODORY_FLOW",
        selector_status="MIN_AND_RIGIDITY_SELECTORS_NEED_A_ROBUST_POLYNOMIAL_DILATION",
        boundary_status="PATCHED_BOUNDARY_NEEDS_A_STANDARD_COMPLEXITY_TRANSFER",
        transfer_status="CONDITIONAL__NOT_ESTABLISHED",
    )
