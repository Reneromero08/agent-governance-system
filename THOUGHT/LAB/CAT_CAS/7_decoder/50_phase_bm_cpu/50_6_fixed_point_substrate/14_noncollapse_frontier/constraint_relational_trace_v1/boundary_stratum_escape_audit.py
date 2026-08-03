from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .reduced_exact_product_phase_flow import (
    ReducedExactProductPhaseState,
    reduced_exact_product_phase_derivative,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


PROPOSAL_KINDS = (
    "stationary_point",
    "moving_invariant_set",
    "asymptotic_numerical_approach",
)

CLASSIFICATIONS = (
    "NOT_INVARIANT",
    "INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION",
    "INVARIANT_STABLE_OR_CENTER_UNRESOLVED",
    "NON_SOLUTION_ATTRACTOR_CANDIDATE",
    "SATISFYING_INVARIANT_SECTION",
)


@dataclass(frozen=True)
class MemoryStratumDescriptor:
    label: str
    short_location: str
    long_location: str
    required_clause_violation: float | None
    forward_compatible_with_public_seed: bool = True

    @property
    def has_interior_memory(self) -> bool:
        return self.short_location == "INTERIOR" or self.long_location == "INTERIOR"


@dataclass(frozen=True)
class NormalExponent:
    coordinate: str
    value: float


@dataclass(frozen=True)
class ClauseBoundaryStratumAudit:
    clause_index: int
    stratum: MemoryStratumDescriptor
    selector_support: tuple[bool, bool, bool]
    clause_violation: float
    clause_violation_derivative: float
    short_zero_exponent: float
    short_one_exponent: float
    long_zero_exponent: float
    long_cap_exponent: float
    memory_coordinate_matches_stratum: bool
    required_violation_matches_stratum: bool
    memory_field_residual: float
    memory_tangency_residual: float
    selector_support_residual: float
    selector_stationarity_residual: float
    selector_field_residual: float
    observed_normal_exponents: tuple[NormalExponent, ...]
    pointwise_boundary_tangent: bool


@dataclass(frozen=True)
class BoundaryStratumEscapeAudit:
    proposal_kind: str
    classification: str
    public_variables: int
    public_clauses: int
    phase_circle_residual: float
    phase_field_residual: float
    maximum_memory_field_residual: float
    maximum_memory_tangency_residual: float
    maximum_selector_stationarity_residual: float
    maximum_selector_field_residual: float
    pointwise_boundary_tangency_verified: bool
    stationary_point_verified: bool
    moving_invariant_set_globally_verified: bool
    numerical_approach_only: bool
    terminal_assignment_satisfies_public_relation: bool
    repelling_directions: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    clause_audits: tuple[ClauseBoundaryStratumAudit, ...]
    status: str
    claim_ceiling: str = CLAIM_CEILING


def forward_compatible_memory_strata(
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
) -> tuple[MemoryStratumDescriptor, ...]:
    """Return the five public-seed-compatible stationary memory classes."""

    return (
        MemoryStratumDescriptor(
            "SHORT_ZERO__LONG_ZERO__C_ARBITRARY",
            "ZERO",
            "ZERO",
            None,
        ),
        MemoryStratumDescriptor(
            "SHORT_ZERO__LONG_CAP__C_ARBITRARY",
            "ZERO",
            "CAP",
            None,
        ),
        MemoryStratumDescriptor(
            "SHORT_ONE__LONG_CAP__C_ARBITRARY",
            "ONE",
            "CAP",
            None,
        ),
        MemoryStratumDescriptor(
            "SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP",
            "INTERIOR",
            "CAP",
            parameters.gamma,
        ),
        MemoryStratumDescriptor(
            "SHORT_ZERO__LONG_INTERIOR_C_EQ_DELTA",
            "ZERO",
            "INTERIOR",
            parameters.delta,
        ),
    )


def _descriptor_by_label(
    label: str,
    parameters: SelfOrganizingFlowParameters,
) -> MemoryStratumDescriptor:
    for descriptor in forward_compatible_memory_strata(parameters):
        if descriptor.label == label:
            return descriptor
    raise ConstraintHoloError(f"unsupported public-seed memory stratum: {label!r}")


def _validate_supports(
    state: ReducedExactProductPhaseState,
    selector_supports: tuple[tuple[bool, bool, bool], ...],
    tolerance: float,
) -> None:
    if len(selector_supports) != len(state.clause_selector):
        raise ConstraintHoloError("selector support count must match public clauses")
    for clause_index, (weights, support) in enumerate(
        zip(state.clause_selector, selector_supports, strict=True)
    ):
        if len(weights) != 3 or len(support) != 3 or not any(support):
            raise ConstraintHoloError(
                f"incomplete selector support for clause {clause_index}"
            )
        if not all(isinstance(value, bool) for value in support):
            raise ConstraintHoloError("selector support entries must be Boolean")
        for weight, included in zip(weights, support, strict=True):
            if included and weight <= tolerance:
                raise ConstraintHoloError(
                    f"selector support omits positive mass for clause {clause_index}"
                )
            if not included and abs(weight) > tolerance:
                raise ConstraintHoloError(
                    f"selector support excludes nonzero mass for clause {clause_index}"
                )


def _validate_state(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
    parameters: SelfOrganizingFlowParameters,
    tolerance: float,
) -> float:
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ConstraintHoloError("boundary-stratum tolerance must be positive and finite")
    n = len(holo.variables)
    m = len(holo.clauses)
    if (
        len(state.phase_cosine) != n
        or len(state.phase_sine) != n
        or len(state.short_memory) != m
        or len(state.long_memory) != m
        or len(state.clause_selector) != m
    ):
        raise ConstraintHoloError("boundary-stratum state dimension mismatch")
    values = (
        state.phase_cosine
        + state.phase_sine
        + state.short_memory
        + state.long_memory
        + tuple(value for weights in state.clause_selector for value in weights)
    )
    if not all(isfinite(value) for value in values):
        raise ConstraintHoloError("boundary-stratum state contains NaN or infinity")
    cap = max(
        1.0,
        parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
    )
    if not all(0.0 <= value <= 1.0 for value in state.short_memory):
        raise ConstraintHoloError("short-memory coordinates must lie in [0,1]")
    if not all(0.0 <= value <= cap for value in state.long_memory):
        raise ConstraintHoloError("long-memory coordinates must lie in [0,L]")
    for weights in state.clause_selector:
        if len(weights) != 3 or any(value < 0.0 for value in weights):
            raise ConstraintHoloError("clause selectors must be nonnegative triples")
        if abs(sum(weights) - 1.0) > tolerance:
            raise ConstraintHoloError("clause selector mass must equal one")
    circle_residual = max(
        (
            abs(cosine * cosine + sine * sine - 1.0)
            for cosine, sine in zip(
                state.phase_cosine,
                state.phase_sine,
                strict=True,
            )
        ),
        default=0.0,
    )
    if circle_residual > tolerance:
        raise ConstraintHoloError("proposed phase state leaves the declared circles")
    return cap


def _memory_location_matches(
    value: float,
    location: str,
    upper: float,
    tolerance: float,
) -> bool:
    if location == "ZERO":
        return abs(value) <= tolerance
    if location in {"ONE", "CAP"}:
        return abs(value - upper) <= tolerance
    if location == "INTERIOR":
        return tolerance < value < upper - tolerance
    raise AssertionError(f"unsupported memory location: {location}")


def _clause_defects(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
    clause_index: int,
) -> tuple[float, float, float]:
    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    clause = holo.clauses[clause_index]
    return tuple(
        1.0
        - (1.0 if literal.positive else -1.0)
        * state.phase_cosine[variable_index[literal.variable]]
        for literal in clause.literals
    )  # type: ignore[return-value]


def _clause_violation_and_derivative(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
    phase_cosine_derivative: tuple[float, ...],
    clause_index: int,
    truth_gain: float,
) -> tuple[float, float, tuple[float, float, float]]:
    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    clause = holo.clauses[clause_index]
    defects = _clause_defects(holo, state, clause_index)
    violation = truth_gain * defects[0] * defects[1] * defects[2] / 8.0
    derivative = 0.0
    for local_index, literal in enumerate(clause.literals):
        others = tuple(index for index in range(3) if index != local_index)
        sign = 1.0 if literal.positive else -1.0
        partial = (
            -truth_gain
            * sign
            * defects[others[0]]
            * defects[others[1]]
            / 8.0
        )
        derivative += partial * phase_cosine_derivative[
            variable_index[literal.variable]
        ]
    return violation, derivative, defects


def _relevant_memory_normals(
    descriptor: MemoryStratumDescriptor,
    short_zero: float,
    short_one: float,
    long_zero: float,
    long_cap: float,
    clause_index: int,
) -> tuple[NormalExponent, ...]:
    normals: list[NormalExponent] = []
    if descriptor.short_location == "ZERO":
        normals.append(NormalExponent(f"clause[{clause_index}].short_from_zero", short_zero))
    elif descriptor.short_location == "ONE":
        normals.append(NormalExponent(f"clause[{clause_index}].short_from_one", short_one))
    if descriptor.long_location == "ZERO":
        normals.append(NormalExponent(f"clause[{clause_index}].long_from_zero", long_zero))
    elif descriptor.long_location == "CAP":
        normals.append(NormalExponent(f"clause[{clause_index}].long_from_cap", long_cap))
    return tuple(normals)


def audit_boundary_stratum_state(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
    stratum_labels: tuple[str, ...],
    selector_supports: tuple[tuple[bool, bool, bool], ...],
    *,
    proposal_kind: str = "stationary_point",
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
    tolerance: float = 1.0e-10,
) -> BoundaryStratumEscapeAudit:
    """Audit one proposed public-seed-compatible boundary-stratum state.

    The audit certifies only pointwise equations. For ``moving_invariant_set`` it can
    establish pointwise tangency but never promotes that observation to global invariant
    closure. ``asymptotic_numerical_approach`` is always treated as non-certifying.
    """

    if proposal_kind not in PROPOSAL_KINDS:
        raise ConstraintHoloError(f"unsupported boundary proposal kind: {proposal_kind!r}")
    cap = _validate_state(holo, state, parameters, tolerance)
    if len(stratum_labels) != len(holo.clauses):
        raise ConstraintHoloError("memory stratum count must match public clauses")
    descriptors = tuple(
        _descriptor_by_label(label, parameters) for label in stratum_labels
    )
    _validate_supports(state, selector_supports, tolerance)

    derivative = reduced_exact_product_phase_derivative(
        holo,
        state,
        parameters=parameters,
        selector_rate=selector_rate,
        boundary_release_rate=boundary_release_rate,
        truth_gain=truth_gain,
    )
    derivative_values = (
        derivative.phase_cosine
        + derivative.phase_sine
        + derivative.short_memory
        + derivative.long_memory
        + tuple(value for weights in derivative.clause_selector for value in weights)
    )
    if not all(isfinite(value) for value in derivative_values):
        raise ConstraintHoloError("boundary-stratum field contains NaN or infinity")

    clause_audits: list[ClauseBoundaryStratumAudit] = []
    all_normals: list[NormalExponent] = []
    unresolved: list[str] = []
    for clause_index, (descriptor, support) in enumerate(
        zip(descriptors, selector_supports, strict=True)
    ):
        violation, violation_derivative, defects = _clause_violation_and_derivative(
            holo,
            state,
            derivative.phase_cosine,
            clause_index,
            truth_gain,
        )
        short_zero = parameters.beta * (violation - parameters.gamma)
        short_one = -short_zero
        long_zero = parameters.alpha * (violation - parameters.delta)
        long_cap = -long_zero
        short_matches = _memory_location_matches(
            state.short_memory[clause_index],
            descriptor.short_location,
            1.0,
            tolerance,
        )
        long_matches = _memory_location_matches(
            state.long_memory[clause_index],
            descriptor.long_location,
            cap,
            tolerance,
        )
        required_matches = (
            descriptor.required_clause_violation is None
            or abs(violation - descriptor.required_clause_violation) <= tolerance
        )
        memory_field_residual = max(
            abs(derivative.short_memory[clause_index]),
            abs(derivative.long_memory[clause_index]),
        )
        memory_tangency_residual = (
            abs(violation_derivative) if descriptor.has_interior_memory else 0.0
        )

        weights = state.clause_selector[clause_index]
        weighted_cost = sum(
            weight * cost for weight, cost in zip(weights, defects, strict=True)
        )
        selector_support_residual = max(
            (
                max(0.0, tolerance - weight)
                if included
                else abs(weight)
                for weight, included in zip(weights, support, strict=True)
            ),
            default=0.0,
        )
        selector_stationarity_residual = max(
            (
                abs(cost - weighted_cost)
                for cost, included in zip(defects, support, strict=True)
                if included
            ),
            default=0.0,
        )
        selector_field_residual = max(
            (abs(value) for value in derivative.clause_selector[clause_index]),
            default=0.0,
        )

        normals = list(
            _relevant_memory_normals(
                descriptor,
                short_zero,
                short_one,
                long_zero,
                long_cap,
                clause_index,
            )
        )
        for literal_index, (cost, included) in enumerate(
            zip(defects, support, strict=True)
        ):
            if not included:
                normals.append(
                    NormalExponent(
                        f"clause[{clause_index}].selector[{literal_index}]_from_face",
                        selector_rate * (weighted_cost - cost),
                    )
                )
        all_normals.extend(normals)

        pointwise_tangent = (
            short_matches
            and long_matches
            and required_matches
            and memory_field_residual <= tolerance
            and memory_tangency_residual <= tolerance
            and selector_support_residual <= tolerance
            and selector_stationarity_residual <= tolerance
            and selector_field_residual <= tolerance
        )
        if descriptor.has_interior_memory and pointwise_tangent:
            unresolved.append(
                f"clause[{clause_index}] higher Lie-derivative closure not certified"
            )
        if sum(support) > 1 and pointwise_tangent:
            unresolved.append(
                f"clause[{clause_index}] supported selector face contains center directions"
            )

        clause_audits.append(
            ClauseBoundaryStratumAudit(
                clause_index=clause_index,
                stratum=descriptor,
                selector_support=support,
                clause_violation=violation,
                clause_violation_derivative=violation_derivative,
                short_zero_exponent=short_zero,
                short_one_exponent=short_one,
                long_zero_exponent=long_zero,
                long_cap_exponent=long_cap,
                memory_coordinate_matches_stratum=short_matches and long_matches,
                required_violation_matches_stratum=required_matches,
                memory_field_residual=memory_field_residual,
                memory_tangency_residual=memory_tangency_residual,
                selector_support_residual=selector_support_residual,
                selector_stationarity_residual=selector_stationarity_residual,
                selector_field_residual=selector_field_residual,
                observed_normal_exponents=tuple(normals),
                pointwise_boundary_tangent=pointwise_tangent,
            )
        )

    circle_residual = max(
        (
            abs(cosine * cosine + sine * sine - 1.0)
            for cosine, sine in zip(
                state.phase_cosine,
                state.phase_sine,
                strict=True,
            )
        ),
        default=0.0,
    )
    phase_field_residual = max(
        (
            abs(value)
            for value in derivative.phase_cosine + derivative.phase_sine
        ),
        default=0.0,
    )
    pointwise_boundary_tangent = all(
        audit.pointwise_boundary_tangent for audit in clause_audits
    )
    stationary_verified = (
        proposal_kind == "stationary_point"
        and pointwise_boundary_tangent
        and phase_field_residual <= tolerance
    )
    moving_pointwise_candidate = (
        proposal_kind == "moving_invariant_set" and pointwise_boundary_tangent
    )
    numerical_only = proposal_kind == "asymptotic_numerical_approach"
    if moving_pointwise_candidate:
        unresolved.append("global moving-set closure is not certified by pointwise tangency")

    assignment = {
        variable: state.phase_cosine[index] > 0.0
        for index, variable in enumerate(holo.variables)
    }
    satisfying = holo.accepts(assignment)
    repelling = tuple(
        normal.coordinate for normal in all_normals if normal.value > tolerance
    )
    observed_normals_strictly_stable = bool(all_normals) and all(
        normal.value < -tolerance for normal in all_normals
    )
    center_present = (
        any(abs(normal.value) <= tolerance for normal in all_normals)
        or any(descriptor.has_interior_memory for descriptor in descriptors)
        or any(sum(support) > 1 for support in selector_supports)
    )

    candidate_invariant = (
        stationary_verified or moving_pointwise_candidate
    ) and not numerical_only
    if not candidate_invariant:
        classification = "NOT_INVARIANT"
    elif satisfying:
        classification = "SATISFYING_INVARIANT_SECTION"
    elif repelling:
        classification = "INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION"
    elif stationary_verified and observed_normals_strictly_stable and not center_present:
        classification = "NON_SOLUTION_ATTRACTOR_CANDIDATE"
        unresolved.append("phase Jacobian and nonlinear basin remain uncertified")
    else:
        classification = "INVARIANT_STABLE_OR_CENTER_UNRESOLVED"

    if classification not in CLASSIFICATIONS:
        raise AssertionError("unexpected boundary-stratum classification")
    status = {
        "NOT_INVARIANT": "BOUNDARY_STRATUM_PROPOSAL_NOT_INVARIANT",
        "INVARIANT_REPELLING_IN_AT_LEAST_ONE_PUBLIC_DIRECTION": (
            "BOUNDARY_STRATUM_POINTWISE_INVARIANT__PUBLIC_REPELLING_DIRECTION_FOUND"
        ),
        "INVARIANT_STABLE_OR_CENTER_UNRESOLVED": (
            "BOUNDARY_STRATUM_POINTWISE_INVARIANT__STABILITY_OR_CENTER_UNRESOLVED"
        ),
        "NON_SOLUTION_ATTRACTOR_CANDIDATE": (
            "NON_SOLUTION_ATTRACTOR_CANDIDATE__FULL_STABILITY_NOT_CERTIFIED"
        ),
        "SATISFYING_INVARIANT_SECTION": "SATISFYING_INVARIANT_SECTION_VERIFIED",
    }[classification]

    return BoundaryStratumEscapeAudit(
        proposal_kind=proposal_kind,
        classification=classification,
        public_variables=len(holo.variables),
        public_clauses=len(holo.clauses),
        phase_circle_residual=circle_residual,
        phase_field_residual=phase_field_residual,
        maximum_memory_field_residual=max(
            (audit.memory_field_residual for audit in clause_audits),
            default=0.0,
        ),
        maximum_memory_tangency_residual=max(
            (audit.memory_tangency_residual for audit in clause_audits),
            default=0.0,
        ),
        maximum_selector_stationarity_residual=max(
            (audit.selector_stationarity_residual for audit in clause_audits),
            default=0.0,
        ),
        maximum_selector_field_residual=max(
            (audit.selector_field_residual for audit in clause_audits),
            default=0.0,
        ),
        pointwise_boundary_tangency_verified=pointwise_boundary_tangent,
        stationary_point_verified=stationary_verified,
        moving_invariant_set_globally_verified=False,
        numerical_approach_only=numerical_only,
        terminal_assignment_satisfies_public_relation=satisfying,
        repelling_directions=repelling,
        unresolved_conditions=tuple(dict.fromkeys(unresolved)),
        clause_audits=tuple(clause_audits),
        status=status,
    )
