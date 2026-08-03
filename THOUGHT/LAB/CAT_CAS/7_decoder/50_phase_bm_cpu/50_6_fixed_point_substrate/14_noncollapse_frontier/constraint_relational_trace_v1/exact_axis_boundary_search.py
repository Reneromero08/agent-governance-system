from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Iterable

from .boundary_stratum_escape_audit import audit_boundary_stratum_state
from .catalytic_existential_trace import CLAIM_CEILING, reference_existential_trace
from .constraint_holo import ConstraintHolo, ConstraintHoloError
from .reduced_exact_product_phase_flow import (
    ReducedExactProductPhaseState,
    public_reduced_exact_product_state,
)
from .self_organizing_clause_flow import SelfOrganizingFlowParameters


AXIS_PHASES = (
    ("FALSE", -1.0, 0.0),
    ("UP", 0.0, 1.0),
    ("DOWN", 0.0, -1.0),
    ("TRUE", 1.0, 0.0),
)


@dataclass(frozen=True)
class ExactAxisStationaryCandidate:
    semantic_digest: str
    presentation_digest: str
    clause_tokens: tuple[tuple[str, str, str], ...]
    reference_witness_count: int
    axis_labels: tuple[str, ...]
    phase_cosine: tuple[float, ...]
    phase_sine: tuple[float, ...]
    stratum_labels: tuple[str, ...]
    selector_supports: tuple[tuple[bool, bool, bool], ...]
    short_memory: tuple[float, ...]
    long_memory: tuple[float, ...]
    classification: str
    phase_field_residual: float
    maximum_memory_field_residual: float
    maximum_selector_field_residual: float
    public_seed_distance_squared: float
    public_seed_threshold_assignment: tuple[tuple[str, bool], ...]
    public_seed_is_terminal_witness: bool
    seed_relevant_obstruction_candidate: bool
    threshold_assignment: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class ExactAxisBoundarySearchResult:
    input_formulae: int
    semantic_unique_formulae: int
    satisfiable_formulae_searched: int
    formulae_public_seed_already_witness: int
    axis_states_audited: int
    selector_combinations_audited: int
    non_solution_stationary_candidates: int
    seed_relevant_non_solution_candidates: int
    truncated: bool
    candidates: tuple[ExactAxisStationaryCandidate, ...]
    status: str
    claim_ceiling: str = CLAIM_CEILING


def _clause_defects(
    holo: ConstraintHolo,
    phase_cosine: tuple[float, ...],
    clause_index: int,
) -> tuple[float, float, float]:
    variable_index = {variable: index for index, variable in enumerate(holo.variables)}
    return tuple(
        1.0
        - (1.0 if literal.positive else -1.0)
        * phase_cosine[variable_index[literal.variable]]
        for literal in holo.clauses[clause_index].literals
    )  # type: ignore[return-value]


def _clause_violation(
    defects: tuple[float, float, float],
    truth_gain: float,
) -> float:
    return truth_gain * defects[0] * defects[1] * defects[2] / 8.0


def _memory_state_from_violation(
    violation: float,
    cap: float,
    parameters: SelfOrganizingFlowParameters,
    tolerance: float,
) -> tuple[str, float, float]:
    if violation < parameters.delta - tolerance:
        return "SHORT_ZERO__LONG_ZERO__C_ARBITRARY", 0.0, 0.0
    if abs(violation - parameters.delta) <= tolerance:
        return "SHORT_ZERO__LONG_INTERIOR_C_EQ_DELTA", 0.0, cap / 2.0
    if violation < parameters.gamma - tolerance:
        return "SHORT_ZERO__LONG_CAP__C_ARBITRARY", 0.0, cap
    if abs(violation - parameters.gamma) <= tolerance:
        return "SHORT_INTERIOR_C_EQ_GAMMA__LONG_CAP", 0.5, cap
    return "SHORT_ONE__LONG_CAP__C_ARBITRARY", 1.0, cap


def _selector_face_options(
    defects: tuple[float, float, float],
    tolerance: float,
) -> tuple[
    tuple[tuple[bool, bool, bool], tuple[float, float, float]], ...
]:
    """Generate deterministic uniform selector faces with equal supported costs."""

    options: list[
        tuple[tuple[bool, bool, bool], tuple[float, float, float]]
    ] = []
    for mask in range(1, 1 << 3):
        support = tuple(bool((mask >> index) & 1) for index in range(3))
        costs = tuple(
            defects[index] for index, included in enumerate(support) if included
        )
        if max(costs) - min(costs) > tolerance:
            continue
        mass = float(len(costs))
        weights = tuple(1.0 / mass if included else 0.0 for included in support)
        options.append((support, weights))  # type: ignore[arg-type]
    return tuple(options)


def _public_seed_distance_squared(
    public_state: ReducedExactProductPhaseState,
    candidate: ReducedExactProductPhaseState,
) -> float:
    values = (
        tuple(
            left - right
            for left, right in zip(
                public_state.phase_cosine,
                candidate.phase_cosine,
                strict=True,
            )
        )
        + tuple(
            left - right
            for left, right in zip(
                public_state.phase_sine,
                candidate.phase_sine,
                strict=True,
            )
        )
        + tuple(
            left - right
            for left, right in zip(
                public_state.short_memory,
                candidate.short_memory,
                strict=True,
            )
        )
        + tuple(
            left - right
            for left, right in zip(
                public_state.long_memory,
                candidate.long_memory,
                strict=True,
            )
        )
        + tuple(
            left - right
            for left_weights, right_weights in zip(
                public_state.clause_selector,
                candidate.clause_selector,
                strict=True,
            )
            for left, right in zip(left_weights, right_weights, strict=True)
        )
    )
    return sum(value * value for value in values)


def _threshold_assignment(
    holo: ConstraintHolo,
    state: ReducedExactProductPhaseState,
) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (variable, state.phase_cosine[index] > 0.0)
        for index, variable in enumerate(holo.variables)
    )


def search_exact_axis_boundary_candidates(
    formulae: Iterable[ConstraintHolo],
    *,
    variable_limit: int = 4,
    clause_limit: int = 6,
    formula_limit: int = 128,
    selector_combination_limit: int = 4096,
    parameters: SelfOrganizingFlowParameters = SelfOrganizingFlowParameters(),
    selector_rate: float = 20.0,
    boundary_release_rate: float = 10.0,
    truth_gain: float = 4.0,
    tolerance: float = 1.0e-10,
) -> ExactAxisBoundarySearchResult:
    """Search a capped exact axis surface for non-solution stationary candidates.

    The search is reference-only and incomplete. It uses the four exact phase-axis points
    per variable and uniform selector faces supported on equal literal-defect costs.
    Memory classes are derived from each exact clause violation; no ``5**m`` memory-label
    enumeration is performed.

    A stationary state is marked seed-relevant only when the declared public-seed
    threshold assignment does not already satisfy the same presented formula.
    """

    integer_limits = (
        variable_limit,
        clause_limit,
        formula_limit,
        selector_combination_limit,
    )
    if any(not isinstance(value, int) or value < 1 for value in integer_limits):
        raise ConstraintHoloError("exact axis search limits must be positive integers")
    controls = (selector_rate, boundary_release_rate, truth_gain, tolerance)
    if not all(isfinite(value) and value > 0.0 for value in controls):
        raise ConstraintHoloError("exact axis search controls must be positive and finite")
    if variable_limit > 4:
        raise ConstraintHoloError("exact axis search variable limit is frozen at four")

    supplied = tuple(formulae)
    unique: dict[str, ConstraintHolo] = {}
    for holo in supplied:
        if len(holo.variables) > variable_limit:
            raise ConstraintHoloError("exact axis search formula exceeds variable cap")
        if len(holo.clauses) > clause_limit:
            raise ConstraintHoloError("exact axis search formula exceeds clause cap")
        unique.setdefault(holo.semantic_digest(), holo)

    truncated = len(unique) > formula_limit
    selected = tuple(unique[key] for key in sorted(unique)[:formula_limit])
    satisfiable_searched = 0
    public_seed_witness_formulae = 0
    axis_states_audited = 0
    selector_combinations_audited = 0
    candidates: list[ExactAxisStationaryCandidate] = []

    for holo in selected:
        reference = reference_existential_trace(
            holo,
            variable_limit=variable_limit,
        )
        if not reference.satisfiable:
            continue
        satisfiable_searched += 1
        cap = max(
            1.0,
            parameters.long_memory_cap_factor * max(1, len(holo.clauses)),
        )
        public_state = public_reduced_exact_product_state(holo)
        public_assignment = _threshold_assignment(holo, public_state)
        public_seed_is_witness = holo.accepts(dict(public_assignment))
        if public_seed_is_witness:
            public_seed_witness_formulae += 1

        for phase_choice in product(AXIS_PHASES, repeat=len(holo.variables)):
            axis_states_audited += 1
            axis_labels = tuple(item[0] for item in phase_choice)
            phase_cosine = tuple(item[1] for item in phase_choice)
            phase_sine = tuple(item[2] for item in phase_choice)

            stratum_labels: list[str] = []
            short_memory: list[float] = []
            long_memory: list[float] = []
            selector_options: list[
                tuple[
                    tuple[tuple[bool, bool, bool], tuple[float, float, float]],
                    ...,
                ]
            ] = []
            for clause_index in range(len(holo.clauses)):
                defects = _clause_defects(holo, phase_cosine, clause_index)
                violation = _clause_violation(defects, truth_gain)
                label, short_value, long_value = _memory_state_from_violation(
                    violation,
                    cap,
                    parameters,
                    tolerance,
                )
                stratum_labels.append(label)
                short_memory.append(short_value)
                long_memory.append(long_value)
                options = _selector_face_options(defects, tolerance)
                if not options:
                    raise AssertionError("every finite defect triple has a selector face")
                selector_options.append(options)

            combination_iterator = product(*selector_options)
            for combination_index, selector_combination in enumerate(
                combination_iterator
            ):
                if combination_index >= selector_combination_limit:
                    truncated = True
                    break
                selector_combinations_audited += 1
                supports = tuple(item[0] for item in selector_combination)
                weights = tuple(item[1] for item in selector_combination)
                state = ReducedExactProductPhaseState(
                    phase_cosine=phase_cosine,
                    phase_sine=phase_sine,
                    short_memory=tuple(short_memory),
                    long_memory=tuple(long_memory),
                    clause_selector=weights,
                )
                audit = audit_boundary_stratum_state(
                    holo,
                    state,
                    tuple(stratum_labels),
                    supports,
                    proposal_kind="stationary_point",
                    parameters=parameters,
                    selector_rate=selector_rate,
                    boundary_release_rate=boundary_release_rate,
                    truth_gain=truth_gain,
                    tolerance=tolerance,
                )
                if (
                    not audit.stationary_point_verified
                    or audit.terminal_assignment_satisfies_public_relation
                ):
                    continue
                assignment = _threshold_assignment(holo, state)
                candidates.append(
                    ExactAxisStationaryCandidate(
                        semantic_digest=holo.semantic_digest(),
                        presentation_digest=holo.presentation_digest(),
                        clause_tokens=tuple(
                            clause.canonical_token() for clause in holo.clauses
                        ),
                        reference_witness_count=reference.witness_count,
                        axis_labels=axis_labels,
                        phase_cosine=phase_cosine,
                        phase_sine=phase_sine,
                        stratum_labels=tuple(stratum_labels),
                        selector_supports=supports,
                        short_memory=tuple(short_memory),
                        long_memory=tuple(long_memory),
                        classification=audit.classification,
                        phase_field_residual=audit.phase_field_residual,
                        maximum_memory_field_residual=(
                            audit.maximum_memory_field_residual
                        ),
                        maximum_selector_field_residual=(
                            audit.maximum_selector_field_residual
                        ),
                        public_seed_distance_squared=_public_seed_distance_squared(
                            public_state,
                            state,
                        ),
                        public_seed_threshold_assignment=public_assignment,
                        public_seed_is_terminal_witness=public_seed_is_witness,
                        seed_relevant_obstruction_candidate=(
                            not public_seed_is_witness
                        ),
                        threshold_assignment=assignment,
                    )
                )

    candidates.sort(
        key=lambda item: (
            item.semantic_digest,
            item.axis_labels,
            item.stratum_labels,
            item.selector_supports,
        )
    )
    seed_relevant = sum(
        candidate.seed_relevant_obstruction_candidate for candidate in candidates
    )
    if seed_relevant:
        status = "EXACT_AXIS_BOUNDARY_SEARCH_FOUND_SEED_RELEVANT_STATIONARY_CANDIDATES"
    elif candidates:
        status = (
            "EXACT_AXIS_BOUNDARY_SEARCH_FOUND_ONLY_PUBLIC_SEED_TERMINAL_OBSTRUCTIONS"
        )
    else:
        status = "EXACT_AXIS_BOUNDARY_SEARCH_FOUND_NO_CANDIDATE_ON_CAPPED_SURFACE"
    return ExactAxisBoundarySearchResult(
        input_formulae=len(supplied),
        semantic_unique_formulae=len(unique),
        satisfiable_formulae_searched=satisfiable_searched,
        formulae_public_seed_already_witness=public_seed_witness_formulae,
        axis_states_audited=axis_states_audited,
        selector_combinations_audited=selector_combinations_audited,
        non_solution_stationary_candidates=len(candidates),
        seed_relevant_non_solution_candidates=seed_relevant,
        truncated=truncated,
        candidates=tuple(candidates),
        status=status,
    )
