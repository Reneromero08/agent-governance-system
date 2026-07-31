from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .catalytic_existential_trace import reference_existential_trace
from .constraint_holo import ClauseRelation, ConstraintHolo, ConstraintHoloError, Literal


SEALED_PHASE_TRANSITION_CASES = (
    (
        12,
        51,
        0,
        "SAT",
        "5c43f19a094aafc9ec0867652cd8a487b9400abdb0e857708ae5189fbc1fe96f",
    ),
    (
        12,
        51,
        12,
        "UNSAT",
        "5dc4f43dadfbd2bba8c407f8f88ae73c4f6594ad10aeb667a3e0a6ac501b0de3",
    ),
)


@dataclass(frozen=True)
class SeededPhaseTransitionCase:
    variable_count: int
    clause_count: int
    seed: int
    expected_status: str
    semantic_digest: str
    witness_count_reference_only: int
    holo: ConstraintHolo


def deterministic_phase_transition_holo(
    variable_count: int,
    clause_count: int,
    seed: int,
) -> ConstraintHolo:
    """Compile one deterministic random-looking exact 3-CNF.

    The native carrier receives only the resulting public clauses. Seed, expected label,
    witness, and reference count remain external corpus instrumentation.
    """

    if variable_count < 3:
        raise ConstraintHoloError("phase-transition corpus requires at least 3 variables")
    if clause_count < 1:
        raise ConstraintHoloError("phase-transition corpus requires at least one clause")
    variables = tuple(f"x{index}" for index in range(variable_count))
    generator = Random(seed)
    clauses: list[ClauseRelation] = []
    seen: set[tuple[str, str, str]] = set()
    attempts = 0
    while len(clauses) < clause_count:
        attempts += 1
        if attempts > 100 * clause_count:
            raise ConstraintHoloError("failed to construct unique deterministic clauses")
        chosen = generator.sample(variables, 3)
        literals = tuple(
            Literal(variable, bool(generator.getrandbits(1)))
            for variable in chosen
        )
        clause = ClauseRelation(literals)  # type: ignore[arg-type]
        token = clause.canonical_token()
        if token in seen:
            continue
        seen.add(token)
        clauses.append(clause)
    return ConstraintHolo.build(variables, clauses)


def certify_phase_transition_case(
    variable_count: int,
    clause_count: int,
    seed: int,
) -> SeededPhaseTransitionCase:
    holo = deterministic_phase_transition_holo(variable_count, clause_count, seed)
    reference = reference_existential_trace(holo)
    return SeededPhaseTransitionCase(
        variable_count=variable_count,
        clause_count=clause_count,
        seed=seed,
        expected_status="SAT" if reference.satisfiable else "UNSAT",
        semantic_digest=holo.semantic_digest(),
        witness_count_reference_only=reference.witness_count,
        holo=holo,
    )


def sealed_phase_transition_cases() -> tuple[SeededPhaseTransitionCase, ...]:
    cases: list[SeededPhaseTransitionCase] = []
    for variable_count, clause_count, seed, expected_status, digest in (
        SEALED_PHASE_TRANSITION_CASES
    ):
        case = certify_phase_transition_case(variable_count, clause_count, seed)
        if case.semantic_digest != digest:
            raise ConstraintHoloError("sealed phase-transition semantic digest mismatch")
        if case.expected_status != expected_status:
            raise ConstraintHoloError("sealed phase-transition label mismatch")
        cases.append(case)
    return tuple(cases)
