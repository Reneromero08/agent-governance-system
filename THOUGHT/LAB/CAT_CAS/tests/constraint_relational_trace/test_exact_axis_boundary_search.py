from __future__ import annotations

from pathlib import Path
import sys

import pytest

PACKAGE_PARENT = (
    Path(__file__).resolve().parents[2]
    / "7_decoder"
    / "50_phase_bm_cpu"
    / "50_6_fixed_point_substrate"
    / "14_noncollapse_frontier"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from constraint_relational_trace_v1.constraint_holo import (  # noqa: E402
    ClauseRelation,
    ConstraintHolo,
    ConstraintHoloError,
    Literal,
)
from constraint_relational_trace_v1.exact_axis_boundary_search import (  # noqa: E402
    search_exact_axis_boundary_candidates,
)


def clause(*literals: Literal) -> ClauseRelation:
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def symmetric_stationary_holo() -> ConstraintHolo:
    return ConstraintHolo.build(
        ("x", "y"),
        (
            clause(Literal("x"), Literal("y"), Literal("y")),
            clause(
                Literal("x", False),
                Literal("y", False),
                Literal("y", False),
            ),
        ),
    )


def test_exact_axis_search_rediscovers_only_seed_terminal_symmetric_obstruction() -> None:
    holo = symmetric_stationary_holo()
    result = search_exact_axis_boundary_candidates((holo,))

    assert result.input_formulae == 1
    assert result.semantic_unique_formulae == 1
    assert result.satisfiable_formulae_searched == 1
    assert result.formulae_public_seed_already_witness == 1
    assert result.axis_states_audited == 16
    assert result.non_solution_stationary_candidates > 0
    assert result.seed_relevant_non_solution_candidates == 0
    assert result.status == (
        "EXACT_AXIS_BOUNDARY_SEARCH_FOUND_ONLY_PUBLIC_SEED_TERMINAL_OBSTRUCTIONS"
    )

    symmetric = tuple(
        candidate
        for candidate in result.candidates
        if candidate.axis_labels in {("UP", "UP"), ("DOWN", "DOWN")}
        and candidate.selector_supports
        == ((True, True, True), (True, True, True))
    )
    assert symmetric
    for candidate in symmetric:
        assert candidate.semantic_digest == holo.semantic_digest()
        assert candidate.presentation_digest == holo.presentation_digest()
        assert candidate.reference_witness_count == 2
        assert candidate.stratum_labels == (
            "SHORT_ONE__LONG_CAP__C_ARBITRARY",
            "SHORT_ONE__LONG_CAP__C_ARBITRARY",
        )
        assert candidate.classification == "INVARIANT_STABLE_OR_CENTER_UNRESOLVED"
        assert candidate.phase_field_residual == 0.0
        assert candidate.maximum_memory_field_residual == 0.0
        assert candidate.maximum_selector_field_residual == 0.0
        assert candidate.public_seed_distance_squared > 0.0
        assert candidate.public_seed_is_terminal_witness
        assert not candidate.seed_relevant_obstruction_candidate
        assert dict(candidate.public_seed_threshold_assignment) == {
            "x": True,
            "y": False,
        }
        assert dict(candidate.threshold_assignment) == {"x": False, "y": False}


def test_wrong_public_seed_formula_without_axis_stationary_candidate_stays_clear() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(
                Literal("x", False),
                Literal("x", False),
                Literal("x", False),
            ),
        ),
    )
    result = search_exact_axis_boundary_candidates((holo,))

    assert result.satisfiable_formulae_searched == 1
    assert result.formulae_public_seed_already_witness == 0
    assert result.non_solution_stationary_candidates == 0
    assert result.seed_relevant_non_solution_candidates == 0
    assert result.status == (
        "EXACT_AXIS_BOUNDARY_SEARCH_FOUND_NO_CANDIDATE_ON_CAPPED_SURFACE"
    )


def test_search_semantically_deduplicates_duplicate_clause_presentations() -> None:
    holo = symmetric_stationary_holo()
    duplicated = holo.with_duplicate_clause(0)
    assert holo.semantic_digest() == duplicated.semantic_digest()
    assert holo.presentation_digest() != duplicated.presentation_digest()

    result = search_exact_axis_boundary_candidates((holo, duplicated))
    assert result.input_formulae == 2
    assert result.semantic_unique_formulae == 1
    assert result.satisfiable_formulae_searched == 1
    assert all(
        candidate.presentation_digest == holo.presentation_digest()
        for candidate in result.candidates
    )


def test_unsat_formula_is_reference_certified_and_not_searched_for_sat_obstruction() -> None:
    holo = ConstraintHolo.build(
        ("x",),
        (
            clause(Literal("x"), Literal("x"), Literal("x")),
            clause(
                Literal("x", False),
                Literal("x", False),
                Literal("x", False),
            ),
        ),
    )
    result = search_exact_axis_boundary_candidates((holo,))
    assert result.satisfiable_formulae_searched == 0
    assert result.formulae_public_seed_already_witness == 0
    assert result.axis_states_audited == 0
    assert result.non_solution_stationary_candidates == 0
    assert result.status == (
        "EXACT_AXIS_BOUNDARY_SEARCH_FOUND_NO_CANDIDATE_ON_CAPPED_SURFACE"
    )


def test_selector_combination_cap_reports_truncation() -> None:
    result = search_exact_axis_boundary_candidates(
        (symmetric_stationary_holo(),),
        selector_combination_limit=1,
    )
    assert result.truncated
    assert result.selector_combinations_audited <= result.axis_states_audited


def test_search_rejects_formula_above_frozen_variable_cap() -> None:
    holo = ConstraintHolo.build(tuple(f"x{index}" for index in range(5)), ())
    with pytest.raises(ConstraintHoloError, match="variable cap"):
        search_exact_axis_boundary_candidates((holo,))


def test_search_rejects_attempt_to_raise_frozen_cap() -> None:
    with pytest.raises(ConstraintHoloError, match="frozen at four"):
        search_exact_axis_boundary_candidates((), variable_limit=5)
